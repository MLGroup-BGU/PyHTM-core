/* ---------------------------------------------------------------------------
 * Pyramid-Model-Core :: bindings/py_PyramidRuntime.cpp
 *
 * The Python face of the C++ pyramid: `htm.bindings.pyramid_engine`.
 *
 *   PyramidEngine(spec: dict)      -- parse the instruction book
 *     .build(reader_factory=None)  -- build encoders + nodes (+sample pass);
 *                                     None -> native CSV from spec['data']
 *     .run(iterations=None, progress_cb=None, progress_every=5000)
 *     .results() / .summaries() / .head_name / .node_names /
 *     .records_processed / .n_workers
 *
 * GIL POLICY (the whole point of this module): build() and run() release
 * the GIL for their entire duration.  The ONLY re-acquisitions are
 *   * PyReaderSource pulling one batch of rows (a few small buffer copies
 *     every `batch_rows` records), and
 *   * the optional progress callback (every `progress_every` records).
 * All stepping, merging and threading in between runs GIL-free, which is
 * what lets the worker pool actually use the machine.
 *
 * DATA INGESTION DESIGN: the Python DatasetReader helper (pandas/pyarrow)
 * feeds batches through PyReaderSource.  A C++ Arrow/Parquet reader was
 * evaluated and rejected -- Arrow is a heavyweight dependency that would
 * multiply the Windows build time and risk; the per-batch GIL cost here is
 * a memcpy every few thousand records (well under 0.1% of a run).  Plain
 * CSVs can bypass Python entirely via the native CsvSource.
 * ------------------------------------------------------------------------ */
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <htm/utils/Log.hpp>

#include <memory>
#include <optional>

#include "../src/pyramid/data/RecordSource.hpp"
#include "pyramid/util/StageLog.hpp"
#include "../src/pyramid/runtime/PyramidRuntime.hpp"

namespace py = pybind11;
using namespace pyramid;

/* ======================= spec dict -> PyramidSpec ======================== */
namespace {

template <typename T>
T get_or(const py::dict &d, const char *k, T fallback) {
    if (!d.contains(k) || d[k].is_none()) return fallback;
    return d[k].cast<T>();
}

template <typename T>
std::optional<T> get_opt(const py::dict &d, const char *k) {
    if (!d.contains(k) || d[k].is_none()) return std::nullopt;
    return d[k].cast<T>();
}

MergeParams parse_merge_params(const py::object &obj) {
    /* The Python side sends the FULLY-OVERLAID dict (DEFAULT_MERGE_PARAMS
     * updated with any user params), so every key present here simply
     * overrides the identical C++ default. */
    MergeParams p;
    if (obj.is_none()) return p;
    const py::dict d = obj.cast<py::dict>();
    p.target_sparsity = get_or<double>(d, "target_sparsity", p.target_sparsity);
    if (d.contains("seed") && !d["seed"].is_none()) {
        p.has_seed = true;
        p.seed = d["seed"].cast<std::uint64_t>();
    }
    p.temperature = get_or<double>(d, "temperature", p.temperature);
    p.alpha = get_or<double>(d, "alpha", p.alpha);
    p.smoothing = get_or<double>(d, "smoothing", p.smoothing);
    p.inhibition_radius =
        get_or<std::int64_t>(d, "inhibition_radius", p.inhibition_radius);
    p.inhibition_strength =
        get_or<double>(d, "inhibition_strength", p.inhibition_strength);
    p.burst_bonus = get_or<double>(d, "burst_bonus", p.burst_bonus);
    p.novelty_weight = get_or<double>(d, "novelty_weight", p.novelty_weight);
    p.segment_size = get_or<std::int64_t>(d, "segment_size", p.segment_size);
    p.segment_threshold =
        get_or<double>(d, "segment_threshold", p.segment_threshold);
    return p;
}

DateAttr parse_date_attr(const py::object &v) {
    DateAttr a;
    if (py::isinstance<py::tuple>(v) || py::isinstance<py::list>(v)) {
        auto seq = v.cast<std::vector<double>>();
        a.width = static_cast<std::int64_t>(seq.at(0));
        if (seq.size() > 1) { a.radius = seq.at(1); a.has_radius = true; }
    } else {
        a.width = v.cast<std::int64_t>();
    }
    return a;
}

FeatureKind parse_kind(const std::string &k) {
    if (k == "rdse") return FeatureKind::Rdse;
    if (k == "adaptive_rdse") return FeatureKind::AdaptiveRdse;
    if (k == "hybrid_rdse") return FeatureKind::HybridRdse;
    if (k == "dual_scalar") return FeatureKind::DualScalar;
    if (k == "linear") return FeatureKind::Linear;
    if (k == "categorical") return FeatureKind::Categorical;
    if (k == "date") return FeatureKind::Date;
    throw std::invalid_argument("unknown encoder kind: " + k);
}

FeatureSpec parse_feature(const py::dict &d) {
    FeatureSpec s;
    s.name = d["name"].cast<std::string>();
    s.kind = parse_kind(d["kind"].cast<std::string>());
    s.size = get_or<std::int64_t>(d, "size", 0);
    s.activeBits = get_or<std::int64_t>(d, "activeBits", 0);
    s.seed = get_or<std::uint64_t>(d, "seed", 0);
    if (d.contains("lin_min") && !d["lin_min"].is_none())
        s.lin_min = d["lin_min"].cast<double>();
    if (d.contains("lin_max") && !d["lin_max"].is_none())
        s.lin_max = d["lin_max"].cast<double>();
    s.lin_p_low = get_or<double>(d, "lin_p_low", s.lin_p_low);
    s.lin_p_high = get_or<double>(d, "lin_p_high", s.lin_p_high);
    if (d.contains("shape") && !d["shape"].is_none())
        s.shape = d["shape"].cast<std::vector<htm::UInt>>();
    if (d.contains("resolution") && !d["resolution"].is_none())
        s.resolution = d["resolution"].cast<double>();
    s.resolution_type =
        get_or<std::string>(d, "resolution_type", s.resolution_type);
    s.dedn = get_or<double>(d, "dedn", s.dedn);
    s.dedw = get_or<double>(d, "dedw", s.dedw);
    s.adaptive_n_quantiles = get_or<std::int64_t>(d, "adaptive_n_quantiles",
                                                  s.adaptive_n_quantiles);
    s.adaptive_index_scale =
        get_or<double>(d, "adaptive_index_scale", s.adaptive_index_scale);
    s.adaptive_gamma = get_or<double>(d, "adaptive_gamma", s.adaptive_gamma);
    s.adaptive_min_resolution = get_or<double>(d, "adaptive_min_resolution",
                                               s.adaptive_min_resolution);
    s.adaptive_clip_input =
        get_or<bool>(d, "adaptive_clip_input", s.adaptive_clip_input);
    s.adaptive_reserve =
        get_or<bool>(d, "adaptive_reserve", s.adaptive_reserve);
    s.hybrid_ratio = get_or<double>(d, "hybrid_ratio", s.hybrid_ratio);
    if (d.contains("dual") && !d["dual"].is_none()) {
        const py::dict dd = d["dual"].cast<py::dict>();
        s.dual.fast_n_bits = get_opt<std::int64_t>(dd, "fast_n_bits");
        s.dual.fast_w = get_opt<std::int64_t>(dd, "fast_w");
        s.dual.slow_n_bits = get_opt<std::int64_t>(dd, "slow_n_bits");
        s.dual.slow_w = get_opt<std::int64_t>(dd, "slow_w");
        s.dual.n_quantiles = get_opt<std::int64_t>(dd, "n_quantiles");
        s.dual.fast_warmup = get_opt<std::int64_t>(dd, "fast_warmup");
        s.dual.fast_alpha = get_opt<double>(dd, "fast_alpha");
        s.dual.fast_z_clip = get_opt<double>(dd, "fast_z_clip");
        s.dual.clip_input = get_opt<bool>(dd, "clip_input");
        s.dual.reserve_for_special_values =
            get_opt<bool>(dd, "reserve_for_special_values");
    }
    if (d.contains("date") && !d["date"].is_none()) {
        const py::dict dc = d["date"].cast<py::dict>();
        if (dc.contains("season")) s.date.season = parse_date_attr(dc["season"]);
        if (dc.contains("dayOfWeek"))
            s.date.dayOfWeek = parse_date_attr(dc["dayOfWeek"]);
        if (dc.contains("timeOfDay"))
            s.date.timeOfDay = parse_date_attr(dc["timeOfDay"]);
        if (dc.contains("weekend"))
            s.date.weekend = dc["weekend"].cast<std::int64_t>();
        if (dc.contains("holiday"))
            s.date.holiday = dc["holiday"].cast<std::int64_t>();
        if (dc.contains("holidays"))
            s.date.holidays =
                dc["holidays"].cast<std::vector<std::vector<int>>>();
    }
    return s;
}

SpSpec parse_sp(const py::dict &d) {
    SpSpec s;
    s.columnDimensions = d["columnDimensions"].cast<std::vector<htm::UInt>>();
    s.potentialPct = d["potentialPct"].cast<double>();
    s.potentialRadius = d["potentialRadius"].cast<std::int64_t>();
    s.globalInhibition = d["globalInhibition"].cast<bool>();
    s.synPermInactiveDec = d["synPermInactiveDec"].cast<double>();
    s.stimulusThreshold = d["stimulusThreshold"].cast<double>();
    s.synPermActiveInc = d["synPermActiveInc"].cast<double>();
    s.synPermConnected = d["synPermConnected"].cast<double>();
    s.boostStrength = d["boostStrength"].cast<double>();
    s.localAreaDensity = d["localAreaDensity"].cast<double>();
    s.wrapAround = d["wrapAround"].cast<bool>();
    return s;
}

TmSpec parse_tm(const py::dict &d) {
    TmSpec t;
    t.cellsPerColumn = d["cellsPerColumn"].cast<std::int64_t>();
    t.activationThreshold = d["activationThreshold"].cast<std::int64_t>();
    t.initialPerm = d["initialPerm"].cast<double>();
    t.permanenceConnected = d["permanenceConnected"].cast<double>();
    t.minThreshold = d["minThreshold"].cast<std::int64_t>();
    t.newSynapseCount = d["newSynapseCount"].cast<std::int64_t>();
    t.permanenceInc = d["permanenceInc"].cast<double>();
    t.permanenceDec = d["permanenceDec"].cast<double>();
    t.predictedSegmentDecrement =
        d["predictedSegmentDecrement"].cast<double>();
    t.maxSegmentsPerCell = d["maxSegmentsPerCell"].cast<std::int64_t>();
    t.maxSynapsesPerSegment = d["maxSynapsesPerSegment"].cast<std::int64_t>();
    return t;
}

NodeSpec parse_node(const std::string &name, const py::dict &d) {
    NodeSpec n;
    n.name = name;
    n.seed = d["seed"].cast<std::uint64_t>();
    n.has_sp = d["has_sp"].cast<bool>();
    if (n.has_sp) n.sp = parse_sp(d["sp"].cast<py::dict>());
    n.tm = parse_tm(d["tm"].cast<py::dict>());
    n.input_dims = d["input_dims"].cast<std::vector<htm::UInt>>();
    n.max_pool = get_or<std::int64_t>(d, "max_pool", 1);
    n.flatten = get_or<bool>(d, "flatten", true);
    n.learn_schedule = get_opt<std::int64_t>(d, "learn_schedule");
    n.anomaly_score = get_or<bool>(d, "anomaly_score", true);
    n.input_predictive = get_or<bool>(d, "input_predictive", false);
    n.return_predictive = get_or<bool>(d, "return_predictive", false);
    n.anomaly_detector = get_or<bool>(d, "anomaly_detector", false);
    n.residual_mode = static_cast<int>(get_or<std::int64_t>(d, "residual_mode", 1));
    n.residual_strength = get_or<double>(d, "residual_strength", 1.0);
    n.summary = get_or<std::string>(d, "summary", "");
    return n;
}

PyramidSpec parse_spec(const py::dict &d) {
    PyramidSpec s;
    s.seed = d["seed"].cast<std::uint64_t>();
    s.layers = d["layers"].cast<std::vector<std::vector<std::string>>>();
    s.children = d["children"]
                     .cast<std::map<std::string, std::vector<std::string>>>();
    s.feature_plan =
        d["feature_plan"]
            .cast<std::map<std::string, std::vector<std::string>>>();

    const py::dict nodes = d["nodes"].cast<py::dict>();
    for (auto item : nodes)
        s.nodes.emplace(item.first.cast<std::string>(),
                        parse_node(item.first.cast<std::string>(),
                                   item.second.cast<py::dict>()));

    const py::dict m = d["merge"].cast<py::dict>();
    s.merge.htm_mode = parse_merge_mode(m["htm_mode"].cast<std::string>());
    s.merge.htm_params = parse_merge_params(
        m.contains("htm_params") ? py::object(m["htm_params"])
                                 : py::object(py::none()));
    if (!m.contains("feature_mode") || m["feature_mode"].is_none()) {
        s.merge.has_feature_mode = false;
    } else {
        s.merge.has_feature_mode = true;
        s.merge.feature_mode =
            parse_merge_mode(m["feature_mode"].cast<std::string>());
    }
    s.merge.feature_params = parse_merge_params(
        m.contains("feature_params") ? py::object(m["feature_params"])
                                     : py::object(py::none()));
    s.merge.concat_axis =
        static_cast<int>(get_or<std::int64_t>(m, "concat_axis", 0));

    const py::dict f = d["run_flags"].cast<py::dict>();
    s.flags.use_predictive = get_or<bool>(f, "use_predictive", false);
    s.flags.residual = get_or<bool>(f, "residual", false);
    s.flags.broadcast_head = get_or<bool>(f, "broadcast_head", false);
    s.flags.recurrent_head = get_or<bool>(f, "recurrent_head", false);
    s.flags.anomaly_boost = get_or<bool>(f, "anomaly_boost", false);
    s.flags.long_term_memory = get_or<bool>(f, "long_term_memory", false);
    s.flags.memory_inertia = get_or<std::int64_t>(f, "memory_inertia", 5);
    s.flags.lateral_exchange = get_or<bool>(f, "lateral_exchange", false);
    s.flags.squeeze_skip = get_or<bool>(f, "squeeze_skip", false);

    for (auto e : d["encoders"].cast<py::list>())
        s.encoders.push_back(parse_feature(e.cast<py::dict>()));

    const py::dict data = d["data"].cast<py::dict>();
    const std::string mode = get_or<std::string>(data, "mode", "py_reader");
    s.data.mode = (mode == "native_csv") ? DataMode::NativeCsv
                                         : DataMode::PyReader;
    s.data.path = get_or<std::string>(data, "path", "");
    s.data.columns = data["columns"].cast<std::vector<std::string>>();
    s.data.column_is_dt = data["column_is_dt"].cast<std::vector<int>>();
    if (data.contains("dt_formats") && !data["dt_formats"].is_none())
        s.data.dt_formats =
            data["dt_formats"].cast<std::vector<std::string>>();
    else
        s.data.dt_formats.assign(s.data.columns.size(), "");
    s.data.row_start = get_opt<std::int64_t>(data, "row_start");
    s.data.row_stop = get_opt<std::int64_t>(data, "row_stop");
    s.data.row_step = get_opt<std::int64_t>(data, "row_step");
    s.data.train_percent = get_or<double>(data, "train_percent", 1.0);
    s.data.batch_rows = get_or<std::int64_t>(data, "batch_rows", 8192);

    s.max_workers = get_opt<std::int64_t>(d, "max_workers");
    s.multiprocess = get_or<bool>(d, "multiprocess", true);
    s.pipeline_depth = get_or<std::int64_t>(d, "pipeline_depth", 4);
    s.progress = get_or<bool>(d, "progress", true);
    return s;
}

/* ===================== PyReaderSource (batched pulls) ==================== */
/* Drives the Python DatasetReader: reader = factory(); reader.next_batch()
 * -> {'n': int, 'num': {col: float64[n]}, 'dt': {col: int32[n,7]}} | None;
 * reader.total_rows -> int | None.  The GIL is held ONLY inside
 * pull_batch() / reset(); everything else serves rows from plain C++
 * buffers. */
class PyReaderSource final : public RecordSource {
public:
    PyReaderSource(py::object factory, std::vector<std::string> columns,
                   std::vector<int> column_is_dt)
        : factory_(std::move(factory)), columns_(std::move(columns)),
          isDt_(std::move(column_is_dt)) {
        numBuf_.resize(columns_.size());
        dtBuf_.resize(columns_.size());
        py::gil_scoped_acquire gil;
        make_reader();
    }

    ~PyReaderSource() override {
        py::gil_scoped_acquire gil;
        reader_ = py::object();
        factory_ = py::object();
    }

    bool next(Record &rec) override {
        if (row_ >= batchN_) {
            if (!pull_batch()) return false;
        }
        rec.nums.resize(columns_.size());
        rec.dts.resize(columns_.size());
        for (std::size_t j = 0; j < columns_.size(); ++j) {
            if (isDt_[j]) {
                const std::int32_t *p = &dtBuf_[j][static_cast<std::size_t>(row_) * 7];
                DateParts &dp = rec.dts[j];
                dp.year = p[0]; dp.month = p[1]; dp.day = p[2];
                dp.hour = p[3]; dp.minute = p[4]; dp.second = p[5];
                dp.valid = p[6] != 0;
            } else {
                rec.nums[j] = numBuf_[j][static_cast<std::size_t>(row_)];
            }
        }
        ++row_;
        return true;
    }

    void reset() override {
        py::gil_scoped_acquire gil;
        make_reader();
        batchN_ = row_ = 0;
    }

    std::int64_t total_rows() const override { return total_; }

private:
    py::object factory_, reader_;
    std::vector<std::string> columns_;
    std::vector<int> isDt_;

    std::vector<std::vector<double>> numBuf_;        // per numeric column
    std::vector<std::vector<std::int32_t>> dtBuf_;   // per dt column, n*7
    std::int64_t batchN_ = 0, row_ = 0, total_ = -1;

    void make_reader() {
        reader_ = factory_();
        const py::object t = reader_.attr("total_rows");
        total_ = t.is_none() ? -1 : t.cast<std::int64_t>();
    }

    bool pull_batch() {
        py::gil_scoped_acquire gil;
        py::object b = reader_.attr("next_batch")();
        if (b.is_none()) return false;
        const py::dict d = b.cast<py::dict>();
        batchN_ = d["n"].cast<std::int64_t>();
        if (batchN_ <= 0) return false;

        const py::dict num = d["num"].cast<py::dict>();
        const py::dict dt = d["dt"].cast<py::dict>();
        for (std::size_t j = 0; j < columns_.size(); ++j) {
            if (isDt_[j]) {
                auto arr = dt[py::str(columns_[j])]
                               .cast<py::array_t<std::int32_t,
                                                 py::array::c_style |
                                                     py::array::forcecast>>();
                NTA_CHECK(arr.ndim() == 2 && arr.shape(0) == batchN_ &&
                          arr.shape(1) == 7)
                    << "dt batch for `" << columns_[j] << "` must be (n, 7)";
                dtBuf_[j].assign(arr.data(),
                                 arr.data() + static_cast<std::size_t>(batchN_) * 7);
            } else {
                auto arr = num[py::str(columns_[j])]
                               .cast<py::array_t<double,
                                                 py::array::c_style |
                                                     py::array::forcecast>>();
                NTA_CHECK(arr.ndim() == 1 && arr.shape(0) == batchN_)
                    << "num batch for `" << columns_[j] << "` must be (n,)";
                numBuf_[j].assign(arr.data(),
                                  arr.data() + static_cast<std::size_t>(batchN_));
            }
        }
        row_ = 0;
        return true;
    }
};

/* ============================ PyramidEngine ============================== */
class PyramidEngine {
public:
    explicit PyramidEngine(const py::dict &spec_dict)
        : spec_(parse_spec(spec_dict)), runtime_(spec_) {}

    // Installs the Ctrl+C hook and removes it on scope exit. The hook briefly
    // re-acquires the GIL and calls PyErr_CheckSignals(); a pending signal
    // (e.g. KeyboardInterrupt) is turned into a C++ throw that unwinds the
    // GIL-released C++ section. Shared by build()/run()/execute() so the whole
    // build+run is interruptible -- including init.
    // Raise Python KeyboardInterrupt. Caller must hold the GIL.
    [[noreturn]] static void raise_keyboard_interrupt() {
        PyErr_SetNone(PyExc_KeyboardInterrupt);
        throw py::error_already_set();
    }

    struct InterruptHook {
        InterruptHook() {
            htm_pyramid::interrupt_hook() = []() {
                // Briefly take the GIL to let Python run its signal handlers.
                // PyErr_CheckSignals() runs them; if one raised (Ctrl+C ->
                // KeyboardInterrupt) we CLEAR it here and throw a plain C++
                // marker instead, so the unwind through GIL-released worker
                // code carries no Python object. The binding re-raises
                // KeyboardInterrupt at the top with the GIL held.
                py::gil_scoped_acquire gil;
                if (PyErr_CheckSignals() != 0) {
                    PyErr_Clear();
                    throw htm_pyramid::PyramidInterrupted{};
                }
            };
        }
        ~InterruptHook() { htm_pyramid::interrupt_hook() = nullptr; }
    };

    void build(const py::object &reader_factory) {
        make_source(reader_factory);
        InterruptHook hook;              // Ctrl+C during init
        try {
            py::gil_scoped_release release;
            runtime_.build(*source_);
        } catch (const htm_pyramid::PyramidInterrupted &) {
            py::gil_scoped_acquire gil;      // GIL was released in the try
            raise_keyboard_interrupt();
        }
    }

    void run(const py::object &iterations, const py::object &progress_cb,
             std::int64_t progress_every) {
        NTA_CHECK(source_) << "run() before build()";
        std::optional<std::int64_t> iters;
        if (!iterations.is_none()) iters = iterations.cast<std::int64_t>();

        std::function<void(std::int64_t, std::int64_t)> cb;
        if (!progress_cb.is_none()) {
            py::object fn = progress_cb;
            cb = [fn](std::int64_t done, std::int64_t total) {
                py::gil_scoped_acquire gil;
                fn(done, total);
            };
        }
        // Make the GIL-released run interruptible (Ctrl+C -> KeyboardInterrupt).
        InterruptHook hook;
        try {
            py::gil_scoped_release release;
            runtime_.run(*source_, iters, cb, progress_every);
        } catch (const htm_pyramid::PyramidInterrupted &) {
            py::gil_scoped_acquire gil;
            raise_keyboard_interrupt();
        }
    }

    /** The single-crossing entry point: build + full run in one Python
     *  call. Equivalent to build(factory); run(...). */
    void execute(const py::object &reader_factory,
                 const py::object &iterations, const py::object &progress_cb,
                 std::int64_t progress_every) {
        // build() and run() each install the hook and translate a Ctrl+C
        // marker into KeyboardInterrupt, so execute() simply chains them.
        build(reader_factory);
        run(iterations, progress_cb, progress_every);
    }

    py::dict results() const {
        py::dict out;
        for (const auto &name : runtime_.node_names_in_layer_order()) {
            const NodeResults r = runtime_.results_for(name);
            py::dict node;
            node["anomaly"] = py::array_t<double>(
                static_cast<py::ssize_t>(r.anomaly.size()), r.anomaly.data());
            node["activity"] = py::array_t<std::int64_t>(
                static_cast<py::ssize_t>(r.activity.size()),
                r.activity.data());
            out[py::str(name)] = node;
        }
        return out;
    }

    py::dict summaries() const {
        py::dict out;
        for (const auto &name : runtime_.node_names_in_layer_order())
            out[py::str(name)] = runtime_.summary_for(name);
        return out;
    }

    std::string head_name() const { return runtime_.head_name(); }
    std::vector<std::string> node_names() const {
        return runtime_.node_names_in_layer_order();
    }
    std::int64_t records_processed() const {
        return runtime_.records_processed();
    }

    /** Total records in the bound source, or None when unknown (e.g. a
     *  chunked CSV reader before exhaustion). Available after build(). */
    py::object total_records() const {
        if (!source_) return py::none();
        const std::int64_t t = source_->total_rows();
        return t < 0 ? py::object(py::none()) : py::object(py::int_(t));
    }
    std::size_t n_workers() const { return runtime_.n_workers(); }

private:
    PyramidSpec spec_;
    PyramidRuntime runtime_;
    std::unique_ptr<RecordSource> source_;

    void make_source(const py::object &reader_factory) {
        if (!reader_factory.is_none()) {
            source_ = std::make_unique<PyReaderSource>(
                reader_factory, spec_.data.columns, spec_.data.column_is_dt);
            return;
        }
        NTA_CHECK(spec_.data.mode == DataMode::NativeCsv)
            << "build(None) requires spec['data']['mode'] == 'native_csv' "
            << "(otherwise pass a DatasetReader factory)";
        source_ = std::make_unique<CsvSource>(
            spec_.data.path, spec_.data.columns, spec_.data.column_is_dt,
            spec_.data.dt_formats, spec_.data.row_start, spec_.data.row_stop,
            spec_.data.row_step);
    }
};

} // namespace

PYBIND11_MODULE(pyramid_engine, m) {
    m.doc() =
        "PyHTM pyramid runtime in C++: the whole per-record loop (encode -> "
        "merge -> SP/TM across all layers -> merge chain extras) executes "
        "under one GIL release, on a pinned worker pool.";

    /* Dims probe for the Python spec builder: single source of truth for
     * encoder output dimensions (the date encoder's size arithmetic lives
     * only in C++).  Sample-dependent kinds are dimensioned by their `size`
     * field, so no samples are needed here. */
    m.def("probe_encoder_dims",
          [](const py::dict &enc_spec) {
              const FeatureSpec fs = parse_feature(enc_spec);
              if (fs.kind == FeatureKind::Date) {
                  auto enc = build_feature_encoder(fs, nullptr);
                  return enc->dimensions();
              }
              if (!fs.shape.empty()) return fs.shape;
              return std::vector<htm::UInt>{
                  static_cast<htm::UInt>(fs.size)};
          },
          py::arg("encoder_spec"),
          "Output dimensions for one encoder spec (no samples required).");


    py::class_<PyramidEngine>(m, "PyramidEngine")
        .def(py::init<const py::dict &>(), py::arg("spec"))
        .def("build", &PyramidEngine::build,
             py::arg("reader_factory") = py::none(),
             "Build encoders (with the training-slice sample pass when "
             "needed) and all SP/TM nodes in parallel.  reader_factory: "
             "zero-arg callable returning a fresh DatasetReader; None uses "
             "the native CSV reader from spec['data'].")
        .def("execute", &PyramidEngine::execute,
             py::arg("reader_factory") = py::none(),
             py::arg("iterations") = py::none(),
             py::arg("progress_cb") = py::none(),
             py::arg("progress_every") = 5000,
             "Build + full run in ONE Python->C++ call (the two-crossing "
             "contract: spec+data down here, results() up after).")
        .def("run", &PyramidEngine::run, py::arg("iterations") = py::none(),
             py::arg("progress_cb") = py::none(),
             py::arg("progress_every") = 5000,
             "Stream the dataset through the pyramid.  progress_cb, when "
             "given, is called with (records_done, total_or_minus1).")
        .def("results", &PyramidEngine::results,
             "Per-node {'anomaly': float64[], 'activity': int64[]} in layer "
             "order.")
        .def("summaries", &PyramidEngine::summaries,
             "Per-node model_summary() strings, layer order.")
        .def_property_readonly("head_name", &PyramidEngine::head_name)
        .def_property_readonly("node_names", &PyramidEngine::node_names)
        .def_property_readonly("records_processed",
                               &PyramidEngine::records_processed)
        .def_property_readonly("total_records",
                               &PyramidEngine::total_records)
        .def_property_readonly("n_workers", &PyramidEngine::n_workers);
}