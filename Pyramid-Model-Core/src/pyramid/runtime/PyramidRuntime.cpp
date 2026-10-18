/* ---------------------------------------------------------------------------
 * Pyramid-Model-Core :: runtime/PyramidRuntime.cpp
 *
 * Implementation; see PyramidRuntime.hpp.
 * ------------------------------------------------------------------------ */
#include "PyramidRuntime.hpp"
#include "../util/StageLog.hpp"
#include <chrono>
#include <cstdio>

#include <htm/utils/Log.hpp>

#include <algorithm>
#include <cmath>

namespace pyramid {

using htm::SDR;
using htm::UInt;

PyramidRuntime::PyramidRuntime(PyramidSpec spec) : spec_(std::move(spec)) {
    nLayers_ = spec_.layers.size();
    NTA_CHECK(nLayers_ >= 1) << "pyramid needs at least one layer";
    const auto &last = spec_.layers[nLayers_ - 1];
    NTA_CHECK(!last.empty()) << "empty head layer";
    /* _resolve_head_name: the LAST name of the LAST layer. */
    headName_ = last.back();
}

PyramidRuntime::~PyramidRuntime() {
    if (pool_) pool_->shutdown();
}

/* ========================== build phase ================================== */
void PyramidRuntime::collect_samples(
        RecordSource &source,
        std::map<std::string, std::vector<double>> &samples) {
    std::vector<std::size_t> want;
    for (std::size_t i = 0; i < spec_.encoders.size(); ++i)
        if (spec_.encoders[i].needs_samples()) want.push_back(i);
    if (want.empty()) return;

    for (std::size_t i : want) samples[spec_.encoders[i].name] = {};

    /* The training slice is df.iloc[:int(len(df) * train_percent)].  With a
     * known total we stop reading at the cut; with an unknown one (open-
     * ended CSV) we collect the whole stream and truncate -- one pass
     * either way, memory bounded by (needed columns x training rows). */
    const std::int64_t total = source.total_rows();
    const std::int64_t known_cut =
        total >= 0 ? static_cast<std::int64_t>(
                         static_cast<double>(total) * spec_.data.train_percent)
                   : -1;

    Record rec;
    std::int64_t n = 0;
    while ((known_cut < 0 || n < known_cut) && source.next(rec)) {
        for (std::size_t i : want)
            samples[spec_.encoders[i].name].push_back(rec.nums[encColIdx_[i]]);
        ++n;
    }
    if (known_cut < 0) {
        const std::int64_t cut = static_cast<std::int64_t>(
            static_cast<double>(n) * spec_.data.train_percent);
        for (auto &kv : samples)
            if (static_cast<std::int64_t>(kv.second.size()) > cut)
                kv.second.resize(static_cast<std::size_t>(cut));
    }
    source.reset();
}

void PyramidRuntime::build_encoders(RecordSource &source) {
    encColIdx_.resize(spec_.encoders.size());
    for (std::size_t i = 0; i < spec_.encoders.size(); ++i) {
        const auto &name = spec_.encoders[i].name;
        auto it = std::find(spec_.data.columns.begin(),
                            spec_.data.columns.end(), name);
        NTA_CHECK(it != spec_.data.columns.end())
            << "feature `" << name << "` is not among the data columns";
        encColIdx_[i] = static_cast<std::size_t>(
            std::distance(spec_.data.columns.begin(), it));
        encByName_[name] = i;
    }

    std::map<std::string, std::vector<double>> samples;
    collect_samples(source, samples);

    encoders_.reserve(spec_.encoders.size());
    encSlots_.reserve(spec_.encoders.size());
    for (const auto &fs : spec_.encoders) {
        const std::vector<double> *sp = nullptr;
        auto it = samples.find(fs.name);
        if (it != samples.end()) sp = &it->second;
        encoders_.push_back(build_feature_encoder(fs, sp));
        encSlots_.emplace_back(encoders_.back()->dimensions());
    }
}

void PyramidRuntime::build_nodes() {
    layers_.resize(nLayers_);
    for (std::size_t li = 0; li < nLayers_; ++li) {
        for (const auto &name : spec_.layers[li]) {
            auto it = spec_.nodes.find(name);
            NTA_CHECK(it != spec_.nodes.end())
                << "node `" << name << "` missing from spec.nodes";
            auto node = std::make_unique<HtmNode>(it->second);
            node->set_record_signals(spec_.record_signals);
            nodePos_[name] = {li, layers_[li].size()};
            layers_[li].push_back(node.get());
            nodes_.emplace(name, std::move(node));
        }
    }
    headLayer_ = nodePos_.at(headName_).layer;
    headPos_ = nodePos_.at(headName_).pos;
}

void PyramidRuntime::build_routing() {
    /* Feature plan, per L0 position. */
    planEnc_.resize(layers_[0].size());
    for (std::size_t p = 0; p < layers_[0].size(); ++p) {
        const auto &name = spec_.layers[0][p];
        auto it = spec_.feature_plan.find(name);
        NTA_CHECK(it != spec_.feature_plan.end())
            << "L0 node `" << name << "` missing from the feature plan";
        for (const auto &feat : it->second) {
            auto eit = encByName_.find(feat);
            NTA_CHECK(eit != encByName_.end())
                << "plan feature `" << feat << "` has no encoder";
            planEnc_[p].push_back(eit->second);
        }
    }

    /* Children as positions in the layer below. */
    childPos_.resize(nLayers_);
    for (std::size_t li = 1; li < nLayers_; ++li) {
        childPos_[li].resize(layers_[li].size());
        for (std::size_t p = 0; p < layers_[li].size(); ++p) {
            const auto &parent = spec_.layers[li][p];
            auto cit = spec_.children.find(parent);
            NTA_CHECK(cit != spec_.children.end() && !cit->second.empty())
                << "node `" << parent << "` has no children in the spec";
            for (const auto &c : cit->second) {
                const Pos cp = nodePos_.at(c);
                NTA_CHECK(cp.layer + 1 == li)
                    << "`" << c << "` is not one layer below `" << parent << "`";
                childPos_[li][p].push_back(cp.pos);
            }
        }
    }
}

void PyramidRuntime::allocate_slots() {
    for (auto &kv : nodes_) {
        auto res = inputSlots_.emplace(kv.first,
                                       SDR(kv.second->input_dims_ref()));
        kv.second->set_input(&res.first->second);
    }

    outView_.resize(nLayers_);
    for (std::size_t li = 0; li < nLayers_; ++li)
        outView_[li].assign(layers_[li].size(), nullptr);
}

void PyramidRuntime::validate_dims() {
    /* L0: concatenated feature dims == spec input dims. */
    for (std::size_t p = 0; p < layers_[0].size(); ++p) {
        std::vector<std::vector<UInt>> dims;
        for (std::size_t ei : planEnc_[p])
            dims.push_back(encoders_[ei]->dimensions());
        std::vector<UInt> merged =
            dims.size() == 1 ? dims[0] : concat_output_dims(dims);
        NTA_CHECK(merged == layers_[0][p]->input_dims_ref())
            << "L0 `" << spec_.layers[0][p]
            << "`: concatenated feature dims != spec input dims";
    }

    /* Parents: concatenated children output dims == spec input dims. */
    for (std::size_t li = 1; li < nLayers_; ++li)
        for (std::size_t p = 0; p < layers_[li].size(); ++p) {
            std::vector<std::vector<UInt>> dims;
            for (std::size_t cp : childPos_[li][p])
                dims.push_back(layers_[li - 1][cp]->output_dims());
            std::vector<UInt> merged =
                dims.size() == 1 ? dims[0] : concat_output_dims(dims);
            NTA_CHECK(merged == layers_[li][p]->input_dims_ref())
                << "`" << spec_.layers[li][p]
                << "`: concatenated child dims != spec input dims";
        }
}

void PyramidRuntime::build(RecordSource &source) {
    using htm_pyramid::Stage;
    using htm_pyramid::stage_banner;
    const auto buildStart = std::chrono::steady_clock::now();

    stage_banner(Stage::Encoders,
                 "ENCODERS -- feature encoders (+ sample pass)");
    build_encoders(source);

    stage_banner(Stage::Architecture,
                 "ARCHITECTURE -- pyramid nodes, routing, slots");
    build_nodes();
    build_routing();
    allocate_slots();
    validate_dims();
    std::printf("[Architecture] %zu nodes across %zu layers\n",
                nodes_.size(), layers_.size());
    std::fflush(stdout);

    /* HTMThreadHive semantics: worker count from get_n_workers, capped by
     * max_workers and L0 width; LPT per layer carrying loads.
     * multiprocess=False forces one worker -- output-identical to the
     * sequential Python path (nodes are independent).
     *
     * TWO pools, as in the build/run contract:
     *   pool #1 (build pool)  -- parallel SP+TM initialization, then
     *                            closed (threads joined) before the run
     *                            pool exists;
     *   pool #2 (run pool)    -- the layer-stepping hive for run().
     * Both pools compute the SAME deterministic LPT partition, so the
     * hand-off cannot affect ownership or results. */
    std::optional<std::int64_t> cap = spec_.max_workers;
    if (!spec_.multiprocess) cap = 1;

    stage_banner(Stage::Workers,
                 "WORKERS -- core discovery & thread<->model assignment "
                 "(build pool)");
    {
        WorkerPool initPool(layers_, cap);
        stage_banner(Stage::Init,
                     "INIT -- SP + TM initialization (parallel)");
        const auto initStart = std::chrono::steady_clock::now();
        // Make the (GIL-released) init interruptible: the main thread polls
        // for Ctrl+C between short waits while the pool builds SP+TM. Without
        // this, Ctrl+C during the multi-second init is ignored until the run
        // loop starts. The hook is installed by the binding for the whole
        // build+run; if none is set (e.g. build() called directly) we pass a
        // no-op and init blocks as before.
        if (htm_pyramid::interrupt_hook())
            initPool.init_all([]() { htm_pyramid::check_interrupt(); });
        else
            initPool.init_all();
        const double initSecs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - initStart).count();
        std::printf("[Init] done in %s (%zu nodes)\n",
                    htm_pyramid::fmt_hms(initSecs).c_str(),
                    nodes_.size());
        std::fflush(stdout);
    }   // build pool joined & destroyed here

    stage_banner(Stage::Workers,
                 "WORKERS -- run pool (same balanced partition)");
    pool_ = std::make_unique<WorkerPool>(layers_, cap);
    allocate_pipeline();

    const double buildSecs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - buildStart).count();
    std::printf("[Build] encoders + architecture + init ready in %s\n",
                htm_pyramid::fmt_hms(buildSecs).c_str());
    std::fflush(stdout);
}

/* =========================== run phase =================================== */
void PyramidRuntime::concat_or_copy(SDR &out) {
    if (mergeInputs_.size() == 1)
        out.setSDR(*mergeInputs_[0]);   // a single input is copied as it is
    else
        concat_.concat(mergeInputs_, out);
}

void PyramidRuntime::encode_record(const Record &rec) {
    for (std::size_t i = 0; i < encoders_.size(); ++i) {
        const std::size_t col = encColIdx_[i];
        if (encoders_[i]->is_datetime())
            encoders_[i]->encode(rec.dts[col], encSlots_[i]);
        else
            encoders_[i]->encode(rec.nums[col], encSlots_[i]);
    }

    for (std::size_t p = 0; p < layers_[0].size(); ++p) {
        SDR &slot = inputSlots_.at(spec_.layers[0][p]);
        mergeInputs_.clear();
        for (std::size_t ei : planEnc_[p]) mergeInputs_.push_back(&encSlots_[ei]);
        concat_or_copy(slot);
    }
}

void PyramidRuntime::run_one_layer(std::size_t li) {
    pool_->run_layer(li, [](HtmNode &n) { n.step(); });

    auto &view = outView_[li];
    for (std::size_t p = 0; p < layers_[li].size(); ++p)
        view[p] = &layers_[li][p]->output();
}

void PyramidRuntime::merge_children_for(std::size_t li, std::size_t parent_pos,
                                        SDR &out) {
    const auto &below = outView_[li - 1];
    mergeInputs_.clear();
    for (std::size_t cp : childPos_[li][parent_pos])
        mergeInputs_.push_back(below[cp]);
    concat_or_copy(out);
}

void PyramidRuntime::merge_into_next(std::size_t li) {
    const std::size_t next = li + 1;
    for (std::size_t p = 0; p < layers_[next].size(); ++p) {
        SDR &slot = inputSlots_.at(spec_.layers[next][p]);
        merge_children_for(next, p, slot);
    }
}

void PyramidRuntime::run(
        RecordSource &source, std::optional<std::int64_t> iterations,
        const std::function<void(std::int64_t, std::int64_t)> &progress,
        std::int64_t progress_every) {
    NTA_CHECK(pool_) << "run() called before build()";

    htm_pyramid::stage_banner(htm_pyramid::Stage::Run,
                              "RUN -- streaming records through the pyramid");
    const auto runStart = std::chrono::steady_clock::now();
    auto lastProgress = runStart;
    /* Terminal -> redraw one line in place. File or pipe -> append a line per
     * step of percent, so an sbatch log stays readable. */
    const bool tty = htm_pyramid::stdout_is_terminal();
    const int pctStep = htm_pyramid::progress_percent_step();
    int lastPct = -1;
    htm_pyramid::RateWindow rateWin;   // current-speed window (~6 s)
    const std::int64_t runFrom = currentIter_;

    const std::int64_t total = source.total_rows();
    const std::int64_t limit =
        iterations.has_value()
            ? (total >= 0 ? std::min(total, *iterations) : *iterations)
            : total;   // -1 == until exhausted

    if (limit >= 0)
        for (auto &kv : nodes_)
            kv.second->reserve_history(static_cast<std::size_t>(limit));

    if (pipelineOn_ && !progress) {
        /* Cross-record pipeline: dependency-released scheduling, no layer
         * barrier; bit-identical per-node sequences (each node processes
         * its records in order with the same inputs as the barrier path).
         * The Python-callback path keeps the barrier loop (callback
         * ordering semantics). */
        const std::int64_t done =
            run_pipelined(source, limit, progress_every, total);
        const double secs = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - runStart).count();
        htm_pyramid::stage_banner(htm_pyramid::Stage::Done,
                                  "DONE -- run complete");
        std::printf("[Run] %lld records in %s (%.1f record/second), "
                    "%lld total processed\n",
                    static_cast<long long>(done),
                    htm_pyramid::fmt_hms(secs).c_str(),
                    secs > 0 ? done / secs : 0.0,
                    static_cast<long long>(currentIter_));
        std::fflush(stdout);
        return;
    }

    Record rec;
    while ((limit < 0 || currentIter_ < limit) && source.next(rec)) {
        encode_record(rec);

        for (std::size_t li = 0; li < nLayers_; ++li) {
            run_one_layer(li);
            if (li + 1 < nLayers_) merge_into_next(li);
        }

        ++currentIter_;
        const bool count_hit =
            progress_every > 0 && currentIter_ % progress_every == 0;
        const bool time_hit =
            progress_every > 0 &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          lastProgress).count() >= htm_pyramid::PROGRESS_REFRESH_SECONDS;
        /* Redrawing in place only reads correctly on a terminal. Sent to a
         * file -- an sbatch log, a pipe -- carriage returns collapse into one
         * unreadable line, so there we append one line per step of percent
         * instead. */
        bool emit = count_hit || time_hit;
        if (emit && !tty && total > 0) {
            const int pct = static_cast<int>(
                100.0 * static_cast<double>(currentIter_) / total);
            emit = pct / pctStep > lastPct / pctStep;
            if (emit) lastPct = pct;
        }
        if (emit) {
            lastProgress = std::chrono::steady_clock::now();
            htm_pyramid::check_interrupt();   // Ctrl+C responsiveness
            if (progress) {
                if (count_hit) progress(currentIter_, total);
            } else {
                /* Native single-line progress (tqdm-style, pure C++): the
                 * line rewrites itself in place; no Python is touched. */
                const double el = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - runStart).count();
                const std::int64_t d = currentIter_ - runFrom;
                rateWin.add(el, d);
                const double cum = el > 0 ? d / el : 0.0;
                const double rate = rateWin.rate(el, d, cum);  // CURRENT speed
                if (total >= 0) {
                    /* ETA from the current (windowed) speed. */
                    const std::int64_t remain = total - currentIter_;
                    const double eta = rate > 0 ? remain / rate : 0.0;
                    char bar[80];
                    const double frac =
                        total > 0 ? static_cast<double>(currentIter_) / total
                                  : 0.0;
                    htm_pyramid::progress_bar(frac, bar, 24);
                    std::printf("%s[Run] %s %5.1f%%  %lld/%lld records  "
                                "%.1f record/second  elapsed %s  "
                                "estimated time remaining %s        %s",
                                tty ? "\r" : "",
                                bar, 100.0 * frac,
                                static_cast<long long>(currentIter_),
                                static_cast<long long>(total), rate,
                                htm_pyramid::fmt_hms(el).c_str(),
                                htm_pyramid::fmt_hms(eta).c_str(),
                                tty ? "" : "\n");
                } else {
                    std::printf("%s[Run] %lld records  %.1f record/second  "
                                "elapsed %s        %s",
                                tty ? "\r" : "",
                                static_cast<long long>(currentIter_), rate,
                                htm_pyramid::fmt_hms(el).c_str(),
                                tty ? "" : "\n");
                }
                std::fflush(stdout);
            }
        }
    }
    if (progress) progress(currentIter_, total);
    else if (tty) std::printf("\n");   // close the in-place progress line

    const double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - runStart).count();
    const std::int64_t done = currentIter_ - runFrom;
    htm_pyramid::stage_banner(htm_pyramid::Stage::Done,
                              "DONE -- run complete");
    std::printf("[Run] %lld records in %s (%.1f record/second), "
                "%lld total processed\n",
                static_cast<long long>(done),
                htm_pyramid::fmt_hms(secs).c_str(),
                secs > 0 ? done / secs : 0.0,
                static_cast<long long>(currentIter_));
    std::fflush(stdout);
}

/* ============================ results ==================================== */
std::vector<std::string> PyramidRuntime::node_names_in_layer_order() const {
    std::vector<std::string> out;
    for (const auto &layer : spec_.layers)
        for (const auto &n : layer) out.push_back(n);
    return out;
}

NodeResults PyramidRuntime::results_for(const std::string &node) const {
    auto it = nodes_.find(node);
    NTA_CHECK(it != nodes_.end()) << "unknown node `" << node << "`";
    NodeResults res;
    res.anomaly = it->second->anomaly_history();
    res.activity = it->second->activity_history();
    res.pred_columns = it->second->pred_columns_history();
    res.pred_hits = it->second->pred_hits_history();
    res.pred_cells = it->second->pred_cells_history();
    res.sp_overlap = it->second->sp_overlap_history();
    return res;
}

std::string PyramidRuntime::summary_for(const std::string &node) const {
    auto it = nodes_.find(node);
    NTA_CHECK(it != nodes_.end()) << "unknown node `" << node << "`";
    return it->second->summary();
}

} // namespace pyramid