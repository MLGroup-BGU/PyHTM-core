#include <string>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <limits>
/* ---------------------------------------------------------------------------
 * Pyramid-Model-Core :: runtime/HtmNode.cpp
 * See HtmNode.hpp for the contract; blocks cite the Python they port.
 * ------------------------------------------------------------------------ */
#include "HtmNode.hpp"

#include <htm/algorithms/Anomaly.hpp>              // computeRawAnomalyScore
#include <htm/utils/Log.hpp>

#include <algorithm>
#include <sstream>

#include "../merge/SdrMerge.hpp"

namespace pyramid {

using htm::SDR;
using htm::UInt;

/* dims helper: product */
static std::int64_t prod(const std::vector<UInt> &d) {
    std::int64_t p = 1;
    for (UInt x : d) p *= static_cast<std::int64_t>(x);
    return p;
}

HtmNode::HtmNode(const NodeSpec &spec) : spec_(spec) {
    /* HTMModule.column_dim: [1, *columnDimensions] when input_predictive,
     * the SP column dims otherwise; without an SP, the input dims. */
    if (spec_.has_sp) {
        if (spec_.input_predictive) {
            columnDims_.push_back(1u);
            for (UInt d : spec_.sp.columnDimensions) columnDims_.push_back(d);
        } else {
            columnDims_ = spec_.sp.columnDimensions;
        }
    } else {
        columnDims_ = spec_.input_dims;
    }

    cellDims_ = columnDims_;
    cellDims_.push_back(static_cast<UInt>(spec_.tm.cellsPerColumn));

    /* The node OUTPUT is its active COLUMNS (Stage 2): one bit per column,
     * columnDims_ wide (2048 bits, 40 on). The TM's cells stay internal --
     * they hold the node's own temporal context and its anomaly score, and
     * the parent builds its own context from the sequence of these columns.
     * Runtime output dims: squeeze the size-1 leading axis
     * (input_predictive), then pool axis 0 and flatten, on the ACTUAL dims. */
    std::vector<UInt> squeezed;
    for (std::size_t i = 0; i < columnDims_.size(); ++i) {
        if (spec_.input_predictive && i == 0 && columnDims_[i] == 1u) continue;
        squeezed.push_back(columnDims_[i]);
    }
    squeezedDims_ = squeezed;

    std::vector<UInt> pooled = squeezed;
    if (spec_.max_pool > 1) {
        NTA_CHECK(pooled[0] % static_cast<UInt>(spec_.max_pool) == 0)
            << "Max pooling dimension must be divisible by ratio, got: ("
            << pooled[0] << ", " << spec_.max_pool << ")";
        pooled[0] /= static_cast<UInt>(spec_.max_pool);
    }
    if (spec_.flatten)
        outDims_ = {static_cast<UInt>(prod(pooled))};
    else
        outDims_ = pooled;

    /* Fixed-dims scratch. */
    activeColumns_.initialize(columnDims_);
    predictiveCells_.initialize(cellDims_);
    predColumns_.initialize(columnDims_);
    outCols_.initialize(columnDims_);
    resTmp1_.initialize(columnDims_);
    resTmp2_.initialize(columnDims_);
    resOut_.initialize(columnDims_);
    poolTmp_.initialize(pooled);
    outFinal_.initialize(outDims_);
    predFinal_.initialize(outDims_);
}

void HtmNode::init_sp() {
    if (!spec_.has_sp) return;
    const SpSpec &p = spec_.sp;
    /* numActiveColumnsPerInhArea=0 (localAreaDensity is used) and
     * spVerbosity=0 keep the binding defaults. minPctOverlapDutyCycle and
     * dutyCyclePeriod come from the config (SpSpec; defaults 0.001 / 1000,
     * the binding defaults). */
    sp_ = std::make_unique<htm::SpatialPooler>(
        spec_.input_dims, spec_.has_sp && spec_.input_predictive
                              ? columnDims_          // [1, *cols]
                              : p.columnDimensions,
        static_cast<UInt>(p.potentialRadius),
        static_cast<htm::Real>(p.potentialPct), p.globalInhibition,
        static_cast<htm::Real>(p.localAreaDensity),
        /*numActiveColumnsPerInhArea=*/0u,
        static_cast<UInt>(p.stimulusThreshold),
        static_cast<htm::Real>(p.synPermInactiveDec),
        static_cast<htm::Real>(p.synPermActiveInc),
        static_cast<htm::Real>(p.synPermConnected),
        static_cast<htm::Real>(p.minPctOverlapDutyCycle),
        static_cast<UInt>(p.dutyCyclePeriod),
        static_cast<htm::Real>(p.boostStrength),
        static_cast<htm::Int>(spec_.seed),
        /*spVerbosity=*/0u, p.wrapAround);
}

void HtmNode::init_tm() {
    const TmSpec &t = spec_.tm;
    tm_ = std::make_unique<htm::TemporalMemory>(
        columnDims_, static_cast<UInt>(t.cellsPerColumn),
        static_cast<UInt>(t.activationThreshold),
        static_cast<htm::Permanence>(t.initialPerm),
        static_cast<htm::Permanence>(t.permanenceConnected),
        static_cast<UInt>(t.minThreshold),
        static_cast<UInt>(t.newSynapseCount),
        static_cast<htm::Permanence>(t.permanenceInc),
        static_cast<htm::Permanence>(t.permanenceDec),
        static_cast<htm::Permanence>(t.predictedSegmentDecrement),
        static_cast<htm::Int>(spec_.seed),
        static_cast<htm::SegmentIdx>(t.maxSegmentsPerCell),
        static_cast<htm::SynapseIdx>(t.maxSynapsesPerSegment),
        /*checkInputs=*/false);
}

bool HtmNode::learn_now() {
    /* learning_hook: without a schedule the module keeps its initial
     * learning=True forever; with one, learn while iteration <= lr_sch. */
    if (!spec_.learn_schedule.has_value()) return learning_;
    learning_ = iteration_ <= *spec_.learn_schedule;
    return learning_;
}

void HtmNode::post_process(const SDR &in, SDR &out) {
    /* post_forward: sdr_max_pool(ratio) then sdr_flatten (dims-only). */
    const SDR *src = &in;
    if (spec_.max_pool > 1) {
        sdr_max_pool_into(in, static_cast<int>(spec_.max_pool), 0, poolTmp_,
                          bits_);
        src = &poolTmp_;
    }
    if (&out != src) {
        /* flatten is a pure reshape: identical flat indices, out's dims. */
        bits_.assign(src->getSparse().begin(), src->getSparse().end());
        out.setSparse(bits_);
    }
}

void HtmNode::residual_into(SDR &out) {
    /* RTM residual in COLUMN space (Stage 2 placeholder until the residual
     * is redesigned): the surprised columns = active columns that were not
     * predicted (outCols_ minus predColumns_). strength <= 0 -> every
     * active column (fresh copy). */
    if (spec_.residual_strength <= 0.0) {
        bits_.assign(outCols_.getSparse().begin(),
                     outCols_.getSparse().end());
        out.setSparse(bits_);
        return;
    }

    if (spec_.residual_mode == 2) {     // 'xor'
        resTmp1_.subtract(outCols_, predColumns_);
        resTmp2_.subtract(predColumns_, outCols_);
        out.set_union(resTmp1_, resTmp2_);
    } else {                            // 'subtract'
        out.subtract(outCols_, predColumns_);
    }

    if (spec_.residual_strength >= 1.0) return;

    /* 0 < strength < 1: keep a uniform random (1-strength) fraction of the
     * predicted-and-active columns on top of the surprise.
     *
     * DOCUMENTED DEVIATION: Python samples this subset from numpy's GLOBAL
     * RandomState; under the thread hive, per-model draw ORDER depends on
     * scheduling, so those Python results were never run-to-run stable.
     * Here each node owns a PCG64 seeded from its model seed -- fully
     * deterministic.  Deterministic strengths (<=0, >=1) are bit-identical
     * to Python on both sides. */
    resTmp1_.intersection(outCols_, predColumns_);
    const auto &expected = resTmp1_.getSparse();
    const std::int64_t n_keep = static_cast<std::int64_t>(
        static_cast<double>(expected.size()) *
        (1.0 - spec_.residual_strength));
    if (!expected.empty() && n_keep > 0) {
        if (!fracRng_)
            fracRng_ = std::make_unique<NumpyGenerator>(spec_.seed);
        const std::int64_t k =
            std::min<std::int64_t>(n_keep,
                                   static_cast<std::int64_t>(expected.size()));
        /* Partial Fisher-Yates over the expected indices. */
        fracIdx_.resize(expected.size());
        for (std::size_t i = 0; i < expected.size(); ++i)
            fracIdx_[i] = static_cast<std::int64_t>(expected[i]);
        for (std::int64_t i = 0; i < k; ++i) {
            const std::int64_t j =
                i + static_cast<std::int64_t>(
                        fracRng_->random() *
                        static_cast<double>(fracIdx_.size() - static_cast<std::size_t>(i)));
            std::swap(fracIdx_[static_cast<std::size_t>(i)],
                      fracIdx_[static_cast<std::size_t>(j)]);
        }
        /* out.sparse = union1d(out.sparse, kept) */
        bits_.assign(out.getSparse().begin(), out.getSparse().end());
        for (std::int64_t i = 0; i < k; ++i)
            bits_.push_back(static_cast<UInt>(fracIdx_[static_cast<std::size_t>(i)]));
        std::sort(bits_.begin(), bits_.end());
        bits_.erase(std::unique(bits_.begin(), bits_.end()), bits_.end());
        out.setSparse(bits_);
    }
}

void HtmNode::step(const SDR &input) {
    const bool learn = learn_now();

    /* --- Spatial Pooler (or encoder pass-through) --- */
    const SDR *activeColumns;
    float spOverlap = std::numeric_limits<float>::quiet_NaN();
    if (sp_) {
        /* compute() returns every column's raw overlap with this input. */
        const auto overlaps = sp_->compute(input, learn, activeColumns_);
        activeColumns = &activeColumns_;
        const auto &win = activeColumns_.getSparse();
        if (recordSignals_ && !win.empty()) {
            double sum = 0.0;
            for (const auto c : win) sum += static_cast<double>(overlaps[c]);
            spOverlap = static_cast<float>(sum / static_cast<double>(win.size()));
        }
    } else {
        activeColumns = &input;
    }
    if (recordSignals_) spOverlapHist_.push_back(spOverlap);
    activityHist_.push_back(
        static_cast<std::int64_t>(activeColumns->getSparse().size()));

    /* --- Temporal Memory: predict, score, then activate.  ORDER IS THE
     * ALGORITHM (see header). --- */
    tm_->activateDendrites(learn);
    {
        SDR p = tm_->getPredictiveCells();
        predictiveCells_.setSDR(p);
        predColumns_.reshape(columnDims_);
        const SDR pc = tm_->cellsToColumns(predictiveCells_);
        predColumns_.setSDR(pc);
        /* signals: how broad the prediction was, and how much of it hit */
        if (recordSignals_) {
        const auto &pa = predColumns_.getSparse();
        const auto &ac = activeColumns->getSparse();
        std::size_t i = 0, j = 0, hits = 0;
        while (i < pa.size() && j < ac.size()) {
            if (pa[i] < ac[j]) ++i; else if (ac[j] < pa[i]) ++j; else { ++hits; ++i; ++j; }
        }
        auto clip16 = [](std::size_t v) {
            return static_cast<std::uint16_t>(std::min<std::size_t>(v, 65535u)); };
        predColsHist_.push_back(clip16(pa.size()));
        predHitsHist_.push_back(clip16(hits));
        predCellsHist_.push_back(clip16(predictiveCells_.getSparse().size()));
        }
    }
    if (spec_.anomaly_score) {
        const double a = static_cast<double>(
            htm::computeRawAnomalyScore(*activeColumns, predColumns_));
        anomalyHist_.push_back(a);
        lastAnomaly_ = a;
    }
    tm_->activateCells(*activeColumns, learn);

    /* --- Output: the active columns, or (RTM) the surprised columns --- */
    outCols_.reshape(columnDims_);     // may hold squeezed dims from last step
    outCols_.setSDR(*activeColumns);

    SDR *rawOut;
    if (spec_.anomaly_detector) {
        resOut_.reshape(columnDims_);
        residual_into(resOut_);
        rawOut = &resOut_;
    } else {
        rawOut = &outCols_;
    }

    /* --- squeeze (input_predictive) + post_forward on each returned SDR --- */
    if (spec_.input_predictive) {
        /* size-1 leading axis drop: flat indices unchanged, dims narrowed */
        rawOut->reshape(squeezedDims_);
        predColumns_.reshape(squeezedDims_);
    }
    post_process(*rawOut, outFinal_);
    if (spec_.return_predictive)
        post_process(predColumns_, predFinal_);
    if (spec_.input_predictive)
        predColumns_.reshape(columnDims_);    // restore for next step

    ++iteration_;
}

std::string HtmNode::summary() const {
    /* Preformatted by the Python spec builder (authoritative -- exact
     * tuple/ndarray reprs); the block below is a debug-only fallback. */
    if (!spec_.summary.empty()) return spec_.summary;

    /* HTMModule.summary(): "[{input_dims} --> {output_dim}] (max_pool: N,
     * R: rad, Boost: b.bb)" with Python tuple formatting.  output_dim here
     * replicates the PROPERTY's arithmetic (including its use of the
     * pre-squeeze dims), not the runtime dims. */
    auto py_tuple = [](const std::vector<std::int64_t> &d) {
        std::ostringstream o;
        o << "(";
        for (std::size_t i = 0; i < d.size(); ++i) {
            o << d[i];
            if (i + 1 < d.size()) o << ", ";
        }
        if (d.size() == 1) o << ",";
        o << ")";
        return o.str();
    };

    std::vector<std::int64_t> in_d(spec_.input_dims.begin(),
                                   spec_.input_dims.end());
    std::vector<std::int64_t> out_d(columnDims_.begin(), columnDims_.end());
    if (spec_.max_pool > 1) out_d[0] /= spec_.max_pool;
    if (spec_.flatten) {
        std::int64_t p = 1;
        for (auto x : out_d) p *= x;
        out_d = {p};
    }
    if (spec_.return_predictive) out_d.insert(out_d.begin(), 2);

    std::ostringstream o;
    o << "[" << py_tuple(in_d) << " --> " << py_tuple(out_d)
      << "] (max_pool: " << spec_.max_pool << ", R: " << spec_.sp.potentialRadius
      << ", Boost: ";
    char b[32];
    std::snprintf(b, sizeof(b), "%.2f", spec_.sp.boostStrength);
    o << b << ")";
    return o.str();
}

void HtmNode::reserve_history(std::size_t n) {
    if (spec_.anomaly_score) anomalyHist_.reserve(n);
    activityHist_.reserve(n);
    if (recordSignals_) {
        predColsHist_.reserve(n); predHitsHist_.reserve(n); predCellsHist_.reserve(n);
        spOverlapHist_.reserve(n);
    }
}

} // namespace pyramid
