/* ---------------------------------------------------------------------------
 * Pyramid-Model-Core :: runtime/HtmNode.cpp
 * See HtmNode.hpp for the contract.
 * ------------------------------------------------------------------------ */
#include "HtmNode.hpp"

#include <htm/algorithms/Anomaly.hpp>              // computeRawAnomalyScore
#include <htm/utils/Log.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace pyramid {

using htm::SDR;
using htm::UInt;

/* dims helper: product */
static std::int64_t prod(const std::vector<UInt> &d) {
    std::int64_t p = 1;
    for (UInt x : d) p *= static_cast<std::int64_t>(x);
    return p;
}

/* Fixed by design (not configurable). The Spatial Pooler values are the ones
 * NuPIC's SPRegion gives NAB's detector: a potential pool over the whole input
 * (potentialRadius = input width, sampled by potentialPct), global inhibition,
 * no boosting, stimulusThreshold 0, wrap-around, and the default duty cycles.
 * The Temporal Memory's connected permanence is NuPIC's default, 0.5. */
static constexpr bool   kGlobalInhibition       = true;
static constexpr UInt   kStimulusThreshold      = 0u;
static constexpr double kMinPctOverlapDutyCycle = 0.001;
static constexpr UInt   kDutyCyclePeriod        = 1000u;
static constexpr double kBoostStrength          = 0.0;
static constexpr bool   kWrapAround             = true;
static constexpr double kPermanenceConnected    = 0.5;

HtmNode::HtmNode(const NodeSpec &spec) : spec_(spec) {
    columnDims_ = spec_.sp.columnDimensions;
    cellDims_ = columnDims_;
    cellDims_.push_back(static_cast<UInt>(spec_.tm.cellsPerColumn));
    /* The output is the active columns, flattened. */
    outDims_ = {static_cast<UInt>(prod(columnDims_))};

    /* Fixed-dims scratch. */
    activeColumns_.initialize(columnDims_);
    predictiveCells_.initialize(cellDims_);
    predColumns_.initialize(columnDims_);
    outFinal_.initialize(outDims_);
}

void HtmNode::init_sp() {
    const SpSpec &p = spec_.sp;
    /* numActiveColumnsPerInhArea=0 (localAreaDensity is used) and
     * spVerbosity=0 keep the binding defaults. */
    sp_ = std::make_unique<htm::SpatialPooler>(
        spec_.input_dims, p.columnDimensions,
        /*potentialRadius=*/static_cast<UInt>(prod(spec_.input_dims)),
        static_cast<htm::Real>(p.potentialPct), kGlobalInhibition,
        static_cast<htm::Real>(p.localAreaDensity),
        /*numActiveColumnsPerInhArea=*/0u,
        kStimulusThreshold,
        static_cast<htm::Real>(p.synPermInactiveDec),
        static_cast<htm::Real>(p.synPermActiveInc),
        static_cast<htm::Real>(p.synPermConnected),
        static_cast<htm::Real>(kMinPctOverlapDutyCycle),
        kDutyCyclePeriod,
        static_cast<htm::Real>(kBoostStrength),
        static_cast<htm::Int>(spec_.seed),
        /*spVerbosity=*/0u, kWrapAround);
}

void HtmNode::init_tm() {
    const TmSpec &t = spec_.tm;
    tm_ = std::make_unique<htm::TemporalMemory>(
        columnDims_, static_cast<UInt>(t.cellsPerColumn),
        static_cast<UInt>(t.activationThreshold),
        static_cast<htm::Permanence>(t.initialPerm),
        static_cast<htm::Permanence>(kPermanenceConnected),
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

void HtmNode::step(const SDR &input) {
    constexpr bool learn = true;

    /* --- Spatial Pooler --- */
    float spOverlap = std::numeric_limits<float>::quiet_NaN();
    {
        /* compute() returns every column's raw overlap with this input. */
        const auto overlaps = sp_->compute(input, learn, activeColumns_);
        const auto &win = activeColumns_.getSparse();
        if (recordSignals_ && !win.empty()) {
            double sum = 0.0;
            for (const auto c : win) sum += static_cast<double>(overlaps[c]);
            spOverlap = static_cast<float>(sum / static_cast<double>(win.size()));
        }
    }
    if (recordSignals_) spOverlapHist_.push_back(spOverlap);
    activityHist_.push_back(
        static_cast<std::int64_t>(activeColumns_.getSparse().size()));

    /* --- Temporal Memory: predict, score, then activate.  ORDER IS THE
     * ALGORITHM (see header). --- */
    tm_->activateDendrites(learn);
    {
        SDR p = tm_->getPredictiveCells();
        predictiveCells_.setSDR(p);
        const SDR pc = tm_->cellsToColumns(predictiveCells_);
        predColumns_.setSDR(pc);
        /* signals: how broad the prediction was, and how much of it hit */
        if (recordSignals_) {
            const auto &pa = predColumns_.getSparse();
            const auto &ac = activeColumns_.getSparse();
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
    anomalyHist_.push_back(static_cast<double>(
        htm::computeRawAnomalyScore(activeColumns_, predColumns_)));
    tm_->activateCells(activeColumns_, learn);

    /* --- Output: the active columns, flattened (identical flat indices). */
    bits_.assign(activeColumns_.getSparse().begin(),
                 activeColumns_.getSparse().end());
    outFinal_.setSparse(bits_);
}

void HtmNode::reserve_history(std::size_t n) {
    anomalyHist_.reserve(n);
    activityHist_.reserve(n);
    if (recordSignals_) {
        predColsHist_.reserve(n); predHitsHist_.reserve(n); predCellsHist_.reserve(n);
        spOverlapHist_.reserve(n);
    }
}

} // namespace pyramid
