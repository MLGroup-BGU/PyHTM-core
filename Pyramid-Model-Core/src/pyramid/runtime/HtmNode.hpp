/* ---------------------------------------------------------------------------
 * Pyramid-Model-Core :: runtime/HtmNode.hpp
 *
 * One pyramid node: a SpatialPooler followed by a TemporalMemory, with the
 * raw anomaly score.
 *
 *   step():  SP.compute -> activateDendrites -> getPredictiveCells ->
 *            cellsToColumns -> computeRawAnomalyScore -> activateCells ->
 *            output (the active columns, flattened) -> histories
 *
 * OUTPUT: the node emits its active columns -- columnDimensions bits (2048),
 * localAreaDensity of them on (40, i.e. 2%). This is the SP's learned code of
 * "which pattern of this node's features is present now". The TM's cells (the
 * node's temporal context) stay inside the node: they drive its anomaly
 * score, and the parent learns its own temporal context from the sequence of
 * these columns. Every node therefore sends the same number of bits whether
 * or not it is surprised, and the output width does not depend on the TM
 * size. htm.core's TMRegion offers the same column-level output
 * (`orColumnOutputs`).
 *
 * ORDER IS THE ALGORITHM: dendrites predict from the t-1 cells; the anomaly
 * compares those predictions with THIS step's columns; only then are cells
 * activated.
 *
 * Learning is always on. The fixed SP / TM constants are in HtmNode.cpp.
 *
 * MEMORY: every per-step object is a reused member; steady-state stepping
 * allocates nothing (beyond what the core SP/TM do internally).
 *
 * THREADING: one node is driven by exactly one worker thread per layer
 * step (the hive ownership contract); nothing here is shared.
 * ------------------------------------------------------------------------ */
#pragma once

#include <htm/algorithms/SpatialPooler.hpp>
#include <htm/algorithms/TemporalMemory.hpp>
#include <htm/types/Sdr.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../spec/PyramidSpec.hpp"

namespace pyramid {

class HtmNode {
public:
    explicit HtmNode(const NodeSpec &spec);

    /* Construct the SP and TM (the build pool runs these in parallel across
     * nodes; construction is deterministic per seed, so parallel init is
     * output-identical to serial). */
    void init_sp();
    void init_tm();

    /* One full step over `input`; the output lands in output(). */
    void step(const htm::SDR &input);

    /* Pointer-based variant for the run loop: the orchestrator wires each
     * node's input slot once, workers just call step(). */
    void set_input(const htm::SDR *input) { input_ = input; }
    const htm::SDR *input_ptr() const { return input_; }
    void step() { step(*input_); }

    const htm::SDR &output() const { return outFinal_; }

    /* Output dimensions: the column count, flattened. */
    const std::vector<htm::UInt> &output_dims() const { return outDims_; }

    /* Spec input dims (the balancer's node weight ~ prod(input_dims)). */
    const std::vector<htm::UInt> &input_dims_ref() const {
        return spec_.input_dims;
    }

    const std::string &name() const { return spec_.name; }

    /* Per-step histories (returned to Python at the end of the run). */
    const std::vector<double> &anomaly_history() const { return anomalyHist_; }
    const std::vector<std::int64_t> &activity_history() const { return activityHist_; }

    /* Per-step signals for offline anomaly-function studies (counts only):
     * predicted columns, predicted-and-active columns, predictive cells, and
     * the mean SP overlap of the winning columns. */
    const std::vector<std::uint16_t> &pred_columns_history() const { return predColsHist_; }
    const std::vector<std::uint16_t> &pred_hits_history() const { return predHitsHist_; }
    const std::vector<std::uint16_t> &pred_cells_history() const { return predCellsHist_; }
    const std::vector<float> &sp_overlap_history() const { return spOverlapHist_; }
    void set_record_signals(bool on) { recordSignals_ = on; }

    /* The model_summary() line for this node (preformatted by Python). */
    const std::string &summary() const { return spec_.summary; }

    void reserve_history(std::size_t n);

private:
    NodeSpec spec_;
    const htm::SDR *input_ = nullptr;      // wired by the runtime

    std::unique_ptr<htm::SpatialPooler> sp_;
    std::unique_ptr<htm::TemporalMemory> tm_;

    std::vector<htm::UInt> columnDims_;    // SP column dims
    std::vector<htm::UInt> cellDims_;      // (*columnDims, cellsPerColumn)
    std::vector<htm::UInt> outDims_;       // {prod(columnDims)}

    std::vector<double> anomalyHist_;
    std::vector<std::int64_t> activityHist_;
    std::vector<std::uint16_t> predColsHist_, predHitsHist_, predCellsHist_;
    std::vector<float> spOverlapHist_;
    /* Signals are recorded only when the config asks for them
     * (general.record_signals); off by default, so a run spends no memory on
     * them unless switched on. Set by the runtime right after construction. */
    bool recordSignals_ = false;

    /* per-step scratch (fixed dims per node) */
    htm::SDR activeColumns_;
    htm::SDR predictiveCells_;            // TM predictions for this step (cells)
    htm::SDR predColumns_;                // ... reduced to columns
    htm::SDR outFinal_;                   // the output: active columns, flat
    std::vector<htm::UInt> bits_;         // index scratch
};

} // namespace pyramid
