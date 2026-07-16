/* ---------------------------------------------------------------------------
 * Pyramid-Model-Core :: runtime/HtmNode.hpp
 *
 * The C++ HTMModule: one (SpatialPooler -> TemporalMemory) unit with anomaly
 * scoring, the optional RTM residual output, max-pool / flatten
 * post-processing, and the learning schedule.
 *
 * step() is a direct port of the stepSpTm orchestrator (the fast path the
 * Python HTMModule already runs) plus HTMModule's own bookkeeping:
 *
 *     learning_hook -> [SP.compute] -> activateDendrites ->
 *     getPredictiveCells -> (cellsToColumns -> computeRawAnomalyScore) ->
 *     activateCells -> getActiveCells / residual -> squeeze ->
 *     max_pool -> flatten -> histories -> iteration++
 *
 * ORDER IS THE ALGORITHM: dendrites predict from the t-1 cells; the anomaly
 * compares those predictions with THIS step's columns; only then are cells
 * activated.  Identical to both Python paths (proven by seam test T6 for
 * stepSpTm, and end-to-end by the pyramid equivalence harness).
 *
 * The fractional residual strength (0 < s < 1) keeps a uniform random
 * (1-s) fraction of the predicted-and-active cells on top of the surprise.
 * Python draws that subset from numpy's GLOBAL RandomState -- under the
 * thread hive the draw order across models is scheduler-dependent, so the
 * Python results there were never reproducible run-to-run.  Here each node
 * draws from its own seed-derived generator: deterministic, and documented
 * as the one intentional deviation (deterministic strengths are
 * bit-identical).
 *
 * MEMORY: every per-step object is a reused member; steady-state stepping
 * allocates nothing (beyond what the core SP/TM do internally, which the
 * earlier optimization rounds already reduced to zero).
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

#include "../rng/NumpyRandom.hpp"
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

    /* One full step over `input`; the post-processed output lands in
     * output() (and predictiveOutput() when return_predictive is on). */
    void step(const htm::SDR &input);

    /* Pointer-based variant for the run loop: the orchestrator wires each
     * node's input slot once (head extras may rewire per record), workers
     * just call step(). */
    void set_input(const htm::SDR *input) { input_ = input; }
    const htm::SDR *input_ptr() const { return input_; }
    void step() { step(*input_); }

    const htm::SDR &output() const { return outFinal_; }
    const htm::SDR &predictiveOutput() const { return predFinal_; }

    /* Post-processed output dimensions (HTMModule.output_dim). */
    const std::vector<htm::UInt> &output_dims() const { return outDims_; }

    /* Spec input dims (the balancer's node weight ~ prod(input_dims)). */
    const std::vector<htm::UInt> &input_dims_ref() const {
        return spec_.input_dims;
    }

    const std::string &name() const { return spec_.name; }
    double last_anomaly() const { return lastAnomaly_; }   // 0.0 before any

    /* Per-step histories (returned to Python at the end of the run). */
    const std::vector<double> &anomaly_history() const { return anomalyHist_; }
    const std::vector<std::int64_t> &activity_history() const { return activityHist_; }

    /* HTMModule.summary() -- the model_summary() line for this node. */
    std::string summary() const;

    void reserve_history(std::size_t n);

private:
    NodeSpec spec_;
    const htm::SDR *input_ = nullptr;      // wired by the runtime
    std::unique_ptr<htm::SpatialPooler> sp_;
    std::unique_ptr<htm::TemporalMemory> tm_;

    std::vector<htm::UInt> columnDims_;    // HTMModule.column_dim
    std::vector<htm::UInt> cellDims_;      // (*column_dim, cellsPerColumn)
    std::vector<htm::UInt> squeezedDims_;  // cellDims_ minus the leading 1
    std::vector<htm::UInt> outDims_;       // post max_pool/flatten (+pred axis)

    std::int64_t iteration_ = 0;
    bool learning_ = true;
    double lastAnomaly_ = 0.0;

    std::vector<double> anomalyHist_;
    std::vector<std::int64_t> activityHist_;

    /* per-step scratch (fixed dims per node) */
    htm::SDR activeColumns_;
    htm::SDR predictiveCells_;
    htm::SDR activeCells_;
    htm::SDR resTmp1_, resTmp2_;
    htm::SDR resOut_;                     // AD residual output (escapes upward)
    htm::SDR outFinal_, predFinal_;
    htm::SDR poolTmp_;                    // pre-flatten pooled scratch
    std::vector<htm::UInt> bits_;         // shared index scratch

    /* fractional-residual sampling (documented deviation; see header) */
    std::unique_ptr<NumpyGenerator> fracRng_;
    std::vector<std::int64_t> fracIdx_;
    std::vector<double> fracP_, fracCdf_, fracX_;
    std::vector<std::int64_t> fracNew_;
    std::vector<std::uint8_t> fracSeen_;

    bool learn_now();                     // learning_hook
    void post_process(const htm::SDR &in, htm::SDR &out);   // pool+flatten
    void residual_into(htm::SDR &out);    // AD residual from active/predictive
};

} // namespace pyramid
