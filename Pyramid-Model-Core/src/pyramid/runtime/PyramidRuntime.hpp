/* ---------------------------------------------------------------------------
 * Pyramid-Model-Core :: runtime/PyramidRuntime.hpp
 *
 * The pyramid itself: ModelPyramid / FeatureSUPyramid's build() and run()
 * re-hosted in C++.  One object owns the whole model -- encoders, nodes,
 * merge machinery, worker pool, per-node input/output slots and results --
 * and executes the exact Python run loop:
 *
 *   per record:
 *     encode features -> feature-plan merges -> L0 input slots
 *     for each layer:
 *       run layer (pinned workers, layer barrier)
 *       lateral_exchange (non-head layers)
 *       record layer outputs (history)
 *       next-layer inputs:  broadcast_head | children merge
 *            [+ anomaly_boost duplicates] [+ residual grandparents]
 *            | squeeze-skip variant
 *       recurrent_head merge + long-term-memory update (head input only)
 *     capture prev-head output (recurrent)
 *
 * plus the build phase: an optional streaming pass over the training slice
 * for sample-derived encoder parameters, encoder construction, node
 * construction (model seeds = pyramid_seed * counter in build order), and
 * parallel SP+TM init across the pool.
 *
 * DESIGN NOTES
 *   * Everything is addressed by (layer, position): node lookup maps are
 *     used only at build time; the per-record loop walks flat vectors, so
 *     steady-state stepping allocates NOTHING in this layer.
 *   * layer outputs / history hold POINTERS into per-node slots that are
 *     valid for the whole record (slots are rewritten only by the next
 *     record's steps).
 *   * use_predictive carries (active, predictive) PAIRS through boost /
 *     residual exactly like Python's zip(*to_merge) -- the pair list is
 *     built first, then unzipped.
 *   * lateral_exchange / broadcast_head are rejected at build when
 *     use_predictive is on: the Python implementation crashes there
 *     (sdr_merge over tuples); the C++ gives the clear error instead.
 *
 * GIL: this class never touches Python.  The bindings layer releases the
 * GIL around build() / run(); PyReaderSource re-acquires it per batch.
 * ------------------------------------------------------------------------ */
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "../data/RecordSource.hpp"
#include "../encoders/EncoderBuild.hpp"
#include "../merge/SdrMerge.hpp"
#include "../spec/PyramidSpec.hpp"
#include "../threading/WorkerPool.hpp"
#include "HtmNode.hpp"

namespace pyramid {

/* Per-node results handed back to Python at the end of a run. */
struct NodeResults {
    std::vector<double> anomaly;          // may be empty (anomaly_score off)
    std::vector<std::int64_t> activity;   // n active columns per step
};

class PyramidRuntime {
public:
    explicit PyramidRuntime(PyramidSpec spec);
    ~PyramidRuntime();

    PyramidRuntime(const PyramidRuntime &) = delete;
    PyramidRuntime &operator=(const PyramidRuntime &) = delete;

    /* Build: optional training-slice sample pass over `source` (only when a
     * feature needs it), encoder construction, node construction, pool
     * spin-up and parallel SP+TM init.  Leaves `source` reset. */
    void build(RecordSource &source);

    /* Run for min(total, iterations) records (all records when nullopt).
     * `progress`, when set, is invoked every `progress_every` records and
     * once at the end with (records_done, total_or_minus1). */
    void run(RecordSource &source, std::optional<std::int64_t> iterations,
             const std::function<void(std::int64_t, std::int64_t)> &progress =
                 nullptr,
             std::int64_t progress_every = 5000);

    /* -- results ---------------------------------------------------------- */
    const std::string &head_name() const { return headName_; }
    std::vector<std::string> node_names_in_layer_order() const;
    NodeResults results_for(const std::string &node) const;
    std::string summary_for(const std::string &node) const;
    std::int64_t records_processed() const { return currentIter_; }
    std::size_t n_workers() const { return pool_ ? pool_->n_workers() : 0; }

private:
    PyramidSpec spec_;
    std::string headName_;
    std::size_t nLayers_ = 0;
    std::size_t headLayer_ = 0, headPos_ = 0;

    /* encoders, in features_cfg order (== spec_.encoders order) */
    std::vector<std::unique_ptr<FeatureEncoder>> encoders_;
    std::vector<std::size_t> encColIdx_;      // encoder -> data column index
    std::vector<htm::SDR> encSlots_;          // per-feature encoding scratch
    std::map<std::string, std::size_t> encByName_;

    /* nodes + layers (position-indexed) */
    std::map<std::string, std::unique_ptr<HtmNode>> nodes_;
    std::vector<std::vector<HtmNode *>> layers_;
    struct Pos { std::size_t layer, pos; };
    std::map<std::string, Pos> nodePos_;

    /* per-node input slots (fixed dims; wired into the nodes at build) */
    std::map<std::string, htm::SDR> inputSlots_;

    /* resolved-at-build routing tables (per layer > 0, per parent):
     * children as (layer==li, pos) indices; residual grandparents as
     * positions in layer li-1; feature-plan encoders per L0 node. */
    std::vector<std::vector<std::vector<std::size_t>>> childPos_;   // [li][p][k]
    std::vector<std::vector<std::vector<std::size_t>>> resPos_;     // [li][p][k]
    std::vector<std::vector<std::size_t>> planEnc_;                 // [L0 pos][k]

    /* merge scratch (allocated per site at build) */
    SdrMerger merger_;
    std::vector<htm::SDR> lateralSlots_;      // flat: (layer,pos) via offsets
    std::vector<std::size_t> lateralOffset_;  // per layer
    std::map<std::string, htm::SDR> mergedActiveSlots_;   // use_predictive
    std::map<std::string, htm::SDR> mergedPredSlots_;     // use_predictive
    std::map<std::string, htm::SDR> skipSlots_;           // squeeze_skip
    htm::SDR headBufA_, headBufB_;            // recurrent / LTM chain
    htm::SDR memorySdr_, memoryBuf_;          // long-term memory
    bool memoryInitialized_ = false;
    htm::SDR prevHeadSdr_;                    // recurrent head feedback
    bool prevHeadValid_ = false;

    /* per-record output views: [layer][pos] -> (active, predictive) */
    using OutPair = std::pair<const htm::SDR *, const htm::SDR *>;
    std::vector<std::vector<OutPair>> outView_;

    std::vector<const htm::SDR *> mergeInputs_;   // reused pointer scratch
    std::vector<OutPair> pairScratch_;            // use_predictive zip list

    std::unique_ptr<WorkerPool> pool_;
    std::int64_t currentIter_ = 0;

    /* -- cross-record pipeline ------------------------------------------
     * Records-in-flight scheduling: a node runs record t as soon as all of
     * its PRODUCERS finished t (dependency release, no layer barrier) and
     * its ring slot t%K is free (all CONSUMERS finished t-K).  Each node
     * still processes its own records strictly in order with inputs
     * identical to the barrier path, so outputs are bit-identical.
     * Enabled when: pipeline_depth != 1 (0 = auto, >1 = explicit K; 1 = off),
     * no lateral_exchange (the only same-record intra-layer dependency),
     * and more than one worker. K itself = clamp(n_workers, K_MIN, K_MAX)
     * in the auto case -- see the K policy block in PipelineRun.cpp. */
    struct WorkerScratch {
        SdrMerger merger;
        std::vector<const htm::SDR *> inputs;
        std::vector<OutPair> pairs;
    };
    bool pipelineOn_ = false;
    std::int64_t pipeK_ = 1;
    std::vector<std::vector<std::vector<htm::SDR>>> outRing_;   // [li][p][k]
    std::vector<std::vector<std::vector<htm::SDR>>> predRing_;  // use_predictive
    std::vector<std::vector<std::vector<double>>>  anomRing_;   // anomaly_boost
    std::vector<std::vector<htm::SDR>> l0InRing_;               // [L0 pos][k]
    std::vector<std::vector<std::unique_ptr<std::atomic<std::int64_t>>>>
        nodeDone_;                                              // last t done
    std::atomic<std::int64_t> srcDone_{-1};
    std::atomic<std::int64_t> tEnd_{0};
    std::vector<std::vector<std::vector<Pos>>> producersOf_;    // [li][p]
    std::vector<std::vector<std::vector<Pos>>> consumersOf_;    // [li][p]
    std::vector<WorkerScratch> ws_;
    std::mutex pipeMx_;
    std::condition_variable pipeCv_;
    std::uint64_t pipeEpoch_ = 0;

    void allocate_pipeline();
    void pipe_notify();
    void pipe_wake();   // lightweight per-record wakeup for parked workers
    bool pipe_ready(std::size_t li, std::size_t p, std::int64_t t) const;
    bool pipe_slot_free(std::size_t li, std::size_t p, std::int64_t t) const;
    void pipe_exec_node(std::size_t li, std::size_t p, std::int64_t t,
                        WorkerScratch &ws);
    bool pipe_produce(RecordSource &source, std::int64_t limit,
                      WorkerScratch &ws);
    void encode_into_l0ring(const Record &rec, std::int64_t t,
                            WorkerScratch &ws);
    void merge_children_ring(std::size_t li, std::size_t p, std::int64_t t,
                             bool with_residual, WorkerScratch &ws,
                             htm::SDR &out);
    const htm::SDR *apply_head_extras_ws(const htm::SDR *current,
                                         WorkerScratch &ws);
    void ws_merge_or_copy(WorkerScratch &ws, MergeMode mode,
                          const MergeParams &params, htm::SDR &out);
    std::int64_t run_pipelined(RecordSource &source, std::int64_t limit,
                               std::int64_t progress_every,
                               std::int64_t total);

    /* -- build helpers ---------------------------------------------------- */
    void collect_samples(RecordSource &source,
                         std::map<std::string, std::vector<double>> &samples);
    void build_encoders(RecordSource &source);
    void build_nodes();
    void build_routing();
    void allocate_slots();
    void validate_dims_and_flags();

    /* -- run helpers ------------------------------------------------------ */
    void encode_record(const Record &rec);
    void run_one_layer(std::size_t li);
    void lateral_exchange(std::size_t li);
    void merge_into_next(std::size_t li);
    void merge_children_for(std::size_t li, std::size_t parent_pos,
                            bool with_residual, htm::SDR &out);
    const htm::SDR *apply_head_extras(const htm::SDR *current);
    void merge_or_copy(MergeMode mode, const MergeParams &params,
                       htm::SDR &out);   // over mergeInputs_
};

} // namespace pyramid