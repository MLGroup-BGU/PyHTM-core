/* ---------------------------------------------------------------------------
 * Pyramid-Model-Core :: runtime/PyramidRuntime.hpp
 *
 * The pyramid itself. One object owns the whole model -- encoders, nodes,
 * the concatenation scratch, the worker pool, per-node input/output slots
 * and results -- and executes the run loop:
 *
 *   per record:
 *     encode features -> concatenate each L0 node's features -> L0 inputs
 *     for each layer:
 *       run layer (pinned workers, layer barrier)
 *       next-layer inputs: each parent gets its children's outputs
 *                          concatenated, in children order
 *
 * plus the build phase: an optional streaming pass over the training slice
 * for sample-derived encoder parameters, encoder construction, node
 * construction (model seeds = pyramid_seed * counter in build order), and
 * parallel SP+TM init across the pool. With more than one worker the run can
 * use the cross-record pipeline instead (PipelineRun.cpp): same per-node
 * inputs, records in flight, bit-identical results.
 *
 * DESIGN NOTES
 *   * Everything is addressed by (layer, position): node lookup maps are
 *     used only at build time; the per-record loop walks flat vectors, so
 *     steady-state stepping allocates NOTHING in this layer.
 *   * layer outputs hold POINTERS into per-node slots that are valid for the
 *     whole record (slots are rewritten only by the next record's steps).
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
    std::vector<double> anomaly;          // raw anomaly score per step
    std::vector<std::int64_t> activity;   // n active columns per step
    std::vector<std::uint16_t> pred_columns, pred_hits, pred_cells;
    std::vector<float> sp_overlap;
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

    /* resolved-at-build routing tables: children of each parent (layer > 0)
     * as positions in the layer below; feature-plan encoders per L0 node. */
    std::vector<std::vector<std::vector<std::size_t>>> childPos_;   // [li][p][k]
    std::vector<std::vector<std::size_t>> planEnc_;                 // [L0 pos][k]

    /* concatenation scratch */
    SdrConcat concat_;

    /* per-record output views: [layer][pos] -> the node's output */
    std::vector<std::vector<const htm::SDR *>> outView_;

    std::vector<const htm::SDR *> mergeInputs_;   // reused pointer scratch

    std::unique_ptr<WorkerPool> pool_;
    std::int64_t currentIter_ = 0;

    /* -- cross-record pipeline ------------------------------------------
     * Records-in-flight scheduling: a node runs record t as soon as all of
     * its PRODUCERS finished t (dependency release, no layer barrier) and
     * its ring slot t%K is free (all CONSUMERS finished t-K).  Each node
     * still processes its own records strictly in order with inputs
     * identical to the barrier path, so outputs are bit-identical.
     * Enabled when: pipeline_depth != 1 (0 = auto, >1 = explicit K; 1 = off)
     * and more than one worker. K itself = clamp(n_workers, K_MIN, K_MAX)
     * in the auto case -- see the K policy block in PipelineRun.cpp. */
    struct WorkerScratch {
        SdrConcat concat;
        std::vector<const htm::SDR *> inputs;
    };
    bool pipelineOn_ = false;
    std::int64_t pipeK_ = 1;
    std::vector<std::vector<std::vector<htm::SDR>>> outRing_;   // [li][p][k]
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
                             WorkerScratch &ws, htm::SDR &out);
    void ws_concat_or_copy(WorkerScratch &ws, htm::SDR &out);
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
    void validate_dims();

    /* -- run helpers ------------------------------------------------------ */
    void encode_record(const Record &rec);
    void run_one_layer(std::size_t li);
    void merge_into_next(std::size_t li);
    void merge_children_for(std::size_t li, std::size_t parent_pos,
                            htm::SDR &out);
    void concat_or_copy(htm::SDR &out);   // over mergeInputs_
};

} // namespace pyramid