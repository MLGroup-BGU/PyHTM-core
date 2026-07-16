/* ---------------------------------------------------------------------------
 * Pyramid-Model-Core :: threading/WorkerPool.hpp
 *
 * The C++ thread hive: a fixed pool of worker threads with PINNED node
 * ownership and a per-layer barrier, replicating HTMThreadHive's contract:
 *
 *   * worker count resolved like get_n_workers (SLURM_CPUS_PER_TASK ->
 *     SLURM_JOB_CPUS_PER_NODE -> sched_getaffinity -> hardware_concurrency),
 *     capped by the caller (max_workers) and by L0 width;
 *   * nodes distributed by balanced_partition_weighted: LPT over
 *     input-dimension weights, walking layers shallow -> deep and CARRYING
 *     cumulative bucket loads across layers (a verbatim port -- identical
 *     partitions to the Python function, verified by the balancer test);
 *   * run_layer(layer) blocks until every node in the layer finished (the
 *     layer barrier);
 *   * a worker exception aborts the run cleanly: the first error is
 *     captured, the barrier still completes, and the orchestrator rethrows.
 *
 * SCHEDULING vs. OUTPUT: Python's ThreadPoolExecutor de-facto work-steals
 * from a shared queue; the computed partition there informs balance but is
 * not enforced per task.  Here ownership IS enforced -- one node, one
 * thread, for the whole run -- which keeps each model's Connections warm in
 * one core's cache.  Scheduling policy cannot affect outputs: every node's
 * step is a pure function of its own state and its input SDR, and all merge
 * RNGs are freshly seeded per call.  The partition itself is the §-contract
 * behavior and matches Python's exactly.
 *
 * Also runs the parallel build phase (init_all_sp: SP + TM construction per
 * node on its owner thread) -- construction is deterministic per seed, so
 * parallel init is output-identical to serial.
 * ------------------------------------------------------------------------ */
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace pyramid {

class HtmNode;

/* Verbatim port of general.balanced_partition_weighted (LPT with cumulative
 * loads).  Returns per-bucket item indices; `cumulative` is updated. */
std::vector<std::vector<std::size_t>> balanced_partition_weighted(
    std::size_t n_items, const std::vector<double> &weights,
    std::size_t n_buckets, std::vector<double> &cumulative);

/* Port of general.get_n_workers. */
std::size_t get_n_workers(int reserve_main,
                          std::optional<std::int64_t> max_workers);

class WorkerPool {
public:
    /* `layers` holds the node pointers per layer (build order); ownership
     * is computed here exactly like HTMThreadHive.assign_models. */
    WorkerPool(const std::vector<std::vector<HtmNode *>> &layers,
               std::optional<std::int64_t> max_workers);
    ~WorkerPool();

    WorkerPool(const WorkerPool &) = delete;
    WorkerPool &operator=(const WorkerPool &) = delete;

    std::size_t n_workers() const { return workers_.size(); }

    /* Owned node POSITIONS per layer for worker idx (the same partition
     * that fills ownedPerLayer) -- used by the pipeline scheduler. */
    const std::vector<std::vector<std::size_t>> &owned_of(
        std::size_t idx) const {
        return workers_[idx].ownedPosPerLayer;
    }

    /* Parallel SP+TM construction across owners (hive.init_all_sp). */
    void init_all(void);
    void init_all(const std::function<void()> &poll);

    /* Run `fn(worker_idx)` on EVERY worker thread simultaneously and block
     * until all return.  Used by the cross-record pipeline scheduler: the
     * pool supplies the pinned threads; the runtime supplies the loop. */
    /* Run `fn(workerIdx)` on every worker; block until all return. If
     * `poll` is set, the calling thread wakes ~every `poll_ms` to call it
     * (used to check for Ctrl+C on the Python MAIN thread while the GIL is
     * released); if `poll` throws, `stop_flag` (if given) is set so the
     * workers can exit, they are joined, and the exception is rethrown. */
    void run_parallel(const std::function<void(std::size_t)> &fn,
                      const std::function<void()> &poll = nullptr,
                      std::atomic<bool> *stop_flag = nullptr,
                      int poll_ms = 30);

    /* Run one layer: every node's step(input) on its owner thread; the
     * caller must have written each node's input SDR beforehand.  Blocks
     * until the whole layer is done (the barrier); rethrows the first
     * worker error. */
    void run_layer(std::size_t layer_idx,
                   const std::function<void(HtmNode &)> &step_fn);

    /* Join all workers (idempotent; also called by the destructor). */
    void shutdown();

private:
    enum class Cmd : std::uint8_t { None, Init, RunLayer, Custom, Stop };

    struct Worker {
        std::thread thread;
        /* Nodes this worker owns, per layer. */
        std::vector<std::vector<HtmNode *>> ownedPerLayer;
        std::vector<std::vector<std::size_t>> ownedPosPerLayer;
        std::size_t ownedTotal = 0;
    };

    std::vector<Worker> workers_;
    std::atomic<std::size_t> initDone_{0};
    std::atomic<bool> abortInit_{false};   // set on Ctrl+C during init
    std::size_t nNodes_ = 0;                 // total nodes (init progress)
    std::vector<int> pinCores_;   // process-allowed cores (pin targets)
    bool pin_ = false;            // 1:1 thread->core pinning active

    /* One shared command "mailbox": the orchestrator publishes a command +
     * generation counter; workers wake, execute their share, and decrement
     * the pending counter.  One wake per worker per layer. */
    std::mutex mx_;
    std::condition_variable cvWork_, cvDone_;
    Cmd cmd_ = Cmd::None;
    std::uint64_t generation_ = 0;
    std::size_t layerIdx_ = 0;
    const std::function<void(HtmNode &)> *stepFn_ = nullptr;
    const std::function<void(std::size_t)> *customFn_ = nullptr;
    std::size_t pending_ = 0;

    std::mutex errMx_;
    std::string firstError_;   // empty == no error
    std::atomic<bool> stop_{false};

    void worker_main(std::size_t idx);
    void dispatch(Cmd cmd, std::size_t layer_idx,
                  const std::function<void(HtmNode &)> *fn);
    void record_error(const std::string &what);
    void rethrow_if_error();
};

} // namespace pyramid
