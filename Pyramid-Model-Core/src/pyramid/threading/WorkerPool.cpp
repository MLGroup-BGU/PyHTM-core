/* ---------------------------------------------------------------------------
 * Pyramid-Model-Core :: threading/WorkerPool.cpp
 * See WorkerPool.hpp.  Blocks cite the Python they port.
 * ------------------------------------------------------------------------ */
#include "WorkerPool.hpp"
#include <chrono>
#include <cstdio>
#include "../util/StageLog.hpp"
#if defined(__linux__)
  #include <pthread.h>
  #include <sched.h>
#elif defined(_WIN32)
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#endif

#include <htm/utils/Log.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>

#ifdef __linux__
#include <sched.h>
#endif

#include "../runtime/HtmNode.hpp"

namespace pyramid {

namespace {
/* Hard thread->core pinning (Linux + Windows; macOS has no strict
 * affinity API, so the OS scheduler is used there -- identical to the
 * original Python hive's behavior everywhere).
 *
 * Safety guards (never regress): pinning happens ONLY when the worker
 * count fits inside the set of cores the PROCESS is allowed to use
 * (taskset/cgroup/SLURM masks respected); otherwise workers stay
 * OS-scheduled. Worker i of EVERY pool pins to the SAME i-th allowed
 * core, so the build pool's first-touch memory pages land on the core
 * that the run pool's owner of the same nodes will use (NUMA locality
 * across the two-pool hand-off). */
inline std::vector<int> allowed_cores() {
    std::vector<int> cores;
#if defined(__linux__)
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        for (int c = 0; c < CPU_SETSIZE; ++c)
            if (CPU_ISSET(c, &set)) cores.push_back(c);
    }
#elif defined(_WIN32)
    DWORD_PTR proc = 0, sys = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys)) {
        for (int c = 0; c < 64; ++c)
            if (proc & (1ULL << c)) cores.push_back(c);
    }
#endif
    return cores;
}

inline bool pin_current_thread(int core) {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#elif defined(_WIN32)
    return SetThreadAffinityMask(GetCurrentThread(), 1ULL << core) != 0;
#else
    (void)core;
    return false;
#endif
}
}  // namespace

/* ------------- general.balanced_partition_weighted (LPT) ----------------- */
std::vector<std::vector<std::size_t>> balanced_partition_weighted(
        std::size_t n_items, const std::vector<double> &weights,
        std::size_t n_buckets, std::vector<double> &cumulative) {
    if (n_items == 0) return {{}};
    NTA_CHECK(weights.size() == n_items) << "items/weights length mismatch";
    for (double w : weights)
        NTA_CHECK(w > 0.0) << "All weights must be > 0";

    /* DELIBERATE DIVERGENCE from the Python original: Python clamps
     * n_buckets to len(items), so a layer narrower than the pool is assigned
     * only into the FIRST len(items) slots -- e.g. a 1-node layer can only
     * ever land on thread 0, a 3-node layer on threads 0..2 -- regardless of
     * the carried loads. Under the barrier model that was harmless (layers
     * run as separate phases), but in the cross-record pipeline it stacks
     * the whole L0_1->L1_1->L2_1->L3_1 chain onto thread 0, making it the
     * per-record bottleneck while other threads idle. Here every candidate
     * slot competes: narrow layers land on the least-loaded threads of the
     * WHOLE pool, so the cumulative per-thread makespan stays balanced.
     * Empty buckets are fine (the ctor just assigns nothing to them).
     * Output-correctness is partition-independent (test_infra contract #2:
     * multi-worker output == single-worker output, bit-exact). */
    n_buckets = std::max<std::size_t>(1, n_buckets);
    NTA_CHECK(cumulative.size() >= n_buckets)
        << "initial_loads shorter than n_buckets";

    /* Sort by weight DESCENDING with a STABLE sort -- equal-weight items
     * keep their original order, exactly like Python's sorted(). */
    std::vector<std::size_t> order(n_items);
    for (std::size_t i = 0; i < n_items; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) {
                         return weights[a] > weights[b];
                     });

    std::vector<std::vector<std::size_t>> buckets(n_buckets);
    for (std::size_t oi = 0; oi < n_items; ++oi) {
        /* min() over the first n_buckets slots; Python's min() keeps the
         * FIRST minimum on ties -- strict '<' does the same. */
        std::size_t target = 0;
        for (std::size_t b = 1; b < n_buckets; ++b)
            if (cumulative[b] < cumulative[target]) target = b;
        buckets[target].push_back(order[oi]);
        cumulative[target] += weights[order[oi]];
    }
    return buckets;
}

/* ------------------------ general.get_n_workers -------------------------- */
static std::optional<std::int64_t> env_first_int(const char *name) {
    const char *v = std::getenv(name);
    if (!v || !*v) return std::nullopt;
    /* SLURM_JOB_CPUS_PER_NODE can look like "16(x2)" or "16,8"; Python takes
     * int() of the string up to the first non-digit via its own parsing --
     * the PyHTM helper int()s the raw value and falls through on failure.
     * Take the leading integer; reject if none. */
    char *end = nullptr;
    const long long n = std::strtoll(v, &end, 10);
    if (end == v || n <= 0) return std::nullopt;
    /* Python's int() would RAISE on "16(x2)" and the helper catches and
     * falls through; only a fully-numeric value counts. */
    for (const char *p = end; *p; ++p)
        if (!std::isspace(static_cast<unsigned char>(*p))) return std::nullopt;
    return n;
}

std::size_t get_n_workers(int reserve_main,
                          std::optional<std::int64_t> max_workers) {
    std::optional<std::int64_t> n = env_first_int("SLURM_CPUS_PER_TASK");
    if (!n) n = env_first_int("SLURM_JOB_CPUS_PER_NODE");
#ifdef __linux__
    if (!n) {
        cpu_set_t set;
        CPU_ZERO(&set);
        if (sched_getaffinity(0, sizeof(set), &set) == 0) {
            const int c = CPU_COUNT(&set);
            if (c > 0) n = c;
        }
    }
#endif
    if (!n) {
        const unsigned c = std::thread::hardware_concurrency();
        n = c > 0 ? static_cast<std::int64_t>(c) : 1;
    }
    std::int64_t v = *n - reserve_main;
    if (max_workers.has_value()) v = std::min(v, *max_workers);
    return static_cast<std::size_t>(std::max<std::int64_t>(1, v));
}

/* ------------------------------ WorkerPool ------------------------------- */
WorkerPool::WorkerPool(const std::vector<std::vector<HtmNode *>> &layers,
                       std::optional<std::int64_t> max_workers) {
    /* Worker count: get_n_workers(reserve_main=0 -- the thread hive default,
     * where the orchestrator mostly waits), capped to L0 width exactly like
     * HTMThreadHive.assign_models. */
    std::size_t n_workers = get_n_workers(/*reserve_main=*/0, max_workers);
    if (!layers.empty() && !layers[0].empty())
        n_workers = std::min(n_workers, layers[0].size());
    n_workers = std::max<std::size_t>(1, n_workers);

    for (const auto &lay : layers) nNodes_ += lay.size();
    workers_.resize(n_workers);
    for (auto &w : workers_) {
        w.ownedPerLayer.resize(layers.size());
        w.ownedPosPerLayer.resize(layers.size());
    }

    /* Layers shallow -> deep, LPT per layer with carried cumulative loads;
     * weight per node ~ prod(input_dims), min 1 (the _node_weight port). */
    std::vector<double> cumulative(n_workers, 0.0);
    /* Producer bias: in the pipeline, worker 0 additionally encodes every
     * record into the L0 ring (pipe_produce). Seed it with a sub-unit load
     * so equal-weight TIES break away from it while any real weight gap
     * (all node weights are >= 1) still dominates. Net effect: thread 0
     * ends up owning the fewest nodes, offsetting its producer duty. */
    if (n_workers > 1) cumulative[0] = 0.5;
    for (std::size_t li = 0; li < layers.size(); ++li) {
        const auto &nodes = layers[li];
        if (nodes.empty()) continue;
        std::vector<double> weights(nodes.size());
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            std::int64_t total = 1;
            for (auto d : nodes[i]->input_dims_ref())
                total *= static_cast<std::int64_t>(d);
            weights[i] = static_cast<double>(std::max<std::int64_t>(1, total));
        }
        const auto buckets =
            balanced_partition_weighted(nodes.size(), weights, n_workers,
                                        cumulative);
        for (std::size_t b = 0; b < buckets.size(); ++b)
            for (std::size_t idx : buckets[b]) {
                workers_[b].ownedPerLayer[li].push_back(nodes[idx]);
                workers_[b].ownedPosPerLayer[li].push_back(idx);
                ++workers_[b].ownedTotal;
            }
    }

    /* Thread->core pinning policy (see allowed_cores above). */
    pinCores_ = allowed_cores();
    pin_ = !pinCores_.empty() && workers_.size() <= pinCores_.size();
#if !defined(__linux__) && !defined(_WIN32)
    pin_ = false;
#endif
    if (pin_)
        std::printf("[WorkerPool] pinning thread[i] -> core %d..%d "
                    "(1:1, %zu allowed cores)\n",
                    pinCores_.front(),
                    pinCores_[workers_.size() - 1],
                    pinCores_.size());
    else
        std::printf("[WorkerPool] OS thread scheduling (%zu workers, "
                    "%zu allowed cores)\n",
                    workers_.size(), pinCores_.size());

    /* Hive-style assignment log: worker count + per-thread ownership. */
    std::printf("[WorkerPool] using %zu worker thread%s "
                "(SLURM/affinity/hardware chain, capped by L0 width%s)\n",
                workers_.size(), workers_.size() == 1 ? "" : "s",
                max_workers ? " and max_workers" : "");
    for (std::size_t i = 0; i < workers_.size(); ++i) {
        std::printf("  thread[%zu]: %zu models (", i, workers_[i].ownedTotal);
        std::size_t shown = 0;
        for (std::size_t li = 0;
             li < workers_[i].ownedPerLayer.size() && shown < 3; ++li)
            for (HtmNode *n : workers_[i].ownedPerLayer[li]) {
                if (shown == 3) break;
                std::printf("%s%s", shown ? ", " : "", n->name().c_str());
                ++shown;
            }
        std::printf("%s)\n", workers_[i].ownedTotal > 3 ? ", ..." : "");
    }
    std::fflush(stdout);

    for (std::size_t i = 0; i < workers_.size(); ++i)
        workers_[i].thread = std::thread(&WorkerPool::worker_main, this, i);
}

WorkerPool::~WorkerPool() { shutdown(); }

void WorkerPool::record_error(const std::string &what) {
    htm_pyramid::error_line(what.c_str());
    std::lock_guard<std::mutex> lk(errMx_);
    if (firstError_.empty()) firstError_ = what;
}

void WorkerPool::rethrow_if_error() {
    std::lock_guard<std::mutex> lk(errMx_);
    if (!firstError_.empty()) {
        const std::string msg = firstError_;
        firstError_.clear();
        NTA_THROW << "pyramid worker error: " << msg;
    }
}

void WorkerPool::worker_main(std::size_t idx) {
    if (pin_ && idx < pinCores_.size())
        pin_current_thread(pinCores_[idx]);
    std::uint64_t seen = 0;
    for (;;) {
        Cmd cmd;
        std::size_t layer;
        const std::function<void(HtmNode &)> *fn;
        const std::function<void(std::size_t)> *fnc;
        {
            std::unique_lock<std::mutex> lk(mx_);
            cvWork_.wait(lk, [&] { return generation_ != seen; });
            seen = generation_;
            cmd = cmd_;
            layer = layerIdx_;
            fn = stepFn_;
            fnc = customFn_;
        }
        if (cmd == Cmd::Stop) return;

        try {
            if (cmd == Cmd::Init) {
                /* hive.init_all_sp: _init_sp + _init_tm per owned model.
                 * Between nodes we check abortInit_ so a Ctrl+C during init
                 * stops enqueuing new nodes at once (the in-flight node, an
                 * uninterruptible C++ SP/TM build, still completes). */
                for (auto &perLayer : workers_[idx].ownedPerLayer) {
                    for (HtmNode *n : perLayer) {
                        if (abortInit_.load(std::memory_order_acquire)) break;
                        n->init_sp();
                        n->init_tm();
                        const std::size_t k = initDone_.fetch_add(
                            1, std::memory_order_relaxed) + 1;
                        std::printf("\r[Init] %zu/%zu nodes", k, nNodes_);
                        std::fflush(stdout);
                    }
                    if (abortInit_.load(std::memory_order_acquire)) break;
                }
            } else if (cmd == Cmd::RunLayer) {
                for (HtmNode *n : workers_[idx].ownedPerLayer[layer])
                    (*fn)(*n);
            } else if (cmd == Cmd::Custom) {
                (*fnc)(idx);
            }
        } catch (const std::exception &e) {
            record_error(e.what());
        } catch (...) {
            record_error("unknown worker exception");
        }

        {
            std::lock_guard<std::mutex> lk(mx_);
            if (--pending_ == 0) cvDone_.notify_all();
        }
    }
}

void WorkerPool::dispatch(Cmd cmd, std::size_t layer_idx,
                          const std::function<void(HtmNode &)> *fn) {
    std::unique_lock<std::mutex> lk(mx_);
    cmd_ = cmd;
    layerIdx_ = layer_idx;
    stepFn_ = fn;
    pending_ = workers_.size();
    ++generation_;
    cvWork_.notify_all();
    cvDone_.wait(lk, [&] { return pending_ == 0; });   // the layer barrier
}

void WorkerPool::init_all() { init_all(nullptr); }

void WorkerPool::init_all(const std::function<void()> &poll) {
    initDone_.store(0, std::memory_order_relaxed);
    abortInit_.store(false, std::memory_order_release);
    {
        std::unique_lock<std::mutex> lk(mx_);
        cmd_ = Cmd::Init;
        layerIdx_ = 0;
        stepFn_ = nullptr;
        pending_ = workers_.size();
        ++generation_;
        cvWork_.notify_all();
        if (!poll) {
            cvDone_.wait(lk, [&] { return pending_ == 0; });
        } else {
            // Poll for Ctrl+C on the calling (Python main) thread while the
            // init pool runs with the GIL released. On interrupt the poll
            // throws; we wait for the in-flight init steps to finish (each is
            // short and bounded) so no worker is left running, then rethrow.
            for (;;) {
                if (cvDone_.wait_for(lk, std::chrono::milliseconds(30),
                                     [&] { return pending_ == 0; }))
                    break;
                lk.unlock();
                try {
                    poll();
                } catch (...) {
                    // Stop workers from starting new nodes, then wait for the
                    // in-flight ones to finish so no thread is left running.
                    abortInit_.store(true, std::memory_order_release);
                    lk.lock();
                    cvDone_.wait(lk, [&] { return pending_ == 0; });
                    lk.unlock();
                    abortInit_.store(false, std::memory_order_release);
                    rethrow_if_error();
                    throw;
                }
                lk.lock();
            }
        }
    }
    if (nNodes_) std::printf("\n");
    rethrow_if_error();
}

void WorkerPool::run_parallel(const std::function<void(std::size_t)> &fn,
                              const std::function<void()> &poll,
                              std::atomic<bool> *stop_flag, int poll_ms) {
    {
        std::unique_lock<std::mutex> lk(mx_);
        cmd_ = Cmd::Custom;
        customFn_ = &fn;
        pending_ = workers_.size();
        ++generation_;
        cvWork_.notify_all();

        if (!poll) {
            cvDone_.wait(lk, [&] { return pending_ == 0; });
        } else {
            // Timed wait so the calling (Python main) thread can poll for
            // signals while the workers run with the GIL released.
            for (;;) {
                if (cvDone_.wait_for(lk, std::chrono::milliseconds(poll_ms),
                                     [&] { return pending_ == 0; }))
                    break;                       // all workers finished
                lk.unlock();
                try {
                    poll();                      // may throw (e.g. Ctrl+C)
                } catch (...) {
                    if (stop_flag)
                        stop_flag->store(true, std::memory_order_release);
                    lk.lock();
                    cvDone_.wait(lk, [&] { return pending_ == 0; });
                    lk.unlock();
                    rethrow_if_error();
                    throw;                       // rethrow the poll exception
                }
                lk.lock();
            }
        }
    }
    rethrow_if_error();
}

void WorkerPool::run_layer(std::size_t layer_idx,
                           const std::function<void(HtmNode &)> &step_fn) {
    dispatch(Cmd::RunLayer, layer_idx, &step_fn);
    rethrow_if_error();
}

void WorkerPool::shutdown() {
    if (workers_.empty()) return;
    bool any_joinable = false;
    for (auto &w : workers_) any_joinable |= w.thread.joinable();
    if (any_joinable) {
        {
            std::unique_lock<std::mutex> lk(mx_);
            cmd_ = Cmd::Stop;
            pending_ = workers_.size();
            ++generation_;
            cvWork_.notify_all();
        }
        for (auto &w : workers_)
            if (w.thread.joinable()) w.thread.join();
    }
    workers_.clear();
}

} // namespace pyramid
