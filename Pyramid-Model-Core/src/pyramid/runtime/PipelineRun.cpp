/* ---------------------------------------------------------------------------
 * Cross-record pipeline (appended implementation section).
 * See the member-block comment in PyramidRuntime.hpp for the model.
 * ------------------------------------------------------------------------- */
#include "PyramidRuntime.hpp"
#include "../util/StageLog.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <thread>

namespace pyramid {

using htm::SDR;

/* =========================================================================
 * PIPELINE DEPTH POLICY  --  K = clamp( floor(n_workers * FACTOR), MIN, MAX )
 * =========================================================================
 * K is HOW MANY RECORDS may be in flight through the pyramid at once. It is
 * NOT the thread count -- that is n_workers = min(L0 width, cores), resolved
 * by the WorkerPool. K is tied to n_workers so every worker can hold roughly
 * one in-flight record instead of parking while upper layers lag: at
 * n_workers=1 the pipeline is off anyway; small pyramids (few workers) still
 * get at least K_MIN depth (measured sweet spot on phase1: K=16); large
 * pyramids get ~n_workers so all threads stay fed; K_MAX caps ring memory on
 * pathologically wide models (K x per-record output buffers; the ring always
 * stays a fraction of the synapse tables, but the marginal throughput past
 * this point is negligible -- measured flat by ~K=32 on phase1).
 *
 * ==> EDIT THESE THREE to retune the pipeline depth for experiments. <==
 * K_WORKER_FACTOR scales depth vs. worker count; floor() keeps K an integer
 * for any factor (e.g. 1.5 -> floor(n_workers*1.5)). A spec pipeline_depth
 * with an EXPLICIT value > 1 (see pyramid_spec) still forces that exact K;
 * the env var PYRAMID_PIPE_DEPTH overrides everything.
 */
static constexpr double        K_WORKER_FACTOR = 1.0;  // K ~= n_workers * this
static constexpr std::int64_t  K_MIN           = 16;   // floor: phase1 optimum
static constexpr std::int64_t  K_MAX           = 64;   // cap:   ring-memory guard

/* K for the AUTO case: floor(n_workers * factor), then clamped to [MIN, MAX]. */
static inline std::int64_t auto_pipeline_depth(std::int64_t n_workers) {
    std::int64_t k = static_cast<std::int64_t>(
        std::floor(static_cast<double>(n_workers) * K_WORKER_FACTOR));
    if (k < K_MIN) return K_MIN;
    if (k > K_MAX) return K_MAX;
    return k;
}

/* Clamp an EXPLICIT override (spec pipeline_depth > 1) to the same bounds. */
static inline std::int64_t clamp_pipeline_depth(std::int64_t k) {
    if (k < K_MIN) return K_MIN;
    if (k > K_MAX) return K_MAX;
    return k;
}

// A single "pause" hint for spin-wait loops: lets the CPU de-prioritize the
// spinning hardware thread (freeing pipeline/issue slots for its hyperthread
// sibling) and cuts power. Far cheaper than sched_yield in a tight loop.
static inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield");
#else
    std::this_thread::yield();
#endif
}

void PyramidRuntime::pipe_notify() {
    {
        std::lock_guard<std::mutex> lk(pipeMx_);
        ++pipeEpoch_;
    }
    pipeCv_.notify_all();
}

// Lightweight wakeup for parked workers. Called at most once per produced
// record by the single producer, so contention is negligible. We still take
// the mutex briefly to publish the epoch under the same lock the waiters use
// to arm their wait (this closes the lost-wakeup race); the cost is tiny at
// O(records) frequency and only when workers are actually parked.
void PyramidRuntime::pipe_wake() {
    {
        std::lock_guard<std::mutex> lk(pipeMx_);
        ++pipeEpoch_;
    }
    pipeCv_.notify_all();
}

void PyramidRuntime::ws_merge_or_copy(WorkerScratch &ws, MergeMode mode,
                                      const MergeParams &params, SDR &out) {
    if (ws.inputs.size() == 1)
        out.setSDR(*ws.inputs[0]);      // Python returns the object itself
    else
        ws.merger.merge(ws.inputs, mode, spec_.merge.concat_axis, params,
                        out);
}

void PyramidRuntime::allocate_pipeline() {
    /* Resolve K (records in flight). Three cases, in priority order:
     *   spec.pipeline_depth <= 1        -> pipeline OFF (untouched below).
     *   spec.pipeline_depth == 0 (AUTO) -> K = clamp(n_workers, K_MIN, K_MAX)
     *   spec.pipeline_depth >  1        -> EXPLICIT override, used verbatim
     *                                      (still clamped to [K_MIN, K_MAX]).
     * pyramid_spec sends 0 as the default ("auto"); a runner may pass an
     * explicit >1 to pin K. The K policy itself (K_MIN/K_MAX/clampK) lives
     * in one place at the top of this file. n_workers is known here because
     * the WorkerPool is already built. */
    const std::int64_t nW =
        pool_ ? static_cast<std::int64_t>(pool_->n_workers()) : 1;
    if (spec_.pipeline_depth <= 1 && spec_.pipeline_depth != 0) {
        pipeK_ = spec_.pipeline_depth;        // <=1 and not the auto sentinel: OFF
    } else if (spec_.pipeline_depth == 0) {
        pipeK_ = auto_pipeline_depth(nW);     // AUTO: tie depth to worker count
    } else {
        pipeK_ = clamp_pipeline_depth(spec_.pipeline_depth); // EXPLICIT override
    }
    /* PYRAMID_PIPE_DEPTH -- optional EXPERIMENT KNOB (environment variable),
     * overrides everything above. K = records in flight. Bit-exact for ANY
     * K >= 2 (each node still steps its records strictly in order; dependency
     * gates are K-independent -- verified by the bit-exact matrix at
     * K=2/4/8). Accepted range [2, K_MAX]. */
    if (const char *e = std::getenv("PYRAMID_PIPE_DEPTH")) {
        char *end = nullptr;
        const long v = std::strtol(e, &end, 10);
        if (end != e && v >= 2 && v <= K_MAX)
            pipeK_ = static_cast<std::int64_t>(v);
    }
    pipelineOn_ = pipeK_ > 1 && !spec_.flags.lateral_exchange &&
                  pool_ && pool_->n_workers() > 1;
    if (!pipelineOn_) return;
    const std::size_t K = static_cast<std::size_t>(pipeK_);

    outRing_.resize(nLayers_);
    if (spec_.flags.use_predictive) predRing_.resize(nLayers_);
    if (spec_.flags.anomaly_boost) anomRing_.resize(nLayers_);
    nodeDone_.resize(nLayers_);
    producersOf_.assign(nLayers_, {});
    consumersOf_.assign(nLayers_, {});
    std::size_t ringBytes = 0;

    for (std::size_t li = 0; li < nLayers_; ++li) {
        const std::size_t n = layers_[li].size();
        outRing_[li].resize(n);
        if (spec_.flags.use_predictive) predRing_[li].resize(n);
        if (spec_.flags.anomaly_boost)
            anomRing_[li].assign(n, std::vector<double>(K, 0.0));
        nodeDone_[li].resize(n);
        producersOf_[li].assign(n, {});
        consumersOf_[li].assign(n, {});
        for (std::size_t p = 0; p < n; ++p) {
            const auto dims = layers_[li][p]->output().dimensions;
            outRing_[li][p].resize(K);
            for (auto &s : outRing_[li][p]) s.initialize(dims);
            ringBytes += K * layers_[li][p]->output().size;
            if (spec_.flags.use_predictive) {
                const auto pd =
                    layers_[li][p]->predictiveOutput().dimensions;
                predRing_[li][p].resize(K);
                for (auto &s : predRing_[li][p]) s.initialize(pd);
                ringBytes += K * layers_[li][p]->predictiveOutput().size;
            }
            nodeDone_[li][p] =
                std::make_unique<std::atomic<std::int64_t>>(-1);
        }
    }

    /* L0 input rings mirror the wired input slots' dims. */
    l0InRing_.resize(layers_[0].size());
    for (std::size_t p = 0; p < layers_[0].size(); ++p) {
        const auto dims = inputSlots_.at(spec_.layers[0][p]).dimensions;
        l0InRing_[p].resize(K);
        for (auto &s : l0InRing_[p]) s.initialize(dims);
        ringBytes += K * inputSlots_.at(spec_.layers[0][p]).size;
    }

    /* Producer lists (who must finish t before I can run t) and the
     * derived consumer lists (who reads my slot t, gating overwrite at
     * t+K).  Mirrors merge_into_next's routing exactly. */
    const bool residual_on =
        spec_.flags.residual && !spec_.flags.squeeze_skip;
    for (std::size_t li = 1; li < nLayers_; ++li) {
        const bool is_head = (li + 1 == nLayers_);
        for (std::size_t p = 0; p < layers_[li].size(); ++p) {
            auto &prod = producersOf_[li][p];
            if (spec_.flags.broadcast_head && is_head &&
                !spec_.flags.squeeze_skip) {
                for (std::size_t l = 0; l < li; ++l)
                    for (std::size_t q = 0; q < layers_[l].size(); ++q)
                        prod.push_back({l, q});
            } else {
                for (std::size_t cp : childPos_[li][p])
                    prod.push_back({li - 1, cp});
                if (residual_on && li >= 2)
                    for (std::size_t gp : resPos_[li][p])
                        prod.push_back({li - 2, gp});
            }
            for (const Pos &q : prod)
                consumersOf_[q.layer][q.pos].push_back({li, p});
        }
    }

    std::printf("[Pipeline] ENABLED: depth K=%lld, "
                "in-flight record buffers %.1f KB\n",
                static_cast<long long>(pipeK_), ringBytes / 1024.0);
    std::fflush(stdout);
}

bool PyramidRuntime::pipe_ready(std::size_t li, std::size_t p,
                                std::int64_t t) const {
    if (li == 0) return srcDone_.load(std::memory_order_acquire) >= t;
    for (const Pos &q : producersOf_[li][p])
        if (nodeDone_[q.layer][q.pos]->load(std::memory_order_acquire) < t)
            return false;
    return true;
}

bool PyramidRuntime::pipe_slot_free(std::size_t li, std::size_t p,
                                    std::int64_t t) const {
    const std::int64_t need = t - pipeK_;
    for (const Pos &c : consumersOf_[li][p])
        if (nodeDone_[c.layer][c.pos]->load(std::memory_order_acquire) <
            need)
            return false;
    return true;
}

void PyramidRuntime::encode_into_l0ring(const Record &rec, std::int64_t t,
                                        WorkerScratch &ws) {
    /* Producer-exclusive: encSlots_ are touched only by the producer. */
    for (std::size_t i = 0; i < encoders_.size(); ++i) {
        const std::size_t col = encColIdx_[i];
        if (encoders_[i]->is_datetime())
            encoders_[i]->encode(rec.dts[col], encSlots_[i]);
        else
            encoders_[i]->encode(rec.nums[col], encSlots_[i]);
    }
    const std::size_t slot = static_cast<std::size_t>(t % pipeK_);
    for (std::size_t p = 0; p < layers_[0].size(); ++p) {
        SDR &out = l0InRing_[p][slot];
        ws.inputs.clear();
        for (std::size_t ei : planEnc_[p])
            ws.inputs.push_back(&encSlots_[ei]);
        if (!spec_.merge.has_feature_mode)
            out.setSDR(*ws.inputs[0]);
        else
            ws_merge_or_copy(ws, spec_.merge.feature_mode,
                             spec_.merge.feature_params, out);
    }
}

bool PyramidRuntime::pipe_produce(RecordSource &source, std::int64_t limit,
                                  WorkerScratch &ws) {
    const std::int64_t t = srcDone_.load(std::memory_order_relaxed) + 1;
    if (t >= tEnd_.load(std::memory_order_acquire)) return false;
    if (limit >= 0 && t >= limit) {
        tEnd_.store(t, std::memory_order_release);
        pipe_notify();
        return false;
    }
    /* source slot t%K free once every L0 node consumed t-K */
    const std::int64_t need = t - pipeK_;
    for (std::size_t p = 0; p < layers_[0].size(); ++p)
        if (nodeDone_[0][p]->load(std::memory_order_acquire) < need)
            return false;

    Record rec;
    if (!source.next(rec)) {
        tEnd_.store(t, std::memory_order_release);
        pipe_notify();
        return false;
    }
    encode_into_l0ring(rec, t, ws);
    srcDone_.store(t, std::memory_order_release);
    // One wakeup per produced record, issued only by the single producer
    // (worker 0). This is O(records), not O(records x nodes) -- it is the
    // cheap signal that lets PARKED workers resume without a hot spin, while
    // avoiding the notify-per-node herd that serialized the pool before.
    pipe_wake();
    return true;
}

void PyramidRuntime::merge_children_ring(std::size_t li, std::size_t p,
                                         std::int64_t t, bool with_residual,
                                         WorkerScratch &ws, SDR &out) {
    const std::size_t k = static_cast<std::size_t>(t % pipeK_);
    const auto &children = childPos_[li][p];
    const std::string &parent = spec_.layers[li][p];
    const bool is_head = (li + 1 == nLayers_);

    if (spec_.flags.broadcast_head && is_head &&
        !spec_.flags.squeeze_skip) {
        ws.inputs.clear();
        for (std::size_t l = 0; l < li; ++l)
            for (std::size_t q = 0; q < layers_[l].size(); ++q)
                ws.inputs.push_back(&outRing_[l][q][k]);
        ws_merge_or_copy(ws, spec_.merge.htm_mode, spec_.merge.htm_params,
                         out);
        return;
    }

    if (spec_.flags.squeeze_skip) {
        ws.inputs.clear();
        for (std::size_t cp : children)
            ws.inputs.push_back(&outRing_[li - 1][cp][k]);
        if (children.size() > 1) {
            SDR &skip = skipSlots_.at(parent);       // owner-exclusive
            ws.merger.merge(ws.inputs, spec_.merge.htm_mode,
                            spec_.merge.concat_axis,
                            spec_.merge.htm_params, skip);
            ws.inputs.push_back(&skip);
        }
        ws_merge_or_copy(ws, spec_.merge.htm_mode, spec_.merge.htm_params,
                         out);
        return;
    }

    ws.pairs.clear();
    for (std::size_t cp : children) {
        OutPair pr{&outRing_[li - 1][cp][k],
                   spec_.flags.use_predictive ? &predRing_[li - 1][cp][k]
                                              : nullptr};
        std::int64_t n_copies = 1;
        if (spec_.flags.anomaly_boost) {
            const double a = anomRing_[li - 1][cp][k];
            n_copies =
                1 + static_cast<std::int64_t>(std::nearbyint(a * 3.0));
        }
        for (std::int64_t c = 0; c < n_copies; ++c) ws.pairs.push_back(pr);
    }
    if (with_residual)
        for (std::size_t gp : resPos_[li][p])
            ws.pairs.push_back(
                {&outRing_[li - 2][gp][k],
                 spec_.flags.use_predictive ? &predRing_[li - 2][gp][k]
                                            : nullptr});

    if (spec_.flags.use_predictive) {
        SDR &ma = mergedActiveSlots_.at(parent);     // owner-exclusive
        SDR &mp = mergedPredSlots_.at(parent);
        ws.inputs.clear();
        for (const auto &pr : ws.pairs) ws.inputs.push_back(pr.first);
        ws_merge_or_copy(ws, spec_.merge.htm_mode, spec_.merge.htm_params,
                         ma);
        ws.inputs.clear();
        for (const auto &pr : ws.pairs) ws.inputs.push_back(pr.second);
        ws_merge_or_copy(ws, spec_.merge.htm_mode, spec_.merge.htm_params,
                         mp);
        ws.inputs.clear();
        ws.inputs.push_back(&ma);
        ws.inputs.push_back(&mp);
        ws.merger.merge(ws.inputs, MergeMode::Stack, 0,
                        spec_.merge.htm_params, out);
        return;
    }

    ws.inputs.clear();
    for (const auto &pr : ws.pairs) ws.inputs.push_back(pr.first);
    ws_merge_or_copy(ws, spec_.merge.htm_mode, spec_.merge.htm_params, out);
}

const SDR *PyramidRuntime::apply_head_extras_ws(const SDR *current,
                                                WorkerScratch &ws) {
    /* Head-owner exclusive; state members (prevHeadSdr_, memorySdr_,
     * headBufA_/B_) are touched only here, strictly in record order. */
    if (spec_.flags.recurrent_head && prevHeadValid_) {
        ws.inputs.clear();
        ws.inputs.push_back(current);
        ws.inputs.push_back(&prevHeadSdr_);
        ws.merger.merge(ws.inputs, spec_.merge.htm_mode,
                        spec_.merge.concat_axis, spec_.merge.htm_params,
                        headBufA_);
        current = &headBufA_;
    }
    if (spec_.flags.long_term_memory) {
        if (!memoryInitialized_) {
            memorySdr_.setSDR(*current);
            memoryInitialized_ = true;
        } else {
            ws.inputs.clear();
            for (std::int64_t c = 0; c < spec_.flags.memory_inertia; ++c)
                ws.inputs.push_back(&memorySdr_);
            ws.inputs.push_back(current);
            ws.merger.merge(ws.inputs, spec_.merge.htm_mode,
                            spec_.merge.concat_axis, spec_.merge.htm_params,
                            memoryBuf_);
            memorySdr_.setSDR(memoryBuf_);
        }
        ws.inputs.clear();
        ws.inputs.push_back(current);
        ws.inputs.push_back(&memorySdr_);
        ws.merger.merge(ws.inputs, spec_.merge.htm_mode,
                        spec_.merge.concat_axis, spec_.merge.htm_params,
                        headBufB_);
        current = &headBufB_;
    }
    return current;
}

void PyramidRuntime::pipe_exec_node(std::size_t li, std::size_t p,
                                    std::int64_t t, WorkerScratch &ws) {
    HtmNode *n = layers_[li][p];
    const std::size_t k = static_cast<std::size_t>(t % pipeK_);
    const bool is_head = (li + 1 == nLayers_);
    const bool with_residual =
        spec_.flags.residual && !spec_.flags.squeeze_skip && li >= 2;

    if (li == 0) {
        inputSlots_.at(spec_.layers[0][p]).setSDR(l0InRing_[p][k]);
    } else {
        SDR &slot = inputSlots_.at(spec_.layers[li][p]);
        merge_children_ring(li, p, t, with_residual, ws, slot);
        if (is_head && !spec_.flags.squeeze_skip)
            n->set_input(apply_head_extras_ws(&slot, ws));
    }

    n->step();

    outRing_[li][p][k].setSDR(n->output());
    if (spec_.flags.use_predictive)
        predRing_[li][p][k].setSDR(n->predictiveOutput());
    if (spec_.flags.anomaly_boost)
        anomRing_[li][p][k] = n->last_anomaly();
    if (is_head && spec_.flags.recurrent_head && !spec_.flags.squeeze_skip) {
        prevHeadSdr_.setSDR(n->output());
        prevHeadValid_ = true;
    }

    /* Publish completion with a release store; workers observe it via their
     * acquire-load busy-poll. NO condition-variable notify here: notifying on
     * every one of the ~(nodes x records) completions makes all workers
     * contend on a single global mutex + notify_all, which serializes the
     * pipeline. The lock-free poll below needs no wakeup on the hot path. */
    nodeDone_[li][p]->store(t, std::memory_order_release);
}

std::int64_t PyramidRuntime::run_pipelined(RecordSource &source,
                                           std::int64_t limit,
                                           std::int64_t progress_every,
                                           std::int64_t total) {
    /* Fresh flight state (run() may be called repeatedly). */
    srcDone_.store(-1, std::memory_order_relaxed);
    tEnd_.store(std::numeric_limits<std::int64_t>::max(),
                std::memory_order_relaxed);
    for (auto &lay : nodeDone_)
        for (auto &a : lay) a->store(-1, std::memory_order_relaxed);
    ws_.clear();
    ws_.resize(pool_->n_workers());

    const auto runStart = std::chrono::steady_clock::now();
    std::atomic<std::int64_t> headDone{-1};
    const std::size_t hL = headLayer_, hP = headPos_;

    auto workerLoop = [&](std::size_t idx) {
        WorkerScratch &ws = ws_[idx];
        const auto &owned = pool_->owned_of(idx);   // [layer][j] -> pos
        std::uint64_t seenEpoch = 0;
        std::uint32_t idleSpins = 0;
        for (;;) {
            bool progressed = false;

            if (idx == 0)
                while (pipe_produce(source, limit, ws)) progressed = true;

            bool allFinished =
                srcDone_.load(std::memory_order_acquire) + 1 >=
                tEnd_.load(std::memory_order_acquire);
            for (std::size_t li = 0; li < nLayers_; ++li) {
                for (std::size_t pos : owned[li]) {
                    std::int64_t t =
                        nodeDone_[li][pos]->load(
                            std::memory_order_relaxed) + 1;
                    const std::int64_t end =
                        tEnd_.load(std::memory_order_acquire);
                    while (t < end && pipe_ready(li, pos, t) &&
                           pipe_slot_free(li, pos, t)) {
                        pipe_exec_node(li, pos, t, ws);
                        if (li == hL && pos == hP)
                            headDone.store(t, std::memory_order_release);
                        ++t;
                        progressed = true;
                    }
                    if (t < end) allFinished = false;
                }
            }
            if (allFinished &&
                (idx != 0 ||
                 srcDone_.load(std::memory_order_acquire) + 1 >=
                     tEnd_.load(std::memory_order_acquire)))
                return;

            if (!progressed) {
                /* No ready node this scan. Back off in two stages:
                 *   1. a SHORT spin (a handful of pause/yield iterations) to
                 *      catch the common case where a dependency completes on
                 *      another core within a few hundred nanoseconds; then
                 *   2. PARK on the condition variable, releasing the physical
                 *      core to the threads that still have work (and to the
                 *      hyperthread sibling). A hot spin here instead starves
                 *      those threads and was measured to REGRESS throughput on
                 *      multi-core hosts. The producer's pipe_wake() (once per
                 *      record) and the end-of-flight notify wake us promptly.
                 * A short bounded timeout on the park guards against any
                 * missed-wakeup edge and bounds tail latency. */
                if (idleSpins < 64) {
                    ++idleSpins;
                    cpu_relax();
                } else {
                    std::unique_lock<std::mutex> lk(pipeMx_);
                    if (pipeEpoch_ == seenEpoch)
                        pipeCv_.wait_for(lk, std::chrono::milliseconds(1));
                    seenEpoch = pipeEpoch_;
                    // reset so we spin briefly again after waking
                    idleSpins = 0;
                }
            } else {
                idleSpins = 0;
            }
        }
    };

    /* Main thread drives the native progress line while workers run. */
    std::atomic<bool> running{true};
    std::thread progressThread;
    if (progress_every > 0) {
        progressThread = std::thread([&] {
            std::int64_t lastShown = -1;
            auto lastT = std::chrono::steady_clock::now();
            htm_pyramid::RateWindow rw;   // current-speed window (~6 s)
            while (running.load(std::memory_order_acquire)) {
                const std::int64_t d = headDone.load(
                                           std::memory_order_acquire) + 1;
                const auto now = std::chrono::steady_clock::now();
                if (d > lastShown && d > 0 &&
                    (d / progress_every > lastShown / progress_every ||
                     std::chrono::duration<double>(now - lastT).count() >=
                         htm_pyramid::PROGRESS_REFRESH_SECONDS)) {
                    lastShown = d;
                    lastT = now;
                    const double el = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - runStart)
                                          .count();
                    rw.add(el, d);
                    const double cum = el > 0 ? d / el : 0.0;
                    const double rate = rw.rate(el, d, cum);   // CURRENT speed
                    if (total >= 0)
                    {
                        const std::int64_t remain = total - d;
                        const double eta = rate > 0 ? remain / rate : 0.0;
                        char bar[80];
                        const double frac =
                            total > 0 ? static_cast<double>(d) / total : 0.0;
                        htm_pyramid::progress_bar(frac, bar, 24);
                        std::printf("\r[Run] %s %5.1f%%  %lld/%lld records  "
                                    "%.1f record/second  elapsed %s  "
                                    "estimated time remaining %s        ",
                                    bar, 100.0 * frac,
                                    static_cast<long long>(d),
                                    static_cast<long long>(total), rate,
                                    htm_pyramid::fmt_hms(el).c_str(),
                                    htm_pyramid::fmt_hms(eta).c_str());
                    }
                    else {
                        std::printf("\r[Run] %lld records  %.1f record/second  "
                                    "elapsed %s        ",
                                    static_cast<long long>(d), rate,
                                    htm_pyramid::fmt_hms(el).c_str());
                    }
                    std::fflush(stdout);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        });
    }

    // Drive the run on the worker pool. The MAIN thread polls for signals
    // (Ctrl+C) between short waits -- PyErr_CheckSignals() only delivers on
    // the Python main thread, so this is what makes interruption work on
    // Windows as well as POSIX. On interrupt we set tEnd_ to the last
    // produced record so workers drain their in-flight slots and exit.
    std::atomic<bool> aborted{false};
    // The poll runs on the Python MAIN thread (only place PyErr_CheckSignals
    // delivers). check_interrupt() throws on Ctrl+C; before that propagates we
    // set tEnd_ so the workers stop pulling new records, drain their in-flight
    // ring slots, and return -- otherwise the pool's post-abort join would wait
    // for a run that never ends. `aborted` guards against a double-store.
    std::function<void()> poll;
    if (htm_pyramid::interrupt_hook()) {
        poll = [&]() {
            try {
                htm_pyramid::check_interrupt();
            } catch (...) {
                if (!aborted.exchange(true))
                    tEnd_.store(srcDone_.load(std::memory_order_acquire) + 1,
                                std::memory_order_release);
                throw;
            }
        };
    }
    try {
        pool_->run_parallel(workerLoop, poll, &aborted);
    } catch (...) {
        running.store(false, std::memory_order_release);
        if (progressThread.joinable()) progressThread.join();
        if (progress_every > 0) std::printf("\n");
        throw;
    }

    running.store(false, std::memory_order_release);
    if (progressThread.joinable()) progressThread.join();
    if (progress_every > 0) std::printf("\n");

    const std::int64_t processed = headDone.load() + 1;
    currentIter_ += processed;
    return processed;
}

}  // namespace pyramid