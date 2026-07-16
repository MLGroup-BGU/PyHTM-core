/* ---------------------------------------------------------------------------
 * Pyramid-Model-Core :: runtime/PyramidRuntime.cpp
 *
 * Implementation.  Blocks cite the Python they port (htm_pyramid.py /
 * feature_su_pyramid.py / data_streamer.py semantics).
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

    /* Children / residual grandparents as positions. */
    childPos_.resize(nLayers_);
    resPos_.resize(nLayers_);
    for (std::size_t li = 1; li < nLayers_; ++li) {
        childPos_[li].resize(layers_[li].size());
        resPos_[li].resize(layers_[li].size());
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

                /* residual: for child in children -> for grandchild in
                 * predecessors(child) -> if in residual_results (layer
                 * li-2): membership by layer, like the Python dict check. */
                auto git = spec_.children.find(c);
                if (git == spec_.children.end()) continue;
                for (const auto &g : git->second) {
                    const Pos gp = nodePos_.at(g);
                    if (li >= 2 && gp.layer == li - 2)
                        resPos_[li][p].push_back(gp.pos);
                }
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

    if (spec_.flags.lateral_exchange) {
        lateralOffset_.assign(nLayers_, 0);
        std::size_t off = 0;
        for (std::size_t li = 0; li + 1 < nLayers_; ++li) {
            lateralOffset_[li] = off;
            off += layers_[li].size();
        }
        lateralSlots_.reserve(off);
        for (std::size_t li = 0; li + 1 < nLayers_; ++li)
            for (HtmNode *n : layers_[li])
                lateralSlots_.emplace_back(n->output_dims());
    }

    if (spec_.flags.use_predictive) {
        for (std::size_t li = 1; li < nLayers_; ++li)
            for (std::size_t p = 0; p < layers_[li].size(); ++p) {
                const auto &firstChild =
                    layers_[li - 1][childPos_[li][p][0]];
                const auto &cd = firstChild->output_dims();
                const auto &name = spec_.layers[li][p];
                mergedActiveSlots_.emplace(name, SDR(cd));
                mergedPredSlots_.emplace(name, SDR(cd));
            }
    }

    if (spec_.flags.squeeze_skip) {
        for (std::size_t li = 1; li < nLayers_; ++li)
            for (std::size_t p = 0; p < layers_[li].size(); ++p)
                if (childPos_[li][p].size() > 1) {
                    const auto &cd =
                        layers_[li - 1][childPos_[li][p][0]]->output_dims();
                    skipSlots_.emplace(spec_.layers[li][p], SDR(cd));
                }
    }

    const auto &headIn = nodes_.at(headName_)->input_dims_ref();
    headBufA_.initialize(headIn);
    headBufB_.initialize(headIn);
    memorySdr_.initialize(headIn);
    memoryBuf_.initialize(headIn);
    prevHeadSdr_.initialize(nodes_.at(headName_)->output_dims());

    outView_.resize(nLayers_);
    for (std::size_t li = 0; li < nLayers_; ++li)
        outView_[li].assign(layers_[li].size(), {nullptr, nullptr});
}

void PyramidRuntime::validate_dims_and_flags() {
    /* Python's sdr_merge over (active, predictive) TUPLES crashes inside
     * lateral_exchange / broadcast_head; reject the combination with a
     * clear message instead. */
    if (spec_.flags.use_predictive) {
        NTA_CHECK(!spec_.flags.lateral_exchange)
            << "use_predictive + lateral_exchange is not a runnable "
            << "combination (the Python pyramid crashes merging tuples)";
        NTA_CHECK(!spec_.flags.broadcast_head)
            << "use_predictive + broadcast_head is not a runnable "
            << "combination (the Python pyramid crashes merging tuples)";
    }

    /* L0: plan-merged feature dims == spec input dims. */
    for (std::size_t p = 0; p < layers_[0].size(); ++p) {
        std::vector<std::vector<UInt>> dims;
        for (std::size_t ei : planEnc_[p])
            dims.push_back(encoders_[ei]->dimensions());
        std::vector<UInt> merged =
            (!spec_.merge.has_feature_mode || dims.size() == 1)
                ? dims[0]
                : merge_output_dims(dims, spec_.merge.feature_mode,
                                    spec_.merge.concat_axis);
        NTA_CHECK(merged == layers_[0][p]->input_dims_ref())
            << "L0 `" << spec_.layers[0][p]
            << "`: plan-merged dims != spec input dims";
    }

    /* Parents: merged children output dims == spec input dims. */
    for (std::size_t li = 1; li < nLayers_; ++li)
        for (std::size_t p = 0; p < layers_[li].size(); ++p) {
            std::vector<std::vector<UInt>> dims;
            for (std::size_t cp : childPos_[li][p])
                dims.push_back(layers_[li - 1][cp]->output_dims());
            std::vector<UInt> merged =
                (dims.size() == 1 && !spec_.flags.use_predictive)
                    ? dims[0]
                    : merge_output_dims(dims, spec_.merge.htm_mode,
                                        spec_.merge.concat_axis);
            if (spec_.flags.use_predictive) {
                std::vector<UInt> stacked = {2u};
                stacked.insert(stacked.end(), merged.begin(), merged.end());
                merged = stacked;
            }
            NTA_CHECK(merged == layers_[li][p]->input_dims_ref())
                << "`" << spec_.layers[li][p]
                << "`: merged child dims != spec input dims";
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
    validate_dims_and_flags();
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
void PyramidRuntime::merge_or_copy(MergeMode mode, const MergeParams &params,
                                   SDR &out) {
    if (mergeInputs_.size() == 1)
        out.setSDR(*mergeInputs_[0]);   // Python returns the object itself
    else
        merger_.merge(mergeInputs_, mode, spec_.merge.concat_axis, params,
                      out);
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

        if (!spec_.merge.has_feature_mode)
            slot.setSDR(*mergeInputs_[0]);   // mode None: the SDR itself
        else
            merge_or_copy(spec_.merge.feature_mode,
                          spec_.merge.feature_params, slot);
    }
}

void PyramidRuntime::run_one_layer(std::size_t li) {
    pool_->run_layer(li, [](HtmNode &n) { n.step(); });

    auto &view = outView_[li];
    for (std::size_t p = 0; p < layers_[li].size(); ++p) {
        HtmNode *n = layers_[li][p];
        view[p] = {&n->output(),
                   spec_.flags.use_predictive ? &n->predictiveOutput()
                                              : nullptr};
    }
}

void PyramidRuntime::lateral_exchange(std::size_t li) {
    /* Neighbor order == the layer's node order (Python dict order). */
    auto &view = outView_[li];
    const std::size_t n = view.size();
    if (n <= 1) return;   // single node keeps its own output

    for (std::size_t i = 0; i < n; ++i) {
        mergeInputs_.clear();
        mergeInputs_.push_back(view[i].first);
        if (i > 0) mergeInputs_.push_back(view[i - 1].first);
        if (i + 1 < n) mergeInputs_.push_back(view[i + 1].first);
        SDR &slot = lateralSlots_[lateralOffset_[li] + i];
        merger_.merge(mergeInputs_, spec_.merge.htm_mode,
                      spec_.merge.concat_axis, spec_.merge.htm_params, slot);
    }
    /* Publish AFTER all merges: Python merges from the pre-exchange dict. */
    for (std::size_t i = 0; i < n; ++i)
        view[i].first = &lateralSlots_[lateralOffset_[li] + i];
}

void PyramidRuntime::merge_children_for(std::size_t li, std::size_t parent_pos,
                                        bool with_residual, SDR &out) {
    const auto &children = childPos_[li][parent_pos];
    const auto &below = outView_[li - 1];
    const std::string &parent = spec_.layers[li][parent_pos];

    /* squeeze-skip variant replaces the standard merge entirely. */
    if (spec_.flags.squeeze_skip) {
        mergeInputs_.clear();
        for (std::size_t cp : children) mergeInputs_.push_back(below[cp].first);
        if (children.size() > 1) {
            SDR &skip = skipSlots_.at(parent);
            merger_.merge(mergeInputs_, spec_.merge.htm_mode,
                          spec_.merge.concat_axis, spec_.merge.htm_params,
                          skip);
            mergeInputs_.push_back(&skip);
        }
        merge_or_copy(spec_.merge.htm_mode, spec_.merge.htm_params, out);
        return;
    }

    /* Build the to_merge list as PAIRS, so anomaly_boost duplication and
     * residual skip inputs flow through use_predictive's zip exactly like
     * Python's `zip(*to_merge)`. */
    pairScratch_.clear();
    for (std::size_t cp : children) {
        const OutPair &pr = below[cp];
        std::int64_t n_copies = 1;
        if (spec_.flags.anomaly_boost) {
            /* 1 copy at a=0, up to 4 at a=1: n = 1 + round(a*3); Python
             * round() is banker's (ties-to-even), matched by nearbyint
             * under the default rounding mode. */
            const double a = layers_[li - 1][cp]->last_anomaly();
            n_copies = 1 + static_cast<std::int64_t>(std::nearbyint(a * 3.0));
        }
        for (std::int64_t k = 0; k < n_copies; ++k) pairScratch_.push_back(pr);
    }
    if (with_residual) {
        const auto &residual = outView_[li - 2];
        for (std::size_t gp : resPos_[li][parent_pos])
            pairScratch_.push_back(residual[gp]);
    }

    if (spec_.flags.use_predictive) {
        SDR &ma = mergedActiveSlots_.at(parent);
        SDR &mp = mergedPredSlots_.at(parent);

        mergeInputs_.clear();
        for (const auto &pr : pairScratch_) mergeInputs_.push_back(pr.first);
        merge_or_copy(spec_.merge.htm_mode, spec_.merge.htm_params, ma);

        mergeInputs_.clear();
        for (const auto &pr : pairScratch_) mergeInputs_.push_back(pr.second);
        merge_or_copy(spec_.merge.htm_mode, spec_.merge.htm_params, mp);

        mergeInputs_.clear();
        mergeInputs_.push_back(&ma);
        mergeInputs_.push_back(&mp);
        merger_.merge(mergeInputs_, MergeMode::Stack, 0,
                      spec_.merge.htm_params, out);
        return;
    }

    mergeInputs_.clear();
    for (const auto &pr : pairScratch_) mergeInputs_.push_back(pr.first);
    merge_or_copy(spec_.merge.htm_mode, spec_.merge.htm_params, out);
}

const SDR *PyramidRuntime::apply_head_extras(const SDR *current) {
    if (spec_.flags.recurrent_head && prevHeadValid_) {
        mergeInputs_.clear();
        mergeInputs_.push_back(current);
        mergeInputs_.push_back(&prevHeadSdr_);
        merger_.merge(mergeInputs_, spec_.merge.htm_mode,
                      spec_.merge.concat_axis, spec_.merge.htm_params,
                      headBufA_);
        current = &headBufA_;
    }

    if (spec_.flags.long_term_memory) {
        if (!memoryInitialized_) {
            memorySdr_.setSDR(*current);   // first step: memory := current
            memoryInitialized_ = true;
        } else {
            /* memory := merge(memory x inertia, current) */
            mergeInputs_.clear();
            for (std::int64_t k = 0; k < spec_.flags.memory_inertia; ++k)
                mergeInputs_.push_back(&memorySdr_);
            mergeInputs_.push_back(current);
            merger_.merge(mergeInputs_, spec_.merge.htm_mode,
                          spec_.merge.concat_axis, spec_.merge.htm_params,
                          memoryBuf_);
            memorySdr_.setSDR(memoryBuf_);
        }
        mergeInputs_.clear();
        mergeInputs_.push_back(current);
        mergeInputs_.push_back(&memorySdr_);
        merger_.merge(mergeInputs_, spec_.merge.htm_mode,
                      spec_.merge.concat_axis, spec_.merge.htm_params,
                      headBufB_);
        current = &headBufB_;
    }
    return current;
}

void PyramidRuntime::merge_into_next(std::size_t li) {
    const std::size_t next = li + 1;
    const bool is_head_next = (li + 2 == nLayers_);

    if (spec_.flags.broadcast_head && is_head_next &&
        !spec_.flags.squeeze_skip) {
        /* head receives ALL previous layer outputs merged (layer order,
         * node order within each layer -- Python's dict .values()). */
        mergeInputs_.clear();
        for (std::size_t l = 0; l <= li; ++l)
            for (const auto &pr : outView_[l])
                mergeInputs_.push_back(pr.first);
        SDR &slot = inputSlots_.at(spec_.layers[next][0]);
        merge_or_copy(spec_.merge.htm_mode, spec_.merge.htm_params, slot);
    } else {
        const bool with_residual = spec_.flags.residual &&
                                   !spec_.flags.squeeze_skip && li >= 1;
        for (std::size_t p = 0; p < layers_[next].size(); ++p) {
            SDR &slot = inputSlots_.at(spec_.layers[next][p]);
            merge_children_for(next, p, with_residual, slot);
        }
    }

    if (is_head_next && !spec_.flags.squeeze_skip) {
        HtmNode *head = layers_[next][0];
        const SDR *cur = &inputSlots_.at(spec_.layers[next][0]);
        head->set_input(apply_head_extras(cur));
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
        std::printf("[Run] %lld records in %.1fs (%.1f record/second), "
                    "%lld total processed\n",
                    static_cast<long long>(done), secs,
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

            if (spec_.flags.lateral_exchange && li + 1 < nLayers_ &&
                !spec_.flags.squeeze_skip)
                lateral_exchange(li);

            if (li + 1 < nLayers_) {
                merge_into_next(li);
            } else if (spec_.flags.recurrent_head &&
                       !spec_.flags.squeeze_skip) {
                /* head just ran: capture output[0] for the next record. */
                prevHeadSdr_.setSDR(*outView_[li][headPos_].first);
                prevHeadValid_ = true;
            }
        }

        ++currentIter_;
        const bool count_hit =
            progress_every > 0 && currentIter_ % progress_every == 0;
        const bool time_hit =
            progress_every > 0 &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          lastProgress).count() >= htm_pyramid::PROGRESS_REFRESH_SECONDS;
        if (count_hit || time_hit) {
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
                    char bar[32];
                    const double frac =
                        total > 0 ? static_cast<double>(currentIter_) / total
                                  : 0.0;
                    htm_pyramid::progress_bar(frac, bar, 24);
                    std::printf("\r[Run] %s %5.1f%%  %lld/%lld records  "
                                "%.1f record/second  elapsed %s  "
                                "estimated time remaining %s        ",
                                bar, 100.0 * frac,
                                static_cast<long long>(currentIter_),
                                static_cast<long long>(total), rate,
                                htm_pyramid::fmt_hms(el).c_str(),
                                htm_pyramid::fmt_hms(eta).c_str());
                } else {
                    std::printf("\r[Run] %lld records  %.1f record/second  "
                                "elapsed %s        ",
                                static_cast<long long>(currentIter_), rate,
                                htm_pyramid::fmt_hms(el).c_str());
                }
                std::fflush(stdout);
            }
        }
    }
    if (progress) progress(currentIter_, total);
    else std::printf("\n");   // finish the in-place progress line

    const double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - runStart).count();
    const std::int64_t done = currentIter_ - runFrom;
    htm_pyramid::stage_banner(htm_pyramid::Stage::Done,
                              "DONE -- run complete");
    std::printf("[Run] %lld records in %.1fs (%.1f record/second), "
                "%lld total processed\n",
                static_cast<long long>(done), secs,
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
    return res;
}

std::string PyramidRuntime::summary_for(const std::string &node) const {
    auto it = nodes_.find(node);
    NTA_CHECK(it != nodes_.end()) << "unknown node `" << node << "`";
    return it->second->summary();
}

} // namespace pyramid
