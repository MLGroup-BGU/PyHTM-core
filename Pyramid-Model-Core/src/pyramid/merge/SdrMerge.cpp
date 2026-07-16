/* ---------------------------------------------------------------------------
 * Pyramid-Model-Core :: merge/SdrMerge.cpp
 *
 * Implementation of every sdr.py merge mode.  See SdrMerge.hpp for the
 * faithfulness / performance / threading contract.  Each mode_* function is
 * annotated with the Python it ports; the float64 operation ORDER inside is
 * deliberate and must not be "simplified" (it reproduces NumPy's arithmetic,
 * including its pairwise `.sum()`), or the sampled bit sets will drift from
 * the Python pyramid's.
 * ------------------------------------------------------------------------ */
#include "SdrMerge.hpp"

#include <htm/utils/Log.hpp>

#include <algorithm>
#include <cmath>

#include "../math/NumpyCompat.hpp"

namespace pyramid {

using htm::SDR;
using htm::UInt;

/* ------------------------------------------------------------------------ */
MergeMode parse_merge_mode(const std::string &m) {
    if (m == "u")   return MergeMode::Union;
    if (m == "i")   return MergeMode::Intersection;
    if (m == "sd" || m == "xor") return MergeMode::SymDiff;
    if (m == "nos") return MergeMode::NonOverlapSum;
    if (m == "c")   return MergeMode::Concat;
    if (m == "s")   return MergeMode::Stack;
    if (m == "su")  return MergeMode::SoftUnion;
    if (m == "aps") return MergeMode::ActivationProportional;
    if (m == "pls") return MergeMode::PowerLaw;
    if (m == "ssu") return MergeMode::StabilizedSoftUnion;
    if (m == "li")  return MergeMode::LateralInhibition;
    if (m == "bws") return MergeMode::BurstWeighted;
    if (m == "pcb") return MergeMode::PredictiveCodingBoost;
    if (m == "di")  return MergeMode::DendriticIntegration;
    NTA_THROW << "Unknown SDR merger mode `" << m << "`";
}

bool is_stochastic_mode(MergeMode m) {
    switch (m) {
        case MergeMode::SoftUnion: case MergeMode::ActivationProportional:
        case MergeMode::PowerLaw:  case MergeMode::StabilizedSoftUnion:
        case MergeMode::LateralInhibition: case MergeMode::BurstWeighted:
        case MergeMode::PredictiveCodingBoost:
        case MergeMode::DendriticIntegration:
            return true;
        default:
            return false;
    }
}

/* Port of _positify_axis. */
static int positify_axis(int axis, int ndim) {
    if (axis < 0) {
        const int old = axis;
        axis = ndim + old;
        NTA_CHECK(axis >= 0) << "Invalid axis " << old << " for shape with "
                             << ndim << " dimensions";
    }
    return axis;
}

/* Port of _check_shapes: sdr_merge requires IDENTICAL dimensions across all
 * inputs, for every mode (including concat/stack -- the Python code runs
 * this check before dispatching). */
static const std::vector<UInt> &check_shapes(
        const std::vector<const SDR *> &inputs) {
    const auto &s0 = inputs[0]->dimensions;
    for (std::size_t i = 1; i < inputs.size(); ++i)
        NTA_CHECK(inputs[i]->dimensions == s0)
            << "All SDRs must have the same dimensions";
    return s0;
}

std::vector<UInt> merge_output_dims(
        const std::vector<std::vector<UInt>> &input_dims,
        MergeMode mode, int axis) {
    NTA_CHECK(!input_dims.empty());
    const auto &d0 = input_dims[0];
    for (const auto &d : input_dims)
        NTA_CHECK(d == d0) << "All SDRs must have the same dimensions";

    if (mode == MergeMode::Concat) {
        const int a = positify_axis(axis, static_cast<int>(d0.size()));
        NTA_CHECK(a < static_cast<int>(d0.size()))
            << "Invalid axis " << axis << " for shapes with " << d0.size()
            << " dimensions";
        auto out = d0;
        out[static_cast<std::size_t>(a)] =
            d0[static_cast<std::size_t>(a)] * static_cast<UInt>(input_dims.size());
        return out;
    }
    if (mode == MergeMode::Stack) {
        std::vector<UInt> out;
        out.reserve(d0.size() + 1);
        out.push_back(static_cast<UInt>(input_dims.size()));
        out.insert(out.end(), d0.begin(), d0.end());
        return out;
    }
    return d0;
}

/* ------------------------------------------------------------------------ */
std::int64_t SdrMerger::accumulate_counts(
        const std::vector<const SDR *> &inputs) {
    const auto &dims = inputs[0]->dimensions;
    std::int64_t total = 1;
    for (UInt d : dims) total *= static_cast<std::int64_t>(d);

    if (static_cast<std::int64_t>(counts_.size()) < total)
        counts_.resize(static_cast<std::size_t>(total), 0.0);

    touched_.clear();
    for (const SDR *in : inputs) {
        for (UInt idx : in->getSparse()) {
            double &c = counts_[idx];
            if (c == 0.0) touched_.push_back(static_cast<std::int64_t>(idx));
            c += 1.0;   // np.stack(dense).sum(axis=0): +1 per occurrence
        }
    }

    /* np.where(counts > 0)[0] yields ASCENDING indices. */
    activeIdx_.assign(touched_.begin(), touched_.end());
    std::sort(activeIdx_.begin(), activeIdx_.end());
    activeCnt_.resize(activeIdx_.size());
    for (std::size_t i = 0; i < activeIdx_.size(); ++i)
        activeCnt_[i] = counts_[static_cast<std::size_t>(activeIdx_[i])];
    return total;
}

void SdrMerger::clear_counts() {
    for (std::int64_t idx : touched_) counts_[static_cast<std::size_t>(idx)] = 0.0;
    touched_.clear();
}

/* ------------------------------------------------------------------------ */
void SdrMerger::sample_and_set(const std::vector<std::int64_t> &candidates,
                               const double *probs, std::int64_t n_sample,
                               const MergeParams &params, SDR &out) {
    /* rng = np.random.default_rng(seed): FRESH generator per merge call.
     * seed=None (has_seed=false) draws OS entropy, exactly like Python. */
    NumpyGenerator rng = params.has_seed ? NumpyGenerator(params.seed)
                                         : NumpyGenerator();
    rng.choice_no_replace_p(static_cast<std::int64_t>(candidates.size()),
                            n_sample, probs, sampled_,
                            pCopy_, cdf_, xDraw_, newIdx_, seen_);

    selected_.clear();
    selected_.reserve(sampled_.size());
    for (std::int64_t s : sampled_)
        selected_.push_back(static_cast<UInt>(candidates[static_cast<std::size_t>(s)]));
    /* Python assigns the (unsorted) sampled ids to SDR.sparse; the binding
     * sorts before handing to the core.  Same here. Uniqueness is guaranteed
     * by sampling without replacement. */
    std::sort(selected_.begin(), selected_.end());
    out.setSparse(selected_);   // swap overload: zero-copy, buffers ping-pong
}

/* ------------------------- basic modes ----------------------------------- */
/* union: SDR(shape).union(inputs) == all bits with count > 0. */
void SdrMerger::mode_union(SDR &out) {
    selected_.clear();
    selected_.reserve(activeIdx_.size());
    for (std::int64_t idx : activeIdx_) selected_.push_back(static_cast<UInt>(idx));
    out.setSparse(selected_);
}

/* intersection: bits active in ALL inputs (count == n_inputs). */
void SdrMerger::mode_intersection(std::size_t n_inputs, SDR &out) {
    const double n = static_cast<double>(n_inputs);
    selected_.clear();
    for (std::size_t i = 0; i < activeIdx_.size(); ++i)
        if (activeCnt_[i] == n) selected_.push_back(static_cast<UInt>(activeIdx_[i]));
    out.setSparse(selected_);
}

/* sd/xor: subtract(union, intersection) == bits with 0 < count < n_inputs. */
void SdrMerger::mode_symdiff(std::size_t n_inputs, SDR &out) {
    const double n = static_cast<double>(n_inputs);
    selected_.clear();
    for (std::size_t i = 0; i < activeIdx_.size(); ++i)
        if (activeCnt_[i] < n) selected_.push_back(static_cast<UInt>(activeIdx_[i]));
    out.setSparse(selected_);
}

/* nos: bits active in EXACTLY one input (count == 1).
 * NOTE: the Python numba kernel (_sdr_non_overlapping_sum_impl) crashes on
 * inputs of rank > 1 (2-D boolean setitem is unsupported by numba); the
 * pyramid only ever merges flat 1-D node outputs, so that limitation never
 * fired.  This port implements the documented semantics for any rank. */
void SdrMerger::mode_nos(SDR &out) {
    selected_.clear();
    for (std::size_t i = 0; i < activeIdx_.size(); ++i)
        if (activeCnt_[i] == 1.0) selected_.push_back(static_cast<UInt>(activeIdx_[i]));
    out.setSparse(selected_);
}

/* c: concatenate along `axis`.  All inputs share one shape (checked), so
 * input j's coordinate on `axis` is offset by j * dims[axis].  Flat indices
 * are re-flattened against the OUT dimensions (row-major). */
void SdrMerger::mode_concat(const std::vector<const SDR *> &inputs, int axis,
                            SDR &out) {
    const auto &in_dims = inputs[0]->dimensions;
    const int ndim = static_cast<int>(in_dims.size());
    const int a = positify_axis(axis, ndim);
    NTA_CHECK(a < ndim) << "Invalid axis " << axis << " for shapes with "
                        << ndim << " dimensions";
    const auto &out_dims = out.dimensions;

    /* Row-major strides for input and output shapes. */
    std::int64_t in_stride[16], out_stride[16];
    NTA_CHECK(ndim <= 16) << "SDR rank too large";
    in_stride[ndim - 1] = out_stride[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; --d) {
        in_stride[d]  = in_stride[d + 1]  * static_cast<std::int64_t>(in_dims[static_cast<std::size_t>(d + 1)]);
        out_stride[d] = out_stride[d + 1] * static_cast<std::int64_t>(out_dims[static_cast<std::size_t>(d + 1)]);
    }

    selected_.clear();
    for (std::size_t j = 0; j < inputs.size(); ++j) {
        const std::int64_t off_a =
            static_cast<std::int64_t>(j) * static_cast<std::int64_t>(in_dims[static_cast<std::size_t>(a)]);
        for (UInt f : inputs[j]->getSparse()) {
            std::int64_t rem = static_cast<std::int64_t>(f), of = 0;
            for (int d = 0; d < ndim; ++d) {
                std::int64_t coord = rem / in_stride[d];
                rem -= coord * in_stride[d];
                if (d == a) coord += off_a;
                of += coord * out_stride[d];
            }
            selected_.push_back(static_cast<UInt>(of));
        }
    }
    std::sort(selected_.begin(), selected_.end());
    out.setSparse(selected_);
}

/* s: stack on a new leading axis == concat(axis=0) then reshape([k, *dims]);
 * flat index = j * prod(dims) + f. Inputs are sorted, so the result is
 * sorted by construction. */
void SdrMerger::mode_stack(const std::vector<const SDR *> &inputs, SDR &out) {
    std::int64_t block = 1;
    for (UInt d : inputs[0]->dimensions) block *= static_cast<std::int64_t>(d);
    selected_.clear();
    for (std::size_t j = 0; j < inputs.size(); ++j) {
        const std::int64_t off = static_cast<std::int64_t>(j) * block;
        for (UInt f : inputs[j]->getSparse())
            selected_.push_back(static_cast<UInt>(off + static_cast<std::int64_t>(f)));
    }
    out.setSparse(selected_);
}

/* ---------------------- stochastic modes --------------------------------- */
/* _sdr_soft_union: softmax(count / max(T, 1e-10)) -> weighted sample. */
void SdrMerger::mode_su(std::int64_t total_bits, const MergeParams &p, SDR &out) {
    const std::int64_t target_active =
        static_cast<std::int64_t>(p.target_sparsity * static_cast<double>(total_bits));
    const std::size_t n = activeIdx_.size();

    const double t = std::max(p.temperature, 1e-10);
    work_.resize(n);
    for (std::size_t i = 0; i < n; ++i) work_[i] = activeCnt_[i] / t;
    double mx = work_[0];
    for (std::size_t i = 1; i < n; ++i) mx = std::max(mx, work_[i]);
    for (std::size_t i = 0; i < n; ++i) work_[i] = std::exp(work_[i] - mx);
    const double s = pairwise_sum(work_.data(), n);
    probs_.resize(n);
    for (std::size_t i = 0; i < n; ++i) probs_[i] = work_[i] / s;

    const std::int64_t k = std::min<std::int64_t>(target_active,
                                                  static_cast<std::int64_t>(n));
    sample_and_set(activeIdx_, probs_.data(), k, p, out);
}

/* _sdr_activation_proportional_sampling: P ~ count. */
void SdrMerger::mode_aps(std::int64_t total_bits, const MergeParams &p, SDR &out) {
    const std::int64_t target_active =
        static_cast<std::int64_t>(p.target_sparsity * static_cast<double>(total_bits));
    const std::size_t n = activeIdx_.size();

    const double s = pairwise_sum(activeCnt_.data(), n);
    probs_.resize(n);
    for (std::size_t i = 0; i < n; ++i) probs_[i] = activeCnt_[i] / s;

    const std::int64_t k = std::min<std::int64_t>(target_active,
                                                  static_cast<std::int64_t>(n));
    sample_and_set(activeIdx_, probs_.data(), k, p, out);
}

/* _sdr_power_law_sampling: P ~ count^alpha. */
void SdrMerger::mode_pls(std::int64_t total_bits, const MergeParams &p, SDR &out) {
    const std::int64_t target_active =
        static_cast<std::int64_t>(p.target_sparsity * static_cast<double>(total_bits));
    const std::size_t n = activeIdx_.size();

    work_.resize(n);
    for (std::size_t i = 0; i < n; ++i) work_[i] = std::pow(activeCnt_[i], p.alpha);
    const double s = pairwise_sum(work_.data(), n);
    probs_.resize(n);
    for (std::size_t i = 0; i < n; ++i) probs_[i] = work_[i] / s;

    const std::int64_t k = std::min<std::int64_t>(target_active,
                                                  static_cast<std::int64_t>(n));
    sample_and_set(activeIdx_, probs_.data(), k, p, out);
}

/* _sdr_stabilized_soft_union: clipped softmax + probability smoothing. */
void SdrMerger::mode_ssu(std::int64_t total_bits, const MergeParams &p, SDR &out) {
    const std::int64_t target_active =
        static_cast<std::int64_t>(p.target_sparsity * static_cast<double>(total_bits));
    const std::size_t n = activeIdx_.size();

    const double t = std::max(p.temperature, 1e-10);
    work_.resize(n);
    for (std::size_t i = 0; i < n; ++i) work_[i] = activeCnt_[i] / t;
    double mx = work_[0];
    for (std::size_t i = 1; i < n; ++i) mx = std::max(mx, work_[i]);
    for (std::size_t i = 0; i < n; ++i) {
        double v = work_[i] - mx;
        v = std::min(std::max(v, -50.0), 50.0);  // np.clip(scaled, -50, 50)
        work_[i] = std::exp(v);
    }
    const double s = pairwise_sum(work_.data(), n) + 1e-10;
    probs_.resize(n);
    for (std::size_t i = 0; i < n; ++i) probs_[i] = work_[i] / s;

    /* p' = (1 - s)*p + s * (1/n), s clipped to [0, 1]; then renormalize. */
    const double sm = std::min(std::max(p.smoothing, 0.0), 1.0);
    const double uniform = 1.0 / static_cast<double>(n);   // np.ones(n)/n
    for (std::size_t i = 0; i < n; ++i)
        probs_[i] = (1.0 - sm) * probs_[i] + sm * uniform;
    const double s2 = pairwise_sum(probs_.data(), n);
    for (std::size_t i = 0; i < n; ++i) probs_[i] /= s2;

    const std::int64_t k = std::min<std::int64_t>(target_active,
                                                  static_cast<std::int64_t>(n));
    sample_and_set(activeIdx_, probs_.data(), k, p, out);
}

/* _sdr_lateral_inhibition: softmax base probs (NO temperature), each bit's
 * probability reduced by higher-count neighbors within a bit-index radius. */
void SdrMerger::mode_li(std::int64_t total_bits, std::size_t n_inputs,
                        const MergeParams &p, SDR &out) {
    const std::int64_t target_active =
        static_cast<std::int64_t>(p.target_sparsity * static_cast<double>(total_bits));
    const std::size_t n = activeIdx_.size();

    /* base_probs = softmax(counts - max) */
    double mx = activeCnt_[0];
    for (std::size_t i = 1; i < n; ++i) mx = std::max(mx, activeCnt_[i]);
    probs_.resize(n);
    for (std::size_t i = 0; i < n; ++i) probs_[i] = std::exp(activeCnt_[i] - mx);
    const double bs = pairwise_sum(probs_.data(), n);
    for (std::size_t i = 0; i < n; ++i) probs_[i] /= bs;

    /* inhibited = base.copy(); windowed suppression. activeIdx_ is ascending,
     * so the |Delta idx| <= radius neighborhood is a contiguous window that
     * two pointers track in O(n) amortized. */
    work_.assign(probs_.begin(), probs_.end());
    const double denom =
        static_cast<double>(static_cast<std::int64_t>(n_inputs) * p.inhibition_radius);
    std::size_t lo = 0, hi = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const std::int64_t idx = activeIdx_[i];
        while (idx - activeIdx_[lo] > p.inhibition_radius) ++lo;
        if (hi < i + 1) hi = i + 1;
        while (hi < n && activeIdx_[hi] - idx <= p.inhibition_radius) ++hi;

        /* neighbors = (distance > 0) & (distance <= radius); the qualifying
         * counts (> my_count) are gathered in ASCENDING index order into a
         * contiguous buffer and pairwise-summed -- exactly what numpy's
         * boolean-mask copy + np.sum does. */
        const double my = activeCnt_[i];
        gather_.clear();
        for (std::size_t jj = lo; jj < hi; ++jj) {
            if (jj == i) continue;                 // distance > 0
            if (activeCnt_[jj] > my) gather_.push_back(activeCnt_[jj]);
        }
        if (hi - lo > 1) {                          // np.any(neighbors)
            double inh = pairwise_sum(gather_.data(), gather_.size()) / denom;
            inh = std::min(inh * p.inhibition_strength, 0.9);
            work_[i] *= (1.0 - inh);
        }
    }
    const double s = pairwise_sum(work_.data(), n);
    for (std::size_t i = 0; i < n; ++i) work_[i] /= s;

    const std::int64_t k = std::min<std::int64_t>(target_active,
                                                  static_cast<std::int64_t>(n));
    sample_and_set(activeIdx_, work_.data(), k, p, out);
}

/* _sdr_burst_weighted: bits present in ALL inputs get a multiplicative bonus
 * before the softmax. */
void SdrMerger::mode_bws(std::int64_t total_bits, std::size_t n_inputs,
                         const MergeParams &p, SDR &out) {
    const std::int64_t target_active =
        static_cast<std::int64_t>(p.target_sparsity * static_cast<double>(total_bits));
    const std::size_t n = activeIdx_.size();
    const double nin = static_cast<double>(n_inputs);

    const double t = std::max(p.temperature, 1e-10);
    work_.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        double c = activeCnt_[i];
        if (c == nin) c *= p.burst_bonus;   // boosted_counts[is_burst] *= bonus
        work_[i] = c / t;
    }
    double mx = work_[0];
    for (std::size_t i = 1; i < n; ++i) mx = std::max(mx, work_[i]);
    for (std::size_t i = 0; i < n; ++i) work_[i] = std::exp(work_[i] - mx);
    const double s = pairwise_sum(work_.data(), n);
    probs_.resize(n);
    for (std::size_t i = 0; i < n; ++i) probs_[i] = work_[i] / s;

    const std::int64_t k = std::min<std::int64_t>(target_active,
                                                  static_cast<std::int64_t>(n));
    sample_and_set(activeIdx_, probs_.data(), k, p, out);
}

/* _sdr_predictive_coding_boost: blend consensus (count/n) with novelty
 * (1 - (count-1)/max(n-1,1), clipped to [0,1]) before the softmax. */
void SdrMerger::mode_pcb(std::int64_t total_bits, std::size_t n_inputs,
                         const MergeParams &p, SDR &out) {
    const std::int64_t target_active =
        static_cast<std::int64_t>(p.target_sparsity * static_cast<double>(total_bits));
    const std::size_t n = activeIdx_.size();
    const double nin = static_cast<double>(n_inputs);
    const double den = static_cast<double>(
        std::max<std::int64_t>(static_cast<std::int64_t>(n_inputs) - 1, 1));

    const double t = std::max(p.temperature, 1e-10);
    work_.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double consensus = activeCnt_[i] / nin;
        double novelty = 1.0 - (activeCnt_[i] - 1.0) / den;
        novelty = std::min(std::max(novelty, 0.0), 1.0);   // np.clip(.., 0, 1)
        const double blended =
            (1.0 - p.novelty_weight) * consensus + p.novelty_weight * novelty;
        work_[i] = blended / t;
    }
    double mx = work_[0];
    for (std::size_t i = 1; i < n; ++i) mx = std::max(mx, work_[i]);
    for (std::size_t i = 0; i < n; ++i) work_[i] = std::exp(work_[i] - mx);
    const double s = pairwise_sum(work_.data(), n);
    probs_.resize(n);
    for (std::size_t i = 0; i < n; ++i) probs_[i] = work_[i] / s;

    const std::int64_t k = std::min<std::int64_t>(target_active,
                                                  static_cast<std::int64_t>(n));
    sample_and_set(activeIdx_, probs_.data(), k, p, out);
}

/* _sdr_dendritic_integration: fixed-size bit segments; only segments whose
 * distinct-active count clears the threshold contribute candidates.  Falls
 * back to plain proportional sampling when no segment qualifies. */
void SdrMerger::mode_di(std::int64_t total_bits, const MergeParams &p, SDR &out) {
    const std::int64_t target_active =
        static_cast<std::int64_t>(p.target_sparsity * static_cast<double>(total_bits));
    const std::int64_t seg = p.segment_size;
    const std::int64_t n_segments = (total_bits + seg - 1) / seg;

    candIdx_.clear();
    candCnt_.clear();
    std::size_t pos = 0;   // cursor into activeIdx_ (ascending)
    for (std::int64_t s = 0; s < n_segments; ++s) {
        const std::int64_t start = s * seg;
        const std::int64_t end = std::min(start + seg, total_bits);
        const std::size_t first = pos;
        while (pos < activeIdx_.size() && activeIdx_[pos] < end) ++pos;
        const std::int64_t n_active_in = static_cast<std::int64_t>(pos - first);
        const std::int64_t seg_len = end - start;

        /* if n_active_in_segment >= segment_threshold * segment_len */
        if (static_cast<double>(n_active_in) >=
            p.segment_threshold * static_cast<double>(seg_len)) {
            for (std::size_t j = first; j < pos; ++j) {
                candIdx_.push_back(activeIdx_[j]);
                candCnt_.push_back(activeCnt_[j]);
            }
        }
    }

    if (candIdx_.empty()) {
        /* Fallback: proportional sampling over all active bits. */
        const std::size_t n = activeIdx_.size();
        const double s = pairwise_sum(activeCnt_.data(), n);
        probs_.resize(n);
        for (std::size_t i = 0; i < n; ++i) probs_[i] = activeCnt_[i] / s;
        const std::int64_t k = std::min<std::int64_t>(
            target_active, static_cast<std::int64_t>(n));
        sample_and_set(activeIdx_, probs_.data(), k, p, out);
        return;
    }

    const std::size_t n = candIdx_.size();
    const double s = pairwise_sum(candCnt_.data(), n);
    probs_.resize(n);
    for (std::size_t i = 0; i < n; ++i) probs_[i] = candCnt_[i] / s;
    const std::int64_t k = std::min<std::int64_t>(target_active,
                                                  static_cast<std::int64_t>(n));
    sample_and_set(candIdx_, probs_.data(), k, p, out);
}

/* ------------------------------------------------------------------------ */
void SdrMerger::merge(const std::vector<const SDR *> &inputs, MergeMode mode,
                      int axis, const MergeParams &params, SDR &out) {
    NTA_CHECK(!inputs.empty()) << "sdr_merge needs at least one input";

    /* Python: `if len(inputs) == 1: return inputs[0]` -- the runtime
     * short-circuits this before calling; copy for API completeness. */
    if (inputs.size() == 1) {
        NTA_CHECK(inputs[0]->size == out.size)
            << "single-input merge: size mismatch";
        out.setSparse(inputs[0]->getSparse());   // template overload: copy
        return;
    }

    check_shapes(inputs);

    /* Concat/stack never look at counts. */
    if (mode == MergeMode::Concat) { mode_concat(inputs, axis, out); return; }
    if (mode == MergeMode::Stack)  { mode_stack(inputs, out);        return; }

    const std::int64_t total_bits = accumulate_counts(inputs);

    /* Stochastic modes: `if len(active_indices) == 0: return SDR(shape)`. */
    if (activeIdx_.empty()) {
        selected_.clear();
        out.setSparse(selected_);
        clear_counts();
        return;
    }

    switch (mode) {
        case MergeMode::Union:          mode_union(out); break;
        case MergeMode::Intersection:   mode_intersection(inputs.size(), out); break;
        case MergeMode::SymDiff:        mode_symdiff(inputs.size(), out); break;
        case MergeMode::NonOverlapSum:  mode_nos(out); break;
        case MergeMode::SoftUnion:      mode_su(total_bits, params, out); break;
        case MergeMode::ActivationProportional:
                                        mode_aps(total_bits, params, out); break;
        case MergeMode::PowerLaw:       mode_pls(total_bits, params, out); break;
        case MergeMode::StabilizedSoftUnion:
                                        mode_ssu(total_bits, params, out); break;
        case MergeMode::LateralInhibition:
                                        mode_li(total_bits, inputs.size(), params, out); break;
        case MergeMode::BurstWeighted:  mode_bws(total_bits, inputs.size(), params, out); break;
        case MergeMode::PredictiveCodingBoost:
                                        mode_pcb(total_bits, inputs.size(), params, out); break;
        case MergeMode::DendriticIntegration:
                                        mode_di(total_bits, params, out); break;
        default:
            clear_counts();
            NTA_THROW << "unreachable merge mode";
    }
    clear_counts();
}

/* ------------------------------------------------------------------------ */
/* NOTE on fidelity: Python's sdr_max_pool assigns the pooled COORDINATES
 * directly; when pooling collapses several bits into one cell the resulting
 * SDR silently carries an unsorted / duplicated internal sparse list (only
 * debug builds assert).  Every downstream consumer reads it through .dense,
 * where order and duplicates vanish -- the semantic content is the bit SET.
 * This port produces that same set, in a properly sorted, de-duplicated SDR. */
void sdr_max_pool_into(const SDR &in, int ratio, int axis, SDR &out,
                       std::vector<UInt> &scratch) {
    NTA_CHECK(ratio >= 1);
    const auto &dims = in.dimensions;
    const int ndim = static_cast<int>(dims.size());
    const int a = positify_axis(axis, ndim);
    const UInt da = dims[static_cast<std::size_t>(a)];
    NTA_CHECK(da % static_cast<UInt>(ratio) == 0)
        << "Max pooling dimension must be divisible by ratio, got: ("
        << da << ", " << ratio << ")";

    std::int64_t in_stride[16], out_stride[16];
    NTA_CHECK(ndim <= 16) << "SDR rank too large";
    const auto &odims = out.dimensions;
    in_stride[ndim - 1] = out_stride[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; --d) {
        in_stride[d]  = in_stride[d + 1]  * static_cast<std::int64_t>(dims[static_cast<std::size_t>(d + 1)]);
        out_stride[d] = out_stride[d + 1] * static_cast<std::int64_t>(odims[static_cast<std::size_t>(d + 1)]);
    }

    scratch.clear();
    for (UInt f : in.getSparse()) {
        std::int64_t rem = static_cast<std::int64_t>(f), of = 0;
        for (int d = 0; d < ndim; ++d) {
            std::int64_t coord = rem / in_stride[d];
            rem -= coord * in_stride[d];
            if (d == a) coord /= ratio;      // new_coords[axis] //= ratio
            of += coord * out_stride[d];
        }
        scratch.push_back(static_cast<UInt>(of));
    }
    std::sort(scratch.begin(), scratch.end());
    scratch.erase(std::unique(scratch.begin(), scratch.end()),
                  scratch.end());            // pooled duplicates collapse
    out.setSparse(scratch);
}

void SdrMerger::max_pool(const SDR &in, int ratio, int axis, SDR &out) {
    sdr_max_pool_into(in, ratio, axis, out, selected_);
}

} // namespace pyramid
