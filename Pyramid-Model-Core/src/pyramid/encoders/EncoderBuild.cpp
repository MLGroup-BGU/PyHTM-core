/* ---------------------------------------------------------------------------
 * Pyramid-Model-Core :: encoders/EncoderBuild.cpp
 * See EncoderBuild.hpp; each block cites the Python it ports.
 * ------------------------------------------------------------------------ */
#include "EncoderBuild.hpp"

#include <htm/utils/Log.hpp>

#include <algorithm>
#include <cmath>

#include "../math/NumpyCompat.hpp"

namespace pyramid {

/* ---------------- get_rdse_resolution (config.py 263-315) ---------------- */
double deduce_rdse_resolution(const std::string &feature,
                              const std::vector<double> &sample,
                              const std::string &resolution_type,
                              double enc_n, double enc_w) {
    NTA_CHECK(resolution_type == "Difference" || resolution_type == "Absolute")
        << "not supported resolution_type in encoders parameters. value "
        << "provided: resolution type - " << resolution_type;

    if (resolution_type == "Difference") {
        /* diffs = |sample[:-1] - sample[1:]| */
        std::vector<double> diffs;
        if (sample.size() > 1) diffs.reserve(sample.size() - 1);
        bool has_nan = false;
        for (std::size_t i = 0; i + 1 < sample.size(); ++i) {
            const double d = std::fabs(sample[i] - sample[i + 1]);
            if (std::isnan(d)) has_nan = true;
            diffs.push_back(d);
        }
        /* np.percentile of a NaN-containing array is NaN; propagate the same
         * way (the downstream RDSE then behaves exactly as it would in
         * Python).  Real datasets in PyHTM contain no NaNs here. */
        if (has_nan) return std::numeric_limits<double>::quiet_NaN();

        int q = 25;
        double resolution = percentile_linear(diffs, static_cast<double>(q));
        if (resolution == 0.0) {
            while (q < 100 && resolution == 0.0) {
                q += 5;
                resolution = percentile_linear(diffs, static_cast<double>(q));
            }
        }
        if (resolution == 0.0) {
            resolution = 1.0;   // "No variation in sample" (constants)
        }
        return resolution;
    }

    /* Absolute: (max - min) / (n - int(n*w) + 1) */
    double mn = sample[0], mx = sample[0];
    for (double x : sample) { mn = std::min(mn, x); mx = std::max(mx, x); }
    const double denom = enc_n - std::trunc(enc_n * enc_w) + 1.0;
    (void)feature;
    return (mx - mn) / denom;
}

/* ------ build_adaptive_params_from_samples (adaptive_rdse.py 223-301) ---- */
std::pair<AdaptiveParams, double> build_adaptive_params_from_samples(
        const std::vector<double> &samples, std::int64_t size,
        std::int64_t activeBits, std::uint64_t seed, std::int64_t n_quantiles,
        double index_scale, double gamma, double min_resolution,
        bool clip_input, bool reserve_for_special_values) {
    std::vector<double> arr;
    arr.reserve(samples.size());
    for (double x : samples)
        if (std::isfinite(x)) arr.push_back(x);

    n_quantiles = std::max<std::int64_t>(2, n_quantiles);
    std::vector<double> qs =
        linspace(0.0, 1.0, static_cast<std::size_t>(n_quantiles));

    std::vector<double> vs;
    if (arr.empty()) {
        vs = {0.0, 1.0};   // fallback for empty / fully non-finite samples
    } else {
        vs = quantile_median_unbiased(arr, qs);
        /* vs += linspace(0, 1e-9, n_quantiles): de-degenerates flat knots */
        const auto eps =
            linspace(0.0, 1e-9, static_cast<std::size_t>(n_quantiles));
        for (std::size_t i = 0; i < vs.size(); ++i) vs[i] += eps[i];
    }

    /* NOTE (Python quirk, kept): when samples are empty, vs has 2 entries
     * but qs keeps n_quantiles entries; the warp below only consumes
     * min(len)-1 segments through diff(vs)/diff(qs) via numpy broadcasting
     * -- which would RAISE in numpy for mismatched lengths.  In Python this
     * path can only be hit with an all-NaN training column, which then
     * raises inside np.diff arithmetic; here it throws the same way. */
    NTA_CHECK(vs.size() == qs.size() || !arr.empty())
        << "adaptive fit received an empty/non-finite sample column";

    /* dv/dq -> tempered seg -> cumulative g (shared with AdaptiveRdse). */
    std::vector<double> g_knots;
    adaptive_warp_knots(qs, vs, gamma, index_scale, g_knots);

    /* resolution = max(min |diff(g_knots)|, min_resolution); 1.0 if empty */
    double resolution;
    {
        double mn = std::numeric_limits<double>::infinity();
        bool any = false;
        for (std::size_t i = 0; i + 1 < g_knots.size(); ++i) {
            const double d = g_knots[i + 1] - g_knots[i];
            if (!std::isfinite(d)) continue;
            mn = std::min(mn, std::fabs(d));
            any = true;
        }
        resolution = any ? std::max(mn, min_resolution) : 1.0;
    }

    AdaptiveParams p;
    p.size = size;
    p.activeBits = activeBits;
    p.seed = seed;
    p.quantiles = std::move(qs);
    p.quantile_values = std::move(vs);
    p.index_scale = index_scale;
    p.gamma = gamma;
    p.clip_input = clip_input;
    p.reserve_for_special_values = reserve_for_special_values;
    return {std::move(p), resolution};
}

/* numpy-default linear-interpolated quantile of a SORTED vector. */
static double quantile_sorted(const std::vector<double> &v, double q) {
    if (v.size() == 1) return v[0];
    const double pos = q * static_cast<double>(v.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(pos);
    const std::size_t hi = std::min(lo + 1, v.size() - 1);
    const double frac = pos - static_cast<double>(lo);
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

/* -------------- EncoderFactory.get_encoder (encoding.py) ----------------- */
static htm::RDSE_Parameters rdse_params(std::uint64_t seed, std::int64_t size,
                                        std::int64_t activeBits,
                                        std::optional<double> resolution,
                                        bool category) {
    htm::RDSE_Parameters p;
    p.seed = static_cast<htm::UInt>(seed);
    p.size = static_cast<htm::UInt>(size);
    p.activeBits = static_cast<htm::UInt>(activeBits);
    if (category) p.category = true;
    else          p.resolution = resolution.value();
    return p;
}

static std::int64_t np_clip_i(std::int64_t v, std::int64_t lo, std::int64_t hi) {
    return std::min(std::max(v, lo), hi);
}

std::unique_ptr<FeatureEncoder> build_feature_encoder(
        const FeatureSpec &spec, const std::vector<double> *samples) {
    NTA_CHECK(!spec.needs_samples() || samples != nullptr)
        << "feature `" << spec.name << "` requires training samples to build";

    auto fe = std::make_unique<FeatureEncoder>();
    fe->kind = spec.kind;
    fe->name = spec.name;

    switch (spec.kind) {
        case FeatureKind::Categorical: {
            fe->rdse = std::make_unique<ShapedRdse>(
                rdse_params(spec.seed, spec.size, spec.activeBits,
                            std::nullopt, /*category=*/true),
                spec.shape);
            break;
        }
        case FeatureKind::Rdse: {
            double res;
            if (spec.resolution.has_value()) {
                res = *spec.resolution;
            } else {
                /* "No resolution supplied - one will be deduced." */
                res = deduce_rdse_resolution(spec.name, *samples,
                                             spec.resolution_type,
                                             spec.dedn, spec.dedw);
            }
            fe->rdse = std::make_unique<ShapedRdse>(
                rdse_params(spec.seed, spec.size, spec.activeBits, res,
                            /*category=*/false),
                spec.shape);
            break;
        }
        case FeatureKind::Linear: {
            double lo, hi;
            if (spec.lin_min.has_value() && spec.lin_max.has_value()) {
                lo = *spec.lin_min;
                hi = *spec.lin_max;
            } else {
                /* Auto-calibration from the training slice: robust
                 * percentiles (default p1/p99) -- the linear twin of the
                 * RDSE resolution deduction. */
                std::vector<double> vals;
                vals.reserve(samples->size());
                for (const double x : *samples)
                    if (!std::isnan(x)) vals.push_back(x);
                NTA_CHECK(!vals.empty())
                    << "feature `" << spec.name << "`: linear auto range "
                    << "needs at least one non-NaN training sample";
                std::sort(vals.begin(), vals.end());
                lo = quantile_sorted(vals, spec.lin_p_low);
                hi = quantile_sorted(vals, spec.lin_p_high);
            }
            fe->linear = std::make_unique<ShapedLinear>(
                spec.size, spec.activeBits, lo, hi, spec.resolution,
                spec.shape);
            break;
        }
        case FeatureKind::AdaptiveRdse: {
            /* build_enc_params computes adaptive_params with seed=0 and the
             * FULL size/activeBits, and stores suggested_resolution as the
             * feature's 'resolution' for the inner RDSE. */
            auto [aparams, sug_res] = build_adaptive_params_from_samples(
                *samples, spec.size, spec.activeBits, /*seed=*/0,
                spec.adaptive_n_quantiles, spec.adaptive_index_scale,
                spec.adaptive_gamma, spec.adaptive_min_resolution,
                spec.adaptive_clip_input, spec.adaptive_reserve);

            auto inner = std::make_unique<ShapedRdse>(
                rdse_params(spec.seed, spec.size, spec.activeBits, sug_res,
                            false),
                spec.shape);
            fe->adaptive = std::make_unique<AdaptiveRdse>(
                std::move(inner), std::move(aparams), spec.shape);
            break;
        }
        case FeatureKind::HybridRdse: {
            auto [aparams, sug_res] = build_adaptive_params_from_samples(
                *samples, spec.size, spec.activeBits, /*seed=*/0,
                spec.adaptive_n_quantiles, spec.adaptive_index_scale,
                spec.adaptive_gamma, spec.adaptive_min_resolution,
                spec.adaptive_clip_input, spec.adaptive_reserve);

            /* Split geometry, verbatim from get_encoder (encoding.py
             * 123-147): floor size split; proportional width split with the
             * two-step rounding fix. */
            const double ratio = spec.hybrid_ratio;
            const std::int64_t total_size = spec.size;
            const std::int64_t total_w = spec.activeBits;
            const std::int64_t size_a = static_cast<std::int64_t>(
                std::floor(static_cast<double>(total_size) * ratio));
            const std::int64_t size_b = total_size - size_a;
            NTA_CHECK(size_a > 0 && size_b > 0)
                << "hybrid_rdse split invalid: size=" << total_size
                << ", ratio=" << ratio << " -> (" << size_a << ", " << size_b << ")";

            std::int64_t w_a = static_cast<std::int64_t>(
                std::floor(static_cast<double>(total_w) * ratio));
            w_a = std::max<std::int64_t>(1, std::min(w_a, size_a));
            std::int64_t w_b = total_w - w_a;
            w_b = std::max<std::int64_t>(1, std::min(w_b, size_b));
            if (w_a + w_b != total_w) {
                std::int64_t delta = total_w - (w_a + w_b);
                w_b = np_clip_i(w_b + delta, 1, size_b);
                if (w_a + w_b != total_w) {
                    delta = total_w - (w_a + w_b);
                    w_a = np_clip_i(w_a + delta, 1, size_a);
                }
            }

            const std::uint64_t base_seed = spec.seed;

            /* Regular half: seed = base + 1. */
            auto enc_b = std::make_unique<ShapedRdse>(
                rdse_params(base_seed + 1, size_b, w_b, sug_res, false),
                std::vector<htm::UInt>{});

            /* Adaptive half: inner RDSE seed = base; params re-sized. */
            auto enc_a_rdse = std::make_unique<ShapedRdse>(
                rdse_params(base_seed, size_a, w_a, sug_res, false),
                std::vector<htm::UInt>{});
            AdaptiveParams half = aparams;
            half.size = size_a;
            half.activeBits = w_a;
            half.seed = base_seed;
            auto enc_a = std::make_unique<AdaptiveRdse>(
                std::move(enc_a_rdse), std::move(half),
                std::vector<htm::UInt>{});

            fe->hybrid = std::make_unique<HybridRdse>(
                std::move(enc_a), std::move(enc_b), total_size, size_a,
                spec.shape);
            break;
        }
        case FeatureKind::DualScalar: {
            /* Split defaults, verbatim from get_encoder (encoding.py
             * 203-238). */
            const std::int64_t total_size = spec.size;
            const std::int64_t total_w = spec.activeBits;
            const auto &d = spec.dual;

            std::int64_t fast_n = d.fast_n_bits.value_or(
                std::max<std::int64_t>(1, total_size / 4));
            std::int64_t fast_w = d.fast_w.value_or(
                std::max<std::int64_t>(1, std::min(fast_n, total_w / 2)));
            std::int64_t slow_n = d.slow_n_bits.value_or(total_size - fast_n);
            std::int64_t slow_w = d.slow_w.value_or(
                std::max<std::int64_t>(1, std::min(slow_n, total_w - fast_w)));

            if (slow_n <= 0) slow_n = std::max<std::int64_t>(1, total_size - fast_n);
            if (slow_n + fast_n != total_size) slow_n = total_size - fast_n;
            slow_w = np_clip_i(slow_w, 1, slow_n);
            fast_w = np_clip_i(fast_w, 1, fast_n);

            DualScalarConfig cfg;
            cfg.slow_n_bits = slow_n;
            cfg.slow_w = slow_w;
            cfg.fast_n_bits = fast_n;
            cfg.fast_w = fast_w;
            cfg.n_quantiles = d.n_quantiles.value_or(513);
            cfg.fast_alpha = d.fast_alpha.value_or(0.01);
            cfg.fast_z_clip = d.fast_z_clip.value_or(5.0);
            cfg.fast_warmup = d.fast_warmup.value_or(25);
            cfg.clip_input = d.clip_input.value_or(true);
            cfg.reserve_for_special_values =
                d.reserve_for_special_values.value_or(true);

            fe->dual = std::make_unique<DualScalarEncoder>(cfg, spec.shape);
            fe->dual->fit(*samples);
            break;
        }
        case FeatureKind::Date: {
            fe->date = std::make_unique<DateFeature>(spec.date);
            break;
        }
    }
    return fe;
}

} // namespace pyramid