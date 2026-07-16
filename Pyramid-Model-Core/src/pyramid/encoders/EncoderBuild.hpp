/* ---------------------------------------------------------------------------
 * Pyramid-Model-Core :: encoders/EncoderBuild.hpp
 *
 * The build-phase half of PyHTM's encoder pipeline:
 *
 *   config.build_enc_params  ->  sample-derived parts run HERE (in C++,
 *                                during the streaming build pass):
 *                                  - get_rdse_resolution   (percentile of
 *                                    consecutive |diffs|, escalating q)
 *                                  - build_adaptive_params_from_samples
 *                                    (median_unbiased quantile knots ->
 *                                     inverse-density warp -> resolution)
 *                                  - DualScalarEncoder.fit samples
 *   encoding.EncoderFactory.get_encoder -> ported as build_feature_encoder
 *                                (hybrid size/width splits, dual splits,
 *                                 inner-RDSE seeding: base and base+1).
 *
 * WHY THE SPLIT
 *   Everything that needs only the YAML config (encoder kind, hyperparams,
 *   size / activeBits with feature weights and Python's banker's rounding,
 *   the final feature seed = pyramid_seed * construction index) is resolved
 *   on the PYTHON side by the spec builder, using the very same code paths
 *   as today -- zero re-implementation risk.  Only quantities that require
 *   the DATA are computed here, from the training slice the record source
 *   streams, with the bit-exact NumpyCompat quantile/percentile math.
 * ------------------------------------------------------------------------ */
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "FeatureEncoders.hpp"

namespace pyramid {

/* Optional per-feature dual_scalar overrides (config 'dual_params'). */
struct DualOverrides {
    std::optional<std::int64_t> fast_n_bits, fast_w, slow_n_bits, slow_w;
    std::optional<std::int64_t> n_quantiles, fast_warmup;
    std::optional<double> fast_alpha, fast_z_clip;
    std::optional<bool> clip_input, reserve_for_special_values;
};

/* Fully-resolved per-feature spec, as delivered by the Python spec builder
 * (htm_source/pipeline/pyramid_spec.py) inside the instruction book. */
struct FeatureSpec {
    std::string name;
    FeatureKind kind = FeatureKind::Rdse;

    std::int64_t size = 0;          // resolved: weights + banker's rounding
    std::int64_t activeBits = 0;
    std::vector<htm::UInt> shape;   // empty -> (size,)
    std::uint64_t seed = 0;         // final: pyramid_seed * encoder index

    /* rdse / categorical */
    std::optional<double> resolution;      // absent -> deduce from samples
    std::string resolution_type = "Difference";   // deduction mode
    double dedn = 0.0, dedw = 0.0;         // encoders n / w (Absolute mode)

    /* adaptive / hybrid hyperparameters (defaults match build_enc_params) */
    std::int64_t adaptive_n_quantiles = 101;
    double adaptive_index_scale = 1.0;
    double adaptive_gamma = 1.0;
    double adaptive_min_resolution = 1e-6;
    bool adaptive_clip_input = true;
    bool adaptive_reserve = false;
    double hybrid_ratio = 0.5;

    /* dual_scalar */
    DualOverrides dual;

    /* datetime */
    DateConfig date;
    std::string dt_format;          // Feature's strptime format

    /* Does building this encoder require the training-slice samples? */
    bool needs_samples() const {
        return kind == FeatureKind::AdaptiveRdse ||
               kind == FeatureKind::HybridRdse ||
               kind == FeatureKind::DualScalar ||
               (kind == FeatureKind::Rdse && !resolution.has_value());
    }
};

/* get_rdse_resolution (config.py): 'Difference' takes np.percentile of the
 * consecutive-absolute-diffs at q=25, escalating by +5 while the result is
 * exactly 0 (constant regions), falling back to 1.0; 'Absolute' is
 * (max-min) / (n - int(n*w) + 1).  NaNs in the sample propagate exactly as
 * numpy propagates them. */
double deduce_rdse_resolution(const std::string &feature,
                              const std::vector<double> &sample,
                              const std::string &resolution_type,
                              double enc_n, double enc_w);

/* build_adaptive_params_from_samples (adaptive_rdse.py): returns the params
 * (quantile knots etc.) and the suggested index-space resolution. */
std::pair<AdaptiveParams, double> build_adaptive_params_from_samples(
    const std::vector<double> &samples, std::int64_t size,
    std::int64_t activeBits, std::uint64_t seed, std::int64_t n_quantiles,
    double index_scale, double gamma, double min_resolution, bool clip_input,
    bool reserve_for_special_values);

/* EncoderFactory.get_encoder, one feature at a time.  `samples` must be the
 * feature's training-slice values when spec.needs_samples(), else null. */
std::unique_ptr<FeatureEncoder> build_feature_encoder(
    const FeatureSpec &spec, const std::vector<double> *samples);

} // namespace pyramid
