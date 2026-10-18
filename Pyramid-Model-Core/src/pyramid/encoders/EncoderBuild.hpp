/* ---------------------------------------------------------------------------
 * Pyramid-Model-Core :: encoders/EncoderBuild.hpp
 *
 * The build-phase half of PyHTM's encoder pipeline:
 *
 *   config.build_enc_params  ->  sample-derived parts run HERE (in C++,
 *                                during the streaming build pass):
 *                                  - get_rdse_resolution   (percentile of
 *                                    consecutive |diffs|, escalating q)
 *   encoding.EncoderFactory.get_encoder -> ported as build_feature_encoder.
 *
 * WHY THE SPLIT
 *   Everything that needs only the YAML config (encoder kind from the
 *   feature's type, size / activeBits with Python's banker's rounding,
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

/* Fully-resolved per-feature spec, as delivered by the Python spec builder
 * (htm_source/pipeline/pyramid_spec.py) inside the instruction book. */
struct FeatureSpec {
    std::string name;
    FeatureKind kind = FeatureKind::Rdse;

    std::int64_t size = 0;          // resolved: banker's rounding
    std::int64_t activeBits = 0;
    std::vector<htm::UInt> shape;   // empty -> (size,)
    std::uint64_t seed = 0;         // final: pyramid_seed * encoder index

    /* rdse / categorical */
    std::optional<double> resolution;      // absent -> deduce from samples
    std::string resolution_type = "Difference";   // deduction mode
    double dedn = 0.0, dedw = 0.0;         // encoders n / w (Absolute mode)

    /* datetime */
    DateConfig date;
    std::string dt_format;          // Feature's strptime format

    /* Does building this encoder require the training-slice samples? */
    bool needs_samples() const {
        return kind == FeatureKind::Rdse && !resolution.has_value();
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

/* EncoderFactory.get_encoder, one feature at a time.  `samples` must be the
 * feature's training-slice values when spec.needs_samples(), else null. */
std::unique_ptr<FeatureEncoder> build_feature_encoder(
    const FeatureSpec &spec, const std::vector<double> *samples);

} // namespace pyramid