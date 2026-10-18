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
        /* At least two training rows are needed for one difference (numpy
         * raises on the empty percentile; this used to read past the end). */
        NTA_CHECK(sample.size() >= 2)
            << "feature `" << feature << "`: deducing its resolution needs at "
            << "least 2 training rows, got " << sample.size()
            << "; give the feature a `resolution` in the data config";
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
    NTA_CHECK(!sample.empty())
        << "feature `" << feature << "`: deducing its resolution needs at "
        << "least 1 training row";
    double mn = sample[0], mx = sample[0];
    for (double x : sample) { mn = std::min(mn, x); mx = std::max(mx, x); }
    const double denom = enc_n - std::trunc(enc_n * enc_w) + 1.0;
    return (mx - mn) / denom;
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
        case FeatureKind::Date: {
            fe->date = std::make_unique<DateFeature>(spec.date);
            break;
        }
    }
    return fe;
}

} // namespace pyramid