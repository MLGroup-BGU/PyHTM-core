/* ---------------------------------------------------------------------------
 * Pyramid-Model-Core :: encoders/FeatureEncoders.hpp
 *
 * C++ ports of every encoder a PyHTM Feature can be configured with:
 *
 *   ShapedRdse        <- encoding.ShapedRDSE + init_rdse   (numeric, categorical)
 *   AdaptiveRdse      <- adaptive_rdse.AdaptiveRDSE        (distribution-aware)
 *   HybridRdse        <- hybrid_rdse.HybridRDSE            (adaptive || rdse)
 *   DualScalarEncoder <- dual_scalar_encoder.DualScalarEncoder (STATEFUL)
 *   DateFeature       <- py/htm/encoders/date.DateEncoder  (composite datetime)
 *
 * plus `FeatureEncoder`, the small dispatcher a pyramid Feature holds.
 *
 * FAITHFULNESS
 *   Every numeric expression is ported in the same float64 operation order
 *   as the Python source, on top of the verified NumpyCompat / core-RDSE /
 *   core-ScalarEncoder primitives, so encodings are bit-identical for
 *   identical inputs.  Two intentional ports of Python *quirks*:
 *     - AdaptiveRDSE's "reserved special value" SDRs are EMPTY: the Python
 *       code does `SDR(size).sparse.fill(k)` -- filling a zero-length array,
 *       a no-op -- so NaN / underflow / overflow encode to all-zero SDRs.
 *     - DualScalarEncoder's EMA state advances on every *normal* encode but
 *       NOT on special-value encodes (NaN / underflow / overflow return
 *       before the update).  Record order therefore matters; the runtime
 *       encodes strictly in stream order.
 *
 * MEMORY: each encoder owns fixed scratch; steady-state encoding performs
 * no heap allocation.  THREADING: encoders are not thread-safe; the runtime
 * encodes on its orchestrator thread (as Python's DataStreamer does on its
 * calling thread).
 * ------------------------------------------------------------------------ */
#pragma once

#include <htm/encoders/RandomDistributedScalarEncoder.hpp>
#include <htm/encoders/ScalarEncoder.hpp>
#include <htm/types/Sdr.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "DateMath.hpp"

namespace pyramid {

/* ======================================================================== */
/* ShapedRDSE: a core RDSE that reports (and reshapes to) a target shape.   */
/* Construction reproduces init_rdse's terminal behavior: the (now fully    */
/* deterministic) collision check either passes or would have failed all 5  */
/* Python retries identically, so one attempt is made and the failure is    */
/* rethrown with the same terminal message.                                 */
/* ======================================================================== */
class ShapedRdse {
public:
    ShapedRdse(const htm::RDSE_Parameters &params,
               std::vector<htm::UInt> shape);

    void encode(double v, htm::SDR &out);   // out.size must equal size()
    htm::UInt size() const { return rdse_->size; }
    const std::vector<htm::UInt> &dimensions() const { return shape_; }

private:
    std::unique_ptr<htm::RandomDistributedScalarEncoder> rdse_;
    std::vector<htm::UInt> shape_;
};

/* ======================================================================== */
/* AdaptiveRDSE parameters (mirrors the AdaptiveRDSEParams dataclass).     */
/* ======================================================================== */
struct AdaptiveParams {
    std::int64_t size = 0;
    std::int64_t activeBits = 0;
    std::uint64_t seed = 0;
    std::vector<double> quantiles;
    std::vector<double> quantile_values;
    double index_scale = 1.0;
    double gamma = 1.0;
    bool clip_input = true;
    bool reserve_for_special_values = false;
};

class AdaptiveRdse {
public:
    /* `rdse` is the inner encoder built with the suggested (index-space)
     * resolution; `shape` empty means (size,). */
    AdaptiveRdse(std::unique_ptr<ShapedRdse> rdse, AdaptiveParams params,
                 std::vector<htm::UInt> shape);

    void encode(double v, htm::SDR &out);
    htm::UInt size() const { return static_cast<htm::UInt>(params_.size); }
    const std::vector<htm::UInt> &dimensions() const { return shape_; }

private:
    std::unique_ptr<ShapedRdse> rdse_;
    AdaptiveParams params_;
    std::vector<htm::UInt> shape_;

    std::vector<double> q_, v_, gKnots_, gSlopes_;
    double vmin_ = 0.0, vmax_ = 1.0;

    double warp_to_index(double x) const;
};

/* Shared with EncoderBuild: the inverse-density cumulative warp from
 * adaptive_rdse.py (both AdaptiveRDSE.__init__ and
 * build_adaptive_params_from_samples run this identical block). */
void adaptive_warp_knots(const std::vector<double> &q,
                         const std::vector<double> &v, double gamma,
                         double index_scale, std::vector<double> &g_knots_out);

/* ======================================================================== */
class HybridRdse {
public:
    HybridRdse(std::unique_ptr<AdaptiveRdse> adaptive,
               std::unique_ptr<ShapedRdse> rdse,
               std::int64_t total_size, std::int64_t size_adaptive,
               std::vector<htm::UInt> shape);

    void encode(double v, htm::SDR &out);
    htm::UInt size() const { return static_cast<htm::UInt>(totalSize_); }
    const std::vector<htm::UInt> &dimensions() const { return shape_; }

private:
    std::unique_ptr<AdaptiveRdse> adaptive_;
    std::unique_ptr<ShapedRdse> rdse_;
    std::int64_t totalSize_, sizeAdaptive_;
    std::vector<htm::UInt> shape_;
    htm::SDR scratchA_, scratchB_;
    std::vector<htm::UInt> joined_;
};

/* ======================================================================== */
/* OverlappingBucketScalarEncoder (dual_scalar_encoder.py).                 */
/* ======================================================================== */
struct OverlappingBucket {
    double minval, maxval;
    std::int64_t n_bits, w;
    bool clip_input;
    std::int64_t n_starts;

    OverlappingBucket(double mn, double mx, std::int64_t n, std::int64_t w_,
                      bool clip);
    /* Returns the first active index; active bits are [start, start + w). */
    std::int64_t encode_start(double x) const;
};

struct DualScalarConfig {
    std::int64_t slow_n_bits = 0, slow_w = 0;
    std::int64_t n_quantiles = 513;
    std::int64_t fast_n_bits = 512, fast_w = 15;
    double fast_alpha = 0.01, fast_z_clip = 5.0;
    std::int64_t fast_warmup = 25;
    bool clip_input = true, reserve_for_special_values = true;
};

class DualScalarEncoder {
public:
    DualScalarEncoder(const DualScalarConfig &cfg,
                      std::vector<htm::UInt> shape);

    /* Offline fit from the training slice (quantile CDF); must precede
     * encode, exactly as in Python. */
    void fit(const std::vector<double> &training_values);
    void encode(double v, htm::SDR &out);

    htm::UInt size() const {
        return static_cast<htm::UInt>(cfg_.slow_n_bits + cfg_.fast_n_bits);
    }
    const std::vector<htm::UInt> &dimensions() const { return shape_; }

private:
    DualScalarConfig cfg_;
    std::vector<htm::UInt> shape_;
    OverlappingBucket slowScalar_, fastScalar_;

    /* _RunningStatsEMA */
    double emaMean_ = 0.0, emaVar_ = 1.0;
    bool emaInit_ = false;
    std::int64_t fastSteps_ = 0;

    bool fitted_ = false;
    std::vector<double> q_, v_;
    double vmin_ = 0.0, vmax_ = 1.0;

    std::vector<htm::UInt> special_[3];   // underflow, overflow, nan patterns
    std::vector<htm::UInt> bits_;

    double cdf_quantile(double x) const;
};

/* ======================================================================== */
/* DateFeature: port of the pure-Python composite DateEncoder.             */
/* Each attribute config value is an int width or a (width, radius) pair,   */
/* exactly as the YAML feature config provides them.                        */
/* ======================================================================== */
struct DateAttr {           // 0 width == disabled (Python's `!= 0` check)
    std::int64_t width = 0;
    double radius = 0.0;    // used only where the Python default is a radius
    bool has_radius = false;
};

struct DateConfig {
    DateAttr season, dayOfWeek, timeOfDay;
    std::int64_t weekend = 0;                  // width only
    std::int64_t holiday = 0;                  // width only
    std::vector<int> customDays;               // Monday=0 list; empty = off
    std::int64_t customDaysWidth = 0;
    /* (month, day) or (year, month, day) holiday dates; default Dec 25. */
    std::vector<std::vector<int>> holidays = {{12, 25}};
};

class DateFeature {
public:
    explicit DateFeature(const DateConfig &cfg);

    void encode(const DateParts &v, htm::SDR &out);
    htm::UInt size() const { return static_cast<htm::UInt>(size_); }
    const std::vector<htm::UInt> &dimensions() const { return dims_; }

private:
    DateConfig cfg_;
    std::int64_t size_ = 0;
    std::vector<htm::UInt> dims_;

    std::unique_ptr<htm::ScalarEncoder> season_, dayOfWeek_, weekend_,
                                        customDays_, holiday_, timeOfDay_;
    htm::SDR scratch_;
    std::vector<htm::UInt> joined_;

    void append_encoded(htm::ScalarEncoder &enc, double value,
                        std::int64_t offset);
};

/* ======================================================================== */
/* FeatureEncoder: what the runtime holds per feature.                      */
/* ======================================================================== */
enum class FeatureKind : std::uint8_t {
    Rdse, AdaptiveRdse, HybridRdse, DualScalar, Categorical, Date
};

class FeatureEncoder {
public:
    FeatureKind kind;
    std::string name;

    std::unique_ptr<ShapedRdse> rdse;            // Rdse / Categorical
    std::unique_ptr<AdaptiveRdse> adaptive;      // AdaptiveRdse
    std::unique_ptr<HybridRdse> hybrid;          // HybridRdse
    std::unique_ptr<DualScalarEncoder> dual;     // DualScalar
    std::unique_ptr<DateFeature> date;           // Date

    bool is_datetime() const { return kind == FeatureKind::Date; }

    void encode(double v, htm::SDR &out);              // numeric family
    void encode(const DateParts &v, htm::SDR &out);    // datetime

    htm::UInt size() const;
    const std::vector<htm::UInt> &dimensions() const;
};

} // namespace pyramid
