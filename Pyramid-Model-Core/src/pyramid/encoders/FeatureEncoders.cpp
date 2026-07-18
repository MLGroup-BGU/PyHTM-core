/* ---------------------------------------------------------------------------
 * Pyramid-Model-Core :: encoders/FeatureEncoders.cpp
 *
 * Implementations.  Every block references the Python lines it ports; the
 * float64 expressions keep the Python operation order verbatim.
 * ------------------------------------------------------------------------ */
#include "FeatureEncoders.hpp"

#include <htm/utils/Log.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "../math/NumpyCompat.hpp"

namespace pyramid {

using htm::SDR;
using htm::UInt;

/* ========================== ShapedRdse ================================== */
ShapedRdse::ShapedRdse(const htm::RDSE_Parameters &params,
                       std::vector<UInt> shape)
    : shape_(std::move(shape)) {
    try {
        rdse_ = std::make_unique<htm::RandomDistributedScalarEncoder>(params);
    } catch (const std::exception &e) {
        /* init_rdse retries ShapedRDSE construction up to 5 times; since the
         * collision check is deterministic (round-2 fix: the trial RNG is
         * seed-derived), all 5 attempts behave identically -- construction
         * either succeeds first try or fails terminally with this message. */
        NTA_THROW << "Failed RDSE random collision check 5 times\n"
                  << "  change rdse params --> " << e.what();
    }
    if (shape_.empty())
        shape_ = {rdse_->size};
}

void ShapedRdse::encode(double v, SDR &out) {
    /* Core checks out.size == size; NaN input yields a zero SDR -- the same
     * behavior the Python binding gives Feature.encode. */
    rdse_->encode(v, out);
}

/* ========================== ShapedLinear ================================ */
ShapedLinear::ShapedLinear(std::int64_t size, std::int64_t activeBits,
                           double minimum, double maximum,
                           std::optional<double> resolution,
                           std::vector<UInt> shape)
    : size_(size), w_(activeBits), min_(minimum), max_(maximum),
      shape_(std::move(shape)) {
    NTA_CHECK(w_ > 0 && size_ > 3 * w_)
        << "linear encoder needs size > 3*activeBits (size=" << size_
        << ", activeBits=" << w_ << "): the range keeps size - 2*activeBits "
        << "bits and each reserved extreme code takes activeBits.";
    if (!(max_ > min_)) {   /* constant / degenerate calibration range */
        const double pad =
            (min_ == 0.0) ? 1.0 : std::abs(min_) * 1e-6 + 1e-12;
        min_ -= pad;
        max_ += pad;
    }
    const std::int64_t room = size_ - 2 * w_;
    htm::ScalarEncoderParameters p;
    p.minimum = min_;
    p.maximum = max_;
    p.activeBits = static_cast<UInt>(w_);
    p.clipInput = true;     /* in-range only; extremes are handled here */
    if (resolution.has_value() && *resolution > 0.0) {
        /* Honor the researcher's granularity: buckets of exactly
         * `resolution`, occupying only the leading bits they need. */
        const double nBuckets = std::ceil((max_ - min_) / *resolution);
        const std::int64_t need =
            static_cast<std::int64_t>(nBuckets) + w_;
        if (need <= room) {
            p.resolution = *resolution;   /* core derives its own size  */
        } else {
            std::printf(
                "[Encoders] WARN linear: resolution %.6g needs %lld bits "
                "but only %lld fit (size %lld); falling back to the "
                "coarsest-fit bucketing -- raise `n` (or the resolution) "
                "to honor it\n",
                *resolution, static_cast<long long>(need),
                static_cast<long long>(room), static_cast<long long>(size_));
            p.size = static_cast<UInt>(room);
        }
    } else {
        p.size = static_cast<UInt>(room);  /* finest buckets that fit */
    }
    core_ = std::make_unique<htm::ScalarEncoder>(p);
    NTA_CHECK(static_cast<std::int64_t>(core_->size) <= room);
    scratch_.initialize({core_->size});
    if (shape_.empty()) shape_ = {static_cast<UInt>(size_)};
    bits_.reserve(static_cast<std::size_t>(w_));
}

void ShapedLinear::encode(double v, SDR &out) {
    NTA_CHECK(out.size == static_cast<UInt>(size_))
        << "ShapedLinear::encode: out.size " << out.size
        << " != encoder size " << size_;
    bits_.clear();
    if (std::isnan(v)) {                       /* NaN -> zero SDR */
        out.setSparse(bits_);
        return;
    }
    if (v < min_) {                            /* reserved "under" code */
        for (std::int64_t i = 0; i < w_; ++i)
            bits_.push_back(static_cast<UInt>(size_ - 2 * w_ + i));
    } else if (v > max_) {                     /* reserved "over" code */
        for (std::int64_t i = 0; i < w_; ++i)
            bits_.push_back(static_cast<UInt>(size_ - w_ + i));
    } else {
        core_->encode(v, scratch_);
        const auto &sp = scratch_.getSparse();
        bits_.assign(sp.begin(), sp.end());    /* region starts at bit 0 */
    }
    out.setSparse(bits_);
}

/* ========================== AdaptiveRdse ================================ */
/* The inverse-density cumulative warp, verbatim from adaptive_rdse.py
 * (lines 109-136; the identical block also runs inside
 * build_adaptive_params_from_samples, lines 257-279). */
void adaptive_warp_knots(const std::vector<double> &q,
                         const std::vector<double> &v, double gamma,
                         double index_scale, std::vector<double> &g) {
    const std::size_t n = q.size();
    std::vector<double> seg(n - 1);

    /* seg = dv / dq, then where(isfinite & > 0, seg, 0) */
    bool any_pos = false;
    for (std::size_t i = 0; i + 1 < n; ++i) {
        const double s = (v[i + 1] - v[i]) / (q[i + 1] - q[i]);
        seg[i] = (std::isfinite(s) && s > 0.0) ? s : 0.0;
        if (seg[i] > 0.0) any_pos = true;
    }

    if (any_pos) {
        /* seg = exp(log(maximum(seg, 1e-12)) / gamma) -- applied to EVERY
         * element, including the zeroed ones (they become exp(log(1e-12)/g),
         * a small positive weight).  Python quirk, kept. */
        for (double &s : seg)
            s = std::exp(std::log(std::max(s, 1e-12)) / gamma);
    } else {
        std::fill(seg.begin(), seg.end(), 1.0);
    }

    /* g_knots[0] = 0; g_knots[1:] = cumsum(seg) */
    g.resize(n);
    g[0] = 0.0;
    double acc = 0.0;
    for (std::size_t i = 0; i + 1 < n; ++i) {
        acc += seg[i];
        g[i + 1] = acc;
    }

    double total = g[n - 1];
    if (!std::isfinite(total) || total <= 0.0) total = 1.0;
    for (double &x : g) x = (x / total) * index_scale;
}

AdaptiveRdse::AdaptiveRdse(std::unique_ptr<ShapedRdse> rdse,
                           AdaptiveParams params, std::vector<UInt> shape)
    : rdse_(std::move(rdse)), params_(std::move(params)),
      shape_(std::move(shape)) {
    if (shape_.empty())
        shape_ = {static_cast<UInt>(params_.size)};

    q_ = params_.quantiles;
    v_ = params_.quantile_values;
    NTA_CHECK(q_.size() == v_.size() && q_.size() >= 2)
        << "quantiles and quantile_values must be 1D arrays of equal length "
        << "with at least 2 quantile knots";
    for (std::size_t i = 0; i + 1 < q_.size(); ++i)
        NTA_CHECK(q_[i + 1] - q_[i] > 0.0) << "quantiles must be strictly increasing";

    /* vmin/vmax = nanmin/nanmax of the knot values. */
    vmin_ = std::numeric_limits<double>::infinity();
    vmax_ = -std::numeric_limits<double>::infinity();
    for (double x : v_) {
        if (std::isnan(x)) continue;
        vmin_ = std::min(vmin_, x);
        vmax_ = std::max(vmax_, x);
    }
    NTA_CHECK(std::isfinite(vmin_) && std::isfinite(vmax_))
        << "quantile_values contain non-finite values";
    if (vmax_ == vmin_) vmax_ = vmin_ + 1e-9;

    NTA_CHECK(std::isfinite(params_.index_scale) && params_.index_scale > 0.0)
        << "index_scale must be a positive finite float";
    NTA_CHECK(std::isfinite(params_.gamma) && params_.gamma > 0.0)
        << "gamma must be a positive finite float";

    adaptive_warp_knots(q_, v_, params_.gamma, params_.index_scale, gKnots_);

    /* slopes = where(isfinite & > 0, diff(g) / maximum(diff(v), 1e-12), 0) */
    gSlopes_.resize(q_.size() - 1);
    for (std::size_t i = 0; i + 1 < q_.size(); ++i) {
        const double s = (gKnots_[i + 1] - gKnots_[i]) /
                         std::max(v_[i + 1] - v_[i], 1e-12);
        gSlopes_[i] = (std::isfinite(s) && s > 0.0) ? s : 0.0;
    }
}

double AdaptiveRdse::warp_to_index(double x) const {
    if (params_.clip_input)
        x = std::min(std::max(x, vmin_), vmax_);   // np.clip

    /* idx = searchsorted(v, x, side='right') - 1, clamped to [0, n-2]. */
    std::int64_t idx = static_cast<std::int64_t>(
                           searchsorted_right(v_.data(), v_.size(), x)) - 1;
    idx = std::max<std::int64_t>(0,
          std::min<std::int64_t>(idx, static_cast<std::int64_t>(v_.size()) - 2));

    const double x0 = v_[static_cast<std::size_t>(idx)];
    const double x1 = v_[static_cast<std::size_t>(idx) + 1];
    const double g0 = gKnots_[static_cast<std::size_t>(idx)];

    if (x1 <= x0) return g0;   // degenerate knot region

    const double dx = x - x0;
    double g = g0 + dx * gSlopes_[static_cast<std::size_t>(idx)];
    return std::min(std::max(g, 0.0), params_.index_scale);   // np.clip
}

void AdaptiveRdse::encode(double v, SDR &out) {
    double xf = v;
    if (!std::isfinite(xf)) {
        if (params_.reserve_for_special_values) {
            /* Python quirk: `SDR(size).sparse.fill(k)` fills a ZERO-LENGTH
             * array -- a no-op -- so the "reserved" NaN / underflow /
             * overflow representations are all-zero SDRs.  Kept faithfully
             * (verified by the encoder equivalence tests). */
            out.zero();
            return;
        }
        xf = vmin_;
    } else if (xf < vmin_) {
        if (params_.reserve_for_special_values) { out.zero(); return; }
    } else if (xf > vmax_) {
        if (params_.reserve_for_special_values) { out.zero(); return; }
    }

    const double g = warp_to_index(xf);
    rdse_->encode(g, out);
}

/* ========================== HybridRdse ================================== */
HybridRdse::HybridRdse(std::unique_ptr<AdaptiveRdse> adaptive,
                       std::unique_ptr<ShapedRdse> rdse,
                       std::int64_t total_size, std::int64_t size_adaptive,
                       std::vector<UInt> shape)
    : adaptive_(std::move(adaptive)), rdse_(std::move(rdse)),
      totalSize_(total_size), sizeAdaptive_(size_adaptive),
      shape_(std::move(shape)),
      scratchA_({static_cast<UInt>(size_adaptive)}),
      scratchB_({static_cast<UInt>(total_size - size_adaptive)}) {
    if (shape_.empty())
        shape_ = {static_cast<UInt>(total_size)};
    NTA_CHECK(total_size > 1) << "HybridRDSE requires size > 1";
    NTA_CHECK(sizeAdaptive_ > 0 && totalSize_ - sizeAdaptive_ > 0)
        << "Hybrid split produced invalid sizes";
    NTA_CHECK(static_cast<std::int64_t>(adaptive_->size()) == sizeAdaptive_)
        << "Adaptive encoder size mismatch";
    NTA_CHECK(static_cast<std::int64_t>(rdse_->size()) == totalSize_ - sizeAdaptive_)
        << "RDSE encoder size mismatch";
}

void HybridRdse::encode(double v, SDR &out) {
    adaptive_->encode(v, scratchA_);
    rdse_->encode(v, scratchB_);

    /* out.sparse = concat(a.sparse, b.sparse + size_adaptive): a's indices
     * are < size_adaptive and b's shifted ones are >= it, so the joined list
     * is sorted by construction (as in Python, where the binding sorts). */
    joined_.clear();
    for (UInt i : scratchA_.getSparse()) joined_.push_back(i);
    for (UInt i : scratchB_.getSparse())
        joined_.push_back(i + static_cast<UInt>(sizeAdaptive_));
    out.setSparse(joined_);   // swap
}

/* ===================== DualScalarEncoder ================================ */
OverlappingBucket::OverlappingBucket(double mn, double mx, std::int64_t n,
                                     std::int64_t w_, bool clip)
    : minval(mn), maxval(mx), n_bits(n), w(w_), clip_input(clip),
      n_starts(n - w_ + 1) {
    NTA_CHECK(std::isfinite(minval) && std::isfinite(maxval) && maxval > minval)
        << "minval/maxval must be finite and maxval > minval";
    NTA_CHECK(n_bits > 0) << "n_bits must be > 0";
    NTA_CHECK(w > 0) << "w must be > 0";
    NTA_CHECK(w <= n_bits) << "w must be <= n_bits";
}

std::int64_t OverlappingBucket::encode_start(double x) const {
    double xf = x;
    if (clip_input)
        xf = std::min(std::max(xf, minval), maxval);   // np.clip

    double t = (xf - minval) / (maxval - minval);
    t = std::min(std::max(t, 0.0), 1.0);               // np.clip

    std::int64_t start = 0;
    if (n_starts > 1)
        start = static_cast<std::int64_t>(
            std::floor(t * static_cast<double>(n_starts - 1)));
    return std::max<std::int64_t>(0, std::min(start, n_starts - 1));
}

DualScalarEncoder::DualScalarEncoder(const DualScalarConfig &cfg,
                                     std::vector<UInt> shape)
    : cfg_(cfg), shape_(std::move(shape)),
      slowScalar_(0.0, 1.0, cfg.slow_n_bits, cfg.slow_w, true),
      fastScalar_(-cfg.fast_z_clip, cfg.fast_z_clip, cfg.fast_n_bits,
                  cfg.fast_w, true) {
    NTA_CHECK(cfg_.fast_warmup >= 0) << "fast_warmup must be >= 0";
    NTA_CHECK(cfg_.n_quantiles >= 2) << "n_quantiles must be >= 2";
    NTA_CHECK(cfg_.fast_z_clip > 0.0 && std::isfinite(cfg_.fast_z_clip))
        << "fast_z_clip must be a positive finite float";

    if (shape_.empty()) shape_ = {size()};

    if (cfg_.reserve_for_special_values) {
        /* w_special = min(3, size); under=[0..w), over=[w..2w)%size,
         * nan=[2w..3w)%size -- verbatim from dual_scalar_encoder.py. */
        const std::int64_t total = cfg_.slow_n_bits + cfg_.fast_n_bits;
        const std::int64_t ws = std::min<std::int64_t>(3, total);
        for (std::int64_t k = 0; k < ws; ++k) {
            special_[0].push_back(static_cast<UInt>(k));
            special_[1].push_back(static_cast<UInt>((ws + k) % total));
            special_[2].push_back(static_cast<UInt>((2 * ws + k) % total));
        }
        for (auto &s : special_) std::sort(s.begin(), s.end());
    }
}

void DualScalarEncoder::fit(const std::vector<double> &training_values) {
    std::vector<double> arr;
    arr.reserve(training_values.size());
    for (double x : training_values)
        if (std::isfinite(x)) arr.push_back(x);

    if (arr.empty()) {
        v_ = {0.0, 1.0};
        q_ = {0.0, 1.0};
    } else {
        q_ = linspace(0.0, 1.0, static_cast<std::size_t>(cfg_.n_quantiles));
        v_ = quantile_median_unbiased(arr, q_);
        /* vs = vs + linspace(0, 1e-9, vs.size): breaks flat/duplicate knots */
        const auto eps = linspace(0.0, 1e-9, v_.size());
        for (std::size_t i = 0; i < v_.size(); ++i) v_[i] = v_[i] + eps[i];
    }

    vmin_ = *std::min_element(v_.begin(), v_.end());
    vmax_ = *std::max_element(v_.begin(), v_.end());
    if (!std::isfinite(vmin_) || !std::isfinite(vmax_) || vmax_ <= vmin_) {
        vmin_ = 0.0;
        vmax_ = 1.0;
    }
    fitted_ = true;
}

double DualScalarEncoder::cdf_quantile(double x) const {
    NTA_CHECK(fitted_) << "DualScalarEncoder.fit(training_values) must be "
                          "called before encode().";
    double xf = x;
    if (cfg_.clip_input)
        xf = std::min(std::max(xf, vmin_), vmax_);

    std::int64_t idx = static_cast<std::int64_t>(
                           searchsorted_right(v_.data(), v_.size(), xf)) - 1;
    idx = std::max<std::int64_t>(0,
          std::min<std::int64_t>(idx, static_cast<std::int64_t>(v_.size()) - 2));

    const double x0 = v_[static_cast<std::size_t>(idx)];
    const double x1 = v_[static_cast<std::size_t>(idx) + 1];
    const double q0 = q_[static_cast<std::size_t>(idx)];
    const double q1 = q_[static_cast<std::size_t>(idx) + 1];

    if (x1 <= x0)
        return std::min(std::max(0.5 * (q0 + q1), 0.0), 1.0);

    const double t = (xf - x0) / (x1 - x0);
    const double q = q0 + t * (q1 - q0);
    return std::min(std::max(q, 0.0), 1.0);
}

void DualScalarEncoder::encode(double v, SDR &out) {
    NTA_CHECK(fitted_) << "DualScalarEncoder.fit(training_values) must be "
                          "called before encode().";
    double xf = v;
    if (!std::isfinite(xf)) {
        if (cfg_.reserve_for_special_values) {
            bits_ = special_[2];   // NaN pattern (EMA is NOT advanced)
            out.setSparse(bits_);
            return;
        }
        xf = vmin_;
    }
    if (cfg_.reserve_for_special_values) {
        if (xf < vmin_) { bits_ = special_[0]; out.setSparse(bits_); return; }
        if (xf > vmax_) { bits_ = special_[1]; out.setSparse(bits_); return; }
    }

    /* --- Slow path: global rarity --- */
    const double q = cdf_quantile(xf);
    const std::int64_t slow_start = slowScalar_.encode_start(q);

    /* --- Fast path: local surprise (EMA state advances here) --- */
    double mean, var;
    if (!emaInit_) {
        emaMean_ = xf;
        emaVar_ = 1.0;
        emaInit_ = true;
        mean = emaMean_;
        var = emaVar_;
    } else {
        double a = cfg_.fast_alpha;
        a = std::min(std::max(a, 1e-6), 1.0);              // np.clip
        const double mean_old = emaMean_;
        const double dx = xf - mean_old;
        const double mean_new = mean_old + a * dx;
        const double var_old = emaVar_;
        double var_new = (1.0 - a) * (var_old + a * dx * (xf - mean_new));
        var_new = std::max(var_new, 1e-8);                  // eps
        emaMean_ = mean_new;
        emaVar_ = var_new;
        mean = mean_new;
        var = var_new;
    }
    fastSteps_ += 1;

    const double std_ = std::sqrt(std::max(var, 1e-8));
    double z = std_ > 0.0 ? (xf - mean) / std_ : 0.0;
    z = std::min(std::max(z, -cfg_.fast_z_clip), cfg_.fast_z_clip);

    /* Warmup fade-in: force z = 0 while stats are immature. */
    if (cfg_.fast_warmup && fastSteps_ <= cfg_.fast_warmup)
        z = 0.0;

    const std::int64_t fast_start = fastScalar_.encode_start(z);

    /* sdr.sparse = unique(concat(slow, fast + slow_n_bits)) -- disjoint
     * ascending ranges, so plain ordered emission is already sorted-unique. */
    bits_.clear();
    for (std::int64_t k = 0; k < cfg_.slow_w; ++k)
        bits_.push_back(static_cast<UInt>(slow_start + k));
    for (std::int64_t k = 0; k < cfg_.fast_w; ++k)
        bits_.push_back(static_cast<UInt>(cfg_.slow_n_bits + fast_start + k));
    out.setSparse(bits_);
}

/* ========================== DateFeature ================================= */
static std::unique_ptr<htm::ScalarEncoder> make_scalar(
        double mn, double mx, bool periodic, bool category,
        std::int64_t activeBits, double radius, bool has_radius) {
    htm::ScalarEncoderParameters p;
    p.minimum = mn;
    p.maximum = mx;
    p.periodic = periodic;
    p.category = category;
    p.activeBits = static_cast<UInt>(activeBits);
    if (has_radius) p.radius = radius;
    return std::make_unique<htm::ScalarEncoder>(p);
}

DateFeature::DateFeature(const DateConfig &cfg) : cfg_(cfg) {
    /* Sub-encoder construction order and parameters are verbatim from
     * py/htm/encoders/date.py __init__. */
    if (cfg_.season.width != 0) {
        season_ = make_scalar(0, 366, true, false, cfg_.season.width,
                              cfg_.season.has_radius ? cfg_.season.radius : 91.5,
                              true);
        size_ += season_->size;
    }
    if (cfg_.dayOfWeek.width != 0) {
        dayOfWeek_ = make_scalar(0, 7, true, false, cfg_.dayOfWeek.width,
                                 cfg_.dayOfWeek.has_radius ? cfg_.dayOfWeek.radius : 1.0,
                                 true);
        size_ += dayOfWeek_->size;
    }
    if (cfg_.weekend != 0) {
        weekend_ = make_scalar(0, 1, false, true, cfg_.weekend, 0, false);
        size_ += weekend_->size;
    }
    if (cfg_.customDaysWidth != 0) {
        customDays_ = make_scalar(0, 1, false, true, cfg_.customDaysWidth, 0, false);
        size_ += customDays_->size;
    }
    if (cfg_.holiday != 0) {
        holiday_ = make_scalar(0, 2, true, false, cfg_.holiday, 1.0, true);
        size_ += holiday_->size;
        for (const auto &h : cfg_.holidays)
            NTA_CHECK(h.size() == 2 || h.size() == 3)
                << "Holidays must be an iterable of length 2 or 3";
    }
    if (cfg_.timeOfDay.width != 0) {
        timeOfDay_ = make_scalar(0, 24, true, false, cfg_.timeOfDay.width,
                                 cfg_.timeOfDay.has_radius ? cfg_.timeOfDay.radius : 4.0,
                                 true);
        size_ += timeOfDay_->size;
    }
    NTA_CHECK(size_ > 0) << "DateEncoder: no attributes enabled";
    dims_ = {static_cast<UInt>(size_)};
}

void DateFeature::append_encoded(htm::ScalarEncoder &enc, double value,
                                 std::int64_t offset) {
    if (static_cast<std::int64_t>(scratch_.size) != static_cast<std::int64_t>(enc.size))
        scratch_.initialize({enc.size});
    enc.encode(value, scratch_);
    for (UInt i : scratch_.getSparse())
        joined_.push_back(i + static_cast<UInt>(offset));
}

void DateFeature::encode(const DateParts &v, SDR &out) {
    if (!v.valid) {          // None / NaN input -> zero SDR
        out.zero();
        return;
    }

    joined_.clear();
    std::int64_t offset = 0;
    const int wday = weekday_monday0(v);
    const int yday = yday_1based(v);
    const double timeOfDay =
        static_cast<double>(v.hour) + static_cast<double>(v.minute) / 60.0;

    if (season_) {
        const double dayOfYear = static_cast<double>(yday - 1);
        append_encoded(*season_, dayOfYear, offset);
        offset += season_->size;
    }
    if (dayOfWeek_) {
        const double hrs = timeOfDay / 24.0;
        double dow = static_cast<double>(wday) + hrs;
        dow -= 0.5;                       // round towards noon
        if (dow < 0) dow += 7;            // Mon-before-noon underflow -> Sun
        append_encoded(*dayOfWeek_, dow, offset);
        offset += dayOfWeek_->size;
    }
    if (weekend_) {
        const double we = (wday == 6 || wday == 5 ||
                           (wday == 4 && timeOfDay > 18)) ? 1.0 : 0.0;
        append_encoded(*weekend_, we, offset);
        offset += weekend_->size;
    }
    if (customDays_) {
        double cd = 0.0;
        for (int d : cfg_.customDays)
            if (d == wday) { cd = 1.0; break; }
        append_encoded(*customDays_, cd, offset);
        offset += customDays_->size;
    }
    if (holiday_) {
        /* Ramp 0->1 the day before, 1 on the day, 1->0 the day after --
         * ported with Python timedelta semantics (days/seconds of the
         * normalized difference), including the loop's break structure. */
        double val = 0.0;
        const std::int64_t now = naive_total_seconds(v);
        for (const auto &h : cfg_.holidays) {
            DateParts hd;
            hd.valid = true;
            if (h.size() == 3) { hd.year = h[0]; hd.month = h[1]; hd.day = h[2]; }
            else               { hd.year = v.year; hd.month = h[0]; hd.day = h[1]; }
            const std::int64_t hsec = naive_total_seconds(hd);

            if (now > hsec) {
                const std::int64_t diff = now - hsec;
                const std::int64_t days = diff / 86400;
                const std::int64_t secs = diff % 86400;
                if (days == 0) { val = 1.0; break; }
                if (days == 1) {
                    val = 1.0 + static_cast<double>(secs) / 86400.0;
                    break;
                }
            } else {
                const std::int64_t diff = hsec - now;
                const std::int64_t days = diff / 86400;
                const std::int64_t secs = diff % 86400;
                if (days == 0)   /* no break in Python here */
                    val = 1.0 - static_cast<double>(secs) / 86400.0;
            }
        }
        append_encoded(*holiday_, val, offset);
        offset += holiday_->size;
    }
    if (timeOfDay_) {
        append_encoded(*timeOfDay_, timeOfDay, offset);
        offset += timeOfDay_->size;
    }

    /* Sub-SDRs are appended in ascending offset order and each is sorted,
     * so the joined list is sorted (matches SDR.concatenate's result). */
    out.setSparse(joined_);
}

/* ========================= FeatureEncoder ================================ */
void FeatureEncoder::encode(double v, SDR &out) {
    switch (kind) {
        case FeatureKind::Rdse:
        case FeatureKind::Categorical: rdse->encode(v, out); return;
        case FeatureKind::AdaptiveRdse: adaptive->encode(v, out); return;
        case FeatureKind::HybridRdse: hybrid->encode(v, out); return;
        case FeatureKind::DualScalar: dual->encode(v, out); return;
        case FeatureKind::Linear: linear->encode(v, out); return;
        case FeatureKind::Date:
            NTA_THROW << "feature `" << name
                      << "` is a Datetime feature; numeric encode called";
    }
}

void FeatureEncoder::encode(const DateParts &v, SDR &out) {
    NTA_CHECK(kind == FeatureKind::Date)
        << "feature `" << name << "` is not a Datetime feature";
    date->encode(v, out);
}

UInt FeatureEncoder::size() const {
    switch (kind) {
        case FeatureKind::Rdse:
        case FeatureKind::Categorical: return rdse->size();
        case FeatureKind::AdaptiveRdse: return adaptive->size();
        case FeatureKind::HybridRdse: return hybrid->size();
        case FeatureKind::DualScalar: return dual->size();
        case FeatureKind::Linear: return linear->size();
        case FeatureKind::Date: return date->size();
    }
    return 0;
}

const std::vector<UInt> &FeatureEncoder::dimensions() const {
    switch (kind) {
        case FeatureKind::Rdse:
        case FeatureKind::Categorical: return rdse->dimensions();
        case FeatureKind::AdaptiveRdse: return adaptive->dimensions();
        case FeatureKind::HybridRdse: return hybrid->dimensions();
        case FeatureKind::DualScalar: return dual->dimensions();
        case FeatureKind::Linear: return linear->dimensions();
        case FeatureKind::Date: return date->dimensions();
    }
    NTA_THROW << "unreachable";
}

} // namespace pyramid