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
        case FeatureKind::Date: return date->size();
    }
    return 0;
}

const std::vector<UInt> &FeatureEncoder::dimensions() const {
    switch (kind) {
        case FeatureKind::Rdse:
        case FeatureKind::Categorical: return rdse->dimensions();
        case FeatureKind::Date: return date->dimensions();
    }
    NTA_THROW << "unreachable";
}

} // namespace pyramid