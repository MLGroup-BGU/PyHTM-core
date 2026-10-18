/* ---------------------------------------------------------------------------
 * Pyramid-Model-Core :: encoders/FeatureEncoders.hpp
 *
 * The encoders a feature gets, chosen by its type:
 *
 *   ShapedRdse   <- encoding.ShapedRDSE + init_rdse    (numeric; categorical
 *                   in the core RDSE's category mode)
 *   DateFeature  <- py/htm/encoders/date.DateEncoder   (timestamp)
 *
 * plus `FeatureEncoder`, the small dispatcher a pyramid Feature holds.
 *
 * FAITHFULNESS
 *   Every numeric expression is ported in the same float64 operation order
 *   as the Python source, on top of the verified NumpyCompat / core-RDSE /
 *   core-ScalarEncoder primitives, so encodings are bit-identical for
 *   identical inputs.
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
#include <optional>
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
enum class FeatureKind : std::uint8_t { Rdse, Categorical, Date };

class FeatureEncoder {
public:
    FeatureKind kind;
    std::string name;

    std::unique_ptr<ShapedRdse> rdse;            // Rdse / Categorical
    std::unique_ptr<DateFeature> date;           // Date

    bool is_datetime() const { return kind == FeatureKind::Date; }

    void encode(double v, htm::SDR &out);              // numeric / categorical
    void encode(const DateParts &v, htm::SDR &out);    // datetime

    htm::UInt size() const;
    const std::vector<htm::UInt> &dimensions() const;
};

} // namespace pyramid