/* ---------------------------------------------------------------------------
 * Pyramid-Model-Core :: encoders/DateMath.hpp
 *
 * Proleptic-Gregorian calendar helpers for the Date feature path.
 *
 * Datetime values reach the runtime as broken-down parts (year, month, day,
 * hour, minute, second): either pre-parsed on the Python side by the dataset
 * reader (Parquet / pandas path -- parsing stays in Python so string-format
 * quirks are byte-identical to `datetime.strptime`), or parsed by the
 * strptime-subset parser here (native CSV path).  From the parts, weekday
 * and day-of-year are computed with Howard Hinnant's civil-date algorithms,
 * which match Python's `datetime.timetuple()` (tm_wday Monday=0, tm_yday
 * 1-based) for all Gregorian dates.
 * ------------------------------------------------------------------------ */
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace pyramid {

/* Broken-down datetime, mirroring the fields Python's timetuple exposes
 * (before derived fields).  `valid=false` represents NaN / missing. */
struct DateParts {
    std::int32_t year = 1970;
    std::int32_t month = 1;    // 1..12
    std::int32_t day = 1;      // 1..31
    std::int32_t hour = 0;
    std::int32_t minute = 0;
    std::int32_t second = 0;
    bool valid = false;
};

/* Days since 1970-01-01 for a civil date (Hinnant, "chrono-compatible"). */
inline std::int64_t days_from_civil(std::int64_t y, unsigned m, unsigned d) {
    y -= m <= 2;
    const std::int64_t era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);              // [0, 399]
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;    // [0, 365]
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;             // [0, 146096]
    return era * 146097 + static_cast<std::int64_t>(doe) - 719468;
}

/* Python tm_wday: Monday = 0 .. Sunday = 6. (1970-01-01 was a Thursday.) */
inline int weekday_monday0(const DateParts &p) {
    const std::int64_t z = days_from_civil(p.year, static_cast<unsigned>(p.month),
                                           static_cast<unsigned>(p.day));
    return static_cast<int>(z >= -3 ? (z + 3) % 7 : (z + 4) % 7 + 6);
}

inline bool is_leap(std::int32_t y) {
    return (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
}

/* Python tm_yday: 1-based day of year. */
inline int yday_1based(const DateParts &p) {
    static const int cum[12] = {0, 31, 59, 90, 120, 151, 181,
                                212, 243, 273, 304, 334};
    int yd = cum[p.month - 1] + p.day;
    if (p.month > 2 && is_leap(p.year)) ++yd;
    return yd;
}

/* Seconds since the epoch of midnight-based parts (proleptic Gregorian,
 * timezone-free -- exactly Python's naive-datetime arithmetic). */
inline std::int64_t naive_total_seconds(const DateParts &p) {
    return days_from_civil(p.year, static_cast<unsigned>(p.month),
                           static_cast<unsigned>(p.day)) * 86400ll +
           p.hour * 3600ll + p.minute * 60ll + p.second;
}

/* strptime-subset parser for the native CSV path.  Supports the directives
 * PyHTM configs actually use (%d %m %y %Y %H %I %M %S %p) plus literal
 * characters; whitespace in the format matches any run of whitespace, as
 * strptime does.  Throws std::invalid_argument on mismatch (mirroring the
 * ValueError `datetime.strptime` raises).  For formats beyond this subset,
 * use the Python dataset reader (force_python_reader). */
DateParts strptime_subset(const std::string &value, const std::string &format);

} // namespace pyramid
