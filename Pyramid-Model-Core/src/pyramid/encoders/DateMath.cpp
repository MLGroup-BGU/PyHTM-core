/* ---------------------------------------------------------------------------
 * Pyramid-Model-Core :: encoders/DateMath.cpp
 * strptime-subset parser (see DateMath.hpp for the contract).
 * ------------------------------------------------------------------------ */
#include "DateMath.hpp"

#include <cctype>

namespace pyramid {

static int read_int(const std::string &s, std::size_t &i, int max_digits) {
    if (i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i])))
        throw std::invalid_argument("time data does not match format (digit expected)");
    int v = 0, n = 0;
    while (i < s.size() && n < max_digits &&
           std::isdigit(static_cast<unsigned char>(s[i]))) {
        v = v * 10 + (s[i] - '0');
        ++i; ++n;
    }
    return v;
}

DateParts strptime_subset(const std::string &value, const std::string &format) {
    DateParts p;
    p.valid = true;
    int hour12 = -1;      // %I value, resolved with %p at the end
    int pm = -1;          // 0 = AM, 1 = PM

    std::size_t vi = 0;
    for (std::size_t fi = 0; fi < format.size(); ++fi) {
        const char fc = format[fi];
        if (fc == '%' && fi + 1 < format.size()) {
            const char d = format[++fi];
            switch (d) {
                case 'd': p.day = read_int(value, vi, 2); break;
                case 'm': p.month = read_int(value, vi, 2); break;
                case 'Y': p.year = read_int(value, vi, 4); break;
                case 'y': {   // strptime: 00-68 -> 2000s, 69-99 -> 1900s
                    const int yy = read_int(value, vi, 2);
                    p.year = yy <= 68 ? 2000 + yy : 1900 + yy;
                    break;
                }
                case 'H': p.hour = read_int(value, vi, 2); break;
                case 'I': hour12 = read_int(value, vi, 2); break;
                case 'M': p.minute = read_int(value, vi, 2); break;
                case 'S': p.second = read_int(value, vi, 2); break;
                case 'p': {
                    if (vi + 1 >= value.size())
                        throw std::invalid_argument("time data does not match format (%p)");
                    const char a = static_cast<char>(std::toupper(static_cast<unsigned char>(value[vi])));
                    const char b = static_cast<char>(std::toupper(static_cast<unsigned char>(value[vi + 1])));
                    if (a == 'A' && b == 'M') pm = 0;
                    else if (a == 'P' && b == 'M') pm = 1;
                    else throw std::invalid_argument("time data does not match format (%p)");
                    vi += 2;
                    break;
                }
                case '%':
                    if (vi >= value.size() || value[vi] != '%')
                        throw std::invalid_argument("time data does not match format (%%)");
                    ++vi;
                    break;
                default:
                    throw std::invalid_argument(
                        std::string("unsupported strptime directive %") + d +
                        " in native CSV parser; use the Python dataset reader");
            }
        } else if (std::isspace(static_cast<unsigned char>(fc))) {
            /* strptime: whitespace in the format matches any (possibly empty
             * only if value also allows) run of whitespace in the input. */
            while (vi < value.size() &&
                   std::isspace(static_cast<unsigned char>(value[vi])))
                ++vi;
        } else {
            if (vi >= value.size() || value[vi] != fc)
                throw std::invalid_argument("time data does not match format (literal)");
            ++vi;
        }
    }
    /* strptime allows (and Python permits) unparsed trailing data? No --
     * datetime.strptime raises on leftovers. Match that. */
    if (vi != value.size())
        throw std::invalid_argument("unconverted data remains");

    if (hour12 >= 0) {
        /* %I with %p: 12AM -> 0, 12PM -> 12; otherwise +12 when PM.
         * Without %p, Python keeps the 1..12 value as the hour. */
        int h = hour12;
        if (pm == 1) h = (h % 12) + 12;
        else if (pm == 0) h = h % 12;
        p.hour = h;
    }
    return p;
}

} // namespace pyramid
