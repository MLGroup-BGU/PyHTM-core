/* ---------------------------------------------------------------------------
 * Pyramid-Model-Core :: data/CsvSource.cpp
 *
 * Native streaming CSV reader.  Kept deliberately strict and simple:
 * comma-separated, one header row, standard double-quote quoting with ""
 * escapes, LF or CRLF line ends.  Numeric parsing matches pandas' default
 * float64 conversion for plain numeric CSVs (strtod; empty/'nan' -> NaN).
 * For CSVs beyond this shape (exotic quoting, encodings, mixed types) the
 * runner can set force_python_reader=True and the pandas-based reader takes
 * over with byte-identical pandas semantics.
 * ------------------------------------------------------------------------ */
#include "RecordSource.hpp"

#include <htm/utils/Log.hpp>

#include <algorithm>
#include <cstdlib>
#include <limits>

namespace pyramid {

CsvSource::CsvSource(std::string path, std::vector<std::string> columns,
                     std::vector<int> column_is_dt,
                     std::vector<std::string> dt_formats,
                     std::optional<std::int64_t> start,
                     std::optional<std::int64_t> stop,
                     std::optional<std::int64_t> step)
    : path_(std::move(path)), columns_(std::move(columns)),
      isDt_(std::move(column_is_dt)), dtFmt_(std::move(dt_formats)) {
    start_ = start.value_or(0);
    stop_ = stop.value_or(-1);            // -1 == unbounded
    step_ = step.value_or(1);
    NTA_CHECK(start_ >= 0 && step_ >= 1)
        << "CSV source supports non-negative start and positive step "
        << "(matching how the runners slice); got start=" << start_
        << " step=" << step_;
    open_and_parse_header();
}

/* Split one CSV line into fields_ honoring standard quoting. */
static void split_csv(const std::string &line, std::vector<std::string> &out) {
    out.clear();
    std::string cur;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (quoted) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    cur.push_back('"');   // escaped quote
                    ++i;
                } else {
                    quoted = false;
                }
            } else {
                cur.push_back(c);
            }
        } else if (c == '"') {
            quoted = true;
        } else if (c == ',') {
            out.push_back(cur);
            cur.clear();
        } else if (c == '\r') {
            /* swallow CR of CRLF */
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
}

void CsvSource::open_and_parse_header() {
    file_.close();
    file_.clear();
    file_.open(path_, std::ios::in | std::ios::binary);
    NTA_CHECK(file_.is_open()) << "CsvSource: cannot open `" << path_ << "`";

    NTA_CHECK(static_cast<bool>(std::getline(file_, line_)))
        << "CsvSource: `" << path_ << "` is empty (no header row)";
    split_csv(line_, fields_);
    nCsvCols_ = fields_.size();

    colIndex_.assign(columns_.size(), -1);
    for (std::size_t j = 0; j < columns_.size(); ++j) {
        for (std::size_t k = 0; k < fields_.size(); ++k) {
            if (fields_[k] == columns_[j]) {
                colIndex_[j] = static_cast<std::int64_t>(k);
                break;
            }
        }
        NTA_CHECK(colIndex_[j] >= 0)
            << "CsvSource: column `" << columns_[j] << "` not found in `"
            << path_ << "` header";
    }
    rowIdx_ = 0;
}

void CsvSource::reset() { open_and_parse_header(); }

bool CsvSource::read_row() {
    if (!std::getline(file_, line_)) return false;
    /* Tolerate a trailing empty line. */
    if (line_.empty() && file_.peek() == std::ifstream::traits_type::eof())
        return false;
    split_csv(line_, fields_);
    return true;
}

static double parse_num(const std::string &s) {
    if (s.empty()) return std::numeric_limits<double>::quiet_NaN();
    char *end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str())
        return std::numeric_limits<double>::quiet_NaN();   // e.g. 'nan', text
    return v;
}

bool CsvSource::next(Record &rec) {
    rec.nums.resize(columns_.size());
    rec.dts.resize(columns_.size());

    for (;;) {
        /* Enforce the iloc slice by data-row index. */
        if (stop_ >= 0 && rowIdx_ >= stop_) return false;
        if (!read_row()) return false;
        const std::int64_t idx = rowIdx_++;
        if (idx < start_) continue;
        if ((idx - start_) % step_ != 0) continue;

        NTA_CHECK(fields_.size() == nCsvCols_)
            << "CsvSource: row " << idx << " has " << fields_.size()
            << " fields, header has " << nCsvCols_;

        for (std::size_t j = 0; j < columns_.size(); ++j) {
            const std::string &f =
                fields_[static_cast<std::size_t>(colIndex_[j])];
            if (isDt_[j]) {
                try {
                    rec.dts[j] = strptime_subset(f, dtFmt_[j]);
                } catch (const std::exception &) {
                    rec.dts[j] = DateParts{};   // valid=false -> NaN path
                }
            } else {
                rec.nums[j] = parse_num(f);
            }
        }
        return true;
    }
}


std::int64_t CsvSource::total_rows() const {
    if (cachedTotal_ != -2) return cachedTotal_;
    std::ifstream f(path_, std::ios::binary);
    if (!f) { cachedTotal_ = -1; return -1; }
    // count newlines after the header; tolerate a missing trailing newline.
    std::int64_t dataLines = 0;
    std::string line;
    bool first = true;
    bool sawAny = false;
    while (std::getline(f, line)) {
        if (first) { first = false; continue; }   // header
        if (!line.empty() || !f.eof()) { ++dataLines; sawAny = true; }
    }
    (void)sawAny;
    // apply the resolved iloc(start, stop, step) slice
    std::int64_t start = start_ < 0 ? 0 : start_;
    std::int64_t stop = (stop_ < 0 || stop_ > dataLines) ? dataLines : stop_;
    std::int64_t step = step_ <= 0 ? 1 : step_;
    std::int64_t n = start >= stop ? 0 : (stop - start + step - 1) / step;
    cachedTotal_ = n;
    return n;
}

} // namespace pyramid
