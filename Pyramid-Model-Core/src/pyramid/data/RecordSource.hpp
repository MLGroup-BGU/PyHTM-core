/* ---------------------------------------------------------------------------
 * Pyramid-Model-Core :: data/RecordSource.hpp
 *
 * Record streaming for the pyramid runtime.
 *
 * A RecordSource yields the dataset's rows -- already sliced by the runner's
 * (start, stop, step) -- one at a time, as parallel arrays matching the
 * spec's column order: a double per numeric/categorical column and DateParts
 * per datetime column.  Sources are RESETTABLE because the build phase may
 * need a first pass over the training slice (sample-derived encoder
 * parameters) before the run pass starts from the beginning again.
 *
 * Implementations:
 *   CsvSource       -- native C++ streaming CSV reader (constant memory).
 *                      Standard quoting; numeric fields via strtod (empty ->
 *                      NaN); datetime fields via the strptime-subset parser.
 *   PyReaderSource  -- (bindings layer) drives the Python DatasetReader
 *                      helper in BATCHES: the GIL is taken once per batch of
 *                      `batch_rows` rows to copy plain buffers out, and
 *                      released for everything else.  This is what makes
 *                      Parquet -- and anything else pandas/pyarrow can read
 *                      -- available without adding a single C++ dependency
 *                      (Apache Arrow C++ was evaluated and rejected: it
 *                      would multiply the Windows build time and risk).
 *
 * THREADING: a RecordSource is pulled by the orchestrator thread only.
 * ------------------------------------------------------------------------ */
#pragma once

#include <cstdint>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "../encoders/DateMath.hpp"

namespace pyramid {

/* One row, laid out per the spec's column order.  For column j:
 * column_is_dt[j] ? dts[j] is meaningful : nums[j] is meaningful. */
struct Record {
    std::vector<double> nums;
    std::vector<DateParts> dts;
};

class RecordSource {
public:
    virtual ~RecordSource() = default;

    /* Fill `rec` with the next row; false at end of stream. */
    virtual bool next(Record &rec) = 0;

    /* Restart the stream from the first (sliced) row. */
    virtual void reset() = 0;

    /* Total sliced rows if cheaply known (Parquet metadata / DataFrame),
     * else -1 (CSV without a stop bound). */
    virtual std::int64_t total_rows() const { return -1; }
};

/* ------------------------------------------------------------------------ */
class CsvSource final : public RecordSource {
public:
    /* `columns` / `column_is_dt` / `dt_formats` follow the spec order;
     * slice = the runner's iloc(start, stop, step) over data rows. */
    CsvSource(std::string path, std::vector<std::string> columns,
              std::vector<int> column_is_dt,
              std::vector<std::string> dt_formats,
              std::optional<std::int64_t> start,
              std::optional<std::int64_t> stop,
              std::optional<std::int64_t> step);

    bool next(Record &rec) override;
    void reset() override;
    /* Count data rows once (cheap line scan), honoring the slice, so the
     * run loop can show an ETA. Cached after the first call; -1 only if the
     * file cannot be opened. */
    std::int64_t total_rows() const override;

private:
    mutable std::int64_t cachedTotal_ = -2;   // -2 = not yet counted
    std::string path_;
    std::vector<std::string> columns_;
    std::vector<int> isDt_;
    std::vector<std::string> dtFmt_;
    std::int64_t start_ = 0, stop_ = -1, step_ = 1;   // resolved slice

    std::ifstream file_;
    std::vector<std::int64_t> colIndex_;   // spec column -> csv column
    std::size_t nCsvCols_ = 0;
    std::int64_t rowIdx_ = 0;              // data-row counter (post header)
    std::string line_;
    std::vector<std::string> fields_;

    void open_and_parse_header();
    bool read_row();                        // parses into fields_
};

} // namespace pyramid
