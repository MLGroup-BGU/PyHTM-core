# 🐍 htm.pyramid — Python support package

Installed by the Python3 build stage as `htm/pyramid`.

## DatasetReader

The batched feeder consumed by the engine's `PyReaderSource`. One class,
three sources with identical semantics:

| Source | `total_rows` | Notes |
|--------|--------------|-------|
| `pd.DataFrame` | known | sliced up-front with `.iloc[start:stop:step]` |
| Parquet (`.parquet`/`.pq`) | known (metadata) | pyarrow `iter_batches`, global-row slicing per batch |
| CSV (anything else) | unknown (`None`) | pandas `chunksize` reader, global-row slicing per batch |

Batch protocol: `next_batch() -> {'n', 'num': {col: f64[n]}, 'dt': {col:
i32[n,7]}} | None`; dt components are `[y, mo, d, h, mi, s, valid]` —
NaT ⇒ `valid=0` (encodes through the NaN path). Datetime parsing happens
HERE with `pandas.to_datetime` — i.e. the exact parser the Python pyramid
used.

The engine holds a zero-arg **factory** and recreates the reader whenever
it needs a rewind (e.g. after the encoder sample pass), so readers are
single-pass by design.

GIL cost: one acquisition per `batch_rows` (default 8192) records for a
few buffer copies — well under 0.1% of a run. Parquet ingestion lives in
this reader (pyarrow), keeping the C++ build free of an Arrow dependency.
the engine's `RecordSource` interface also accepts native sources
(`CsvSource` today).