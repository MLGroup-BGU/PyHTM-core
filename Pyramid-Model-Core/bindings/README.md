# 🔗 bindings — htm.bindings.pyramid_engine

`py_PyramidRuntime.cpp` exposes:

| Symbol | Purpose |
|--------|---------|
| `PyramidEngine(spec: dict)` | Parse the instruction book (dict → `PyramidSpec`) |
| `.build(reader_factory=None)` | Encoders (+ sample pass on the training slice) and all SP/TM nodes, in parallel. `None` ⇒ native CSV from `spec['data']` |
| `.run(iterations=None, progress_cb=None, progress_every=5000)` | Stream records; callable repeatedly — continues the same stream up to `iterations` total |
| `.results()` / `.summaries()` | Per-node `{'anomaly': f64[], 'activity': i64[]}` / `model_summary()` strings, layer order |
| `.head_name / .node_names / .records_processed / .n_workers` | Introspection |
| `probe_encoder_dims(enc_spec)` | Encoder output dims without samples — the spec builder's single source of truth for date-encoder sizes |

`PyReaderSource` drives a `DatasetReader` factory: GIL only per batch pull
and reader recreation; rows are served from plain C++ buffers.

Implementation note: optional-dict lookups are written as
`cond ? py::object(d[key]) : py::object(py::none())`. A bare
`cond ? d[key] : py::none()` resolves the C++ ternary to `py::none` and
type-checks the accessor, raising "not an instance of 'none'" for real
dict values.
