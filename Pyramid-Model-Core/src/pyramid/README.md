# 🧠 src/pyramid — the C++ pyramid runtime

One folder per concern.

| Folder | What lives here |
|--------|-----------------|
| `rng/` | numpy-exact `SeedSequence` (wrapping-subtraction mix), PCG64 (portable 128-bit, XSL-RR), `choice(replace=False, p=…)` cdf-resampling |
| `math/` | numpy-exact `pairwise_sum`, `cumsum`, `searchsorted_right`, `linspace`, `quantile_median_unbiased` (numpy's *literal* virtual-index formula — do **not** simplify it algebraically), `percentile_linear` |
| `merge/` | `SdrMerger` — all 14 modes (`u i sd nos c s su aps pls ssu li bws pcb di`) + `sdr_max_pool_into`. Scratch buffers, touched-index clearing, zero steady-state allocation |
| `encoders/` | `FeatureEncoders` (rdse/adaptive/hybrid/dual/categorical/date), `EncoderBuild` (sample-derived params: deduced resolution, adaptive quantile table, dual fit), `DateMath` |
| `spec/` | `PyramidSpec` structs — the parsed instruction book, incl. preformatted per-node `summary` strings from the Python builder |
| `data/` | `RecordSource` + `CsvSource` (native CSV: header map, dt strptime, row slicing) |
| `runtime/` | `HtmNode` (SP→TM step order, squeeze/restore, max-pool, residual modes) and `PyramidRuntime` (per-record loop: encode → feature merge → layers → extras chain → results) |
| `threading/` | `WorkerPool` — pinned ownership, weighted LPT with carried loads (verbatim `balanced_partition_weighted` arithmetic), mailbox + layer barrier, first-error rethrow |

⚠️ Ordering rules:
* `SDR::setSparse` (non-const overload) **swaps** — pass copies when the
  caller still needs the vector. `reshape` mutates in place.
* The extras chain order in `PyramidRuntime::run` mirrors the Python run
  loop **exactly**: children → residual → boost → memory/recurrent at the
  head.