# 🧠 src/pyramid — the C++ pyramid runtime

One folder per concern

| Folder | What lives here | Verified by |
|--------|-----------------|-------------|
| `rng/` | numpy-exact `SeedSequence` (wrapping-subtraction mix), PCG64 (portable 128-bit, XSL-RR), `choice(replace=False, p=…)` cdf-resampling | `test_numpy_rng.py` |
| `math/` | numpy-exact `pairwise_sum`, `cumsum`, `searchsorted_right`, `linspace`, `quantile_median_unbiased` (numpy's *literal* virtual-index formula — do **not** simplify it algebraically), `percentile_linear` | `test_numpy_math.py` |
| `merge/` | `SdrMerger` — all 14 modes (`u i sd nos c s su aps pls ssu li bws pcb di`) + `sdr_max_pool_into`; scratch buffers, touched-index clearing, zero steady-state allocation | `test_merge_equiv.py` (1950 cases) |
| `encoders/` | `FeatureEncoders` (rdse/adaptive/hybrid/dual/categorical/date), `EncoderBuild` (sample-derived params: deduced resolution, adaptive quantile table, dual fit), `DateMath` | `test_encoder_equiv.py` (2743 cases) |
| `spec/` | `PyramidSpec` structs — the parsed instruction book, incl. preformatted per-node `summary` strings from the Python builder | `test_spec_builder.py` |
| `data/` | `RecordSource` + `CsvSource` (native CSV: header map, dt strptime, row slicing) | sources-equivalence test |
| `runtime/` | `HtmNode` (SP→TM step order, squeeze/restore, max-pool, residual modes) and `PyramidRuntime` (per-record loop: encode → feature merge → layers → extras chain → results) | full pyramid matrix |
| `threading/` | `WorkerPool` — pinned ownership, weighted LPT with carried loads (verbatim `balanced_partition_weighted` arithmetic), mailbox + layer barrier, first-error rethrow | `test_infra.py` |

⚠️ Ordering rules:
* `SDR::setSparse` (non-const overload) **swaps** — pass copies when the
  caller still needs the vector; `reshape` mutates in place.
* The extras chain order in `PyramidRuntime::run` mirrors the Python run
  loop **exactly**: children → residual → boost → memory/recurrent at the
  head — changing it breaks bit-exactness.
