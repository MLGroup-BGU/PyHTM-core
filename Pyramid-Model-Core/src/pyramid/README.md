# 🧠 src/pyramid — the C++ pyramid runtime

One folder per concern.

| Folder | What lives here |
|--------|-----------------|
| `math/` | numpy-exact `pairwise_sum`, `cumsum`, `searchsorted_right`, `linspace`, `quantile_median_unbiased` (numpy's *literal* virtual-index formula — do **not** simplify it algebraically), `percentile_linear` |
| `merge/` | `SdrConcat` — concatenation along axis 0, the pyramid's one merge: an L0 node's feature encodings, and a parent's children outputs, each input in its own slice. Scratch buffer, zero steady-state allocation |
| `encoders/` | `FeatureEncoders` (RDSE / categorical, the RDSE in category mode / date — chosen by the feature's type), `EncoderBuild` (the deduced RDSE resolution from the training slice), `DateMath` |
| `spec/` | `PyramidSpec` structs — the parsed instruction book, incl. preformatted per-node `summary` strings from the Python builder |
| `data/` | `RecordSource` + `CsvSource` (native CSV: header map, dt strptime, an optional `[start:stop]` row window) |
| `runtime/` | `HtmNode` (SP→TM step order; output = the active columns; the fixed SP / TM values) and `PyramidRuntime` (per-record loop: encode → concatenate per L0 node → layers → concatenate per parent → results) |
| `threading/` | `WorkerPool` — pinned ownership, weighted LPT with carried loads, mailbox + layer barrier, first-error rethrow |

### 🎲 Determinism

Nothing in the run loop draws random numbers: concatenation is exact, and every node's
Spatial Pooler, Temporal Memory and every encoder are seeded from `general.seed`
(`seed * counter`). Two runs of one configuration therefore match byte for byte under any
worker count and any thread interleaving. `general.seed: 0` is the exception: htm.core
treats a seed of 0 as "a new random seed", so such a run is not reproducible.

⚠️ Ordering rules:
* `SDR::setSparse` (non-const overload) **swaps** — pass copies when the
  caller still needs the vector. `reshape` mutates in place.
* `HtmNode::step` keeps the algorithm's order: dendrites predict from the
  previous step's cells, the anomaly compares those predictions with this
  step's columns, and only then are the cells activated.
