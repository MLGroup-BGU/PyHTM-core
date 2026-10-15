# 🔺 Pyramid-Model-Core

The PyHTM pyramid runtime in C++: encoding, feature merges, SP/TM steps
across all layers, the run-loop extensions (residual / broadcast /
recurrent / anomaly-boost / long-term memory / lateral exchange /
squeeze-skip), data ingestion, and threading — executing under a single
GIL release on a pinned worker pool.

## 🤝 Division of responsibilities

| Side | Responsibility |
|------|----------------|
| **Python** (`htm_source.pipeline.pyramid_spec`) | Resolves every configuration-derived value (per-layer configs, timescale overrides, `potentialRadius` rounding, seeds, dims chaining, per-node summary strings) and hands the engine a fully-resolved spec dict. |
| **C++** (this module) | Computes every data-derived value (deduced RDSE resolutions, adaptive quantile tables, dual-scalar fits), builds all nodes in parallel, and streams the dataset through the pyramid. |

Result equivalence with the PyHTM Python implementation
(`htm_source.model.htm_pyramid_python`) is exact — anomaly and activity
arrays compare element-for-element across the flag × encoder matrix. See

## 📁 Layout

| Path | Contents |
|------|----------|
| `src/pyramid/rng/` | numpy-equivalent RNG stack: `SeedSequence`, PCG64, `Generator.choice(replace=False, p=…)` |
| `src/pyramid/math/` | numpy-equivalent math: `pairwise_sum`, `cumsum`, `searchsorted`, `linspace`, `quantile(median_unbiased)`, `percentile(linear)` |
| `src/pyramid/merge/` | All 14 SDR merge modes + `max_pool`, allocation-free steady state |
| `src/pyramid/encoders/` | RDSE / adaptive / hybrid / dual-scalar / categorical / date encoders + sample-derived parameter builders |
| `src/pyramid/spec/` | `PyramidSpec` — the parsed spec structs |
| `src/pyramid/data/` | `RecordSource` interface + native `CsvSource` |
| `src/pyramid/runtime/` | `HtmNode` (SP+TM step; the node output is its active columns, or the surprised columns under RTM; squeeze, max-pool) + `PyramidRuntime` (per-record loop and the extension chain) |
| `src/pyramid/threading/` | `WorkerPool`: pinned ownership, weighted-LPT partition, layer barrier |
| `bindings/` | pybind11 module `htm.bindings.pyramid_engine` (`PyramidEngine`, `probe_encoder_dims`) + `PyReaderSource` |
| `py/pyramid_support/` | `htm.pyramid.DatasetReader` — the batched Python feeder |

Details per folder: [`src/pyramid/`](src/pyramid/README.md) ·
[`bindings/`](bindings/README.md) ·
[`py/pyramid_support/`](py/pyramid_support/README.md).

## ⚙️ Build

Built by the standard two-stage flow (see the
[repository root README](../README.md)). Stage 2 (`BINDING_BUILD=Python3`)
compiles this module into the wheel as `htm.bindings.pyramid_engine` and
installs `htm/pyramid`.

Build switches (CMake cache variable or environment variable):

| Switch | Values | Effect |
|--------|--------|--------|
| `HTM_LTO` | `AUTO` (default) / `FULL` / `OFF` | `AUTO` applies LTO across this module's translation units only, leaving the library untouched. `FULL` includes the library, which shifts its floating-point results slightly. |
| `HTM_MARCH` | `auto` (default) / `off` / `x86-64` / `x86-64-v2` / `x86-64-v3` / `x86-64-v4` / `native` | CPU SIMD level for both cores. `auto` probes the build host (CPUID + XGETBV). Explicit `-march`//`arch:` in `CXXFLAGS`/`CL` takes precedence over this switch. See [`DetectMarch.cmake`](../DetectMarch.cmake). |

Dependencies: pybind11, CMake ≥ 3.21, a C++17 compiler, Threads —
the same stack as the rest of the repository. Parquet ingestion uses the
Python-side `DatasetReader` (pyarrow), so the C++ build carries no Arrow
dependency.

## 🚀 Direct use

Runners use the `htm_source.model` classes (thin delegates with the
original signatures) and do not touch this module directly. Direct API:

```python
from htm.bindings.pyramid_engine import PyramidEngine
from htm.pyramid import DatasetReader
from htm_source.pipeline import build_pyramid_spec

spec = build_pyramid_spec(..., data=dict(mode="py_reader",
                                         columns=cols, column_is_dt=flags))
eng = PyramidEngine(spec)
eng.build(lambda: DatasetReader(df, cols, flags))   # None → native CSV
eng.run()                     # optional: iterations=, progress_cb=
scores = eng.results()        # {node: {'anomaly': f64[], 'activity': i64[]}}
```

## 🧵 GIL & threading model

* `build()` / `run()` release the GIL for their entire duration.
* GIL re-acquisitions: `DatasetReader` batch pulls (one per `batch_rows`
  records) and the optional progress callback.
* Workers are pinned: each node is owned by exactly one worker for the
  whole run (weighted-LPT partition per layer, cumulative loads carried
  across layers — the arithmetic of
  `htm_source.utils.general.balanced_partition_weighted`), so worker
  count cannot affect results. A layer barrier orders inter-layer
  hand-offs.
* Thread→core pinning: on Linux and Windows, worker *i* of **both** pools
  is pinned 1:1 to the *i*-th process-allowed core (taskset/cgroup/SLURM
  masks respected) whenever the worker count fits the allowed set — so
  memory first-touched by the build pool stays local to the run pool's
  owner of the same nodes (NUMA). When workers exceed allowed cores, or
  on macOS (no strict affinity API), the OS scheduler is used.
* Cross-record pipeline: with more than one worker (and no
  `lateral_exchange`), scheduling is dependency-released instead of
  layer-barriered — a node runs record *t* the moment its producers
  finished *t* and its ring slot is free, so up to `K` records are in
  flight across layers. The spec key `pipeline_depth` selects `K`:
  `0` (the default) = auto, `1` disables the pipeline, `>1` forces an
  explicit depth. Auto resolves to
  `K = clamp(floor(n_workers × K_WORKER_FACTOR), K_MIN, K_MAX)`, today
  `factor 1.0, K_MIN 16, K_MAX 64`; an explicit value is clamped to the
  same bounds. The three constants live at the top of
  `src/pyramid/runtime/PipelineRun.cpp`, and the environment variable
  `PYRAMID_PIPE_DEPTH` overrides everything. Each node still processes its own records
  strictly in order with identical inputs, so outputs are bit-identical
  to the barrier path (proven by the cross-mode equality tests). Ring
  memory: `K ×` per-node output SDRs — kilobytes.
* Worker count resolution: `SLURM_CPUS_PER_TASK` → CPU affinity →
  hardware concurrency, capped by `max_workers` and the widest layer.
  `multiprocess=False` forces a single worker.

## 🧩 Extending the model spec

Adding a future model parameter touches exactly four small places, in
order. Omitting a parameter never breaks anything:

1. **YAML / runner** — the value enters the model configuration as usual.
2. **Builder** (`htm_source/pipeline/pyramid_spec.py`) — accept the kwarg
   and write it into the spec dict (top-level or per-node).
3. **Spec parse** (`bindings/py_PyramidRuntime.cpp`) — read it with the
   tolerant accessor pattern already used everywhere:
   `s.field = get_or<T>(d, "key", default);` — a **missing key falls back
   to the default**, and **unknown keys in the dict are ignored** by
   construction, so spec and engine versions can evolve independently.
4. **Runtime** — consume the new `PyramidSpec` field where it applies
   (`src/pyramid/spec/PyramidSpec.hpp` + the relevant runtime site).

`multiprocess` illustrates the pattern end to end:
`get_or<bool>(d, "multiprocess", true)`. `false` forces a single worker
thread, `true` sizes the pool from the environment. The PyHTM surface
also accepts `multithreading` as the preferred alias for the same switch.

## 🔗 What enters and leaves a node

- **Into an L0 node** (`feature_merge_mode: c`): the feature encodings are concatenated,
  one slice per feature, in the node's feature order (numpy.concatenate rule: inputs may
  differ in length along the concat axis). Nothing is sampled or overlaid, so equal values
  always give an identical node input. Every other merge mode overlays its inputs on one
  index space and therefore requires identical input dimensions.
- **Into a parent node** (`htm_merge_mode: c`): the children's outputs concatenated, one
  2048-bit slice per child in connection order (a single child is copied). As NuPIC's Network
  API and htm.core's "fanin" links do, each source keeps its own portion of the input buffer.
- **Out of every node**: its active columns (`columnDimensions` bits, `localAreaDensity`
  of them on), or under RTM (`anomaly_detector`) the surprised columns (active, not
  predicted). The TM's cells stay internal: they hold the node's temporal context and
  produce its anomaly score. Every node therefore sends the same number of bits whether
  it is surprised or not, and the output width does not depend on the TM size. htm.core's
  region API offers the same column-level output (`TMRegion`, `orColumnOutputs`).
- **Per-node signals (optional)**: with `record_signals` in the run flags (the model config's
  `general.record_signals`), every node
  also records per record its predicted columns, predicted-and-active columns, predictive cells and
  the mean SP overlap of its winning columns (10 bytes per node and record, handed to Python once
  at the end of the run, like the scores). Off unless the config asks for it.
- **SP settings that come from the config** include `dutyCyclePeriod` and
  `minPctOverlapDutyCycle` (defaults 1000 / 0.001 when absent). With
  `stimulusThreshold: 0` every node activates exactly `localAreaDensity` of its columns on
  every record; ties at the cut-off go to the higher column index, as in NuPIC and
  htm.core, which keeps the same input mapped to the same columns.

## 📌 Behavioral notes

| Item | Detail |
|------|--------|
| Fractional anomaly-boost residual | Sampled from a per-node PCG64 stream (seeded, deterministic). Boost copy counts use banker's rounding. |
| `use_predictive` + `lateral_exchange` / `broadcast_head` | Rejected at build time with a "not a runnable combination" error. |
| Timescale suffixes | A node name without a timescale suffix resolves to the `'short'` parameter set when one exists in the `timescales` block (`get_node_config` semantics). |
| Per-layer `max_pool` | Unpacked per layer only when passed as a **tuple** (`isinstance(..., tuple)` in the build loop). |
| Feature names | Must not contain `feature_join_str` (default `'_'`). |
| Datetime features | Require all five date-encoder keys (`season`, `dayOfWeek`, `weekend`, `holiday`, `timeOfDay`) plus a `format` parameter. |
| `std::exp/log/pow` | Scalar libm paths. On AVX-512 hosts a ≤ 1 ULP difference from numpy is possible in these calls. FMA contraction is disabled project-wide under march levels (`-ffp-contract=off`), keeping numeric output identical across every `HTM_MARCH` level. |
| Library SIMD | The `htm_core` library compiles at `x86-64-v3` max. On v4 hosts its segment-histogram hot loop (`Connections::computeActivity`) uses an AVX-512 (`vpconflictd` + gather/scatter) integer kernel compiled per-file — order-independent integer addition, bit-identical to the scalar loop, measured ×1.24 on end-to-end run for 2048-column models (single worker). |