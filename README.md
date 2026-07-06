# 🧠 PyHTM-core

> The C++ HTM engine behind **[PyHTM](https://github.com/MLGroup-BGU/htm-streamer.git)** (`htm-streamer`).
> A trimmed, lightly-modified fork of **[htm.core](https://github.com/htm-community/htm.core)**, adapted to power the PyHTM pyramid architecture.

---

## 📋 Table of Contents

- [What this is](#-what-this-is)
- [Relationship to htm.core](#-relationship-to-htmcore)
- [What changed vs. htm.core](#-what-changed-vs-htmcore)
- [What was removed](#-what-was-removed)
- [Project structure](#-project-structure)
- [Building](#-building)
- [License](#-license)

---

## 🎯 What this is

PyHTM-core is the C++ engine (Spatial Pooler, Temporal Memory, encoders,
SDR, NetworkAPI) that PyHTM compiles and calls into through Python bindings.
It is **not** a standalone project — it exists to be built and consumed by
PyHTM (`htm-streamer`), which clones it, builds it, and installs the
resulting wheel.

It began as a direct copy of **htm.core** and keeps htm.core's source layout
intact. The differences are deliberate and minimal: a handful of performance
and build changes needed for PyHTM's multithreaded pyramid of HTM machines,
plus the removal of everything not required to build or run the engine.

---

## 🔗 Relationship to htm.core

This repository is a **modified fork** of:

> **htm.core** — HTM Community Edition of NuPIC
> https://github.com/htm-community/htm.core
> Copyright © 2013–2024, Numenta, Inc. and the htm.core community.
> Licensed under the **GNU Affero General Public License v3 (AGPLv3)**.

Because PyHTM-core is a derivative work of AGPLv3 software, it is distributed
under the **same license — AGPLv3** (see [License](#-license)). All original
copyright notices and source-file headers from htm.core are preserved.

> ℹ️ **Attribution & scope.** This fork does not claim authorship of htm.core.
> It restates htm.core's code under the same AGPLv3 terms, documents every
> substantive change below, and keeps upstream structure so changes remain
> auditable. The original work belongs to Numenta and the htm.core
> contributors; see [`NOTICE`](NOTICE) and the per-file headers.

---

## 🔧 What changed vs. htm.core

High-level only — *what* was adapted, not line-by-line diffs. The intent is
that anyone familiar with htm.core can quickly see the nature of the changes.

- **GIL release in the Python bindings.** The heavy C++ compute calls in the
  Spatial Pooler and Temporal Memory bindings now release Python's GIL while
  the C++ work runs. This is what lets PyHTM drive many HTM machines on real
  threads instead of separate processes.

- **Connections: buffer pre-reservation.** Synapse/segment storage is sized
  up front instead of growing dynamically during model build, removing
  repeated reallocation+copy during the (very allocation-heavy) build phase.

- **Connections: CSR-style data structures.** On the hot path, several
  hash-map structures keyed by contiguous indices were replaced with directly
  index-addressable vectors, cutting lookup overhead and cache misses during
  the run.

- **Linux build mode fix.** The build now compiles in **Release** on Linux
  (assertions off), matching the Windows build. Previously the Linux build
  left internal debug assertions active, which dominated runtime.

- **Hot-path allocation removal (2026 optimization pass).** Verified
  bit-exact against the previous build (identical outputs, fixed seeds,
  including pickle round-trips):
  - `TemporalMemory::getPredictiveCells()` no longer builds a `std::set`
    (one heap allocation per inserted node, every step) — the active-segment
    list is already sorted by cell, so a linear adjacent-duplicate skip
    produces the same result with zero tree allocations.
  - `SpatialPooler::initialize()` reuses hoisted buffers instead of
    allocating two dense input-sized vectors per column (~72 KB of
    malloc/free churn per column removed — the allocator contention this
    causes across a multi-threaded pyramid build is exactly what mimalloc
    was brought in to fight; now there is far less of it to fight).
  - `SpatialPooler::updateDutyCycles_()` no longer constructs a temporary
    SDR every learning step.
  - Global inhibition keeps its index scratch between steps and drops a
    pointless `shrink_to_fit` (an extra allocation + copy per step).
  - Boost-factor updates early-out entirely when `boostStrength == 0`
    (PyHTM's default), skipping a no-op full-column loop every step.
  - `SpatialPooler::compute()` returns a movable overlaps vector; together
    with the binding change this removes two full vector copies per step.

- **Wider GIL release in the bindings (2026 optimization pass).**
  `getPredictiveCells`, `cellsToColumns`, `getWinnerCells`, and
  `RDSE.encode` now also run their C++ work with the GIL released (only
  `compute`/`activateDendrites`/`activateCells`/`getActiveCells` did
  before). Measured: ~15% less GIL-held time per forward() step — time that
  previously serialized every other worker thread in the hive.

- **New C++ primitives for PyHTM's per-step Python work.**
  - `SDR.subtract(minuend, subtrahend)` — sorted-sparse set difference,
    GIL-released. Replaces the Python-set / `np.setdiff1d` logic in PyHTM's
    residual (RTM) path: measured ×117 faster than the set approach,
    ×58 vs `setdiff1d`, identical results.
  - `htm.bindings.algorithms.computeRawAnomalyScore(active, predicted)` —
    the core's own anomaly formula exposed directly (GIL-released), so
    PyHTM's `calc_anomaly_score` can become a single C++ call instead of
    an SDR allocation + intersection + two length reads under the GIL.

> None of these change HTM logic or model outputs — they affect how the
> engine threads, allocates memory, and is compiled.

---

## 🗑️ What was removed

To keep this repository lean and focused on building the engine, the
following content from upstream htm.core was **not** carried over. None of it
is needed to compile or run the engine for PyHTM's use case:

- **Examples** — both the C++ examples (`src/examples/`) and the Python
  examples (`py/htm/examples/`, `py/htm/advanced/examples/`).
- **Tests / unit-tests** — `src/test/`, `py/tests/`, `bindings/py/tests/`,
  and the bindings `plugin/unittest/`.
- **Docs** — the upstream `docs/` tree (Doxygen, NetworkAPI docs, images).
- **CI & automation** — `ci/`, `.github/` workflows, and `githooks/`.
- **Docker** — `Dockerfile`, `Dockerfile-pypi`, `.dockerignore`.
- Assorted upstream top-level docs (changelogs, contributing, developer
  notes) and the C++ formatting config.

**2026 dead-code elimination pass.** PyHTM drives the engine directly
through three binding modules (`algorithms`, `sdr`, `encoders`); everything
that only served upstream's Network/Region API never executed in a PyHTM
run and was removed, together with the third-party dependencies that only
it pulled in:

- **`src/htm/engine/`** — the whole NetworkAPI (Network, Region, Link,
  Spec, RESTapi, Watcher, …).
- **`src/htm/regions/`** — all region wrappers (SPRegion, TMRegion,
  encoder regions, DatabaseRegion, file I/O regions, TestNode).
- **`src/htm/ntypes/`** — Array/Value/Dimensions plumbing used only by the
  engine (and with it the **libyaml** dependency).
- **`SDRClassifier`** and **`SimHashDocumentEncoder`** (+ its **digestpp**
  dependency) — not used by PyHTM.
- **Binding modules `engine_internal` and `math`**, plus the PyBindRegion
  plugin. The shared pybind helpers moved to `bindings/py_utils.hpp`; the
  `NTA_LOG_LEVEL` definition (oddly housed in `engine/Network.cpp`
  upstream) moved to its natural home, `src/htm/utils/Log.cpp`.
- **`py/htm/advanced/`**, **`py/htm/optimization/`**, bindings
  `tools/`+`regions/`, and the unused encoders (coordinate, grid-cell,
  SimHash, the legacy `date_encoder.py`) and legacy Python
  anomaly modules.
- **External fetch scripts** for **eigen, mnist, gtest, sqlite3,
  cpp-httplib, digestpp, libyaml** — none of the remaining code references
  them (eigen/mnist appeared only in comments). Only **cereal**
  (serialization) and the bundled **MurmurHash3** remain; notably, the
  eigen fetch was the build's only gitlab.com dependency — every remaining
  download is GitHub/PyPI.
- Python package dependencies shrank from `numpy, hexy, prettytable,
  wcwidth` to **`numpy` only** (the others served removed code).

Net effect: **225 → ~60 source files**, three binding modules instead of
five, a ~1.25 MB wheel, and materially faster clean builds — with the
`htm_install.py` flow and the import surface PyHTM uses left completely
untouched.

Everything **kept** retains its original htm.core path and structure, so
tracking upstream changes stays straightforward.

---

## 📁 Project structure

```
PyHTM-core/
├── 📂 src/                  # C++ engine — the core of everything
│   ├── htm/                # algorithms (SP, TM, Connections, anomaly),
│   │                       #   encoders (RDSE, Scalar, Date), types (SDR),
│   │                       #   utils (Random, Topology, …), minimal os
│   └── README.md           # → what lives in the C++ source tree
├── 📂 bindings/             # pybind11 layer exposing C++ to Python
│   └── py/
│       ├── cpp_src/         # algorithms / sdr / encoders modules
│       │                    #   (incl. the GIL-release edits)
│       └── README.md        # → how the Python bindings are organized
├── 📂 py/                   # the importable `htm` Python package
├── 📂 external/             # cereal fetch script + bundled MurmurHash3
├── CMakeLists.txt           # top-level build entry
├── CommonCompilerConfig.cmake
├── htm_install.py           # orchestrates the C++ build + wheel
├── pyproject.toml           # build/package definition (scikit-build-core)
├── get_version.py / VERSION
├── LICENSE                  # AGPLv3 (full text)
└── NOTICE                   # fork attribution + summary of changes
```

Every directory carries its own README — start from
**[`src/htm/`](src/htm/README.md)** (engine map),
**[`bindings/py/cpp_src/`](bindings/py/cpp_src/README.md)** (modules + GIL strategy),
**[`py/htm/`](py/htm/README.md)** (import surface),
**[`external/`](external/README.md)** (dependencies) and
**[`verification/`](verification/README.md)** (the bit-exactness harness).

---

## 🛠️ Building

You normally **don't build this directly**. PyHTM (`htm-streamer`) builds it
for you: its setup runner clones this repository as a sibling of PyHTM and
compiles it via `htm_install.py`, auto-selecting the best SIMD level for your
CPU. See the PyHTM repository's setup instructions.

For reference, the engine is built through the standard htm.core flow
(`htm_install.py` → CMake configure/build → wheel), which produces a static
`htm_core` library and the Python extension modules.

---

## 📜 License

PyHTM-core is licensed under the **GNU Affero General Public License v3
(AGPLv3)** — the same license as upstream htm.core, as required for a
derivative work. See the [`LICENSE`](LICENSE.txt) file for the full text and
[`NOTICE`](NOTICE) for attribution and the summary of modifications.

In short: you are free to use, study, modify, and redistribute this software,
provided that derivative works remain under AGPLv3 and that network-served
modified versions make their source available (AGPL §13).
