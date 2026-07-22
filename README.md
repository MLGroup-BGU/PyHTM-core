# 🧠 PyHTM-core

> The C++ HTM engine behind **[PyHTM](https://github.com/MLGroup-BGU/htm-streamer.git)** (`htm-streamer`).

---

## 📋 Table of Contents

- [What this is](#-what-this-is)
- [Project structure](#-project-structure)
- [System overview](#-system-overview)
- [Language & interface support](#-language--interface-support)
- [Limitations](#-limitations)
- [Building](#-building)
- [License & attribution](#-license--attribution)

---

## 🎯 What this is

PyHTM-core is a C++ implementation of Hierarchical Temporal Memory with
Python bindings. It ships **two engines in one wheel**:

* **Single-HTM-Core** — the HTM building blocks: Spatial Pooler, Temporal
  Memory, `Connections`, anomaly scoring, the SDR type, and the scalar /
  RDSE / date encoders.
* **Pyramid-Model-Core** — a multi-HTM *pyramid* runtime built on top of
  Single-HTM-Core: it wires many HTM nodes into layers, encodes and merges
  their inputs, runs them on a pinned worker pool, and streams a dataset
  through the whole structure natively.

The Single engine is a self-contained HTM library usable on its own; the
pyramid runtime is what PyHTM (`htm-streamer`) compiles and calls into for
its anomaly-detection experiments.

---

## 📁 Project structure

```
PyHTM-core/
├── 📂 Single-HTM-Core/       # the single-HTM engine
│   ├── src/                 # C++ algorithms (SP, TM, Connections, anomaly),
│   │                        #   encoders (RDSE, Scalar, Date), types (SDR),
│   │                        #   utils (Random, Topology, …), minimal os
│   ├── bindings/py/cpp_src/ # pybind11 modules: algorithms / sdr / encoders
│   ├── py/                  # the importable `htm` Python package
│   └── external/            # cereal fetch script + bundled MurmurHash3
├── 📂 Pyramid-Model-Core/    # 🔺 the FULL pyramid runtime in C++
│   ├── src/pyramid/         # rng / math / merge / encoders / spec / data /
│   │                        #   runtime / threading (see its README)
│   ├── bindings/            # htm.bindings.pyramid_engine (PyramidEngine)
│   └── py/pyramid_support/  # htm.pyramid (DatasetReader batched feeder)
├── CMakeLists.txt            # thin dispatcher: CPP_Only / Python3 stages
├── CommonCompilerConfig.cmake
├── DetectMarch.cmake         # CPUID/XGETBV SIMD-level probe
├── htm_install.py            # orchestrates the C++ build + wheel
├── pyproject.toml            # build/package definition (scikit-build-core)
├── get_version.py / VERSION
├── LICENSE                   # AGPLv3 (full text)
└── NOTICE                    # attribution
```

Each engine directory carries its own README — start from
**[`Pyramid-Model-Core/`](Pyramid-Model-Core/README.md)** (the pyramid
runtime: contract, GIL model, threading, flags),
**[`Single-HTM-Core/src/htm/`](Single-HTM-Core/src/htm/README.md)** (engine map),
**[`Single-HTM-Core/bindings/py/cpp_src/`](Single-HTM-Core/bindings/py/cpp_src/README.md)** (modules + GIL strategy), and
**[`Single-HTM-Core/py/htm/`](Single-HTM-Core/py/htm/README.md)** (import surface).


---

## 🧩 System overview

Two engines, one wheel:

* **Single-HTM-Core** exposes the HTM building blocks as
  `htm.bindings.sdr / .encoders / .algorithms`.
* **Pyramid-Model-Core** links the Single-HTM-Core static library and
  implements only what a pyramid adds on top: numpy-equivalent RNG/math,
  the 14 SDR merge modes, sample-derived encoder builders, the per-record
  run loop with its extensions, data ingestion, and the pinned worker
  pool. It is exposed as `htm.bindings.pyramid_engine` + the `htm.pyramid`
  support package.

Execution contract (the "instruction book"):

1. The caller assembles a fully-resolved **spec dict** — architecture
   (nodes, layers, feature plan), per-node SP/TM parameters, encoder
   definitions, merge modes and parameters, run-loop flags, and a data
   descriptor (native CSV path or a batched reader). PyHTM builds this
   dict via `htm_source.pipeline.build_pyramid_spec`.
2. `PyramidEngine(spec)` parses it; `execute()` constructs encoders (with
   a sampling pass over the training slice when parameters are
   data-derived), builds all SP/TM nodes in parallel, and streams the
   dataset through the pyramid — all under a single GIL release. Data is
   read natively in C++, one record at a time; the caller passes only a
   fully-resolved absolute path.
3. `results()` / `summaries()` return per-node anomaly + activity arrays
   and summary strings for downstream metrics and plotting.

There are exactly two Python↔C++ crossings per run: the spec goes down in
`execute()`, and the results come back up in `results()`.

Field-level spec documentation and the runtime model:
[`Pyramid-Model-Core/`](Pyramid-Model-Core/README.md).

## 🌐 Language & interface support

| Interface | Requirement |
|-----------|-------------|
| C++ static library (`htm_core` + the pyramid runtime sources) | Any C++17 compiler; CMake ≥ 3.21. No language restriction beyond that — the library is callable from any environment with C++ interop (C++ executables, JNI, .NET C++/CLI, Rust `cxx`, etc.). |
| Shipped Python bindings (`htm.bindings.*`, `htm.pyramid`) | CPython ≥ 3.9 (pybind11-based; no upper cap — see Requirements below). |
| Data formats | Native CSV reader in C++; DataFrame / Parquet / chunked CSV through `htm.pyramid.DatasetReader` (pandas/pyarrow). |

## ⚠️ Limitations

* The prebuilt binding modules target **CPython** (pybind11); PyPy and
  other interpreters are not targeted.
* The pyramid engine consumes a **fully-resolved spec** — YAML parsing and
  parameter resolution live with the caller (PyHTM keeps them in Python).
* Stochastic merge modes are reproducible only when a `seed` is supplied
  in the merge parameters; without one, each call draws OS entropy.
* On AVX-512 hosts, scalar `std::exp/log/pow` calls may differ from
  numpy by ≤ 1 ULP (see the
  [behavioral notes](Pyramid-Model-Core/README.md#-behavioral-notes)).

## 🛠️ Building

You normally **don't build this directly**. PyHTM (`htm-streamer`) builds it
for you: its setup runner clones this repository as a sibling of PyHTM and
compiles it via `htm_install.py`, auto-selecting the best SIMD level for your
CPU. See the PyHTM repository's setup instructions.

For reference, the engine is built through `htm_install.py` → CMake
configure/build → wheel, which produces a static `htm_core` library and the
Python extension modules.

### 📋 Requirements

* **Python ≥ 3.9** (enforced by `pyproject.toml` and `htm_install.py`; no
  upper cap — classifiers cover 3.9–3.13, and newer interpreters work as
  the dependencies below publish support).
* **Dependencies** (open floors): `numpy>=2.0` at runtime;
  `scikit-build-core>=0.10.7` + `pybind11>=2.13.6` at build time.
* **Toolchain**: CMake ≥ 3.21, a C++17 compiler.

### 🎛️ Build switches

Set as CMake cache variables (`-D…`) or environment variables — both build
stages honor them:

| Switch | Values | Effect |
|--------|--------|--------|
| `HTM_MARCH` | `auto` (default) / `off` / `x86-64` / `x86-64-v2` / `x86-64-v3` / `x86-64-v4` / `native` | CPU SIMD level for the generated code. `auto` probes the build host (CPUID + XGETBV, see [`DetectMarch.cmake`](DetectMarch.cmake)). The `htm_core` library itself is capped at `x86-64-v3` (v4 codegen measurably perturbs its floating-point results); its integer AVX-512 histogram kernel opts in per-file on v4 hosts, and the pyramid runtime uses the full detected level. Explicit `-march`/`/arch:` already present in `CXXFLAGS`/`CL` takes precedence and disables the adaptive selection. |
| `HTM_TYPE` | `multi` (default) / `single` | `multi` builds both engines. `single` builds only the Single-HTM-Core bindings (`htm.bindings.sdr/.encoders/.algorithms`), skipping [`Pyramid-Model-Core`](Pyramid-Model-Core/README.md); PyHTM's model layer detects the missing engine and uses its Python pyramid implementation. |
| `HTM_LTO` | `AUTO` (default) / `FULL` / `OFF` | `AUTO` link-time-optimizes the pyramid runtime's own translation units (measured ~×1.4 on the run loop) while the `htm_core` library's codegen — and its fixed-seed digests — stay unchanged. `FULL` extends LTO into the library for ~13% more, at the cost of ~1-ULP float changes inside it. |

---

## 📜 License & attribution

PyHTM-core is licensed under the **GNU Affero General Public License v3
(AGPLv3)**. See [`LICENSE`](LICENSE.txt) for the full text.

The Single-HTM-Core engine derives from
[htm.core](https://github.com/htm-community/htm.core) (HTM Community
Edition of NuPIC; © 2013–2024 Numenta, Inc. and the htm.core community),
which is itself AGPLv3. Original copyright notices and source-file headers
are preserved, and [`NOTICE`](NOTICE) carries the attribution. As required
by the AGPL, derivative works remain under AGPLv3, and network-served
modified versions must make their source available (AGPL §13).
