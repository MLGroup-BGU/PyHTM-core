# Single-HTM-Core

A C++ implementation of the core **Hierarchical Temporal Memory (HTM)**
building blocks, with Python bindings. This is a self-contained engine: the
Spatial Pooler, Temporal Memory, `Connections`, anomaly scoring, the SDR
type, and the scalar / RDSE / date encoders.

It can be built and used entirely on its own (this README), and it is the
foundation the [Pyramid-Model-Core](../Pyramid-Model-Core/README.md) runtime
is layered on top of within [PyHTM-core](../README.md).

---

## What's inside

```
Single-HTM-Core/
├── src/htm/
│   ├── algorithms/     # SpatialPooler, TemporalMemory, Connections, Anomaly
│   ├── encoders/       # ScalarEncoder, RDSE, DateEncoder, base
│   ├── types/          # SDR (Sparse Distributed Representation), Sdr metrics
│   ├── utils/          # Random, Topology, MovingAverage, ...
│   ├── os/             # minimal Directory / Path helpers
│   └── CMakeLists.txt  # the library target
├── bindings/py/cpp_src/ # pybind11 modules: algorithms / sdr / encoders
├── py/htm/             # the importable `htm` Python package (thin wrappers)
├── external/           # cereal fetch + bundled MurmurHash3
├── CMakeLists.txt          # standalone build entry (this directory as root)
├── CommonCompilerConfig.cmake
├── DetectMarch.cmake       # CPUID/XGETBV SIMD-level probe
└── htm_install.py          # standalone build/install script
```

The subtree is **self-contained**: it carries its own
`CommonCompilerConfig.cmake` and `DetectMarch.cmake`, so it configures with
this directory as the CMake source root without any parent-repository files.

---

## Building

The quickest path (auto-detects OS, compiler, and SIMD level):

```bash
python htm_install.py                # build the C++ library only
python htm_install.py --bindings     # also build the Python modules
```

Options: `--march <off|x86-64|x86-64-v2|x86-64-v3|x86-64-v4|native>` to force a
SIMD level (default is the auto-probe), `--build-type Debug`, `--clean`.

Or drive CMake directly:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBINDING_BUILD=CPP_Only
cmake --build build --config Release
# -> build/src/libhtm_core.a  (htm_core.lib on Windows)
```

Set `-DBINDING_BUILD=Python3` to also build the pybind11 extension modules.

### Requirements

* CMake ≥ 3.21 and a C++17 compiler (MSVC on Windows; gcc/clang on
  Linux/macOS). The Python bindings additionally need CPython ≥ 3.9 and
  pybind11.

### SIMD level

`HTM_MARCH=auto` (the default) probes the build host with CPUID + XGETBV and
selects `x86-64-v2/v3/v4`. The library is capped at `x86-64-v3` because v4
codegen measurably perturbs its floating-point results; the integer
histogram hot loop in `Connections::computeActivity` opts into an AVX-512
(`vpconflictd` + gather/scatter) kernel per-file on v4 hosts. Floating-point
contraction is pinned off under march levels so results are identical across
levels.

---

## Using it

**C++** — link the static `htm_core` library and include `src/htm`:

```cpp
#include <htm/algorithms/SpatialPooler.hpp>
#include <htm/algorithms/TemporalMemory.hpp>
#include <htm/types/Sdr.hpp>
// build an SDR, run SP -> TM, read anomaly. See the headers in src/htm/.
```

**Python** (after `--bindings`) — the modules install under `htm.bindings`:

```python
from htm.bindings.sdr import SDR
from htm.bindings.algorithms import SpatialPooler, TemporalMemory
from htm.bindings.encoders import RDSE, RDSE_Parameters

enc = RDSE(RDSE_Parameters())   # configure resolution/size/sparsity
sp  = SpatialPooler(...)        # input -> active columns
tm  = TemporalMemory(...)       # columns -> predictive cells, anomaly
```

---

## License & attribution

Single-HTM-Core derives from
[htm.core](https://github.com/htm-community/htm.core) (HTM Community Edition
of NuPIC; © 2013–2024 Numenta, Inc. and the htm.core community) and is
licensed under the **GNU Affero General Public License v3 (AGPLv3)**.
Original copyright notices and source-file headers are preserved. As required
by the AGPL, derivative works remain under AGPLv3.
