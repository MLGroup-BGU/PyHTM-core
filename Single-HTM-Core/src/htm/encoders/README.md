# 🔡 `encoders/` — Values → SDRs

> C++ encoders that turn raw inputs into Sparse Distributed Representations.

## 📄 Files

| File | Role |
|---|---|
| `RandomDistributedScalarEncoder.{hpp,cpp}` | **RDSE** — the encoder PyHTM uses for numeric features. Hash-based bucketing (MurmurHash3 from [`external/common`](../../../external/README.md)). Nearby values share bits. |
| `ScalarEncoder.{hpp,cpp}` | Classic bounded scalar encoder. Also the building block the pure-Python `DateEncoder` composes. |
| `DateEncoder.{hpp,cpp}` | C++ date/time encoder (season, day-of-week, weekend, holiday, time-of-day…). |
| `BaseEncoder.hpp` | The tiny common interface (`size`, `dimensions`, `encode`). |

> ℹ️ **Two DateEncoders exist.** PyHTM currently imports the *pure-Python*
> one ([`py/htm/encoders/date.py`](../../../py/htm/encoders/README.md)),
> which composes ScalarEncoders. The C++ `DateEncoder` here is bound and
> available as `htm.bindings.encoders.DateEncoder`. Switching PyHTM to it is
> a candidate for the PyHTM-side optimization phase (needs parameter-mapping
> validation first).

The `RDSE.encode` binding releases the GIL — encoding different features on
different hive threads runs truly in parallel.

---

## 🎯 RDSE Bucket Index: a PyHTM-core Design

> **This behavior belongs to PyHTM-core, not to htm.core.** Upstream htm.core computes the RDSE
> bucket index with a plain cast. PyHTM-core replaces that one line in
> `RandomDistributedScalarEncoder::encode()` with the computation below. The rest of `encode()`,
> the MurmurHash3 window of `activeBits` consecutive buckets, is the upstream code.

### Scope

Every RDSE in PyHTM-core runs this code. That covers the `htm.bindings.encoders.RDSE` binding
and the pyramid runtime's `rdse`, `adaptive_rdse`, `hybrid_rdse` and categorical encoders, which
wrap this class
([`FeatureEncoders.cpp`](../../../../Pyramid-Model-Core/src/pyramid/encoders/FeatureEncoders.cpp)).
Both Python modules link the static `htm_core` library, so a rebuild of `htm_core` carries the
behavior into both.

`htm_install.py` skips the C++ build while `build/Release/lib/libhtm_core.a` (`htm_core.lib` on
Windows) exists, and skips the wheel build while `dist/` holds a wheel. In a tree built before
this change, deleting that library file makes the next `htm_install.py` rebuild `htm_core`, clear
`dist/`, and rebuild and install both modules. A fresh clone builds everything on its own.

### The upstream line

```cpp
const UInt index = (UInt) (input / args_.resolution);   // htm.core [5]
```

`UInt` is 32 bits ([`types/Types.hpp`](../types/README.md)). Converting a floating-point value that
does not fit the destination type is undefined behavior in C++, and modulo arithmetic does not
apply even for an unsigned destination [1]. The result is whatever the conversion instruction of
the build returns:

| Build | Negative quotient | Positive quotient from 2^32 to 2^63 | Quotient of magnitude 2^63 or more |
|---|---|---|---|
| x86-64-v3, the default `htm_core` build | the low 32 bits of a 64-bit truncation, which is a bucket below 0 on the ring while the magnitude stays below 2^32 | the low 32 bits, an arbitrary bucket | bucket 0, the exact code of 0 [2] |
| x86-64-v4 (AVX-512) | bucket 0xFFFFFFFF | bucket 0xFFFFFFFF | bucket 0xFFFFFFFF [3] |

Bucket 0xFFFFFFFF sits next to bucket 0 on the ring, so its window shares all but one of its
`activeBits` hash inputs with the window of bucket 0 (163 of 164 at n = 8192, w = 0.02). On a v4
build every negative input and every quotient from 2^32 up is therefore encoded almost as 0. On
the default build a quotient of magnitude 2^63 or more is encoded exactly as 0, and a quotient
from 2^32 to 2^63 lands on an arbitrary bucket.

**Measured case.** WADI `2A_AIT_002_PV` is float residue near 1e-41 on 99.67% of its train rows,
so the resolution meta derives for it is 1.807e-41, and a reading of 0.2 gives a quotient near
1.1e40. With the upstream line, all 293 distinct real readings (0.0078 to 7.9, 2,930 rows) receive
the exact code of 0, and a PyramidEngine run is bit-identical whether those readings are present
or replaced by 0.0.

### What PyHTM-core computes

```cpp
constexpr Real64 kMaxBucket = 1073741824.0;   // 2^30
Real64 quotient = input / args_.resolution;
if( quotient > kMaxBucket ) {
  quotient = kMaxBucket;
}
else if( quotient < -kMaxBucket ) {
  quotient = -kMaxBucket;
}
const Int64 signed_index = static_cast<Int64>( quotient );   // truncates toward zero
const UInt  index = static_cast<UInt>( static_cast<UInt64>( signed_index ) );
```

- **Inside ±2^30 buckets** the index equals the one the default x86-64-v3 build of the upstream
  line gives: truncation toward zero in 64 bits, low 32 bits kept. Every in-range encoding is
  bit-identical to upstream on that build.
- **Beyond +2^30** an input takes the top bucket of the scale, and **beyond -2^30** the bottom
  bucket. This is the most extreme encoding the feature has.
- **The ends continue the scale.** Each bucket one step closer to 0 shares one bit fewer with the
  top (or bottom) code, exactly as any two RDSE neighbors do.
- **±inf** follows the same rule. **NaN** still gives an empty SDR.
- **Every conversion is defined**, so an x86-64-v3 build and an x86-64-v4 build return the same
  encoding for every input.

### Why the index is computed this way

- **An out-of-range value takes the end bucket of the scale.** This is the rule Purdy gives for
  bucketed encoders: *"use the smallest bucket for values below the range and the largest bucket
  for values above the range. Thus all values above the range will have the same representation,
  and similarly for those below the range"* [4]. A value above the scale shares bits with the
  largest in-range values, in line with the first encoder principle, *"semantically similar data
  should result in SDRs with overlapping active bits"* [4]. A reserved code placed away from the
  scale would share no bits with them.
- **The bound is 2^30 on each side.** The ring has 2^32 positions. The top bucket sits at 2^30 and
  the bottom bucket at 3·2^30 (the wrap of -2^30), so the two are 2^31 apart and each is 2^30 away
  from bucket 0. No two of the three windows overlap for any `activeBits` below 2^30. A bound of
  2^31 would bring the top and bottom buckets together on the ring, with the same or almost the
  same code.
- **The index stays 32 bits.** `hash_buffer` is hashed over its own byte size, so a 64-bit index
  would change every encoding and break comparison with every existing result.
- **Values beyond an end share one code.** The encoder marks a reading as above (or below) the
  scale and does not rank readings beyond it. A resolution that keeps every reading inside ±2^30
  buckets keeps the full ranking.

### Verification

An unmodified and a patched PyHTM-core, each built with `htm_install.py` (default x86-64-v3
library) and compared on the same inputs:

| Check | Result |
|---|---|
| 135,940 synthetic in-range encodings: negatives, zeros, bucket edges, category mode, sizes 1,000 to 16,384 | bit-identical |
| 1,380,734 encodings of real WADI values, all 81 features of the original config at their original resolutions, negative values included | bit-identical |
| PyramidEngine run, three L0 nodes of the WADI runner under one head, 10,000 rows, original resolutions | every node's anomaly score bit-identical |
| `2A_AIT_002_PV` at 1.807e-41, 293 distinct real readings | unmodified: every one equals the code of 0. Patched: every one equals the top code, which shares 5 bits with the code of 0 |
| Negative inputs beyond the scale | patched: the bottom code, which shares 2 bits with the code of 0 and 4 with the top code |
| The same source compiled at x86-64-v3 and at x86-64-v4 | patched: identical encodings. Unmodified at v4: every negative input maps to one code that shares 161 bits with the code of 0 |
| `htm_install.py` in a tree built before the change, with only the `.cpp` replaced | C++ build and wheel build skipped, the old wheel reinstalled, and a PyramidEngine run still encodes the readings as 0 |
| The same tree after deleting `build/Release/lib/libhtm_core.a` | `htm_core` and both modules rebuilt, and the PyramidEngine run encodes the readings apart from 0 |

### Sources

1. cppreference, *Implicit conversions*, floating-integral conversions (ISO C++ [conv.fpint]):
   *"If the truncated value cannot fit into the destination type, the behavior is undefined (even
   when the destination type is unsigned, modulo arithmetic does not apply)."*
   [en.cppreference.com/w/cpp/language/implicit_conversion](https://en.cppreference.com/w/cpp/language/implicit_conversion)
2. Intel SDM, *CVTTSD2SI*: a result beyond the signed 64-bit range returns the indefinite integer
   80000000_00000000H. [felixcloutier.com/x86/cvttsd2si](https://www.felixcloutier.com/x86/cvttsd2si)
3. Intel SDM, *VCVTTSD2USI*: a result that cannot be represented returns 2^w - 1.
   [felixcloutier.com/x86/vcvttsd2usi](https://www.felixcloutier.com/x86/vcvttsd2usi)
4. Purdy, S. *Encoding Data for HTM Systems.* Numenta, 2016.
   [arXiv:1602.05925](https://arxiv.org/abs/1602.05925)
5. htm.core, `RandomDistributedScalarEncoder.cpp`.
   [github.com/htm-community/htm.core](https://github.com/htm-community/htm.core/blob/master/src/htm/encoders/RandomDistributedScalarEncoder.cpp)