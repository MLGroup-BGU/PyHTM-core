"""
Correctness tests for the APIs added for PyHTM:
  * SDR.subtract(minuend, subtrahend)          (sorted-sparse set difference)
  * algorithms.computeRawAnomalyScore(a, p)    (1 - |a&p| / |a|)

Verified against numpy references over randomized cases + edge cases,
including in-place usage (output SDR is also an input).
"""
import numpy as np
from htm.bindings.sdr import SDR
from htm.bindings.algorithms import computeRawAnomalyScore

rng = np.random.default_rng(7)
N = 2000

fails = 0
for trial in range(300):
    size = int(rng.integers(2, N))
    ka = int(rng.integers(0, max(1, size // 3)))
    kb = int(rng.integers(0, max(1, size // 3)))
    a_idx = np.sort(rng.choice(size, size=ka, replace=False)).astype(np.uint32)
    b_idx = np.sort(rng.choice(size, size=kb, replace=False)).astype(np.uint32)

    A = SDR(size); A.sparse = a_idx
    B = SDR(size); B.sparse = b_idx

    # --- subtract ---
    X = SDR(size)
    X.subtract(A, B)
    ref = np.setdiff1d(a_idx, b_idx, assume_unique=True)
    if not np.array_equal(np.asarray(X.sparse), ref):
        print(f"FAIL subtract trial={trial}"); fails += 1

    # chaining returns self
    Y = SDR(size).subtract(A, B)
    if not np.array_equal(np.asarray(Y.sparse), ref):
        print(f"FAIL subtract-chaining trial={trial}"); fails += 1

    # in-place: output is also the minuend
    Z = SDR(size); Z.sparse = a_idx
    Z.subtract(Z, B)
    if not np.array_equal(np.asarray(Z.sparse), ref):
        print(f"FAIL subtract-inplace-minuend trial={trial}"); fails += 1

    # in-place: output is also the subtrahend
    W = SDR(size); W.sparse = b_idx
    W.subtract(A, W)
    if not np.array_equal(np.asarray(W.sparse), ref):
        print(f"FAIL subtract-inplace-subtrahend trial={trial}"); fails += 1

    # --- computeRawAnomalyScore ---
    got = computeRawAnomalyScore(A, B)
    if ka == 0:
        want = 0.0
    else:
        want = 1.0 - len(np.intersect1d(a_idx, b_idx, assume_unique=True)) / ka
    if abs(got - want) > 1e-6:
        print(f"FAIL anomaly trial={trial}: got {got} want {want}"); fails += 1

# edge cases
E = SDR(10)
F = SDR(10)
E.sparse = np.array([], dtype=np.uint32)
F.sparse = np.array([1, 2], dtype=np.uint32)
G = SDR(10); G.subtract(E, F)
assert len(G.sparse) == 0, "empty minuend"
G.subtract(F, E)
assert list(G.sparse) == [1, 2], "empty subtrahend"
assert computeRawAnomalyScore(E, F) == 0.0, "empty active -> 0"

# dimension mismatch must raise
try:
    bad = SDR(11)
    G2 = SDR(10); G2.subtract(bad, F)
    print("FAIL: dimension mismatch not caught"); fails += 1
except RuntimeError:
    pass

print("ALL NEW-API TESTS PASSED" if fails == 0 else f"{fails} FAILURES")
