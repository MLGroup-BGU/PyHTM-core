"""RSS-based leak smoke: long steady-state run + construct/destroy cycles.
A leak in the per-step path shows as monotonic RSS growth between windows;
a leak in construction shows as growth across build/teardown cycles."""
import gc, resource, sys
import numpy as np
from htm.bindings.sdr import SDR
from htm.bindings.algorithms import SpatialPooler, TemporalMemory, computeRawAnomalyScore
from htm.bindings.encoders import RDSE, RDSE_Parameters

def rss_mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
def cur_rss_mb():
    with open('/proc/self/statm') as f: return int(f.read().split()[1]) * 4096 / 1e6

def make(seed):
    rp = RDSE_Parameters(); rp.size=2048; rp.sparsity=0.02; rp.resolution=0.25; rp.seed=seed
    enc = RDSE(rp)
    sp = SpatialPooler(inputDimensions=[2048], columnDimensions=[1024], potentialPct=0.3,
                       potentialRadius=2048, globalInhibition=True, localAreaDensity=0.02,
                       boostStrength=0.0, wrapAround=True, seed=seed+1)
    tm = TemporalMemory(columnDimensions=[1024], cellsPerColumn=6, activationThreshold=13,
                        initialPermanence=0.21, connectedPermanence=0.13, minThreshold=10,
                        maxNewSynapseCount=32, permanenceIncrement=0.1, permanenceDecrement=0.1,
                        predictedSegmentDecrement=0.001, seed=seed+2,
                        maxSegmentsPerCell=128, maxSynapsesPerSegment=64)
    return enc, sp, tm

# --- steady-state loop: measure RSS at window boundaries AFTER warm-up ---
enc, sp, tm = make(7)
inp, act, res = SDR(2048), SDR([1024]), SDR([1024*6])
def step(t):
    enc.encode(float(np.sin(t*0.05)*10 + (t%13)), inp)
    sp.compute(inp, True, act)
    tm.activateDendrites(True)
    pred = tm.getPredictiveCells()
    pcols = tm.cellsToColumns(pred)
    _ = computeRawAnomalyScore(act, pcols)
    tm.activateCells(act, True)
    ac = tm.getActiveCells()
    res.reshape(ac.dimensions); res.subtract(ac, pred)

for t in range(3000): step(t)          # warm-up: segments grow to steady state
gc.collect(); base = cur_rss_mb()
marks = []
for w in range(4):
    for t in range(3000, 3000+2000): step(t + w*2000)
    gc.collect(); marks.append(cur_rss_mb())
growth = marks[-1] - marks[0]
print(f"steady-state RSS after warmup: {base:.1f} MB; windows: {[f'{m:.1f}' for m in marks]}; "
      f"growth over last {3*2000} steps: {growth:+.2f} MB")
steady_ok = growth < 3.0   # TM still grows a few segments; must be near-flat

# --- construct/destroy cycles ---
del enc, sp, tm; gc.collect()
c0 = cur_rss_mb(); cyc = []
for i in range(6):
    e,s,m = make(100+i)
    a2 = SDR([1024]); i2 = SDR(2048)
    for t in range(50):
        e.encode(float(t), i2); s.compute(i2, True, a2); m.activateDendrites(True); m.activateCells(a2, True)
    del e,s,m,a2,i2; gc.collect(); cyc.append(cur_rss_mb())
cyc_growth = cyc[-1] - cyc[1]   # ignore first-cycle allocator retention
print(f"construct/destroy cycles RSS: start {c0:.1f} MB; per-cycle: {[f'{m:.1f}' for m in cyc]}; "
      f"growth cycles 2..6: {cyc_growth:+.2f} MB")
cycles_ok = cyc_growth < 2.0
print("LEAK SMOKE: " + ("PASS" if (steady_ok and cycles_ok) else "FAIL"))
sys.exit(0 if (steady_ok and cycles_ok) else 1)
