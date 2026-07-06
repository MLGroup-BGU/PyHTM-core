"""Decisive leak isolation: after warm-up, run windows with learn=False.
No learning => no new segments/synapses => RSS must be FLAT if (and only if)
the per-step path itself is allocation-leak-free."""
import gc, sys
import numpy as np
from htm.bindings.sdr import SDR
from htm.bindings.algorithms import SpatialPooler, TemporalMemory, computeRawAnomalyScore
from htm.bindings.encoders import RDSE, RDSE_Parameters

def cur_rss_mb():
    with open('/proc/self/statm') as f: return int(f.read().split()[1]) * 4096 / 1e6

rp = RDSE_Parameters(); rp.size=2048; rp.sparsity=0.02; rp.resolution=0.25; rp.seed=7
enc = RDSE(rp)
sp = SpatialPooler(inputDimensions=[2048], columnDimensions=[1024], potentialPct=0.3,
                   potentialRadius=2048, globalInhibition=True, localAreaDensity=0.02,
                   boostStrength=0.0, wrapAround=True, seed=8)
tm = TemporalMemory(columnDimensions=[1024], cellsPerColumn=6, activationThreshold=13,
                    initialPermanence=0.21, connectedPermanence=0.13, minThreshold=10,
                    maxNewSynapseCount=32, permanenceIncrement=0.1, permanenceDecrement=0.1,
                    predictedSegmentDecrement=0.001, seed=9,
                    maxSegmentsPerCell=128, maxSynapsesPerSegment=64)
inp, act, res = SDR(2048), SDR([1024]), SDR([1024*6])
def step(t, learn):
    enc.encode(float(np.sin(t*0.05)*10 + (t%13)), inp)
    sp.compute(inp, learn, act)
    tm.activateDendrites(learn)
    pred = tm.getPredictiveCells()
    pcols = tm.cellsToColumns(pred)
    _ = computeRawAnomalyScore(act, pcols)
    tm.activateCells(act, learn)
    ac = tm.getActiveCells()
    res.reshape(ac.dimensions); res.subtract(ac, pred)

for t in range(3000): step(t, True)      # warm-up WITH learning
gc.collect(); m0 = cur_rss_mb()
marks = []
for w in range(4):                        # inference-only windows
    for t in range(2000): step(3000 + w*2000 + t, False)
    gc.collect(); marks.append(cur_rss_mb())
growth = marks[-1] - m0
print(f"learn=False windows RSS: start {m0:.1f} MB -> {[f'{m:.1f}' for m in marks]}; growth: {growth:+.3f} MB")
print("PER-STEP PATH: " + ("LEAK-FREE ✓" if abs(growth) < 0.5 else "SUSPECT ✗"))
sys.exit(0 if abs(growth) < 0.5 else 1)
