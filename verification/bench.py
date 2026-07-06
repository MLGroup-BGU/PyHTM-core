"""Micro-benchmark: SP init time + per-step throughput in PyHTM's forward() pattern."""
import sys
import time

import numpy as np
from htm.bindings.sdr import SDR
from htm.bindings.algorithms import SpatialPooler, TemporalMemory
from htm.bindings.encoders import RDSE, RDSE_Parameters

INPUT, COLS, CPC, STEPS = 4096, 2048, 4, 300


def build():
    t0 = time.perf_counter()
    sp = SpatialPooler(
        inputDimensions=[INPUT], columnDimensions=[COLS],
        potentialPct=0.3, potentialRadius=INPUT, globalInhibition=True,
        localAreaDensity=0.02, synPermInactiveDec=0.006, synPermActiveInc=0.04,
        synPermConnected=0.13, boostStrength=0.0, wrapAround=True, seed=1956)
    t_sp = time.perf_counter() - t0
    tm = TemporalMemory(
        columnDimensions=[COLS], cellsPerColumn=CPC, activationThreshold=13,
        initialPermanence=0.21, connectedPermanence=0.13, minThreshold=10,
        maxNewSynapseCount=32, permanenceIncrement=0.1, permanenceDecrement=0.1,
        predictedSegmentDecrement=0.001, seed=1960,
        maxSegmentsPerCell=128, maxSynapsesPerSegment=64)
    return sp, tm, t_sp


def main():
    rp = RDSE_Parameters(); rp.size = INPUT; rp.sparsity = 0.02
    rp.resolution = 0.25; rp.seed = 42
    enc = RDSE(rp)

    sp, tm, t_init = build()
    inp, active = SDR(INPUT), SDR([COLS])

    # warmup (also grows TM segments so the steady-state is realistic)
    for t in range(60):
        enc.encode(float(np.sin(t * 0.05) * 10), inp)
        sp.compute(inp, True, active)
        tm.activateDendrites(True)
        _ = tm.cellsToColumns(tm.getPredictiveCells())
        tm.activateCells(active, True)
        _ = tm.getActiveCells()

    t0 = time.perf_counter()
    for t in range(STEPS):
        enc.encode(float(np.sin(t * 0.05) * 10), inp)
        sp.compute(inp, True, active)
        tm.activateDendrites(True)
        pred = tm.getPredictiveCells()
        _ = tm.cellsToColumns(pred)
        tm.activateCells(active, True)
        _ = tm.getActiveCells()
    dt = time.perf_counter() - t0

    print(f"sp_init_s={t_init:.3f} steps_per_s={STEPS/dt:.1f} step_ms={1000*dt/STEPS:.2f}")


if __name__ == "__main__":
    sys.exit(main())
