"""
GIL-contention benchmark.

In PyHTM's thread hive, any time a worker holds the GIL it stalls EVERY other
worker's Python glue. This measures exactly that: a probe thread runs a pure
Python counting loop; a worker thread runs PyHTM's forward() pattern in a
loop. The probe's throughput drop (vs. running alone) equals the fraction of
wall time the worker holds the GIL. Works on a single core, since the GIL is
about holding, not parallel hardware.

Reported: gil_held_fraction  (lower = more of the step overlaps with other
threads' Python work = better hive scaling).
"""
import sys
import threading
import time

import numpy as np
from htm.bindings.sdr import SDR
from htm.bindings.algorithms import SpatialPooler, TemporalMemory
from htm.bindings.encoders import RDSE, RDSE_Parameters

INPUT, COLS, CPC = 4096, 2048, 4
DURATION = 6.0


def make_model():
    rp = RDSE_Parameters(); rp.size = INPUT; rp.sparsity = 0.02
    rp.resolution = 0.25; rp.seed = 42
    enc = RDSE(rp)
    sp = SpatialPooler(
        inputDimensions=[INPUT], columnDimensions=[COLS],
        potentialPct=0.3, potentialRadius=INPUT, globalInhibition=True,
        localAreaDensity=0.02, synPermInactiveDec=0.006, synPermActiveInc=0.04,
        synPermConnected=0.13, boostStrength=0.0, wrapAround=True, seed=1956)
    tm = TemporalMemory(
        columnDimensions=[COLS], cellsPerColumn=CPC, activationThreshold=13,
        initialPermanence=0.21, connectedPermanence=0.13, minThreshold=10,
        maxNewSynapseCount=32, permanenceIncrement=0.1, permanenceDecrement=0.1,
        predictedSegmentDecrement=0.001, seed=1960,
        maxSegmentsPerCell=128, maxSynapsesPerSegment=64)
    return enc, sp, tm


def probe(stop, out):
    n = 0
    while not stop.is_set():
        n += 1
    out.append(n)


def probe_rate(worker=None, warm_args=None):
    stop = threading.Event()
    out = []
    th_probe = threading.Thread(target=probe, args=(stop, out))
    threads = [th_probe]
    if worker:
        threads.append(threading.Thread(target=worker, args=(stop,) + warm_args))
    for t in threads:
        t.start()
    time.sleep(DURATION)
    stop.set()
    for t in threads:
        t.join()
    return out[0] / DURATION


def forward_worker(stop, enc, sp, tm, inp, active, counter):
    t = 0
    while not stop.is_set():
        enc.encode(float(np.sin(t * 0.05) * 10), inp)
        sp.compute(inp, True, active)
        tm.activateDendrites(True)
        pred = tm.getPredictiveCells()
        _ = tm.cellsToColumns(pred)
        tm.activateCells(active, True)
        _ = tm.getActiveCells()
        t += 1
    counter.append(t)


def main():
    enc, sp, tm = make_model()
    inp, active = SDR(INPUT), SDR([COLS])
    # warm up / grow segments
    for t in range(60):
        enc.encode(float(np.sin(t * 0.05) * 10), inp)
        sp.compute(inp, True, active)
        tm.activateDendrites(True)
        _ = tm.cellsToColumns(tm.getPredictiveCells())
        tm.activateCells(active, True)
        _ = tm.getActiveCells()

    alone = probe_rate()

    steps = []
    contended = probe_rate(worker=forward_worker,
                           warm_args=(enc, sp, tm, inp, active, steps))
    frac = 1.0 - contended / alone
    step_rate = steps[0] / DURATION
    gil_ms_per_step = 1000.0 * frac / step_rate if step_rate else float("nan")
    print(f"probe_alone={alone:.0f}/s probe_contended={contended:.0f}/s "
          f"gil_held_fraction={frac:.3f} worker_steps_per_s={step_rate:.0f} "
          f"gil_ms_per_step={gil_ms_per_step:.3f}")


if __name__ == "__main__":
    sys.exit(main())
