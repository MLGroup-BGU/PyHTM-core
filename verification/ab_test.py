"""
A/B bit-exactness test between two htm.core builds.

Runs deterministic fixed-seed SP+TM+RDSE scenarios and digests EVERY output
(active columns, predictive cells, active cells, winner cells, raw overlaps,
anomaly scores, and a pickle round-trip) into one SHA-256. Two builds are
behaviourally identical for these code paths iff the digests match.

Scenarios chosen to cover every code path touched by the optimization pass:
  S1: global inhibition + boostStrength=0   (PyHTM's default config path;
      exercises B1 getPredictiveCells, B4 duty cycles, B5 global inhibition,
      B6 boost early-out, B10 compute move chain, B3 init buffers)
  S2: global inhibition + boostStrength=2.0 (exercises the boost-ENABLED
      branch that B6 must leave untouched, incl. exp() boosting + bumping)
  S3: local inhibition + boostStrength=1.0  (exercises updateBoostFactorsLocal_
      past the early-out + the untouched local-inhibition path)
Each scenario also pickles SP+TM mid-run, restores, and continues -- proving
serialization still works and the restored state computes identically.
"""
import hashlib
import json
import pickle
import sys

import numpy as np
from htm.bindings.sdr import SDR
from htm.bindings.algorithms import SpatialPooler, TemporalMemory
from htm.bindings.encoders import RDSE, RDSE_Parameters


def run_scenario(h, *, input_size, columns, cells_per_col, steps,
                 global_inh, boost, seed_base, pickle_at, rdse_sparsity=0.06):
    rp = RDSE_Parameters()
    rp.size = input_size
    rp.sparsity = rdse_sparsity
    rp.resolution = 0.25
    rp.seed = seed_base
    enc = RDSE(rp)

    sp = SpatialPooler(
        inputDimensions=[input_size],
        columnDimensions=[columns],
        potentialPct=0.5,
        potentialRadius=input_size,
        globalInhibition=global_inh,
        localAreaDensity=0.04,
        synPermInactiveDec=0.006,
        synPermActiveInc=0.04,
        synPermConnected=0.13,
        boostStrength=boost,
        wrapAround=True,
        seed=seed_base + 1,
        dutyCyclePeriod=100,
        minPctOverlapDutyCycle=0.001,
    )
    tm = TemporalMemory(
        columnDimensions=[columns],
        cellsPerColumn=cells_per_col,
        activationThreshold=13,
        initialPermanence=0.21,
        connectedPermanence=0.13,
        minThreshold=10,
        maxNewSynapseCount=32,
        permanenceIncrement=0.10,
        permanenceDecrement=0.10,
        predictedSegmentDecrement=0.001,
        seed=seed_base + 2,
        maxSegmentsPerCell=64,
        maxSynapsesPerSegment=64,
    )

    inp = SDR(input_size)
    active = SDR([columns])
    anomalies = []

    for t in range(steps):
        val = float(np.sin(t * 0.05) * 10.0 + (5.0 if t % 97 == 0 else 0.0))
        enc.encode(val, inp)

        overlaps = sp.compute(inp, True, active)

        tm.activateDendrites(True)
        pred_cells = tm.getPredictiveCells()
        pred_cols = tm.cellsToColumns(pred_cells)

        if len(active.sparse) == 0:
            a = 0.0
        else:
            inter = SDR(active.dimensions)
            inter.intersection(active, pred_cols)
            a = 1.0 - len(inter.sparse) / len(active.sparse)
        anomalies.append(a)

        tm.activateCells(active, True)
        act_cells = tm.getActiveCells()
        win_cells = tm.getWinnerCells()

        h.update(np.asarray(active.sparse, dtype=np.uint32).tobytes())
        h.update(np.asarray(pred_cells.sparse, dtype=np.uint32).tobytes())
        h.update(np.asarray(pred_cols.sparse, dtype=np.uint32).tobytes())
        h.update(np.asarray(act_cells.sparse, dtype=np.uint32).tobytes())
        h.update(np.asarray(win_cells.sparse, dtype=np.uint32).tobytes())
        h.update(np.asarray(overlaps, dtype=np.uint64).tobytes())
        h.update(np.float64(a).tobytes())

        if t == pickle_at:
            sp = pickle.loads(pickle.dumps(sp))
            tm = pickle.loads(pickle.dumps(tm))
            h.update(b"pickled")

    return anomalies


def main():
    h = hashlib.sha256()

    a1 = run_scenario(h, input_size=600, columns=1024, cells_per_col=8,
                      steps=600, global_inh=True, boost=0.0,
                      seed_base=1956, pickle_at=300)
    a2 = run_scenario(h, input_size=400, columns=512, cells_per_col=6,
                      steps=350, global_inh=True, boost=2.0,
                      seed_base=2718, pickle_at=175)
    # S3 uses sparsity=0.10 (30 active bits at size 300): comfortably inside
    # the RDSE collision-check acceptance region, so scenario construction is
    # robust on ANY build -- including baselines that predate the
    # deterministic-check fix, where borderline params were a per-run coin
    # flip.
    a3 = run_scenario(h, input_size=300, columns=400, cells_per_col=4,
                      steps=250, global_inh=False, boost=1.0,
                      seed_base=3141, pickle_at=125, rdse_sparsity=0.10)

    out = {
        "digest": h.hexdigest(),
        "s1_anomaly_mean_last100": float(np.mean(a1[-100:])),
        "s2_anomaly_mean_last100": float(np.mean(a2[-100:])),
        "s3_anomaly_mean_last100": float(np.mean(a3[-100:])),
        "s1_anomaly_first10_mean": float(np.mean(a1[:10])),
    }
    print(json.dumps(out))


if __name__ == "__main__":
    sys.exit(main())
