# ✅ `verification/` — Proof & Regression Suite

> Self-contained scripts that validated the 2026 performance pass — and the
> harness for proving any *future* change is behaviour-preserving.

The suite validated the pass described in the [root README](../README.md#-what-changed-vs-htmcore). Run them
against any installed `htm` wheel (`pip install <wheel>` first).

| Script | What it proves |
|---|---|
| `ab_test.py` | **Bit-exactness.** Three fixed-seed SP+TM+RDSE scenarios (PyHTM's default global/boost-0 path, boost-enabled, local inhibition), each with a mid-run pickle round-trip. Digests every output (active/predictive/winner cells, raw overlaps, anomaly) into one SHA-256. Two builds are behaviourally identical for these paths **iff the digests match**. Reference digest of the current build: `2b89a6ce28e854241775b24240b7a7c9c0c873b5ab753b168472294b24659a76` (Python 3.12 / linux-x86_64; the digest is platform- and Python-version-sensitive, so compare builds on the same machine). |
| `newapi_test.py` | **Correctness of the new APIs** — `SDR.subtract` (300 randomized trials + in-place variants + edge cases, vs `np.setdiff1d`) and `computeRawAnomalyScore` (vs a numpy reference). |
| `gilbench.py` | **GIL-held time per forward step.** A pure-Python probe thread measures how much a worker running PyHTM's forward() pattern starves it — i.e. the fraction of each step that serializes the thread hive. |
| `bench.py` | Single-thread step throughput + SP init time (sanity/regression numbers). |
| `leak_test.py` | **Memory-leak smoke.** Long steady-state run (RSS windows) + repeated construct/destroy cycles. Learning growth (new segments/synapses) is expected; cycle growth is not. |
| `leak_test2.py` | **Decisive leak isolation.** After warm-up, runs inference-only (`learn=False`) windows: with no structural learning the RSS must be flat -- and it is, to the kilobyte, proving the per-step path performs no leaking allocations. |

## 🔁 Typical A/B workflow for a future change

```bash
pip install --force-reinstall dist/htm-<before>.whl && python verification/ab_test.py   # digest A
pip install --force-reinstall dist/htm-<after>.whl  && python verification/ab_test.py   # digest B
# identical digests == the change did not alter model behaviour
```
