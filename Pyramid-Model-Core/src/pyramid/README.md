# 🧠 src/pyramid — the C++ pyramid runtime

One folder per concern.

| Folder | What lives here |
|--------|-----------------|
| `rng/` | numpy-exact `SeedSequence` (wrapping-subtraction mix), PCG64 (portable 128-bit, XSL-RR), `choice(replace=False, p=…)` cdf-resampling, and `SeedMix` — the counter-based per-call seed for the stochastic merge modes |
| `math/` | numpy-exact `pairwise_sum`, `cumsum`, `searchsorted_right`, `linspace`, `quantile_median_unbiased` (numpy's *literal* virtual-index formula — do **not** simplify it algebraically), `percentile_linear` |
| `merge/` | `SdrMerger` — all 14 modes (`u i sd nos c s su aps pls ssu li bws pcb di`) + `sdr_max_pool_into`. Scratch buffers, touched-index clearing, zero steady-state allocation. Every call carries a `MergeCall` (node, slot, record index) which the stochastic modes fold into their seed |
| `encoders/` | `FeatureEncoders` (rdse/adaptive/hybrid/dual/categorical/date), `EncoderBuild` (sample-derived params: deduced resolution, adaptive quantile table, dual fit), `DateMath` |
| `spec/` | `PyramidSpec` structs — the parsed instruction book, incl. preformatted per-node `summary` strings from the Python builder |
| `data/` | `RecordSource` + `CsvSource` (native CSV: header map, dt strptime, row slicing) |
| `runtime/` | `HtmNode` (SP→TM step order; output = active columns, or surprised columns under RTM; squeeze/restore, max-pool, residual modes) and `PyramidRuntime` (per-record loop: encode → feature merge → layers → extras chain → results) |
| `threading/` | `WorkerPool` — pinned ownership, weighted LPT with carried loads (verbatim `balanced_partition_weighted` arithmetic), mailbox + layer barrier, first-error rethrow |

### 🎲 Merge seeding

The sparsity-controlled modes keep a fixed budget of bits and **draw** which
ones to keep, so each call needs a random stream. The stream is derived, not
taken directly from the config:

```
seed_for_call = mix(base_seed, node_key, slot, iteration)
```

| Component | Meaning | Source |
|-----------|---------|--------|
| `base_seed` | the experimenter's seed, constant across the run | `htm_merge_params.seed` / `feature_merge_params.seed` |
| `node_key` | **where** in the pyramid | layer + node index, packed into disjoint 32-bit halves then mixed |
| `slot` | **which** merge, for nodes running several per record | fixed constant per call site (`MergeSlot`) |
| `iteration` | **which** record | the serial loop counter, or the pipeline's per-node record index `t` |

Why each part is needed:

* **`node_key`** keeps sibling nodes independent. It is read from the node
  NAME (`L{layer}_{node}`) once at build time, and the **timescale suffix is
  excluded on purpose** — the same structural position must hash identically
  whether it runs as `_short` or `_long`, so architectures that differ only in
  timescale assignment stay comparable. Packing the two indices into separate
  32-bit halves is what makes `L1_12` and `L11_2` distinct, which a decimal or
  textual join would not.
* **`slot`** keeps a node's several merges independent. With
  `use_predictive`, `broadcast_head`, `recurrent_head` or `long_term_memory`
  a single node performs up to five stochastic merges for one record; without
  this component they would all share one stream and their draws would move
  together.
* **`iteration`** is what stops a repeating input from producing a repeating
  output. A node whose input is unchanged still draws a different subset each
  record, so the bits that lose one draw are not excluded permanently.

Every component is a static property of the configuration and the record
index. Nothing is read from shared state, no counter is incremented at run
time, and no value depends on which thread arrives first — so the result is
identical under any worker count and any interleaving, and two runs of one
configuration match byte for byte. `seed: null` keeps its separate meaning:
fresh OS entropy per call, deliberately non-reproducible.

The budget short-circuit is unaffected. When the candidate set is no larger
than `target_sparsity * total_bits` every candidate is kept, no draw decides
anything, and the seed cannot change the output.

⚠️ Ordering rules:
* `SDR::setSparse` (non-const overload) **swaps** — pass copies when the
  caller still needs the vector. `reshape` mutates in place.
* The extras chain order in `PyramidRuntime::run` mirrors the Python run
  loop **exactly**: children → residual → boost → memory/recurrent at the
  head.