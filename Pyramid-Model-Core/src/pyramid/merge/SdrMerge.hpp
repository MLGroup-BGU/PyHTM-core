/* ---------------------------------------------------------------------------
 * Pyramid-Model-Core :: merge/SdrMerge.hpp
 *
 * The C++ port of PyHTM's `htm_source/data/sdr.py` merge machinery -- every
 * mode the pyramid can be configured with:
 *
 *   basic:       u (union), i (intersection), sd|xor (symmetric difference),
 *                nos (non-overlapping sum), c (concatenate), s (stack)
 *   stochastic:  su, aps, pls, ssu, li, bws, pcb, di
 *   utilities:   max_pool (sdr_max_pool)
 *
 * FAITHFULNESS
 *   The stochastic modes are line-for-line ports of the Python functions:
 *   identical probability math (float64, numpy pairwise `.sum()`, numpy exp/
 *   clip semantics, the same operation ORDER), and the identical sampler --
 *   NumpyGenerator::choice_no_replace_p seeded exactly like
 *   `np.random.default_rng(seed)`, constructed FRESH PER CALL just as the
 *   Python code does.  Given the same seed and inputs, the selected bit set
 *   is the same set NumPy selects.  Verified per mode against the actual
 *
 * PERFORMANCE / MEMORY
 *   One SdrMerger instance owns all scratch buffers and is reused across
 *   calls: steady-state merging performs ZERO heap allocations once buffers
 *   reach their high-water size.  The activation-count workspace is a
 *   full-size float64 vector cleared via a touched-index list (never a full
 *   memset), so cost scales with active bits, not SDR size.
 *
 * THREAD-SAFETY
 *   An SdrMerger is NOT thread-safe (by design -- it owns scratch).  The
 *   pyramid runtime performs all merges on its orchestrator thread, exactly
 *   where the Python pyramid performs them (its main thread).  Because every
 *   stochastic call constructs a fresh, identically-seeded generator, merge
 *   ORDER has no effect on results.
 * ------------------------------------------------------------------------ */
#pragma once

#include <htm/types/Sdr.hpp>

#include <cstdint>
#include <string>
#include <vector>

#include "../rng/NumpyRandom.hpp"

namespace pyramid {

/* Resolved merge parameters = DEFAULT_MERGE_PARAMS overlaid with the user's
 * merge_params dict (the overlay happens on the Python side / spec parser;
 * this struct always carries a full, final set). */
struct MergeParams {
    double        target_sparsity     = 0.02;
    bool          has_seed            = false;  // Python seed=None -> entropy
    std::uint64_t seed                = 0;
    double        temperature         = 1.0;    // su / ssu / bws / pcb
    double        alpha               = 1.5;    // pls
    double        smoothing           = 0.1;    // ssu
    std::int64_t  inhibition_radius   = 50;     // li
    double        inhibition_strength = 0.5;    // li
    double        burst_bonus         = 2.0;    // bws
    double        novelty_weight      = 0.3;    // pcb
    std::int64_t  segment_size        = 64;     // di
    double        segment_threshold   = 0.3;    // di
};

/* Compact mode enum (parsed once from the config string at build time so the
 * per-record hot path never touches strings). */
enum class MergeMode : std::uint8_t {
    Union, Intersection, SymDiff, NonOverlapSum, Concat, Stack,
    SoftUnion, ActivationProportional, PowerLaw, StabilizedSoftUnion,
    LateralInhibition, BurstWeighted, PredictiveCodingBoost,
    DendriticIntegration
};

MergeMode parse_merge_mode(const std::string &mode);  // throws on unknown
bool      is_stochastic_mode(MergeMode m);

/* sdr_max_pool as a free function (shared by SdrMerger::max_pool and the
 * per-node post-processing): divide coordinates on `axis` by `ratio`,
 * collapsing duplicates.  `scratch` avoids per-call allocation. */
void sdr_max_pool_into(const htm::SDR &in, int ratio, int axis, htm::SDR &out,
                       std::vector<htm::UInt> &scratch);

/* Output shape of `sdr_merge` for a given mode (ports concat_shapes and the
 * stack shape rule; identity for all in-place modes). Validates like Python. */
std::vector<htm::UInt> merge_output_dims(
    const std::vector<std::vector<htm::UInt>> &input_dims,
    MergeMode mode, int axis);

class SdrMerger {
public:
    SdrMerger() = default;

    /* Merge `inputs` (>= 1) into `out`.
     *
     * - `out` must already carry the correct dimensions (merge_output_dims);
     *   for every mode except Concat/Stack that is simply the input dims.
     * - With a single input, Python's sdr_merge returns the input object
     *   itself; here the caller is expected to short-circuit that case (the
     *   runtime does), but if called anyway the input is copied into `out`.
     * - Duplicate input pointers are legal and meaningful (anomaly-boost
     *   passes the same child several times; each occurrence increments the
     *   activation counts exactly as np.stack of repeated .dense does).
     */
    void merge(const std::vector<const htm::SDR *> &inputs, MergeMode mode,
               int axis, const MergeParams &params, htm::SDR &out);

    /* sdr_max_pool: divide coordinates on `axis` by `ratio` (>= 2), collapsing
     * duplicates (set semantics).  `out` must carry the pooled dimensions.
     * ratio == 1 is the caller's short-circuit (Python returns the input). */
    void max_pool(const htm::SDR &in, int ratio, int axis, htm::SDR &out);

private:
    /* -- activation-count workspace (see class comment) ------------------- */
    std::vector<double>       counts_;      // full-size, cleared via touched_
    std::vector<std::int64_t> touched_;     // indices written in counts_
    std::vector<std::int64_t> activeIdx_;   // sorted ascending (np.where)
    std::vector<double>       activeCnt_;   // counts at activeIdx_

    /* -- probability / sampling scratch ----------------------------------- */
    std::vector<double>       probs_;       // per-mode probabilities
    std::vector<double>       work_;        // mode-local temporaries
    std::vector<double>       pCopy_, cdf_, xDraw_;      // choice scratch
    std::vector<std::int64_t> sampled_, newIdx_;         // choice scratch
    std::vector<std::uint8_t> seen_;                     // choice scratch
    std::vector<htm::UInt>    selected_;    // final bit set (sorted)

    std::vector<double>       gather_;      // li: qualifying neighbor counts
    std::vector<std::int64_t> candIdx_;     // di: candidate bit ids
    std::vector<double>       candCnt_;     // di: candidate counts

    /* Build counts_ / touched_ / activeIdx_ / activeCnt_ from inputs.
     * Returns total bit count of the (shared) input shape. */
    std::int64_t accumulate_counts(const std::vector<const htm::SDR *> &inputs);
    void clear_counts();  // zero only touched_ entries

    /* Sample `n_sample` indices from activeIdx_ (or a candidate list) with
     * probabilities `probs`, write the chosen GLOBAL bit ids into selected_
     * (sorted), and set them on `out`. */
    void sample_and_set(const std::vector<std::int64_t> &candidates,
                        const double *probs, std::int64_t n_sample,
                        const MergeParams &params, htm::SDR &out);

    /* -- per-mode implementations (ports of the sdr.py functions) --------- */
    void mode_union(htm::SDR &out);
    void mode_intersection(std::size_t n_inputs, htm::SDR &out);
    void mode_symdiff(std::size_t n_inputs, htm::SDR &out);
    void mode_nos(htm::SDR &out);
    void mode_concat(const std::vector<const htm::SDR *> &inputs, int axis,
                     htm::SDR &out);
    void mode_stack(const std::vector<const htm::SDR *> &inputs, htm::SDR &out);

    void mode_su (std::int64_t total_bits, const MergeParams &p, htm::SDR &out);
    void mode_aps(std::int64_t total_bits, const MergeParams &p, htm::SDR &out);
    void mode_pls(std::int64_t total_bits, const MergeParams &p, htm::SDR &out);
    void mode_ssu(std::int64_t total_bits, const MergeParams &p, htm::SDR &out);
    void mode_li (std::int64_t total_bits, std::size_t n_inputs,
                  const MergeParams &p, htm::SDR &out);
    void mode_bws(std::int64_t total_bits, std::size_t n_inputs,
                  const MergeParams &p, htm::SDR &out);
    void mode_pcb(std::int64_t total_bits, std::size_t n_inputs,
                  const MergeParams &p, htm::SDR &out);
    void mode_di (std::int64_t total_bits, const MergeParams &p, htm::SDR &out);
};

} // namespace pyramid
