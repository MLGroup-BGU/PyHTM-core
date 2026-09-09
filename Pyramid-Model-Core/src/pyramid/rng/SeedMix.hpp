/* ---------------------------------------------------------------------
 * PyHTM-core -- Pyramid-Model-Core
 *
 * Counter-based seeding for the stochastic SDR merge modes.
 * ------------------------------------------------------------------- */

/** @file
Derivation of the per-call random seed used by the sparsity-controlled merge
modes (`su`, `aps`, `pls`, `ssu`, `li`, `bws`, `pcb`, `di`).

Those modes keep a fixed budget of `target_sparsity * total_bits` bits and draw
that many from the candidate set.  The draw needs a random stream, and WHICH
stream a given call gets is what this header decides.

A single scalar seed cannot serve that role.  The merge is a pure function of
its inputs, so one scalar means one stream for every call in the run: a node
whose input repeats emits the identical output bits every time, and the
n - budget bits that lose the draw lose it permanently.  Drawing fresh OS
entropy per call avoids that but gives up reproducibility, and makes the result
depend on which thread reached the call first.

The seed is therefore derived from four independent components:

    seed_for_call = mix(base_seed, node_key, slot, iteration)

    base_seed  -- `htm_merge_params.seed` / `feature_merge_params.seed`.
                  Chosen by the experimenter; identical across a run.
    node_key   -- WHERE in the pyramid. Derived from the node's layer and
                  index (see node_key_from_name).
    slot       -- WHICH merge, for nodes that perform more than one in the
                  same record (see MergeSlot).
    iteration  -- WHICH record. The index of the record being processed.

Every component is a static property of the configuration and the record
index, so the derived seed does not depend on thread count, scheduling, or the
order in which nodes are visited.  Two runs of the same configuration with the
same base seed produce byte-identical output; two records, two nodes, or two
merges within one node-step each get an independent stream.

`base_seed == none` keeps the previous meaning: fresh OS entropy per call,
deliberately non-reproducible, for confirming that a result is not an artefact
of one seed.
*/

#ifndef PYRAMID_RNG_SEEDMIX_HPP
#define PYRAMID_RNG_SEEDMIX_HPP

#include <cstdint>
#include <string>

namespace pyramid {

/** SplitMix64 finalizer.  Full avalanche: neighbouring inputs (t, t+1) give
 *  streams with no usable correlation, which is what lets the iteration
 *  component be a plain counter. */
inline std::uint64_t mix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

/** Pack (layer, node) into one 64-bit key.
 *
 *  The two indices occupy disjoint 32-bit halves before mixing, so the key is
 *  collision-free for any layer/node count -- unlike a decimal or textual
 *  concatenation, where L1_12 and L11_2 would collide. */
inline std::uint64_t node_key(std::uint64_t layer, std::uint64_t node) {
    return mix64(((layer & 0xFFFFFFFFULL) << 32) | (node & 0xFFFFFFFFULL));
}

/** Node key from the naming convention `L{layer}_{node}_{timescale}`.
 *
 *  The timescale is deliberately NOT part of the key: the same structural
 *  position must hash identically whether it runs as `_short` or `_long`, so
 *  that architectures which differ only in timescale assignment remain
 *  comparable.
 *
 *  Names that do not follow the convention fall back to the positional
 *  indices the runtime already holds, which keeps the scheme total for
 *  architectures that name their nodes differently. */
inline std::uint64_t node_key_from_name(const std::string &name,
                                        std::size_t fallback_layer,
                                        std::size_t fallback_pos) {
    if (name.size() >= 4 && (name[0] == 'L' || name[0] == 'l')) {
        std::size_t i = 1, layer = 0;
        bool have_layer = false;
        for (; i < name.size() && name[i] >= '0' && name[i] <= '9'; ++i) {
            layer = layer * 10 + static_cast<std::size_t>(name[i] - '0');
            have_layer = true;
        }
        if (have_layer && i < name.size() && name[i] == '_') {
            ++i;
            std::size_t node = 0;
            bool have_node = false;
            for (; i < name.size() && name[i] >= '0' && name[i] <= '9'; ++i) {
                node = node * 10 + static_cast<std::size_t>(name[i] - '0');
                have_node = true;
            }
            /* Anything after the node number (the timescale suffix, or
             * nothing) is ignored on purpose. */
            if (have_node)
                return node_key(static_cast<std::uint64_t>(layer),
                                static_cast<std::uint64_t>(node));
        }
    }
    return node_key(static_cast<std::uint64_t>(fallback_layer),
                    static_cast<std::uint64_t>(fallback_pos));
}

/** Which merge a node is performing, for the record it is on.
 *
 *  Several flags let one node run more than one stochastic merge per record;
 *  without this component those merges would share a stream and their draws
 *  would be locked together.  The values are fixed constants tied to the call
 *  site in the source -- never a runtime counter, which would make the result
 *  depend on execution order and destroy reproducibility. */
enum class MergeSlot : std::uint32_t {
    FeatureMerge     = 0,   // L0: feature encoders -> node input
    Children         = 1,   // standard parent <- children merge
    ChildrenPred     = 2,   // use_predictive: the predictive half
    SqueezeSkip      = 3,   // squeeze_skip: the skip-connection merge
    BroadcastHead    = 4,   // broadcast_head: head <- every lower node
    LateralExchange  = 5,   // lateral_exchange between siblings
    RecurrentHead    = 6,   // recurrent_head: head <- previous head output
    MemoryUpdate     = 7,   // long_term_memory: memory := memory x k + current
    MemoryApply      = 8,   // long_term_memory: current := current + memory
};

/** The three per-call components, assembled by the runtime at each merge. */
struct MergeCall {
    std::uint64_t node_key  = 0;
    std::int64_t  iteration = 0;
    MergeSlot     slot      = MergeSlot::Children;
};

/** Fold the four components into the seed handed to the generator.
 *
 *  Each component is mixed in its own round so that a change in any one of
 *  them re-randomises the whole stream. */
inline std::uint64_t derive_merge_seed(std::uint64_t base_seed,
                                       const MergeCall &call) {
    std::uint64_t s = mix64(base_seed ^ call.node_key);
    s = mix64(s ^ (static_cast<std::uint64_t>(call.slot) * 0xC2B2AE3D27D4EB4FULL));
    s = mix64(s ^ (static_cast<std::uint64_t>(call.iteration) *
                   0xD1B54A32D192ED03ULL));
    return s;
}

}  // namespace pyramid

#endif  // PYRAMID_RNG_SEEDMIX_HPP
