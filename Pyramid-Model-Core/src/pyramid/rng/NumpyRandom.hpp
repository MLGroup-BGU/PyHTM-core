/* ---------------------------------------------------------------------------
 * Pyramid-Model-Core :: rng/NumpyRandom.hpp
 *
 * A bit-exact C++ port of the exact slice of NumPy's random stack that the
 * PyHTM pyramid consumes on its hot path:
 *
 *     np.random.SeedSequence(int_seed)      -> SeedSequence
 *     np.random.PCG64(seed_sequence)        -> Pcg64
 *     np.random.Generator.random(k)         -> NumpyGenerator::random_fill
 *     np.random.Generator.choice(n, size=k, replace=False, p=probs)
 *                                           -> NumpyGenerator::choice_no_replace_p
 *
 * WHY THIS EXISTS
 *   The stochastic SDR-merge modes (su / aps / pls / ssu / li / bws / pcb /
 *   di) in PyHTM's Python implementation draw their sampled bits from
 *   `np.random.default_rng(seed)` -- a fresh PCG64 generator per merge call.
 *   For the C++ pyramid runtime to be output-faithful to the Python pyramid,
 *   the SAME seed must select the SAME set of bits.  That requires
 *   reproducing NumPy's seeding, its raw bit stream, its double generation,
 *   and its weighted sampling-without-replacement algorithm exactly.
 *
 * FAITHFULNESS
 *   Every algorithm below is a line-for-line port of the corresponding NumPy
 *   source (numpy/random/_bit_generator.pyx, pcg64.c/.h, _generator.pyx as of
 *   NumPy 2.x; the algorithms are unchanged since NumPy 1.17).
 *
 * THREAD-SAFETY
 *   Objects here are cheap value types with no global state; each merge call
 *   constructs its own generator (exactly as the Python code constructs a
 *   fresh default_rng per call), so no synchronization is ever needed.
 * ------------------------------------------------------------------------ */
#pragma once

#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace pyramid {

/* ---------------------------------------------------------------------------
 * U128 -- minimal portable unsigned 128-bit integer.
 *
 * PCG64 needs 128-bit state arithmetic (mul + add).  GCC/Clang provide
 * __uint128_t; MSVC (the user's Windows build) does not, so a two-limb
 * portable implementation is used everywhere for identical behavior across
 * compilers.  Only the operations PCG64 needs are implemented.
 * ------------------------------------------------------------------------ */
struct U128 {
    std::uint64_t hi = 0;
    std::uint64_t lo = 0;

    constexpr U128() = default;
    constexpr U128(std::uint64_t h, std::uint64_t l) : hi(h), lo(l) {}

    // 64x64 -> 128 multiply (portable schoolbook on 32-bit halves).
    static U128 mul64(std::uint64_t a, std::uint64_t b) {
        const std::uint64_t a_lo = static_cast<std::uint32_t>(a);
        const std::uint64_t a_hi = a >> 32;
        const std::uint64_t b_lo = static_cast<std::uint32_t>(b);
        const std::uint64_t b_hi = b >> 32;

        const std::uint64_t p0 = a_lo * b_lo;
        const std::uint64_t p1 = a_lo * b_hi;
        const std::uint64_t p2 = a_hi * b_lo;
        const std::uint64_t p3 = a_hi * b_hi;

        const std::uint64_t mid = p1 + (p0 >> 32) + static_cast<std::uint32_t>(p2);
        U128 r;
        r.lo = (mid << 32) | static_cast<std::uint32_t>(p0);
        r.hi = p3 + (mid >> 32) + (p2 >> 32);
        return r;
    }

    // 128x128 multiply keeping the LOW 128 bits (what PCG's LCG step needs).
    U128 mul_lo(const U128 &o) const {
        U128 r = mul64(lo, o.lo);                 // full lo*lo
        r.hi += lo * o.hi + hi * o.lo;            // cross terms (low 64 only)
        return r;
    }

    U128 add(const U128 &o) const {
        U128 r;
        r.lo = lo + o.lo;
        r.hi = hi + o.hi + (r.lo < lo ? 1u : 0u); // carry
        return r;
    }

    U128 shl1() const {  // left shift by 1 (used to build the PCG increment)
        return U128((hi << 1) | (lo >> 63), lo << 1);
    }
};

/* ---------------------------------------------------------------------------
 * SeedSequence -- port of np.random.SeedSequence for integer entropy.
 *
 * PyHTM only ever seeds with a single Python int (the merge_params['seed']),
 * so only the int-entropy path is ported: the entropy int is decomposed into
 * little-endian uint32 words (at least one word), mixed into a 4-word pool,
 * and generate_state() hashes uint64 outputs from the pool.  Constants and
 * loop structure are verbatim from numpy/random/bit_generator.pyx.
 * ------------------------------------------------------------------------ */
class SeedSequence {
public:
    explicit SeedSequence(std::uint64_t entropy) {
        // _int_to_uint32_array: little-endian 32-bit words, >= 1 word.
        std::vector<std::uint32_t> words;
        if (entropy == 0) {
            words.push_back(0u);
        } else {
            std::uint64_t v = entropy;
            while (v > 0) {
                words.push_back(static_cast<std::uint32_t>(v & 0xffffffffu));
                v >>= 32;
            }
        }
        mix_entropy(words);
    }

    // generate_state(n_words, dtype=np.uint64): 2*n uint32 words, paired
    // little-endian into uint64 (word0 = low half). Verbatim hashing loop.
    std::vector<std::uint64_t> generate_state_u64(std::size_t n_words) const {
        const std::size_t n32 = n_words * 2;
        std::vector<std::uint32_t> out32(n32);
        std::uint32_t hash_const = INIT_B;
        for (std::size_t i_dst = 0; i_dst < n32; ++i_dst) {
            std::uint32_t data_val = pool_[i_dst % POOL_SIZE];
            data_val ^= hash_const;
            hash_const *= MULT_B;
            data_val *= hash_const;
            data_val ^= data_val >> XSHIFT;
            out32[i_dst] = data_val;
        }
        std::vector<std::uint64_t> out(n_words);
        for (std::size_t i = 0; i < n_words; ++i) {
            out[i] = static_cast<std::uint64_t>(out32[2 * i]) |
                     (static_cast<std::uint64_t>(out32[2 * i + 1]) << 32);
        }
        return out;
    }

private:
    static constexpr std::size_t POOL_SIZE = 4;   // numpy DEFAULT_POOL_SIZE
    static constexpr std::uint32_t XSHIFT = 16;   // 32 bits / 2
    static constexpr std::uint32_t INIT_A = 0x43b0d7e5u;
    static constexpr std::uint32_t MULT_A = 0x931e8875u;
    static constexpr std::uint32_t INIT_B = 0x8b51f9ddu;
    static constexpr std::uint32_t MULT_B = 0x58f38dedu;
    static constexpr std::uint32_t MIX_MULT_L = 0xca01f9ddu;
    static constexpr std::uint32_t MIX_MULT_R = 0x4973f715u;

    std::uint32_t pool_[POOL_SIZE] = {0, 0, 0, 0};

    void mix_entropy(const std::vector<std::uint32_t> &entropy) {
        std::uint32_t hash_const = INIT_A;

        auto hashmix = [&hash_const](std::uint32_t value) -> std::uint32_t {
            value ^= hash_const;
            hash_const *= MULT_A;
            value *= hash_const;
            value ^= value >> XSHIFT;
            return value;
        };
        auto mix = [](std::uint32_t x, std::uint32_t y) -> std::uint32_t {
            // numpy: result = (MIX_MULT_L * x - MIX_MULT_R * y), uint32 wrap.
            std::uint32_t result = MIX_MULT_L * x - MIX_MULT_R * y;
            result ^= result >> XSHIFT;
            return result;
        };

        // Seed the pool from the first POOL_SIZE entropy words (0-padded).
        for (std::size_t i = 0; i < POOL_SIZE; ++i)
            pool_[i] = hashmix(i < entropy.size() ? entropy[i] : 0u);

        // Full cross-mix of the pool with itself.
        for (std::size_t i_src = 0; i_src < POOL_SIZE; ++i_src)
            for (std::size_t i_dst = 0; i_dst < POOL_SIZE; ++i_dst)
                if (i_src != i_dst)
                    pool_[i_dst] = mix(pool_[i_dst], hashmix(pool_[i_src]));

        // Fold in any entropy words beyond the pool size.
        for (std::size_t i_src = POOL_SIZE; i_src < entropy.size(); ++i_src)
            for (std::size_t i_dst = 0; i_dst < POOL_SIZE; ++i_dst)
                pool_[i_dst] = mix(pool_[i_dst], hashmix(entropy[i_src]));
    }
};

/* ---------------------------------------------------------------------------
 * Pcg64 -- port of numpy's PCG64 (pcg_setseq_128 with the XSL-RR output).
 * Seeded from a SeedSequence exactly as numpy does (4 uint64 words: the first
 * pair is the 128-bit initstate high|low, the second pair the initseq).
 * ------------------------------------------------------------------------ */
class Pcg64 {
public:
    explicit Pcg64(const SeedSequence &ss) {
        const auto w = ss.generate_state_u64(4);
        // PCG_128BIT_CONSTANT(high, low): w[0]=state hi, w[1]=state lo, etc.
        seed(U128(w[0], w[1]), U128(w[2], w[3]));
    }
    explicit Pcg64(std::uint64_t int_seed) : Pcg64(SeedSequence(int_seed)) {}

    // One raw 64-bit draw: step the LCG, then XSL-RR the NEW state.
    std::uint64_t next64() {
        step();
        const unsigned rot = static_cast<unsigned>(state_.hi >> 58); // state>>122
        const std::uint64_t xored = state_.hi ^ state_.lo;
        return rotr64(xored, rot);
    }

    // numpy next_double: top 53 bits scaled to [0, 1).
    double next_double() {
        return static_cast<double>(next64() >> 11) * (1.0 / 9007199254740992.0);
    }

private:
    // PCG_DEFAULT_MULTIPLIER_128 = 0x2360ed051fc65da44385df649fccf645
    static constexpr std::uint64_t MULT_HI = 0x2360ed051fc65da4ull;
    static constexpr std::uint64_t MULT_LO = 0x4385df649fccf645ull;

    U128 state_{};
    U128 inc_{};

    void seed(const U128 &initstate, const U128 &initseq) {
        state_ = U128(0, 0);
        inc_ = initseq.shl1();
        inc_.lo |= 1u;
        step();
        state_ = state_.add(initstate);
        step();
    }

    void step() { state_ = state_.mul_lo(U128(MULT_HI, MULT_LO)).add(inc_); }

    static std::uint64_t rotr64(std::uint64_t v, unsigned rot) {
        return (v >> rot) | (v << ((-static_cast<int>(rot)) & 63));
    }
};

/* ---------------------------------------------------------------------------
 * NumpyGenerator -- the Generator-level algorithms PyHTM's merges use.
 * ------------------------------------------------------------------------ */
class NumpyGenerator {
public:
    // Seeded generator: identical stream to np.random.default_rng(seed).
    explicit NumpyGenerator(std::uint64_t seed) : bitgen_(seed) {}
    // Unseeded (matches Python's seed=None semantics: fresh OS entropy per
    // construction; by definition non-reproducible, exactly like Python).
    NumpyGenerator() : bitgen_(entropy_seed()) {}

    double random() { return bitgen_.next_double(); }

    void random_fill(double *out, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) out[i] = bitgen_.next_double();
    }

    /* Generator.choice(pop_size, size=k, replace=False, p=probs)
     *
     * Verbatim port of numpy's cdf-resampling loop (numpy _generator.pyx):
     *   - validates p (length, NaN, Kahan-sum within sqrt(eps) of 1)
     *   - repeatedly: draw uniforms, zero already-found probabilities,
     *     sequential cumsum, normalize by the last element, searchsorted
     *     side='right', keep first occurrences in draw order.
     * Output order matches numpy's (draw order of first occurrences); the
     * merge layer only uses the SET of indices, but order is kept faithful.
     *
     * `p` is taken by value-copy internally (numpy copies too, then mutates).
     * Scratch buffers are caller-provided to keep the merge hot path
     * allocation-free at steady state.
     */
    void choice_no_replace_p(std::int64_t pop_size, std::int64_t k,
                             const double *p, std::vector<std::int64_t> &out,
                             std::vector<double> &p_scratch,
                             std::vector<double> &cdf_scratch,
                             std::vector<double> &x_scratch,
                             std::vector<std::int64_t> &new_scratch,
                             std::vector<std::uint8_t> &seen_scratch) {
        if (k > pop_size)
            throw std::invalid_argument(
                "Cannot take a larger sample than population when replace is False");
        validate_p(p, pop_size);
        // numpy: if np.count_nonzero(p > 0) < size -> ValueError
        std::int64_t nonzero = 0;
        for (std::int64_t i = 0; i < pop_size; ++i)
            if (p[i] > 0.0) ++nonzero;
        if (nonzero < k)
            throw std::invalid_argument("Fewer non-zero entries in p than size");

        out.resize(static_cast<std::size_t>(k));
        p_scratch.assign(p, p + pop_size);          // numpy: p = p.copy()
        cdf_scratch.resize(static_cast<std::size_t>(pop_size));
        seen_scratch.assign(static_cast<std::size_t>(pop_size), 0u);

        std::int64_t n_uniq = 0;
        while (n_uniq < k) {
            const std::size_t need = static_cast<std::size_t>(k - n_uniq);
            x_scratch.resize(need);
            random_fill(x_scratch.data(), need);    // x = self.random((k-n_uniq,))

            if (n_uniq > 0)                          // p[found[:n_uniq]] = 0
                for (std::int64_t i = 0; i < n_uniq; ++i)
                    p_scratch[static_cast<std::size_t>(out[static_cast<std::size_t>(i)])] = 0.0;

            // cdf = np.cumsum(p)  (strictly sequential float64 accumulation)
            double acc = 0.0;
            for (std::size_t i = 0; i < cdf_scratch.size(); ++i) {
                acc += p_scratch[i];
                cdf_scratch[i] = acc;
            }
            // cdf /= cdf[-1]
            const double last = cdf_scratch.back();
            for (double &c : cdf_scratch) c /= last;

            // new = cdf.searchsorted(x, side='right')
            new_scratch.resize(need);
            for (std::size_t i = 0; i < need; ++i)
                new_scratch[i] = searchsorted_right(cdf_scratch, x_scratch[i]);

            // np.unique(new, return_index=True) -> first occurrences, then
            // unique_indices.sort() + take == first occurrences in draw order.
            for (std::size_t i = 0; i < need && n_uniq < k; ++i) {
                const std::int64_t v = new_scratch[i];
                std::uint8_t &flag = seen_scratch[static_cast<std::size_t>(v)];
                if (!flag) {
                    flag = 1u;
                    out[static_cast<std::size_t>(n_uniq++)] = v;
                }
            }
            // (numpy re-derives duplicates via p-zeroing on the next pass;
            //  the seen_ flags implement the identical "first occurrence"
            //  selection without re-sorting.)
        }
    }

private:
    Pcg64 bitgen_;

    static std::uint64_t entropy_seed() {
        std::random_device rd;
        return (static_cast<std::uint64_t>(rd()) << 32) ^ rd();
    }

    static std::int64_t searchsorted_right(const std::vector<double> &cdf, double x) {
        // upper_bound: first index with cdf[idx] > x.
        std::size_t lo = 0, hi = cdf.size();
        while (lo < hi) {
            const std::size_t mid = (lo + hi) / 2;
            if (cdf[mid] <= x) lo = mid + 1; else hi = mid;
        }
        return static_cast<std::int64_t>(lo);
    }

    static void validate_p(const double *p, std::int64_t n) {
        // numpy: kahan_sum, NaN check, |sum-1| <= sqrt(eps).
        double sum = p[0], c = 0.0;
        for (std::int64_t i = 1; i < n; ++i) {
            const double y = p[i] - c;
            const double t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        if (sum != sum)  // NaN
            throw std::invalid_argument("probabilities contain NaN");
        const double atol = 1.4901161193847656e-08;  // sqrt(float64 eps)
        if (sum > 1.0 + atol || sum < 1.0 - atol)
            throw std::invalid_argument("probabilities do not sum to 1");
        for (std::int64_t i = 0; i < n; ++i)
            if (p[i] < 0.0)
                throw std::invalid_argument("probabilities are not non-negative");
    }
};

} // namespace pyramid
