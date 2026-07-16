/* ---------------------------------------------------------------------------
 * Pyramid-Model-Core :: math/NumpyCompat.hpp
 *
 * Float64 primitives that reproduce NumPy's results bit-for-bit, for the
 * places where the Python pyramid's numbers flow through NumPy reductions:
 *
 *   np.ndarray.sum()          -> pairwise_sum      (numpy's pairwise blocks)
 *   np.cumsum                 -> cumsum            (strictly sequential)
 *   np.searchsorted 'right'   -> searchsorted_right
 *   np.linspace               -> linspace
 *   np.quantile  (median_unbiased) -> quantile_median_unbiased
 *   np.percentile (linear)         -> percentile_linear
 *
 * WHY BIT-EXACTNESS MATTERS HERE
 *   The stochastic merge modes normalize probabilities with `.sum()` and the
 *   sampled bit set depends on every last ulp of those probabilities (they
 *   feed a cumsum + searchsorted draw).  Likewise the sample-derived encoder
 *   parameters (adaptive quantile knots, deduced RDSE resolutions) shape the
 *   entire encoding space.  Reproducing NumPy's exact summation ORDER and
 *   quantile interpolation removes any drift between the Python pyramid and
 *   this runtime.
 *
 * SOURCES
 *   pairwise_sum: numpy/_core/src/umath/loops_utils.h.src (verbatim port of
 *   the scalar 8-accumulator pairwise algorithm; numpy's SIMD variants were
 *   designed to preserve this exact ordering).
 *   quantile machinery: numpy/lib/_function_base_impl.py (_compute_virtual_
 *   index, _get_indexes, _get_gamma, _lerp).
 *   All verified value-for-value by verification/pyramid/test_numpy_math.py.
 *
 * THREAD-SAFETY: pure functions; no state.
 * ------------------------------------------------------------------------ */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace pyramid {

/* np.ndarray.sum() for contiguous float64 -- numpy's pairwise summation.
 * Block of 8 accumulators for n <= 128, recursive halving (rounded down to a
 * multiple of 8) above.  The result is the value numpy's add.reduce returns
 * (numpy then adds the 0.0 identity, a no-op for finite inputs). */
inline double pairwise_sum(const double *a, std::size_t n) {
    if (n < 8) {
        double res = -0.0;   // preserve -0.0 for all -0.0 inputs (numpy comment)
        for (std::size_t i = 0; i < n; ++i) res += a[i];
        return res;
    } else if (n <= 128) {   // PW_BLOCKSIZE
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        std::size_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) +
                     ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    } else {
        std::size_t n2 = n / 2;
        n2 -= n2 % 8;
        return pairwise_sum(a, n2) + pairwise_sum(a + n2, n - n2);
    }
}

inline double pairwise_sum(const std::vector<double> &v) {
    return pairwise_sum(v.data(), v.size());
}

/* np.cumsum -- strictly sequential accumulation (numpy add.accumulate). */
inline void cumsum(const double *in, double *out, std::size_t n) {
    double acc = 0.0;
    for (std::size_t i = 0; i < n; ++i) { acc += in[i]; out[i] = acc; }
}

/* np.searchsorted(side='right'): first index where a[idx] > x. */
inline std::size_t searchsorted_right(const double *a, std::size_t n, double x) {
    std::size_t lo = 0, hi = n;
    while (lo < hi) {
        const std::size_t mid = (lo + hi) / 2;
        if (a[mid] <= x) lo = mid + 1; else hi = mid;
    }
    return lo;
}

/* np.linspace(start, stop, num) with endpoint=True -- numpy's exact form:
 * y[i] = i * ((stop-start)/(num-1)) + start, then y[num-1] forced to stop. */
inline std::vector<double> linspace(double start, double stop, std::size_t num) {
    std::vector<double> y(num);
    if (num == 0) return y;
    if (num == 1) { y[0] = start; return y; }
    const double step = (stop - start) / static_cast<double>(num - 1);
    for (std::size_t i = 0; i < num; ++i)
        y[i] = static_cast<double>(i) * step + start;
    y[num - 1] = stop;
    return y;
}

/* numpy _lerp: a + (b-a)*t, switched to b - (b-a)*(1-t) when t >= 0.5
 * (numerically symmetric interpolation; matters for exactness). */
inline double np_lerp(double a, double b, double t) {
    const double diff = b - a;
    double r = a + diff * t;
    if (t >= 0.5) r = b - diff * (1.0 - t);
    return r;
}

/* Shared quantile core over an ALREADY SORTED ascending sample, given the
 * method's virtual index (numpy's bounds fixup + _lerp). */
inline double quantile_sorted_virt(const double *sorted, std::size_t n,
                                   double virt) {
    const double nn = static_cast<double>(n);

    std::size_t prev, next;
    if (virt >= nn - 1.0) {          // above bounds -> last element
        prev = next = n - 1;
    } else if (virt < 0.0) {         // below bounds -> first element
        prev = next = 0;
    } else {
        prev = static_cast<std::size_t>(std::floor(virt));
        next = prev + 1;
    }
    const double gamma = virt - std::floor(virt);
    return np_lerp(sorted[prev], sorted[next], gamma);
}

/* np.quantile(arr, q, method='median_unbiased'): numpy computes the virtual
 * index through _compute_virtual_index with alpha = beta = 1/3:
 *     virt = n*q + (alpha + q*(1 - alpha - beta)) - 1
 * (this exact expression and operation order -- it is NOT floating-point
 * equivalent to the simplified (n + 1/3)*q - 2/3). */
inline std::vector<double> quantile_median_unbiased(std::vector<double> data,
                                                    const std::vector<double> &qs) {
    std::sort(data.begin(), data.end());
    const double nn = static_cast<double>(data.size());
    constexpr double a = 1.0 / 3.0;
    std::vector<double> out(qs.size());
    for (std::size_t i = 0; i < qs.size(); ++i) {
        const double q = qs[i];
        const double virt = nn * q + (a + q * (1.0 - a - a)) - 1.0;
        out[i] = quantile_sorted_virt(data.data(), data.size(), virt);
    }
    return out;
}

/* np.percentile(arr, q) with the default 'linear' method: numpy special-cases
 * this method's virtual index as literally (n - 1) * quantile (again, the
 * exact expression matters -- the generic alpha=beta=1 formula rounds
 * differently). q is in PERCENT, divided by 100 first as numpy does. */
inline double percentile_linear(std::vector<double> data, double q_percent) {
    std::sort(data.begin(), data.end());
    const double q = q_percent / 100.0;
    const double virt = (static_cast<double>(data.size()) - 1.0) * q;
    return quantile_sorted_virt(data.data(), data.size(), virt);
}

} // namespace pyramid
