/* ---------------------------------------------------------------------------
 * Pyramid-Model-Core :: merge/SdrMerge.cpp
 * Concatenation along axis 0; see SdrMerge.hpp.
 * ------------------------------------------------------------------------ */
#include "SdrMerge.hpp"

#include <htm/utils/Log.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace pyramid {

using htm::SDR;
using htm::UInt;

std::vector<UInt> concat_output_dims(
        const std::vector<std::vector<UInt>> &input_dims) {
    NTA_CHECK(!input_dims.empty());
    const auto &d0 = input_dims[0];
    const int ndim = static_cast<int>(d0.size());
    NTA_CHECK(ndim >= 1) << "concat: zero-rank SDR";
    std::uint64_t along = 0;
    for (const auto &d : input_dims) {
        NTA_CHECK(d.size() == d0.size())
            << "All shapes must have same number of dimensions";
        for (int k = 1; k < ndim; ++k)
            NTA_CHECK(d[static_cast<std::size_t>(k)] == d0[static_cast<std::size_t>(k)])
                << "Shapes must be equal in all axes except the "
                << "concatenated axis, got different shapes for axis " << k;
        along += d[0];
    }
    NTA_CHECK(along <= static_cast<std::uint64_t>(std::numeric_limits<UInt>::max()))
        << "concat: output too large";
    auto out = d0;
    out[0] = static_cast<UInt>(along);
    return out;
}

/* Validates the runtime inputs against the preallocated output without
 * allocating (called once per merge). */
static void check_concat_shapes(const std::vector<const SDR *> &inputs,
                                const SDR &out) {
    const auto &d0 = inputs[0]->dimensions;
    const int ndim = static_cast<int>(d0.size());
    NTA_CHECK(ndim >= 1) << "concat: zero-rank SDR";
    NTA_CHECK(out.dimensions.size() == d0.size())
        << "concat: output rank != input rank";
    std::uint64_t along = 0;
    for (const SDR *in : inputs) {
        const auto &d = in->dimensions;
        NTA_CHECK(d.size() == d0.size())
            << "All shapes must have same number of dimensions";
        for (int k = 1; k < ndim; ++k)
            NTA_CHECK(d[static_cast<std::size_t>(k)] == out.dimensions[static_cast<std::size_t>(k)])
                << "Shapes must be equal in all axes except the "
                << "concatenated axis, got different shapes for axis " << k;
        along += d[0];
    }
    NTA_CHECK(along == out.dimensions[0])
        << "concat: output size on axis 0 (" << out.dimensions[0]
        << ") != sum of input sizes (" << along << ")";
}

void SdrConcat::concat(const std::vector<const SDR *> &inputs, SDR &out) {
    NTA_CHECK(inputs.size() >= 2) << "concat needs at least two inputs";
    check_concat_shapes(inputs, out);
    const int ndim = static_cast<int>(inputs[0]->dimensions.size());
    const auto &out_dims = out.dimensions;

    /* Row-major strides of the output shape. */
    std::int64_t in_stride[16], out_stride[16];
    NTA_CHECK(ndim <= 16) << "SDR rank too large";
    out_stride[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; --d)
        out_stride[d] = out_stride[d + 1] * static_cast<std::int64_t>(out_dims[static_cast<std::size_t>(d + 1)]);

    selected_.clear();
    std::int64_t off = 0;                     // running offset along axis 0
    for (std::size_t j = 0; j < inputs.size(); ++j) {
        const auto &in_dims = inputs[j]->dimensions;
        in_stride[ndim - 1] = 1;
        for (int d = ndim - 2; d >= 0; --d)
            in_stride[d] = in_stride[d + 1] * static_cast<std::int64_t>(in_dims[static_cast<std::size_t>(d + 1)]);
        for (UInt f : inputs[j]->getSparse()) {
            std::int64_t rem = static_cast<std::int64_t>(f), of = 0;
            for (int d = 0; d < ndim; ++d) {
                std::int64_t coord = rem / in_stride[d];
                rem -= coord * in_stride[d];
                if (d == 0) coord += off;
                of += coord * out_stride[d];
            }
            selected_.push_back(static_cast<UInt>(of));
        }
        off += static_cast<std::int64_t>(in_dims[0]);
    }
    std::sort(selected_.begin(), selected_.end());
    out.setSparse(selected_);
}

} // namespace pyramid
