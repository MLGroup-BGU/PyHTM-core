/* ---------------------------------------------------------------------------
 * Pyramid-Model-Core :: merge/SdrMerge.hpp
 *
 * The pyramid's one merge: concatenation along axis 0.
 *
 *   * An L0 node's input is the concatenation of its features' encodings.
 *   * A parent's input is the concatenation of its children's outputs (their
 *     active columns), in children order.
 *
 * Every input keeps its own slice of the result, so equal inputs always give
 * an equal merged SDR and no input can overwrite another. The shape rule is
 * numpy.concatenate's: every input has the same rank and the same size on
 * every axis except axis 0, and the output's size on axis 0 is the sum of the
 * inputs' sizes there (a 1024-bit RDSE next to a date encoder of another
 * width is fine). This is what htm.core's SDR::concatenate and NuPIC's
 * MultiEncoder do.
 *
 * A single input is copied as it is (the runtime does that itself).
 *
 * THREADING: one SdrConcat per thread (it owns index scratch); the free
 * function is pure.
 * ------------------------------------------------------------------------ */
#pragma once

#include <htm/types/Sdr.hpp>

#include <vector>

namespace pyramid {

/* Output dims of concatenating inputs with `input_dims` along axis 0. */
std::vector<htm::UInt> concat_output_dims(
    const std::vector<std::vector<htm::UInt>> &input_dims);

class SdrConcat {
public:
    /* Concatenate `inputs` (>= 2) into `out`, which must already carry
     * concat_output_dims of the inputs' dims (checked). */
    void concat(const std::vector<const htm::SDR *> &inputs, htm::SDR &out);

private:
    std::vector<htm::UInt> selected_;   // output bit ids (sorted)
};

} // namespace pyramid
