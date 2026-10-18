/* ---------------------------------------------------------------------------
 * Pyramid-Model-Core :: spec/PyramidSpec.hpp
 *
 * The "instruction book": everything the C++ pyramid runtime needs to build
 * and run, fully resolved.  The Python side (htm_source/pipeline/
 * pyramid_spec.py) assembles this from the YAML configs using the SAME code
 * paths the Python pyramid uses (get_layer_config / get_node_config /
 * banker's rounding / final feature and model seeds / the
 * predictedSegmentDecrement formula), so parameter resolution is identical by
 * construction. What the pyramid fixes by design is not in the spec at all:
 * merges are concatenation (merge/SdrMerge.hpp), every node is an SP followed
 * by a TM whose output is its active columns, and the fixed SP / TM constants
 * live in HtmNode.  The C++ side receives only final numbers; the only values
 * derived here are the SAMPLE-dependent encoder parameters (computed during
 * the build pass with the bit-exact NumpyCompat math).
 *
 * Everything is plain data -- no pybind types -- so the runtime core stays
 * independent of Python; the bindings layer translates the Python dict into
 * these structs.
 * ------------------------------------------------------------------------ */
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "../encoders/EncoderBuild.hpp"

namespace pyramid {

/* The configurable SpatialPooler parameters of one node (post
 * get_layer_config + get_node_config). The fixed ones (potential radius,
 * inhibition, boosting, stimulus threshold, wrap-around, duty cycles) are set
 * in HtmNode. */
struct SpSpec {
    std::vector<htm::UInt> columnDimensions;
    double potentialPct = 0.0;
    double synPermInactiveDec = 0.0;
    double synPermActiveInc = 0.0;
    double synPermConnected = 0.0;
    double localAreaDensity = 0.0;
};

/* The configurable TemporalMemory parameters of one node; the connected
 * permanence is fixed in HtmNode. predictedSegmentDecrement arrives resolved
 * (2 * permanenceInc * localAreaDensity, computed by the spec builder). */
struct TmSpec {
    std::int64_t cellsPerColumn = 0;
    std::int64_t activationThreshold = 0;
    double initialPerm = 0.0;
    std::int64_t minThreshold = 0;
    std::int64_t newSynapseCount = 0;
    double permanenceInc = 0.0;
    double permanenceDec = 0.0;
    double predictedSegmentDecrement = 0.0;
    std::int64_t maxSegmentsPerCell = 0;
    std::int64_t maxSynapsesPerSegment = 0;
};

/* One pyramid node: a Spatial Pooler followed by a Temporal Memory. */
struct NodeSpec {
    std::string name;
    std::uint64_t seed = 0;               // pyramid_seed * model_counter
    SpSpec sp;
    TmSpec tm;
    std::vector<htm::UInt> input_dims;    // resolved by the Python builder
    /* model_summary() line, preformatted by the Python spec builder. */
    std::string summary;
};

/* Where the records come from. */
enum class DataMode : std::uint8_t { NativeCsv, PyReader };

struct DataSpec {
    DataMode mode = DataMode::PyReader;
    std::string path;                     // dataset path ('' for DataFrame)
    std::vector<std::string> columns;     // X column order (row layout)
    std::vector<int> column_is_dt;        // 1 = datetime column
    std::vector<std::string> dt_formats;  // per-column strptime format ('' = n/a)
    std::optional<std::int64_t> row_start, row_stop;   // iloc[start:stop] window
    double train_percent = 1.0;
    std::int64_t batch_rows = 8192;
};

/* The full instruction book. */
struct PyramidSpec {
    std::uint64_t seed = 0;
    std::vector<std::vector<std::string>> layers;          // layer -> node names
    std::map<std::string, std::vector<std::string>> children;   // predecessors order
    std::map<std::string, std::vector<std::string>> feature_plan;// L0 -> features
    std::map<std::string, NodeSpec> nodes;
    bool record_signals = false;          // general.record_signals (per-node signals)
    std::vector<FeatureSpec> encoders;    // features_cfg order (seed order)
    DataSpec data;
    std::optional<std::int64_t> max_workers;
    bool multiprocess = true;             // false -> single-thread run path
    std::int64_t pipeline_depth = 0;      // 0 = auto (K=clamp(n_workers));
                                          // 1 = off; >1 = explicit K override
    bool progress = true;
};

} // namespace pyramid