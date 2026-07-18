/* ---------------------------------------------------------------------------
 * Pyramid-Model-Core :: spec/PyramidSpec.hpp
 *
 * The "instruction book": everything the C++ pyramid runtime needs to build
 * and run, fully resolved.  The Python side (htm_source/pipeline/
 * pyramid_spec.py) assembles this from the YAML configs using the SAME code
 * paths the Python pyramid uses today (get_layer_config / get_node_config /
 * potentialRadius rounding / feature weights & banker's rounding / final
 * feature and model seeds), so parameter resolution is identical by
 * construction.  The C++ side receives only final numbers; the only values
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
#include "../merge/SdrMerge.hpp"

namespace pyramid {

/* Fully-resolved SpatialPooler parameters for one node (post
 * get_layer_config + get_node_config + potentialRadius rounding). */
struct SpSpec {
    std::vector<htm::UInt> columnDimensions;
    double potentialPct = 0.0;
    std::int64_t potentialRadius = 0;   // already round(pr * prod(input_dims))
    bool globalInhibition = true;
    double synPermInactiveDec = 0.0;
    double stimulusThreshold = 0.0;
    double synPermActiveInc = 0.0;
    double synPermConnected = 0.0;
    double boostStrength = 0.0;
    double localAreaDensity = 0.0;
    bool wrapAround = true;
};

/* Fully-resolved TemporalMemory parameters for one node. */
struct TmSpec {
    std::int64_t cellsPerColumn = 0;
    std::int64_t activationThreshold = 0;
    double initialPerm = 0.0;
    double permanenceConnected = 0.0;
    std::int64_t minThreshold = 0;
    std::int64_t newSynapseCount = 0;
    double permanenceInc = 0.0;
    double permanenceDec = 0.0;
    double predictedSegmentDecrement = 0.0;
    std::int64_t maxSegmentsPerCell = 0;
    std::int64_t maxSynapsesPerSegment = 0;
};

/* One pyramid node (== one HTMModule). */
struct NodeSpec {
    std::string name;
    std::uint64_t seed = 0;               // pyramid_seed * model_counter
    bool has_sp = true;
    SpSpec sp;
    TmSpec tm;
    std::vector<htm::UInt> input_dims;    // resolved by the Python builder
    std::int64_t max_pool = 1;            // per-layer value, already selected
    bool flatten = true;
    std::optional<std::int64_t> learn_schedule;
    bool anomaly_score = true;
    bool input_predictive = false;        // use_predictive && layer > 0
    bool return_predictive = false;       // use_predictive
    bool anomaly_detector = false;
    int residual_mode = 1;                // 1 = subtract, 2 = xor
    double residual_strength = 1.0;
    /* model_summary() line, preformatted by the Python spec builder with the
     * ORIGINAL property logic (tuple vs ndarray reprs and all); when empty,
     * HtmNode::summary() falls back to a C++ approximation (debug only). */
    std::string summary;
};

/* Merge configuration (both merge levels). */
struct MergeSpec {
    MergeMode htm_mode = MergeMode::Union;
    MergeParams htm_params;               // DEFAULT_MERGE_PARAMS overlaid
    bool has_feature_mode = true;         // false == streamer mode None
    MergeMode feature_mode = MergeMode::Union;
    MergeParams feature_params;           // SU streamer: user params;
                                          // plain streamer: pure defaults
    int concat_axis = 0;
};

/* Run-loop feature flags (ModelPyramid extras). */
struct RunFlags {
    bool use_predictive = false;
    bool residual = false;
    bool broadcast_head = false;
    bool recurrent_head = false;
    bool anomaly_boost = false;
    bool long_term_memory = false;
    std::int64_t memory_inertia = 5;
    bool lateral_exchange = false;
    bool squeeze_skip = false;            // FeatureSUSqueezeSkipPyramid.run
};

/* Where the records come from. */
enum class DataMode : std::uint8_t { NativeCsv, PyReader };

struct DataSpec {
    DataMode mode = DataMode::PyReader;
    std::string path;                     // dataset path ('' for DataFrame)
    std::vector<std::string> columns;     // X column order (row layout)
    std::vector<int> column_is_dt;        // 1 = datetime column
    std::vector<std::string> dt_formats;  // per-column strptime format ('' = n/a)
    std::optional<std::int64_t> row_start, row_stop, row_step;   // iloc slice
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
    MergeSpec merge;
    RunFlags flags;
    std::vector<FeatureSpec> encoders;    // features_cfg order (seed order)
    DataSpec data;
    std::optional<std::int64_t> max_workers;
    bool multiprocess = true;             // false -> single-thread run path
    std::int64_t pipeline_depth = 0;      // 0 = auto (K=clamp(n_workers));
                                          // 1 = off; >1 = explicit K override
    bool progress = true;
};

} // namespace pyramid