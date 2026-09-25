/**
 * @file graph_schema.h
 * @brief Zero-Code Graph Definition Schema
 *
 * Defines the serializable structure for a model's computation graph.
 * This allows models to be defined via config (JSON/Protobuf) rather than C++ code.
 */

#ifndef DENSECORE_GRAPH_SCHEMA_H
#define DENSECORE_GRAPH_SCHEMA_H

#include "densecore/hal/operation_graph.h"
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace densecore {
namespace graph {

using ParamValue = std::variant<int, float, bool, std::string, std::vector<int>, std::vector<float>>;

/**
 * @brief Configuration for a single node in the graph
 */
struct NodeConfig {
    std::string name;                  ///< Unique node name (e.g. "layer_0_attn_softmax")
    OpType op;                         ///< Operation type (e.g. Softmax)
    std::vector<std::string> inputs;   ///< Input tensor names
    std::vector<std::string> outputs;  ///< Output tensor names

    ///< Operation-specific parameters (e.g. axis, epsilon, num_heads)
    std::unordered_map<std::string, ParamValue> params;

    ///< Optional: Condition for conditional execution (e.g. "arch_flags.requires_q_norm")
    std::string condition;
};

/**
 * @brief Complete Graph Configuration
 */
struct GraphConfig {
    std::string name;
    std::vector<std::string> inputs;   ///< Graph-level inputs
    std::vector<std::string> outputs;  ///< Graph-level outputs
    std::vector<NodeConfig> nodes;     ///< Topological list of nodes

    // TODO: Subgraphs for control flow (if needed later)
};

}  // namespace graph
}  // namespace densecore

#endif  // DENSECORE_GRAPH_SCHEMA_H
