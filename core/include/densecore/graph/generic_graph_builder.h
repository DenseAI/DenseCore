/**
 * @file generic_graph_builder.h
 * @brief Builder that constructs OperationGraph from GraphConfig schema
 */

#ifndef DENSECORE_GENERIC_GRAPH_BUILDER_H
#define DENSECORE_GENERIC_GRAPH_BUILDER_H

#include "densecore/graph/graph_schema.h"
#include "densecore/hal/operation_graph.h"
#include "densecore/hal/tensor.h"
#include <memory>
#include <unordered_map>
#include <vector>

namespace densecore {
namespace graph {

class GenericGraphBuilder {
public:
    /**
     * @brief Build an OperationGraph from a config and input tensors
     * 
     * @param config The declarative graph configuration
     * @param inputs Map of input name to actual Tensor object
     * @return Constructed OperationGraph
     */
    static std::unique_ptr<OperationGraph> Build(const GraphConfig& config,
                                                 const std::unordered_map<std::string, Tensor>& inputs);

private:
    // Helper to convert schema params to OpParams variant
    static OpParams ConvertParams(OpType op, const std::unordered_map<std::string, ParamValue>& params);
};

}  // namespace graph
}  // namespace densecore

#endif  // DENSECORE_GENERIC_GRAPH_BUILDER_H
