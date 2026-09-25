#ifndef DENSECORE_LLM_GRAPH_PLANNING_H
#define DENSECORE_LLM_GRAPH_PLANNING_H

#include "densecore/runtime/inference.h"

namespace densecore::llm::graph {

TransformerGraphExecutionPlan ResolveExecutionPlan(const TransformerModel* model);
std::unique_ptr<TransformerGraphBuilder> InstantiateRegistryBuilder(const TransformerGraphExecutionPlan& plan,
                                                                    std::string* error_reason);

}  // namespace densecore::llm::graph

#endif  // DENSECORE_LLM_GRAPH_PLANNING_H
