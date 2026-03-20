/**
 * @file llm_config_generator.h
 * @brief Generates GraphConfig for Decoder-only Transformer models (LLMs)
 */

#ifndef DENSECORE_LLM_CONFIG_GENERATOR_H
#define DENSECORE_LLM_CONFIG_GENERATOR_H

#include "densecore/graph/graph_schema.h"
#include "model_types.h"  // For TransformerModel

namespace densecore {

class LlmConfigGenerator {
public:
    static graph::GraphConfig Generate(const TransformerModel* model);

private:
    static void AddLayer(graph::GraphConfig& config, const TransformerModel* model, int layer_idx,
                         const std::string& input_name, std::string& output_name);
};

}  // namespace densecore

#endif  // DENSECORE_LLM_CONFIG_GENERATOR_H
