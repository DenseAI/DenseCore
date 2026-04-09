/**
 * @file model_graph_bridge.h
 * @brief Bridge connecting loaded model weights to graph builders
 *
 * After model loading, ModelGraphBridge registers weight-aware graph builders
 * with the GraphRegistry. This enables VIT, SIGLIP, and Whisper models to
 * execute through the Universal Graph Execution path.
 *
 * Usage (called from model_loader.cpp):
 * @code
 *   TransformerModel* model = LoadModel(path);
 *   ModelGraphBridge::RegisterFromModel(model);  // Registers builders with weights
 *   // Now ExecuteGraph("vit_base", inputs) will use real weights
 * @endcode
 *
 * Copyright (c) 2025 DenseCore Authors
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DENSECORE_MODELS_MODEL_GRAPH_BRIDGE_H
#define DENSECORE_MODELS_MODEL_GRAPH_BRIDGE_H

#include <string>

// Forward declarations
#include "model_types.h"

namespace densecore {

/**
 * @brief Mapping from ModelArch to graph name for the registry
 */
class ModelGraphBridge {
public:
    /**
     * @brief Register graph builders for a loaded model
     *
     * Inspects the model type and registers appropriate builders:
     * - VIT/CLIP/SIGLIP: Registers "vit_base", "siglip", etc.
 * - Whisper: Registers config-driven "whisper_encoder"/"whisper_decoder"
     *
     * @param model Loaded model with vision/whisper weights
     * @return true if registration succeeded
     */
    static bool RegisterFromModel(const TransformerModel* model);

    /**
     * @brief Check if a model architecture supports graph execution
     *
     * @param arch Model architecture enum
     * @return true if graph execution is supported
     */
    static bool IsGraphModel(ModelArch arch);

    /**
     * @brief Check if a loaded model instance supports graph execution
     *
     * This is more precise than the enum-only overload for architecture
     * families like GEMMA where Gemma4 requires a specialized inline graph.
     *
     * @param model Loaded model instance
     * @return true if graph execution is supported for this specific model
     */
    static bool IsGraphModel(const TransformerModel* model);

    /**
     * @brief Get the graph name for a model architecture
     *
     * @param arch Model architecture enum
     * @return Graph name for registry (e.g., "vit_base")
     */
    static const char* GetGraphName(ModelArch arch);

    /**
     * @brief Get the graph name for a specific loaded model instance
     *
     * Some architecture families share an enum but require different graph
     * paths depending on model flags. This overload keeps those variants from
     * being advertised as generic graph-compatible.
     *
     * @param model Loaded model instance
     * @return Graph name for registry, or nullptr if no generic graph applies
     */
    static const char* GetGraphName(const TransformerModel* model);

    /**
     * @brief Check if a loaded model has whisper weights registered for graph execution
     */
    static bool SupportsWhisperGraph(const TransformerModel* model) {
        return model && model->has_whisper && model->whisper_model != nullptr;
    }

    /**
     * @brief Unregister builders for a model (cleanup on model unload)
     *
     * @param model Model to unregister
     */
    static void UnregisterFromModel(const TransformerModel* model);

private:
    // Register vision encoder builder with actual weights
    static bool RegisterVisionBuilder(const TransformerModel* model);

    // Register config-driven whisper builders with actual weights
    static bool RegisterWhisperBuilder(const TransformerModel* model);

    // Register LLM builder with actual weights
    static bool RegisterLlmBuilder(const TransformerModel* model);

    // Register operation-template builders for non-LLM domains
    static bool RegisterUniversalTemplateBuilders(const TransformerModel* model);
};

}  // namespace densecore

#endif  // DENSECORE_MODELS_MODEL_GRAPH_BRIDGE_H
