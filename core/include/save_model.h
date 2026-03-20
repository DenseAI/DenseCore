#ifndef DENSECORE_SAVE_MODEL_H
#define DENSECORE_SAVE_MODEL_H

#include <string>

#include "model_types.h"

namespace densecore {

/**
 * @brief Save a TransformerModel to GGUF format.
 *
 * This function writes the model weights and metadata to a GGUF file,
 * enabling persistence of pruned/quantized models.
 *
 * @param model The model to save
 * @param output_path Path to the output GGUF file
 * @return true if successful, false otherwise
 */
bool SaveGGUFModel(const TransformerModel& model, const std::string& output_path);

/**
 * @brief Progress callback for long-running save operations.
 */
using SaveProgressCallback = void (*)(int current, int total, const char* message);

/**
 * @brief Save a TransformerModel to GGUF format with progress reporting.
 *
 * @param model The model to save
 * @param output_path Path to the output GGUF file
 * @param callback Progress callback (can be nullptr)
 * @return true if successful, false otherwise
 */
bool SaveGGUFModelWithProgress(const TransformerModel& model, const std::string& output_path,
                               SaveProgressCallback callback);

}  // namespace densecore

#endif  // DENSECORE_SAVE_MODEL_H
