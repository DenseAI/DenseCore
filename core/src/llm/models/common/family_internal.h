#ifndef DENSECORE_LLM_MODELS_COMMON_FAMILY_INTERNAL_H
#define DENSECORE_LLM_MODELS_COMMON_FAMILY_INTERNAL_H

#include <string_view>

struct TransformerModel;

namespace densecore {
namespace llm {
namespace models {

bool IsHybridSSMQkvWeightName(const char* weight_name);
bool ShouldForcePlainGgmlForHybridSSMQkv();
bool IsGemma4SharedKVSourceLayer(const TransformerModel* model, int layer_idx);
void ResetHybridSSMQkvForceGgmlCache();

}  // namespace models
}  // namespace llm
}  // namespace densecore

#endif  // DENSECORE_LLM_MODELS_COMMON_FAMILY_INTERNAL_H
