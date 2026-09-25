#include "llm/models/common/family_internal.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string_view>

namespace densecore::models {
bool IsGemma4SlidingLayer(const TransformerModel* model, int layer_idx);
int Gemma4KVSourceLayer(const TransformerModel* model, int layer_idx);
}  // namespace densecore::models

namespace densecore {
namespace llm {
namespace models {
namespace {

bool Contains(std::string_view value, std::string_view needle) {
    return value.find(needle) != std::string_view::npos;
}

bool IsTruthyEnv(const char* value) {
    return value && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

std::atomic<int> g_force_plain_ggml_cache{-1};
std::mutex g_force_plain_ggml_mu;

}  // namespace

bool IsHybridSSMQkvWeightName(const char* weight_name) {
    if (!weight_name || weight_name[0] == '\0') {
        return false;
    }
    const std::string_view name(weight_name);
    return Contains(name, "attn_qkv") || Contains(name, "qkv_mixed") || Contains(name, "linear_attn.in_proj_qkv") ||
           Contains(name, "attn_gate") || Contains(name, "in_proj_z") || name == "z" || Contains(name, ".z") ||
           Contains(name, "ssm_out") || Contains(name, "linear_attn.out_proj");
}

bool ShouldForcePlainGgmlForHybridSSMQkv() {
    int cached = g_force_plain_ggml_cache.load(std::memory_order_acquire);
    if (cached >= 0) {
        return cached == 1;
    }

    std::lock_guard<std::mutex> lock(g_force_plain_ggml_mu);
    cached = g_force_plain_ggml_cache.load(std::memory_order_relaxed);
    if (cached >= 0) {
        return cached == 1;
    }

    const bool force = IsTruthyEnv(std::getenv("DENSECORE_HYBRID_SSM_QKV_FORCE_GGML"));
    g_force_plain_ggml_cache.store(force ? 1 : 0, std::memory_order_release);
    return force;
}

bool IsGemma4SharedKVSourceLayer(const TransformerModel* model, int layer_idx) {
    return densecore::models::Gemma4KVSourceLayer(model, layer_idx) == layer_idx;
}

void ResetHybridSSMQkvForceGgmlCache() {
    std::lock_guard<std::mutex> lock(g_force_plain_ggml_mu);
    g_force_plain_ggml_cache.store(-1, std::memory_order_release);
}

}  // namespace models
}  // namespace llm
}  // namespace densecore
