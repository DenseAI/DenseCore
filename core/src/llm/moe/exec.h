#pragma once
#include <cstdint>
#include <string>

void ResetMoECallbackEntryCount();
uint64_t GetMoECallbackEntryCount();
uint64_t GetMoECallbackMissingUserdataCount();
uint64_t GetMoECallbackMissingBackendCount();
uint64_t GetMoECallbackMissingExpertsCount();
uint64_t GetMoECallbackRoutingFailureCount();
uint64_t GetMoECallbackEmptyRoutingCount();
uint64_t GetMoECallbackFailClosedCount();
void ResetMoEStrictFailureState();
bool ConsumeMoEStrictFailureState(std::string* message);

bool IsMoEDebugLoggingEnabled();

struct ggml_tensor;
struct MoEUserData;
namespace densecore::moe {
struct MoERouteResult;
}
bool RouteMoESoftmaxTopK(const struct ggml_tensor* gate_logits, const MoEUserData* ud,
                         densecore::moe::MoERouteResult* routing);
bool RouteMoEGroupedSigmoid(const struct ggml_tensor* gate_logits, const MoEUserData* ud,
                            densecore::moe::MoERouteResult* routing);
bool RouteMoEGemma4TopK(const struct ggml_tensor* gate_logits, const MoEUserData* ud,
                        densecore::moe::MoERouteResult* routing);
