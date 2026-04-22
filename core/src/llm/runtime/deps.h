#ifndef DENSECORE_LLM_RUNTIME_DEPS_H
#define DENSECORE_LLM_RUNTIME_DEPS_H

#include "densecore/runtime/inference.h"

namespace densecore::llm::runtime {

const InferenceConfig& ResolveInferenceConfig(const BatchSpec* batch);
densecore::HardwareTopology& ResolveHardwareTopology(const BatchSpec* batch);
densecore::BackendRegistry& ResolveBackendRegistry(const BatchSpec* batch);
densecore::DeviceType ResolvePreferredDevice(const BatchSpec* batch);
densecore::DeviceType ResolvePreferredMatmulDevice(const BatchSpec* batch);
densecore::DeviceType ResolvePreferredAttentionDevice(const BatchSpec* batch);
densecore::DeviceType ResolvePreferredNormDevice(const BatchSpec* batch);
bool IsMixedRoutingEnabled(const BatchSpec* batch);

}  // namespace densecore::llm::runtime

#endif  // DENSECORE_LLM_RUNTIME_DEPS_H
