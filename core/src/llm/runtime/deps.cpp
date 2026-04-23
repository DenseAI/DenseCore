#include "llm/runtime/deps.h"

#include "densecore/backend/hardware_topology.h"
#include "densecore/hal/backend_registry.h"
#include "llm/config/runtime_config.h"

const InferenceConfig& densecore::llm::runtime::ResolveInferenceConfig(const BatchSpec* batch) {
    if (batch && batch->deps && batch->deps->config) {
        return *batch->deps->config;
    }
    return InferenceConfig::Instance();
}

densecore::HardwareTopology& densecore::llm::runtime::ResolveHardwareTopology(const BatchSpec* batch) {
    if (batch && batch->deps && batch->deps->hardware_topology) {
        return *batch->deps->hardware_topology;
    }
    return densecore::HardwareTopology::GetInstance();
}

densecore::BackendRegistry& densecore::llm::runtime::ResolveBackendRegistry(const BatchSpec* batch) {
    if (batch && batch->deps && batch->deps->backend_registry) {
        return *batch->deps->backend_registry;
    }
    return densecore::BackendRegistry::Instance();
}

const densecore::llm::config::FastPathRuntimeConfig&
densecore::llm::runtime::ResolveFastPathRuntimeConfig(const BatchSpec* batch) {
    if (batch && batch->deps && batch->deps->fast_path_config) {
        return *batch->deps->fast_path_config;
    }
    thread_local densecore::llm::config::FastPathRuntimeConfig config;
    config = densecore::llm::config::LoadFastPathRuntimeConfig();
    return config;
}

densecore::DeviceType densecore::llm::runtime::ResolvePreferredDevice(const BatchSpec* batch) {
    if (batch && batch->deps) {
        return batch->deps->preferred_device;
    }
    return densecore::DeviceType::CPU;
}

densecore::DeviceType densecore::llm::runtime::ResolvePreferredMatmulDevice(const BatchSpec* batch) {
    if (batch && batch->deps) {
        return batch->deps->preferred_matmul_device;
    }
    return densecore::DeviceType::CPU;
}

densecore::DeviceType densecore::llm::runtime::ResolvePreferredAttentionDevice(const BatchSpec* batch) {
    if (batch && batch->deps) {
        return batch->deps->preferred_attention_device;
    }
    return densecore::DeviceType::CPU;
}

densecore::DeviceType densecore::llm::runtime::ResolvePreferredNormDevice(const BatchSpec* batch) {
    if (batch && batch->deps) {
        return batch->deps->preferred_norm_device;
    }
    return densecore::DeviceType::CPU;
}

bool densecore::llm::runtime::IsMixedRoutingEnabled(const BatchSpec* batch) {
    if (batch && batch->deps) {
        return batch->deps->mixed_operation_routing;
    }
    return false;
}
