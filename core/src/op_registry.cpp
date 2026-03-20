/**
 * @file op_registry.cpp
 * @brief OpRegistry Implementation - Runtime Vendor Plugin System
 *
 * This file is part of DenseCore Reference Implementation.
 * Licensed under Apache 2.0 (Open Source) or Commercial License.
 */

#include "densecore/hal/op_registry.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <unordered_set>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

namespace densecore {

namespace {

bool ShouldLogRegistryRegistration() {
    const char* env = std::getenv("DENSECORE_LOG_OP_REGISTRY");
    if (!env) {
        return false;
    }
    return std::string(env) == "1";
}

}  // namespace

// Static member initialization
std::atomic<bool> OpRegistry::initialized_{false};
std::unique_ptr<OpRegistry> OpRegistry::test_instance_;

std::vector<std::function<void(OpRegistry&)>>& OpRegistry::GetGlobalRegistrars() {
    static std::vector<std::function<void(OpRegistry&)>> registrars;
    return registrars;
}

void OpRegistry::AddGlobalRegistrar(std::function<void(OpRegistry&)> registrar) {
    GetGlobalRegistrars().push_back(registrar);
}

OpRegistry::OpRegistry() {
    for (const auto& registrar : GetGlobalRegistrars()) {
        registrar(*this);
    }
}

OpRegistry& OpRegistry::Instance() {
    // Return test instance if set (for dependency injection in tests)
    if (test_instance_) {
        return *test_instance_;
    }

    static OpRegistry instance;
    if (!initialized_.load(std::memory_order_acquire)) {
        initialized_.store(true, std::memory_order_release);
    }
    return instance;
}

void OpRegistry::SetTestInstance(std::unique_ptr<OpRegistry> instance) {
    test_instance_ = std::move(instance);
}

void OpRegistry::ResetToGlobalInstance() {
    test_instance_.reset();
}

bool OpRegistry::HasTestInstance() {
    return test_instance_ != nullptr;
}

bool OpRegistry::IsInitialized() {
    return initialized_.load(std::memory_order_acquire);
}

void OpRegistry::Init() {
    // Force initialization
    (void)Instance();
}

void OpRegistry::RegisterImpl(OpType op, DeviceType device, std::shared_ptr<DenseCoreOp> impl) {
    if (!impl) {
        std::cerr << "[OpRegistry] Warning: Attempted to register null implementation for " << OpTypeName(op)
                  << std::endl;
        return;
    }

    Key key{op, device};

    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        registry_[key].push_back(std::move(impl));
    }

    // Invalidate cache for this key (new registration may change "best")
    {
        std::unique_lock<std::shared_mutex> cache_lock(cache_mutex_);
        for (auto it = best_cache_.begin(); it != best_cache_.end();) {
            if (it->first.op == op) {
                it = best_cache_.erase(it);
            } else {
                ++it;
            }
        }
    }

    if (ShouldLogRegistryRegistration()) {
        std::cerr << "[OpRegistry] Registered " << OpTypeName(op) << " for " << DeviceTypeName(device) << std::endl;
    }
}

DenseCoreOp* OpRegistry::Get(OpType op, DeviceType device) {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    Key key{op, device};

    auto it = registry_.find(key);
    if (it == registry_.end() || it->second.empty()) {
        return nullptr;
    }

    // Return the first registered kernel (simple lookup)
    return it->second.front().get();
}

DenseCoreOp* OpRegistry::GetBest(OpType op, DeviceType preferred_device) {
    Key key{op, preferred_device};

    // Fast path: check cache first
    {
        std::shared_lock<std::shared_mutex> cache_lock(cache_mutex_);
        auto it = best_cache_.find(key);
        if (it != best_cache_.end()) {
            RecordDispatchTelemetry(op, preferred_device, it->second.resolved_device, true);
            return it->second.op;
        }
    }

    // Slow path: compute best and cache
    SelectionCriteria criteria;
    ResolveResult result;
    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        result = ResolveBestLocked(op, preferred_device, criteria);
    }

    // Cache the result (including nullptr + resolved device)
    {
        std::unique_lock<std::shared_mutex> cache_lock(cache_mutex_);
        best_cache_[key] = CachedBest{result.op, result.resolved_device};
    }

    RecordDispatchTelemetry(op, preferred_device, result.resolved_device, false);
    return result.op;
}

static bool MatchesCriteria(const DenseCoreOp* op, const OpRegistry::SelectionCriteria& criteria) {
    if (!op) {
        return false;
    }

    if (criteria.dtype != DType::UNKNOWN && !op->SupportsDType(criteria.dtype)) {
        return false;
    }

    if (criteria.layout != TensorLayout::UNKNOWN && !op->SupportsLayout(criteria.layout)) {
        return false;
    }

    if (criteria.batch > 0) {
        const OpCapabilities caps = op->GetCapabilities();
        if (caps.max_batch_size > 0 && static_cast<size_t>(criteria.batch) > caps.max_batch_size) {
            return false;
        }
    }

    return true;
}

static std::vector<DeviceType> BuildFallbackOrder(DeviceType preferred_device) {
    std::vector<DeviceType> ordered;
    ordered.reserve(4);

    switch (preferred_device) {
    case DeviceType::CPU: ordered.push_back(DeviceType::CPU); break;
    case DeviceType::NPU:
        ordered.push_back(DeviceType::NPU);
        ordered.push_back(DeviceType::METAL);
        ordered.push_back(DeviceType::CPU);
        break;
    case DeviceType::METAL:
        ordered.push_back(DeviceType::METAL);
        ordered.push_back(DeviceType::NPU);
        ordered.push_back(DeviceType::CPU);
        break;
    case DeviceType::ASIC:
        ordered.push_back(DeviceType::ASIC);
        ordered.push_back(DeviceType::NPU);
        ordered.push_back(DeviceType::METAL);
        ordered.push_back(DeviceType::CPU);
        break;
    default:
        ordered.push_back(preferred_device);
        ordered.push_back(DeviceType::CPU);
        break;
    }

    std::unordered_set<uint8_t> seen;
    std::vector<DeviceType> deduped;
    deduped.reserve(ordered.size());
    for (DeviceType d : ordered) {
        const uint8_t key = static_cast<uint8_t>(d);
        if (seen.insert(key).second) {
            deduped.push_back(d);
        }
    }

    return deduped;
}

OpRegistry::ResolveResult OpRegistry::ResolveBestLocked(OpType op, DeviceType preferred_device,
                                                        const SelectionCriteria& criteria) const {
    auto find_best_on_device = [&](DeviceType device) -> DenseCoreOp* {
        Key key{op, device};
        auto it = registry_.find(key);
        if (it == registry_.end() || it->second.empty()) {
            return nullptr;
        }

        auto& kernels = it->second;
        auto best = std::max_element(kernels.begin(), kernels.end(), [&](const auto& a, const auto& b) {
            const bool a_ok = MatchesCriteria(a.get(), criteria);
            const bool b_ok = MatchesCriteria(b.get(), criteria);
            if (a_ok != b_ok) {
                return !a_ok && b_ok;
            }
            return a->GetCapabilities().priority < b->GetCapabilities().priority;
        });
        if (best != kernels.end() && MatchesCriteria(best->get(), criteria)) {
            return best->get();
        }
        return nullptr;
    };

    const std::vector<DeviceType> fallback_order = BuildFallbackOrder(preferred_device);
    for (DeviceType device : fallback_order) {
        if (DenseCoreOp* best = find_best_on_device(device)) {
            return ResolveResult{best, device};
        }
    }

    return ResolveResult{};
}

DenseCoreOp* OpRegistry::GetBest(OpType op, DeviceType preferred_device, const SelectionCriteria& criteria) {
    ResolveResult result;
    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        result = ResolveBestLocked(op, preferred_device, criteria);
    }
    RecordDispatchTelemetry(op, preferred_device, result.resolved_device, false);
    return result.op;
}

const std::vector<OpType>& OpRegistry::GetRequiredOpsForProfile(InferenceProfile profile) {
    return RequiredOpsForProfile(profile);
}

OpRegistry::AdmissionReport OpRegistry::CheckProfileAdmission(InferenceProfile profile, DeviceType preferred_device,
                                                              const AdmissionPolicy& policy) const {
    AdmissionReport report;
    report.requested_device = preferred_device;
    const auto& required = RequiredOpsForProfile(profile);

    std::unordered_set<uint8_t> seen_ops;
    for (OpType op : required) {
        const uint8_t op_key = static_cast<uint8_t>(op);
        if (policy.unique_ops_only && !seen_ops.insert(op_key).second) {
            continue;
        }

        report.examined_ops++;
        SelectionCriteria criteria;
        ResolveResult result;
        {
            std::shared_lock<std::shared_mutex> lock(mutex_);
            result = ResolveBestLocked(op, preferred_device, criteria);
        }
        if (!result.op) {
            report.missing_ops.push_back(op);
            continue;
        }
        if (result.resolved_device != preferred_device) {
            report.used_fallback = true;
            report.fallback_ops.push_back(op);
        }
    }

    report.admitted = report.missing_ops.empty() && (policy.allow_fallback || report.fallback_ops.empty());
    return report;
}

OpRegistry::AdmissionReport OpRegistry::CheckProfileAdmission(InferenceProfile profile,
                                                              DeviceType preferred_device) const {
    AdmissionPolicy policy;
    return CheckProfileAdmission(profile, preferred_device, policy);
}

OpRegistry::AdmissionReport OpRegistry::CheckGraphAdmission(const OperationGraph& graph, DeviceType preferred_device,
                                                            const AdmissionPolicy& policy) const {
    AdmissionReport report;
    report.requested_device = preferred_device;

    std::unordered_set<uint8_t> seen_ops;
    for (const GraphNode& node : graph.Nodes()) {
        const uint8_t op_key = static_cast<uint8_t>(node.op);
        if (policy.unique_ops_only && !seen_ops.insert(op_key).second) {
            continue;
        }

        report.examined_ops++;
        SelectionCriteria criteria;
        if (!node.inputs.empty() && node.inputs[0] < graph.TensorCount()) {
            const Tensor& t = graph.GetTensor(node.inputs[0]);
            criteria.dtype = t.dtype;
            criteria.layout = t.layout;
            if (t.ndim > 0) {
                criteria.batch = t.shape[0];
                criteria.shape.assign(t.shape.begin(), t.shape.begin() + static_cast<size_t>(t.ndim));
            }
        }

        ResolveResult result;
        {
            std::shared_lock<std::shared_mutex> lock(mutex_);
            result = ResolveBestLocked(node.op, preferred_device, criteria);
        }
        if (!result.op) {
            report.missing_ops.push_back(node.op);
            continue;
        }
        if (result.resolved_device != preferred_device) {
            report.used_fallback = true;
            report.fallback_ops.push_back(node.op);
        }
    }

    report.admitted = report.missing_ops.empty() && (policy.allow_fallback || report.fallback_ops.empty());
    return report;
}

OpRegistry::AdmissionReport OpRegistry::CheckGraphAdmission(const OperationGraph& graph,
                                                            DeviceType preferred_device) const {
    AdmissionPolicy policy;
    return CheckGraphAdmission(graph, preferred_device, policy);
}

void OpRegistry::RecordDispatchTelemetry(OpType op, DeviceType requested_device, DeviceType resolved_device,
                                         bool cache_hit) const {
    total_requests_.fetch_add(1, std::memory_order_relaxed);
    if (cache_hit) {
        total_cache_hits_.fetch_add(1, std::memory_order_relaxed);
    }

    if (resolved_device == DeviceType::UNKNOWN) {
        total_misses_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    if (resolved_device != requested_device) {
        total_fallbacks_.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(telemetry_mutex_);
        TelemetryKey key{op, requested_device, resolved_device};
        fallback_counts_[key]++;
    }
}

OpRegistry::DispatchTelemetrySnapshot OpRegistry::GetDispatchTelemetry() const {
    DispatchTelemetrySnapshot snapshot;
    snapshot.total_requests = total_requests_.load(std::memory_order_relaxed);
    snapshot.cache_hits = total_cache_hits_.load(std::memory_order_relaxed);
    snapshot.misses = total_misses_.load(std::memory_order_relaxed);
    snapshot.fallbacks = total_fallbacks_.load(std::memory_order_relaxed);

    std::lock_guard<std::mutex> lock(telemetry_mutex_);
    snapshot.fallback_records.reserve(fallback_counts_.size());
    for (const auto& [key, count] : fallback_counts_) {
        snapshot.fallback_records.push_back(FallbackTelemetryRecord{
            .op = key.op,
            .requested_device = key.requested_device,
            .resolved_device = key.resolved_device,
            .count = count,
        });
    }
    std::sort(snapshot.fallback_records.begin(), snapshot.fallback_records.end(),
              [](const FallbackTelemetryRecord& a, const FallbackTelemetryRecord& b) {
                  if (a.count != b.count) {
                      return a.count > b.count;
                  }
                  if (a.op != b.op) {
                      return static_cast<uint8_t>(a.op) < static_cast<uint8_t>(b.op);
                  }
                  if (a.requested_device != b.requested_device) {
                      return static_cast<uint8_t>(a.requested_device) < static_cast<uint8_t>(b.requested_device);
                  }
                  return static_cast<uint8_t>(a.resolved_device) < static_cast<uint8_t>(b.resolved_device);
              });
    return snapshot;
}

void OpRegistry::ResetDispatchTelemetry() {
    total_requests_.store(0, std::memory_order_relaxed);
    total_cache_hits_.store(0, std::memory_order_relaxed);
    total_misses_.store(0, std::memory_order_relaxed);
    total_fallbacks_.store(0, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(telemetry_mutex_);
    fallback_counts_.clear();
}

std::vector<std::string> OpRegistry::ListRegistered() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    std::vector<std::string> result;
    result.reserve(registry_.size());

    for (const auto& [key, impls] : registry_) {
        std::ostringstream oss;
        oss << OpTypeName(key.op) << "@" << DeviceTypeName(key.device) << " (" << impls.size() << " impl(s))";
        result.push_back(oss.str());
    }

    return result;
}

DeviceType OpRegistry::DetectDeviceType() {
    if (const char* env = std::getenv("DENSECORE_PREFERRED_DEVICE")) {
        std::string value(env);
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        if (value == "CPU") return DeviceType::CPU;
        if (value == "METAL") return DeviceType::METAL;
        if (value == "NPU") return DeviceType::NPU;
        if (value == "ASIC") return DeviceType::ASIC;
        // AUTO or unknown value -> runtime detection path.
    }

    // Runtime hardware detection
    // Needs more sophisticated logic in real environment

#if defined(__APPLE__)
#if TARGET_OS_OSX || TARGET_OS_IOS
    // Apple Silicon (M1-M4) or iOS
    return DeviceType::METAL;
#else
    return DeviceType::CPU;
#endif
#elif defined(__ANDROID__) || defined(ANDROID)
    // Android - Qualcomm/Mali GPU etc.
    // Need to check Vulkan/OpenCL support in reality
    return DeviceType::CPU;  // default
#elif defined(__linux__)
    // Linux x86/ARM (including AWS Graviton): CPU path by default.
    return DeviceType::CPU;
#else
    return DeviceType::CPU;
#endif
}

}  // namespace densecore
