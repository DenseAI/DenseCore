#pragma once

#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include "densecore/exceptions.h"
#include "densecore/runtime/resource_budget.h"

namespace densecore::runtime {
namespace resource_detail {
struct Ledger {
    std::mutex mutex;
    std::unordered_map<const void*, RuntimeMemoryPlan> plans;
    uint64_t graph_used = 0;
    uint64_t repack_used = 0;
};
inline Ledger& ProcessLedger() {
    // Process lifetime: static repack cache deleters may run during DSO teardown.
    static Ledger* ledger = new Ledger;
    return *ledger;
}
inline uint64_t Add(uint64_t a, uint64_t b) { return b > UINT64_MAX-a ? UINT64_MAX : a+b; }
inline uint64_t PlannedBytesLocked(const Ledger& ledger) {
    uint64_t total = 0;
    for (const auto& entry : ledger.plans) total = Add(total,entry.second.RequiredBytes());
    return total;
}
}
inline uint64_t ProcessPlannedMemoryBytes() {
    auto& ledger = resource_detail::ProcessLedger();
    std::lock_guard<std::mutex> lock(ledger.mutex);
    return resource_detail::PlannedBytesLocked(ledger);
}
// One model owns one plan. All consumers charge its categories rather than
// independently assigning the same residual process memory to their caches.
class ResourcePlanLease {
public:
    explicit ResourcePlanLease(const RuntimeMemoryPlan& plan) {
        if (plan.RequiredBytes() == 0) return; // Platforms without a detector retain the legacy path.
        auto& ledger = resource_detail::ProcessLedger();
        std::lock_guard<std::mutex> lock(ledger.mutex);
        const uint64_t capacity = plan.resource_capacity_bytes ? plan.resource_capacity_bytes : plan.available_bytes;
        const auto previous = resource_detail::PlannedBytesLocked(ledger);
        if (previous > capacity || plan.RequiredBytes() > capacity-previous)
            throw densecore::OutOfMemoryException("resource plan concurrent reservation required_bytes=" +
                   std::to_string(resource_detail::Add(previous,plan.RequiredBytes())) +
                   " available_bytes=" + std::to_string(capacity));
        ledger.plans.emplace(this,plan);
        registered_ = true;
    }
    ResourcePlanLease(const ResourcePlanLease&) = delete;
    ResourcePlanLease& operator=(const ResourcePlanLease&) = delete;
    ~ResourcePlanLease() {
        if (!registered_) return;
        auto& ledger = resource_detail::ProcessLedger();
        std::lock_guard<std::mutex> lock(ledger.mutex);
        ledger.plans.erase(this);
    }
    bool ExtendMTP(uint64_t bytes, std::string* error = nullptr) {
        auto& ledger = resource_detail::ProcessLedger();
        std::lock_guard<std::mutex> lock(ledger.mutex);
        auto it = ledger.plans.find(this);
        if (it == ledger.plans.end()) {
            if (error) *error = "resource plan is unavailable for MTP reservation";
            return false;
        }
        const uint64_t capacity = it->second.resource_capacity_bytes ? it->second.resource_capacity_bytes : it->second.available_bytes;
        const uint64_t previous = resource_detail::PlannedBytesLocked(ledger);
        if (previous > capacity || bytes > capacity-previous || bytes > UINT64_MAX-it->second.mtp_checkpoint_bytes) {
            if (error) *error = "MTP resource plan required_bytes=" + std::to_string(resource_detail::Add(previous,bytes)) +
                                " available_bytes=" + std::to_string(capacity);
            return false;
        }
        it->second.mtp_checkpoint_bytes += bytes;
        return true;
    }
private:
    bool registered_ = false;
};

enum class AllocationCategory { Graph, Repack };
// Charge actual allocation lifetimes, including evicted but still referenced
// weights and simultaneous cache builders. Inactive/standalone kernel tests use
// the pre-existing local cache policy without an engine reservation.
class ResourceAllocationLease {
public:
    static std::shared_ptr<ResourceAllocationLease> Acquire(AllocationCategory category, uint64_t bytes) {
        auto& ledger = resource_detail::ProcessLedger();
        std::lock_guard<std::mutex> lock(ledger.mutex);
        uint64_t capacity = ledger.plans.empty() ? UINT64_MAX : 0;
        for (const auto& entry : ledger.plans)
            capacity = resource_detail::Add(capacity, category == AllocationCategory::Graph
                                                           ? entry.second.graph_bytes : entry.second.repack_bytes);
        uint64_t& used = category == AllocationCategory::Graph ? ledger.graph_used : ledger.repack_used;
        if (used > capacity || bytes > capacity-used) return {};
        // Allocate the control block before charging so bad_alloc cannot strand a charge.
        auto result = std::shared_ptr<ResourceAllocationLease>(new ResourceAllocationLease(category,bytes));
        used += bytes;
        result->charged_ = true;
        return result;
    }
    ~ResourceAllocationLease() {
        if (!charged_) return;
        auto& ledger = resource_detail::ProcessLedger();
        std::lock_guard<std::mutex> lock(ledger.mutex);
        uint64_t& used = category_ == AllocationCategory::Graph ? ledger.graph_used : ledger.repack_used;
        used = bytes_ > used ? 0 : used-bytes_;
    }
private:
    ResourceAllocationLease(AllocationCategory category, uint64_t bytes) : category_(category), bytes_(bytes) {}
    AllocationCategory category_;
    uint64_t bytes_;
    bool charged_ = false;
};
template<class Weight> std::shared_ptr<Weight> MakeBudgetedRepackedWeight(size_t bytes) {
    auto lease = ResourceAllocationLease::Acquire(AllocationCategory::Repack,bytes);
    if (!lease) return {};
    return std::shared_ptr<Weight>(new Weight, [lease = std::move(lease)](Weight* weight) { delete weight; });
}
} // namespace densecore::runtime

namespace densecore::runtime {
namespace resource_detail {
struct GraphOwners {
    std::mutex mutex;
    std::unordered_map<const void*, std::shared_ptr<ResourceAllocationLease>> leases;
};
inline GraphOwners& ProcessGraphOwners() { static auto* owners = new GraphOwners; return *owners; }
}
inline void RetainGraphAllocation(const void* owner, std::shared_ptr<ResourceAllocationLease> lease) {
    auto& owners = resource_detail::ProcessGraphOwners();
    std::shared_ptr<ResourceAllocationLease> previous;
    { std::lock_guard<std::mutex> lock(owners.mutex);
      auto& slot = owners.leases[owner]; previous = std::move(slot); slot = std::move(lease); }
}
inline void ReleaseGraphAllocation(const void* owner) {
    auto& owners = resource_detail::ProcessGraphOwners();
    std::shared_ptr<ResourceAllocationLease> previous;
    { std::lock_guard<std::mutex> lock(owners.mutex);
      auto it = owners.leases.find(owner);
      if (it != owners.leases.end()) { previous = std::move(it->second); owners.leases.erase(it); } }
}
} // namespace densecore::runtime

namespace densecore::runtime {
struct ResourceLedgerUsage { uint64_t planned_bytes, graph_bytes, repack_bytes; };
inline ResourceLedgerUsage ResourceLedgerUsageSnapshot() {
    auto& ledger = resource_detail::ProcessLedger();
    std::lock_guard<std::mutex> lock(ledger.mutex);
    return {resource_detail::PlannedBytesLocked(ledger),ledger.graph_used,ledger.repack_used};
}
} // namespace densecore::runtime
