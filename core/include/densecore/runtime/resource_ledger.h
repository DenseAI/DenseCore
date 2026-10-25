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
    uint64_t admission_used = 0;
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
        return ExtendLocked(ledger,bytes,&RuntimeMemoryPlan::mtp_checkpoint_bytes,"MTP",error);
    }
    bool EnsureKVBytes(uint64_t bytes, std::string* error = nullptr) {
        auto& ledger = resource_detail::ProcessLedger();
        std::lock_guard<std::mutex> lock(ledger.mutex);
        const auto it = ledger.plans.find(this);
        if (it == ledger.plans.end()) {
            if (error) *error = "resource plan is unavailable for KV reservation";
            return false;
        }
        const uint64_t current = it->second.kv_bytes;
        return ExtendLocked(ledger,bytes > current ? bytes-current : 0,&RuntimeMemoryPlan::kv_bytes,"KV",error);
    }
private:
    bool ExtendLocked(resource_detail::Ledger& ledger, uint64_t bytes, uint64_t RuntimeMemoryPlan::* category,
                      const char* name, std::string* error) {
        auto it = ledger.plans.find(this);
        if (it == ledger.plans.end()) {
            if (error) *error = std::string("resource plan is unavailable for ") + name + " reservation";
            return false;
        }
        const uint64_t capacity = it->second.resource_capacity_bytes ? it->second.resource_capacity_bytes : it->second.available_bytes;
        const uint64_t previous = resource_detail::PlannedBytesLocked(ledger);
        if (previous > capacity || bytes > capacity-previous || bytes > UINT64_MAX-(it->second.*category)) {
            if (error) *error = std::string(name) + " resource plan required_bytes=" + std::to_string(resource_detail::Add(previous,bytes)) +
                                " available_bytes=" + std::to_string(capacity);
            return false;
        }
        it->second.*category += bytes;
        return true;
    }

    bool registered_ = false;
};

enum class AllocationCategory { Graph, Repack, Admission };
// No-model graph adapters have no committed allocator plan. Keep their
// admission bounded, but never manufacture capacity from exhaustion/read failure.
inline uint64_t StandaloneAdmissionCapacity(const ResourceBudget& budget) {
    constexpr uint64_t ceiling = 128ULL*1024*1024;
    if (budget.memory.state == ResourceState::Unknown) return 0;
    if (budget.memory.state == ResourceState::Unlimited) return ceiling;
    return std::min<uint64_t>(ceiling,budget.memory.bytes/32);
}
// Charge actual allocation lifetimes, including evicted but still referenced
// weights and simultaneous cache builders. Inactive/standalone kernel tests use
// the pre-existing local cache policy without an engine reservation.
class ResourceAllocationLease {
public:
    static std::shared_ptr<ResourceAllocationLease> Acquire(AllocationCategory category, uint64_t bytes) {
        auto& ledger = resource_detail::ProcessLedger();
        std::lock_guard<std::mutex> lock(ledger.mutex);
        uint64_t capacity = ledger.plans.empty() ? (category == AllocationCategory::Admission ? StandaloneAdmissionCapacity(DetectResourceBudget()) : UINT64_MAX) : 0;
        for (const auto& entry : ledger.plans)
            capacity = resource_detail::Add(capacity, CategoryCapacity(entry.second,category));
        uint64_t& used = CategoryUsed(ledger,category);
        if (used > capacity || bytes > capacity-used) return {};
        // Allocate the control block before charging so bad_alloc cannot strand a charge.
        auto result = std::shared_ptr<ResourceAllocationLease>(new ResourceAllocationLease(category,bytes));
        used += bytes;
        result->charged_ = true;
        return result;
    }
    static uint64_t AvailableBytes(AllocationCategory category) {
        auto& ledger = resource_detail::ProcessLedger();
        std::lock_guard<std::mutex> lock(ledger.mutex);
        if (ledger.plans.empty()) {
            if (category != AllocationCategory::Admission) return UINT64_MAX;
            const auto capacity=StandaloneAdmissionCapacity(DetectResourceBudget());
            return ledger.admission_used >= capacity ? 0 : capacity-ledger.admission_used;
        }
        uint64_t capacity = 0;
        for (const auto& entry : ledger.plans)
            capacity = resource_detail::Add(capacity,CategoryCapacity(entry.second,category));
        const uint64_t used = CategoryUsed(ledger,category);
        return used >= capacity ? 0 : capacity-used;
    }
    ~ResourceAllocationLease() {
        if (!charged_) return;
        auto& ledger = resource_detail::ProcessLedger();
        std::lock_guard<std::mutex> lock(ledger.mutex);
        uint64_t& used = CategoryUsed(ledger,category_);
        used = bytes_ > used ? 0 : used-bytes_;
    }
private:
    static uint64_t CategoryCapacity(const RuntimeMemoryPlan& plan, AllocationCategory category) {
        if (category == AllocationCategory::Graph) return plan.graph_bytes;
        if (category == AllocationCategory::Repack) return plan.repack_bytes;
        // Partition the existing allocator reserve; this is not extra headroom.
        return plan.allocator_bytes / 2;
    }
    static uint64_t& CategoryUsed(resource_detail::Ledger& ledger, AllocationCategory category) {
        if (category == AllocationCategory::Graph) return ledger.graph_used;
        if (category == AllocationCategory::Repack) return ledger.repack_used;
        return ledger.admission_used;
    }
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
struct ResourceLedgerUsage { uint64_t planned_bytes, graph_bytes, repack_bytes, admission_bytes; };
inline ResourceLedgerUsage ResourceLedgerUsageSnapshot() {
    auto& ledger = resource_detail::ProcessLedger();
    std::lock_guard<std::mutex> lock(ledger.mutex);
    return {resource_detail::PlannedBytesLocked(ledger),ledger.graph_used,ledger.repack_used,ledger.admission_used};
}
} // namespace densecore::runtime
