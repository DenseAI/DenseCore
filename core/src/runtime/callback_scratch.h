#pragma once

#include "densecore/runtime/resource_ledger.h"
#include <atomic>
#include <memory>
#include <mutex>

namespace densecore::runtime {
// Numeric ownership only. In particular this object never retains an engine,
// model, request, graph, backend or work context. OpenMP TLS can outlive all of
// those owners, so actual TLS workspaces retain this token independently.
class CallbackScratchBudget {
public:
    CallbackScratchBudget() = default;
    CallbackScratchBudget(const CallbackScratchBudget&) = delete;
    ~CallbackScratchBudget() {
        auto& ledger=resource_detail::ProcessLedger();
        std::lock_guard<std::mutex> lock(ledger.mutex);
        ledger.plans.erase(this);
    }
    uint64_t Capacity() const {
        auto& ledger=resource_detail::ProcessLedger();std::lock_guard<std::mutex> lock(ledger.mutex);
        auto it=ledger.plans.find(this);return it==ledger.plans.end()?0:it->second.callback_scratch_bytes;
    }
    // The model's accepted plan already contains the delta. Transfer it under
    // the same ledger lock; global required bytes never increase here.
    bool Adopt(ResourcePlanLease& source,uint64_t envelope,std::string* error=nullptr) {
        return source.TransferCallbackScratch(this,envelope,error);
    }
    bool TryGrow(uint64_t old_bytes,uint64_t new_bytes) noexcept {
        if(new_bytes<=old_bytes)return true;
        try {
            auto& ledger=resource_detail::ProcessLedger();std::lock_guard<std::mutex> lock(ledger.mutex);
            auto it=ledger.plans.find(this);const uint64_t capacity=it==ledger.plans.end()?0:it->second.callback_scratch_bytes;
            if(used_>capacity || new_bytes-old_bytes>capacity-used_)return false;
            used_+=new_bytes-old_bytes;return true;
        } catch(...) {return false;}
    }
    void Release(uint64_t bytes) noexcept {
        auto& ledger=resource_detail::ProcessLedger();std::lock_guard<std::mutex> lock(ledger.mutex);
        used_=bytes>used_?0:used_-bytes;
    }
    bool BindMaster(uint64_t token) noexcept {
        auto& ledger=resource_detail::ProcessLedger();std::lock_guard<std::mutex> lock(ledger.mutex);
        if(!token || (master_ && master_!=token))return false;
        master_=token;return true;
    }
    uint64_t Used() const {
        auto& ledger=resource_detail::ProcessLedger();std::lock_guard<std::mutex> lock(ledger.mutex);return used_;
    }
private:
    uint64_t used_=0,master_=0;
};

inline uint64_t CallbackScratchMasterToken() {
    static std::atomic<uint64_t> next{1};
    static thread_local const uint64_t token=next.fetch_add(1,std::memory_order_relaxed);
    return token;
}

// Suballocates once inside an already transferred global reservation. This is
// not ResourceAllocationLease::Acquire and does not charge global bytes twice.
class CallbackScratchSlice {
public:
    ~CallbackScratchSlice(){Reset();}
    bool Grow(const std::shared_ptr<CallbackScratchBudget>& owner,uint64_t bytes) noexcept {
        if(!owner)return false;
        if(owner==owner_) {
            if(!owner->TryGrow(bytes_,bytes))return false;
            bytes_=std::max(bytes_,bytes);return true;
        }
        // A reused OpenMP thread can enter a different engine. Its retained
        // capacities must be covered by the new owner before the old one drops.
        if(!owner->TryGrow(0,bytes))return false;
        Reset();owner_=owner;bytes_=bytes;return true;
    }
    void Reset() noexcept {if(owner_)owner_->Release(bytes_);bytes_=0;owner_.reset();}
    uint64_t Bytes() const{return bytes_;}
    bool IsOwner(const std::shared_ptr<CallbackScratchBudget>& owner) const{return owner_==owner;}
private:
    std::shared_ptr<CallbackScratchBudget> owner_;
    uint64_t bytes_=0;
};
}
