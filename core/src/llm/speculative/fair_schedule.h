#pragma once
#include <algorithm>
#include <cstddef>
#include <vector>

namespace densecore::llm::speculative {

struct SchedulingIdentity {
    int sequence;
    int request;
    bool operator==(const SchedulingIdentity& other) const {
        return sequence == other.sequence && request == other.request;
    }
};

// Persistent arrival order, rather than a numeric-ID cursor: a stream of new
// higher IDs must not postpone the wraparound to an older live request forever.
class FairSchedule {
public:
    enum class Phase { None, Prefill, Decode };

    template <typename IsLive> void Prune(IsLive is_live) {
        order_.erase(std::remove_if(order_.begin(), order_.end(),
                                   [&](SchedulingIdentity identity) { return !is_live(identity); }), order_.end());
    }
    void Forget(SchedulingIdentity identity) {
        order_.erase(std::remove(order_.begin(), order_.end(), identity), order_.end());
    }
    std::vector<int> Order(const std::vector<SchedulingIdentity>& eligible) {
        for (const auto identity : eligible) {
            // Sequence IDs may be recycled after a request finishes.
            order_.erase(std::remove_if(order_.begin(), order_.end(), [&](SchedulingIdentity previous) {
                return previous.sequence == identity.sequence && previous.request != identity.request;
            }), order_.end());
            if (std::find(order_.begin(), order_.end(), identity) == order_.end()) order_.push_back(identity);
        }
        std::vector<int> result;
        result.reserve(eligible.size());
        for (const auto identity : order_)
            if (std::find(eligible.begin(), eligible.end(), identity) != eligible.end()) result.push_back(identity.sequence);
        return result;
    }
    void DecodeExecuted(SchedulingIdentity identity) {
        const auto item = std::find(order_.begin(), order_.end(), identity);
        if (item != order_.end()) std::rotate(item, item+1, order_.end());
        next_ = Phase::Prefill;
    }
    void PrefillExecuted() { next_ = Phase::Decode; }
    void NoExecution(Phase attempted) {
        if (attempted == Phase::Decode) next_ = Phase::Prefill;
        else if (attempted == Phase::Prefill) next_ = Phase::Decode;
    }
    bool PreferDecode() const { return next_ == Phase::Decode; }
    Phase Choose(bool has_prefill, bool has_decode) const {
        if (has_prefill && has_decode) return next_;
        if (has_decode) return Phase::Decode;
        if (has_prefill) return Phase::Prefill;
        return Phase::None;
    }
    std::size_t Size() const { return order_.size(); }
private:
    std::vector<SchedulingIdentity> order_;
    Phase next_ = Phase::Decode;
};
}
