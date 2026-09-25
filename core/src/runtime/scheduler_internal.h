#ifndef DENSECORE_SCHEDULER_INTERNAL_H
#define DENSECORE_SCHEDULER_INTERNAL_H

#include "densecore/runtime/scheduler.h"

#include <cstddef>
#include <unordered_set>

namespace densecore::scheduler_internal {


std::size_t ScheduledSeqCount(const SchedulerOutput& output);
int CountExpertOverlap(const std::vector<int>& experts, const std::unordered_set<int>& active_experts);
int CountNewExperts(const std::vector<int>& experts, const std::unordered_set<int>& active_experts);
float ComputeMoEDeferScore(int request_key);
bool ShouldDeferForMoEBudget(int request_key, const std::vector<int>& experts,
                             const std::unordered_set<int>& active_experts, const SchedulerConfig& config);

}  // namespace densecore::scheduler_internal

#endif  // DENSECORE_SCHEDULER_INTERNAL_H
