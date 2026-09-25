#pragma once

#include "densecore/moe/moe_routing.h"
#include "densecore/moe/profiler.h"

#include <cstdint>
#include <memory>
#include <unordered_set>
#include <vector>

namespace densecore {

struct MoEActiveExpertWork {
    int expert_id = -1;
    int start = 0;
    int count = 0;
    int numa_node = -1;
    float ema_load = 0.0f;
    bool local_hot = false;
};

struct MoELocalityOrderingOutcome {
    bool considered = false;
    bool applied = false;
    bool skipped_small_batch = false;
    bool skipped_low_reuse = false;
    uint64_t numa_switches_before = 0;
    uint64_t numa_switches_after = 0;
};

struct MoEForwardExecutionPlan {
    std::vector<MoEActiveExpertWork> active_work;
    std::vector<int> current_batch_experts;
    std::unordered_set<int> previous_batch_set;
    int reuse_intersection = 0;
    int reuse_union = 0;
    int max_expert_batch = 0;
    int local_hot_count = 0;
    int worker_threads = 1;
    bool small_decode_step = false;
    bool prefer_inner_parallel_prefill = false;
    bool parallelize_experts = false;
    MoELocalityOrderingOutcome ordering;
};

MoEForwardExecutionPlan BuildMoEForwardExecutionPlan(const moe::MoEReorderMapView& reorder_map, int num_experts,
                                                     int batch_size, int total_assignments, int worker_threads,
                                                     bool registry_present,
                                                     const std::unordered_set<int>& local_hot_experts,
                                                     const std::vector<int>& previous_batch_experts,
                                                     const std::shared_ptr<moe::ExpertProfiler>& profiler);

}  // namespace densecore
