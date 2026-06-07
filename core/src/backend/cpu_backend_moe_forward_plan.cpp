#include "backend/cpu_backend_moe_forward_plan.h"

#include "backend/cpu_backend_internal.h"

#include <algorithm>

namespace densecore {
namespace {

uint64_t CountMoENumaSwitches(const std::vector<MoEActiveExpertWork>& work_items) {
    uint64_t switches = 0;
    for (size_t i = 1; i < work_items.size(); ++i) {
        const int prev = work_items[i - 1].numa_node;
        const int cur = work_items[i].numa_node;
        if (prev >= 0 && cur >= 0 && prev != cur) {
            ++switches;
        }
    }
    return switches;
}

void ApplyMoELocalityOrdering(MoEForwardExecutionPlan* plan) {
    if (!plan || plan->small_decode_step || !internal::IsMoELocalityOrderingEnabled()) {
        return;
    }

    plan->ordering.considered = true;
    const bool enough_active_experts =
        static_cast<int>(plan->active_work.size()) >= internal::GetMoELocalityOrderingMinActiveExperts();
    const bool enough_reuse_signal =
        plan->reuse_intersection >= internal::GetMoELocalityOrderingMinReuseIntersection() || plan->local_hot_count > 0;

    if (!enough_active_experts) {
        plan->ordering.skipped_small_batch = true;
        return;
    }
    if (!enough_reuse_signal) {
        plan->ordering.skipped_low_reuse = true;
        return;
    }

    plan->ordering.numa_switches_before = CountMoENumaSwitches(plan->active_work);
    std::stable_sort(plan->active_work.begin(), plan->active_work.end(),
                     [plan](const MoEActiveExpertWork& lhs, const MoEActiveExpertWork& rhs) {
                         const auto ordering_score = [plan](const MoEActiveExpertWork& work) {
                             const bool reused =
                                 plan->previous_batch_set.find(work.expert_id) != plan->previous_batch_set.end();
                             int score = 0;
                             if (work.local_hot) score += 32;
                             if (reused) score += 24;
                             if (work.numa_node >= 0) score += 4;
                             score += std::min(work.count, 4) * 3;
                             return score;
                         };
                         const int lhs_score = ordering_score(lhs);
                         const int rhs_score = ordering_score(rhs);
                         if (lhs_score != rhs_score) return lhs_score > rhs_score;
                         if (lhs.ema_load != rhs.ema_load) return lhs.ema_load < rhs.ema_load;
                         if (lhs.count != rhs.count) return lhs.count < rhs.count;
                         if (lhs.numa_node != rhs.numa_node) return lhs.numa_node < rhs.numa_node;
                         return lhs.expert_id < rhs.expert_id;
                     });
    plan->ordering.numa_switches_after = CountMoENumaSwitches(plan->active_work);
    plan->ordering.applied = true;
}

void BalanceMoEParallelExpertWork(std::vector<MoEActiveExpertWork>* active_work, int active_threads) {
    if (!active_work || active_threads <= 1 || active_work->size() <= 1) {
        return;
    }

    const int work_per_thread = (static_cast<int>(active_work->size()) + active_threads - 1) / active_threads;
    std::vector<MoEActiveExpertWork> by_cost = *active_work;
    std::stable_sort(by_cost.begin(), by_cost.end(),
                     [](const MoEActiveExpertWork& lhs, const MoEActiveExpertWork& rhs) {
                         if (lhs.count != rhs.count) {
                             return lhs.count > rhs.count;
                         }
                         return lhs.expert_id < rhs.expert_id;
                     });

    std::vector<std::vector<MoEActiveExpertWork>> buckets(static_cast<size_t>(active_threads));
    std::vector<int64_t> bucket_cost(static_cast<size_t>(active_threads), 0);
    for (const MoEActiveExpertWork& work : by_cost) {
        int best_bucket = -1;
        for (int bucket = 0; bucket < active_threads; ++bucket) {
            if (static_cast<int>(buckets[static_cast<size_t>(bucket)].size()) >= work_per_thread) {
                continue;
            }
            if (best_bucket < 0 ||
                bucket_cost[static_cast<size_t>(bucket)] < bucket_cost[static_cast<size_t>(best_bucket)]) {
                best_bucket = bucket;
            }
        }
        if (best_bucket < 0) {
            best_bucket = active_threads - 1;
        }
        buckets[static_cast<size_t>(best_bucket)].push_back(work);
        bucket_cost[static_cast<size_t>(best_bucket)] += std::max(1, work.count);
    }

    active_work->clear();
    active_work->reserve(by_cost.size());
    for (auto& bucket : buckets) {
        active_work->insert(active_work->end(), bucket.begin(), bucket.end());
    }
}

}  // namespace

MoEForwardExecutionPlan BuildMoEForwardExecutionPlan(const moe::MoEReorderMapView& reorder_map, int num_experts,
                                                     int batch_size, int total_assignments, int worker_threads,
                                                     bool registry_present,
                                                     const std::unordered_set<int>& local_hot_experts,
                                                     const std::vector<int>& previous_batch_experts,
                                                     const std::shared_ptr<moe::ExpertProfiler>& profiler) {
    MoEForwardExecutionPlan plan;
    plan.worker_threads = std::max(1, worker_threads);
    plan.active_work.reserve(static_cast<size_t>(num_experts));
    plan.current_batch_experts.reserve(static_cast<size_t>(num_experts));

    for (int expert_id = 0; expert_id < num_experts; ++expert_id) {
        const int start = reorder_map.expert_offsets[static_cast<size_t>(expert_id)];
        const int end = reorder_map.expert_offsets[static_cast<size_t>(expert_id + 1)];
        const int count = end - start;
        if (count <= 0) {
            continue;
        }

        MoEActiveExpertWork work;
        work.expert_id = expert_id;
        work.start = start;
        work.count = count;
        work.local_hot = local_hot_experts.find(expert_id) != local_hot_experts.end();
        if (profiler) {
            work.numa_node = profiler->GetExpertNumaNode(expert_id);
            work.ema_load = profiler->GetEmaLoad(expert_id);
        }
        plan.local_hot_count += work.local_hot ? 1 : 0;
        plan.active_work.push_back(work);
        plan.current_batch_experts.push_back(expert_id);
    }

    if (!previous_batch_experts.empty()) {
        plan.previous_batch_set.insert(previous_batch_experts.begin(), previous_batch_experts.end());
        for (int expert_id : plan.current_batch_experts) {
            if (plan.previous_batch_set.find(expert_id) != plan.previous_batch_set.end()) {
                ++plan.reuse_intersection;
            }
        }
    }
    plan.reuse_union =
        static_cast<int>(plan.current_batch_experts.size() + previous_batch_experts.size() - plan.reuse_intersection);
    plan.max_expert_batch = reorder_map.max_expert_batch;
    plan.small_decode_step = batch_size <= 4 && total_assignments <= 8 && plan.max_expert_batch <= 1;

    ApplyMoELocalityOrdering(&plan);

    plan.prefer_inner_parallel_prefill =
        !plan.small_decode_step && batch_size > 1 && plan.active_work.size() < static_cast<size_t>(plan.worker_threads);
    plan.parallelize_experts = !plan.prefer_inner_parallel_prefill && !plan.small_decode_step && batch_size > 1 &&
                               plan.active_work.size() >= static_cast<size_t>(std::max(4, plan.worker_threads / 2)) &&
                               registry_present;

    if (plan.parallelize_experts && plan.active_work.size() > 1) {
        const int active_threads =
            std::max(1, std::min(plan.worker_threads, static_cast<int>(plan.active_work.size())));
        BalanceMoEParallelExpertWork(&plan.active_work, active_threads);
    }

    return plan;
}

}  // namespace densecore
