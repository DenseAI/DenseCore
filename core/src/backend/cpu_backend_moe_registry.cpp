#include "backend/cpu_backend_internal.h"

namespace densecore {

namespace {

static constexpr size_t MOVE_PAGES_BATCH_SIZE = 4096;
static constexpr float REBALANCE_HYSTERESIS_RATIO = 0.15f;
static constexpr float REBALANCE_MIGRATION_STABLE_CHURN = 0.10f;
static constexpr int REBALANCE_MIGRATION_STABLE_CYCLES = 2;

int QueryDominantInitialNumaNode(CpuBackend* backend, const CpuBackend::ExpertWeight& weight) {
#if defined(__linux__) && defined(DENSECORE_HAS_NUMA)
    if (!backend || !weight.ptr || weight.size == 0) {
        return -1;
    }
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        return -1;
    }
    const uintptr_t raw_start = reinterpret_cast<uintptr_t>(weight.ptr);
    const uintptr_t raw_end = raw_start + weight.size;
    const uintptr_t start =
        (raw_start + static_cast<uintptr_t>(page_size) - 1) & ~(static_cast<uintptr_t>(page_size) - 1);
    const uintptr_t end = raw_end & ~(static_cast<uintptr_t>(page_size) - 1);
    if (raw_end < raw_start || end <= start) {
        return -1;
    }

    constexpr size_t kSamples = 16;
    const size_t page_count = (end - start) / static_cast<size_t>(page_size);
    const size_t sample_count = std::min(page_count, kSamples);
    std::vector<int> counts(static_cast<size_t>(std::max(0, backend->GetNumaNodeCount())), 0);
    int valid_samples = 0;
    for (size_t sample = 0; sample < sample_count; ++sample) {
        const size_t page_index = sample_count == 1 ? 0 : (sample * (page_count - 1)) / (sample_count - 1);
        void* page = reinterpret_cast<void*>(start + page_index * static_cast<size_t>(page_size));
        const int node = backend->QueryMemoryNumaNode(page);
        if (node >= 0 && static_cast<size_t>(node) < counts.size()) {
            ++counts[static_cast<size_t>(node)];
            ++valid_samples;
        }
    }
    if (valid_samples < static_cast<int>(sample_count * 3 / 4)) {
        return -1;
    }
    const auto winner = std::max_element(counts.begin(), counts.end());
    if (winner == counts.end() || *winner * 4 < valid_samples * 3) {
        return -1;
    }
    return static_cast<int>(std::distance(counts.begin(), winner));
#else
    (void)backend;
    (void)weight;
    return -1;
#endif
}

std::vector<int> SelectLocalExpertsWithHysteresis(const std::vector<int>& previous_local,
                                                  const std::vector<int>& hot_expert_ids,
                                                  const std::vector<float>& ema_loads, int n_experts) {
    const size_t target_size = hot_expert_ids.size();
    std::vector<std::pair<float, int>> local_ranked;
    local_ranked.reserve(previous_local.size());
    std::unordered_set<int> local_set;

    const auto score = [&](int expert_id) {
        return expert_id >= 0 && static_cast<size_t>(expert_id) < ema_loads.size()
                   ? ema_loads[static_cast<size_t>(expert_id)]
                   : 0.0f;
    };

    for (int expert_id : previous_local) {
        if (expert_id < 0 || expert_id >= n_experts) continue;
        if (local_set.insert(expert_id).second) {
            local_ranked.emplace_back(score(expert_id), expert_id);
        }
    }

    if (local_ranked.size() > target_size) {
        std::partial_sort(local_ranked.begin(), local_ranked.begin() + target_size, local_ranked.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        local_ranked.resize(target_size);
    }

    local_set.clear();
    for (const auto& entry : local_ranked) {
        local_set.insert(entry.second);
    }

    std::vector<std::pair<float, int>> candidates;
    candidates.reserve(hot_expert_ids.size());
    for (int expert_id : hot_expert_ids) {
        if (expert_id < 0 || expert_id >= n_experts) continue;
        if (local_set.find(expert_id) == local_set.end()) {
            candidates.emplace_back(score(expert_id), expert_id);
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::sort(local_ranked.begin(), local_ranked.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    for (const auto& candidate : candidates) {
        if (local_ranked.size() < target_size) {
            local_ranked.push_back(candidate);
            local_set.insert(candidate.second);
            std::sort(local_ranked.begin(), local_ranked.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
            continue;
        }
        if (local_ranked.empty()) break;

        const auto& lowest_local = local_ranked.front();
        if (candidate.first > lowest_local.first * (1.0f + REBALANCE_HYSTERESIS_RATIO)) {
            local_set.erase(lowest_local.second);
            local_ranked.front() = candidate;
            local_set.insert(candidate.second);
            std::sort(local_ranked.begin(), local_ranked.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
        }
    }

    std::vector<int> selected;
    selected.reserve(local_ranked.size());
    for (const auto& entry : local_ranked) {
        selected.push_back(entry.second);
    }
    return selected;
}

float ComputeHotSetChurn(const std::vector<int>& previous_hot, const std::vector<int>& current_hot) {
    if (previous_hot.empty() && current_hot.empty()) return 0.0f;
    if (previous_hot.empty() || current_hot.empty()) return 1.0f;

    std::unordered_set<int> previous(previous_hot.begin(), previous_hot.end());
    size_t overlap = 0;
    for (int expert_id : current_hot) {
        if (previous.find(expert_id) != previous.end()) ++overlap;
    }
    const size_t union_size = previous.size() + current_hot.size() - overlap;
    return union_size > 0 ? 1.0f - static_cast<float>(overlap) / static_cast<float>(union_size) : 0.0f;
}

int NextRebalanceIntervalMs(int current_interval_ms, int base_interval_ms, bool any_high, bool all_low) {
    const int min_interval_ms = std::max(250, base_interval_ms / 4);
    const int max_interval_ms = std::max(base_interval_ms, base_interval_ms * 4);
    if (any_high) {
        return std::max(min_interval_ms, static_cast<int>(current_interval_ms * 0.7));
    }
    if (all_low) {
        return std::min(max_interval_ms, static_cast<int>(current_interval_ms * 1.3));
    }
    return current_interval_ms;
}

int NextMigrationStableCycles(int current_cycles, bool has_previous_hot, float churn, uint64_t delta_hits) {
    if (!has_previous_hot || delta_hits == 0 || churn > REBALANCE_MIGRATION_STABLE_CHURN) {
        return 0;
    }
    return std::min(current_cycles + 1, REBALANCE_MIGRATION_STABLE_CYCLES);
}

}  // namespace

#ifdef DENSECORE_TEST_BUILD
namespace testing {

std::vector<int> SelectLocalExpertsWithHysteresisForTest(const std::vector<int>& previous_local,
                                                         const std::vector<int>& hot_expert_ids,
                                                         const std::vector<float>& ema_loads, int n_experts) {
    return SelectLocalExpertsWithHysteresis(previous_local, hot_expert_ids, ema_loads, n_experts);
}

float ComputeHotSetChurnForTest(const std::vector<int>& previous_hot, const std::vector<int>& current_hot) {
    return ComputeHotSetChurn(previous_hot, current_hot);
}

int NextRebalanceIntervalMsForTest(int current_interval_ms, int base_interval_ms, bool any_high, bool all_low) {
    return NextRebalanceIntervalMs(current_interval_ms, base_interval_ms, any_high, all_low);
}

int NextMigrationStableCyclesForTest(int current_cycles, bool has_previous_hot, float churn, uint64_t delta_hits) {
    return NextMigrationStableCycles(current_cycles, has_previous_hot, churn, delta_hits);
}

}  // namespace testing
#endif

std::shared_ptr<CpuBackend::MoELayerRegistry> CpuBackend::GetMoELayerRegistry(const TransformerLayer* layer_key) const {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    auto it = moe_registries_.find(layer_key);
    if (it == moe_registries_.end()) {
        return nullptr;
    }
    return it->second;
}

std::shared_ptr<CpuBackend::MoELayerRegistry> CpuBackend::GetOrCreateMoELayerRegistry(const TransformerLayer* layer_key,
                                                                                      int n_experts, float ema_alpha) {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    auto it = moe_registries_.find(layer_key);
    if (it != moe_registries_.end()) {
        auto& registry = it->second;
        if (registry) {
            std::lock_guard<std::mutex> registry_lock(registry->mutex);
            if (n_experts > 0 && (!registry->profiler || registry->profiler->GetNumExperts() != n_experts)) {
                registry->profiler = std::make_shared<moe::ExpertProfiler>(n_experts, ema_alpha);
            }
            registry->ema_alpha = ema_alpha;
        }
        return registry;
    }

    auto registry = std::make_shared<MoELayerRegistry>();
    registry->ema_alpha = ema_alpha;
    if (n_experts > 0) {
        registry->profiler = std::make_shared<moe::ExpertProfiler>(n_experts, ema_alpha);
    }
    moe_registries_.emplace(layer_key, registry);
    return registry;
}

void CpuBackend::RecordExpertAccess(const int* expert_ids, int count) {
    RecordExpertAccess(nullptr, expert_ids, count);
}

void CpuBackend::RecordExpertAccess(const TransformerLayer* layer_key, const int* expert_ids, int count) {
    auto registry = GetMoELayerRegistry(layer_key);
    if (!registry || !expert_ids || count <= 0) {
        return;
    }

    std::shared_ptr<moe::ExpertProfiler> profiler;
    {
        std::lock_guard<std::mutex> lock(registry->mutex);
        profiler = registry->profiler;
    }
    if (profiler) {
        profiler->RecordHitBatch(expert_ids, count);
    }
}

int CpuBackend::RebalanceExperts(const std::vector<int>& hot_expert_ids,
                                 const std::vector<ExpertWeights>& expert_weights, int n_experts,
                                 int target_numa_node) {
    return RebalanceExperts(nullptr, hot_expert_ids, expert_weights, n_experts, target_numa_node);
}

int CpuBackend::RebalanceExperts(const TransformerLayer* layer_key, const std::vector<int>& hot_expert_ids,
                                 const std::vector<ExpertWeights>& expert_weights, int n_experts,
                                 int target_numa_node) {
    auto registry = GetMoELayerRegistry(layer_key);
    if (!registry) {
        return 0;
    }

    std::shared_ptr<moe::ExpertProfiler> profiler;
    std::vector<int> local_expert_ids;
    {
        std::lock_guard<std::mutex> lock(registry->mutex);
        profiler = registry->profiler;
        local_expert_ids = registry->local_expert_ids;
    }

    const int migrated = RebalanceExpertsInternal(hot_expert_ids, expert_weights, n_experts, target_numa_node,
                                                  profiler ? profiler.get() : nullptr, &local_expert_ids);
    {
        std::lock_guard<std::mutex> lock(registry->mutex);
        registry->local_expert_ids = std::move(local_expert_ids);
    }
    return migrated;
}

int CpuBackend::RebalanceExpertsInternal(const std::vector<int>& hot_expert_ids,
                                         const std::vector<ExpertWeights>& expert_weights, int n_experts,
                                         int target_numa_node, moe::ExpertProfiler* profiler,
                                         std::vector<int>* local_expert_ids) {
    if (rebalance_disabled_.load()) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(rebalance_mutex_);

    if (hot_expert_ids.empty() || expert_weights.empty()) {
        return 0;
    }

    std::vector<int> experts_to_migrate = hot_expert_ids;
    const std::vector<int> previous_local = local_expert_ids ? *local_expert_ids : std::vector<int>{};
    if (profiler && local_expert_ids && !hot_expert_ids.empty()) {
        if (previous_local.empty()) {
            *local_expert_ids = hot_expert_ids;
        } else {
            std::vector<float> ema_loads(static_cast<size_t>(std::max(0, n_experts)), 0.0f);
            for (int expert_id = 0; expert_id < n_experts; ++expert_id) {
                ema_loads[static_cast<size_t>(expert_id)] = profiler->GetEmaLoad(expert_id);
            }
            *local_expert_ids = SelectLocalExpertsWithHysteresis(previous_local, hot_expert_ids, ema_loads, n_experts);
        }

        if (!local_expert_ids->empty()) {
            std::unordered_set<int> previous_set(previous_local.begin(), previous_local.end());
            experts_to_migrate.clear();
            for (int expert_id : *local_expert_ids) {
                if (previous_set.find(expert_id) == previous_set.end()) {
                    experts_to_migrate.push_back(expert_id);
                }
            }
        }
    }

#if defined(__linux__) && defined(DENSECORE_HAS_NUMA)
    if (target_numa_node < 0) {
        int cpu = sched_getcpu();
        if (cpu >= 0) {
            target_numa_node = numa_node_of_cpu(cpu);
        }
        if (target_numa_node < 0) {
            target_numa_node = 0;
        }
    }
    if (numa_available() < 0) {
        std::cerr << "[CpuBackend] NUMA not available on this system" << std::endl;
        if (local_expert_ids) {
            *local_expert_ids = previous_local;
        }
        return 0;
    }

    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        std::cerr << "[CpuBackend] Failed to get page size" << std::endl;
        if (local_expert_ids) {
            *local_expert_ids = previous_local;
        }
        return 0;
    }

    int total_pages_migrated = 0;
    int total_pages_failed = 0;
    static bool logged_permission_error = false;

    struct BufferMigrationResult {
        int pages_on_target = 0;
        bool verified = true;
    };
    auto migrate_buffer = [&](const ExpertWeight& weight) -> BufferMigrationResult {
        if (!weight.ptr || weight.size == 0) return {};

        uintptr_t start_addr = reinterpret_cast<uintptr_t>(weight.ptr);
        uintptr_t aligned_start = (start_addr + page_size - 1) & ~(page_size - 1);
        uintptr_t end_addr = start_addr + weight.size;
        uintptr_t aligned_end = end_addr & ~(page_size - 1);
        size_t num_pages = aligned_end > aligned_start ? (aligned_end - aligned_start) / page_size : 0;

        if (num_pages == 0) return {0, false};

        BufferMigrationResult migration;
        std::vector<void*> pages(std::min(num_pages, MOVE_PAGES_BATCH_SIZE));
        std::vector<int> nodes(pages.size(), target_numa_node);
        std::vector<int> status(pages.size(), -1);

        for (size_t batch_start = 0; batch_start < num_pages; batch_start += MOVE_PAGES_BATCH_SIZE) {
            size_t batch_size = std::min(MOVE_PAGES_BATCH_SIZE, num_pages - batch_start);

            for (size_t i = 0; i < batch_size; i++) {
                uintptr_t page_addr = aligned_start + (batch_start + i) * page_size;
                pages[i] = reinterpret_cast<void*>(page_addr);
                nodes[i] = target_numa_node;
                status[i] = -1;
            }

            long result = -1;
#ifdef DENSECORE_TEST_BUILD
            if (move_pages_test_hook_) {
                result = move_pages_test_hook_(batch_size, pages.data(), nodes.data(), status.data());
            } else
#endif
            {
                result = move_pages(0, batch_size, pages.data(), nodes.data(), status.data(), MPOL_MF_MOVE);
            }

            if (result >= 0) {
                for (size_t i = 0; i < batch_size; i++) {
                    if (status[i] == target_numa_node) {
                        migration.pages_on_target++;
                    } else {
                        total_pages_failed++;
                        migration.verified = false;
                    }
                }
            } else if (result < 0) {
                migration.verified = false;
                total_pages_failed += static_cast<int>(batch_size);
                int err = errno;
                if (err == EACCES || err == EPERM) {
                    rebalance_disabled_.store(true);
                    if (!logged_permission_error) {
                        std::cerr << "[CpuBackend] move_pages: permission denied - disabling NUMA rebalance "
                                  << "(need CAP_SYS_NICE)" << std::endl;
                        logged_permission_error = true;
                    }
                } else if (err == ESRCH) {
                    std::cerr << "[CpuBackend] move_pages: ESRCH - process not found" << std::endl;
                } else if (err != EINVAL && err != EACCES) {
                    std::cerr << "[CpuBackend] move_pages batch failed (errno=" << err << "): " << strerror(err)
                              << std::endl;
                }
            }
        }

        int verified_node = -1;
#ifdef DENSECORE_TEST_BUILD
        if (query_numa_node_range_test_hook_) {
            verified_node = query_numa_node_range_test_hook_(weight.ptr, weight.size);
        } else
#endif
        {
            verified_node = QueryMemoryNumaNodeRange(weight.ptr, weight.size);
        }
        migration.verified = migration.verified && migration.pages_on_target == static_cast<int>(num_pages) &&
                             verified_node == target_numa_node;
        return migration;
    };

    std::unordered_set<int> successfully_migrated;
    for (int expert_id : experts_to_migrate) {
        if (expert_id < 0 || expert_id >= n_experts || static_cast<size_t>(expert_id) >= expert_weights.size()) {
            continue;
        }

        // Hot-set churn can evict and later re-admit an expert even though its
        // pages never left this node. Do not issue move_pages() again for a
        // placement that the last fully verified migration already committed.
        if (profiler && target_numa_node >= 0 && profiler->GetExpertNumaNode(expert_id) == target_numa_node) {
            successfully_migrated.insert(expert_id);
            continue;
        }

        const ExpertWeights& weights = expert_weights[expert_id];
        const BufferMigrationResult w1 = migrate_buffer(weights.w1);
        const BufferMigrationResult w2 = migrate_buffer(weights.w2);
        const BufferMigrationResult w3 = migrate_buffer(weights.w3);
        const size_t expert_pages = static_cast<size_t>(w1.pages_on_target + w2.pages_on_target + w3.pages_on_target);
        total_pages_migrated += expert_pages;

        const bool migrated = w1.verified && w2.verified && w3.verified && expert_pages > 0;
        if (profiler && target_numa_node >= 0 && migrated) {
            profiler->SetExpertNumaNode(expert_id, target_numa_node);
            successfully_migrated.insert(expert_id);
        }
#if DENSECORE_NUMA_DEBUG
        if (!migrated) {
            std::cerr << "[NUMA] page migration was not verified for expert " << expert_id << " on node "
                      << target_numa_node << std::endl;
        }
#endif
    }

    if (local_expert_ids) {
        const std::unordered_set<int> previous_set(previous_local.begin(), previous_local.end());
        const size_t target_size = local_expert_ids->size();
        std::vector<int> committed;
        committed.reserve(target_size);
        std::unordered_set<int> committed_set;
        for (int expert_id : *local_expert_ids) {
            const bool was_local = previous_set.find(expert_id) != previous_set.end();
            const bool migration_succeeded = successfully_migrated.find(expert_id) != successfully_migrated.end();
            if ((was_local || migration_succeeded) && committed_set.insert(expert_id).second) {
                committed.push_back(expert_id);
            }
        }
        // A failed newcomer must not evict a previously committed sticky expert.
        for (int expert_id : previous_local) {
            if (committed.size() >= target_size) break;
            if (expert_id >= 0 && expert_id < n_experts && committed_set.insert(expert_id).second) {
                committed.push_back(expert_id);
            }
        }
        *local_expert_ids = std::move(committed);
    }

    if (total_pages_migrated > 0 || total_pages_failed > 0) {
        size_t bytes_migrated = total_pages_migrated * page_size;
        std::cerr << "[CpuBackend] NUMA rebalance: " << total_pages_migrated << " pages ("
                  << (bytes_migrated / (1024 * 1024)) << " MB) migrated to node " << target_numa_node;
        if (total_pages_failed > 0) {
            std::cerr << " (" << total_pages_failed << " pages failed)";
        }
        std::cerr << std::endl;
    }

    return total_pages_migrated;
#else
    if (local_expert_ids) {
        *local_expert_ids = previous_local;
    }
    if (target_numa_node < 0) {
        target_numa_node = 0;
    }
    size_t total_bytes = 0;
    for (int expert_id : experts_to_migrate) {
        if (expert_id >= 0 && expert_id < n_experts && static_cast<size_t>(expert_id) < expert_weights.size()) {
            const ExpertWeights& w = expert_weights[expert_id];
            total_bytes += w.w1.size + w.w2.size + w.w3.size;
        }
    }

    if (total_bytes > 0) {
        std::cerr << "[CpuBackend] NUMA rebalance (no-op): " << (total_bytes / (1024 * 1024))
                  << " MB targeted for node " << target_numa_node << std::endl;
    }
    return 0;
#endif
}

void CpuBackend::StartRebalanceThread(int interval_ms, int top_k, bool enable_page_migration) {
    if (interval_ms <= 0) {
        interval_ms = 5000;
    }
    if (top_k <= 0) {
        top_k = 4;
    }

    if (rebalance_running_.load()) {
        return;
    }
    if (rebalance_disabled_.load()) {
        std::cerr << "[CpuBackend] NUMA rebalance disabled; thread not started" << std::endl;
        return;
    }

    rebalance_stop_.store(false);
    rebalance_running_.store(true);
#ifdef DENSECORE_TEST_BUILD
    rebalance_test_cycles_.store(0);
    rebalance_test_interval_decreases_.store(0);
    rebalance_test_interval_increases_.store(0);
    rebalance_test_current_interval_ms_.store(interval_ms);
#endif

    rebalance_thread_ = std::thread([this, interval_ms, top_k, enable_page_migration]() {
        struct LayerRebalanceState {
            std::vector<int> previous_hot;
            uint64_t previous_total_hits = 0;
            std::chrono::steady_clock::time_point previous_time;
            double hit_rate_ema = 0.0;
            int migration_stable_cycles = 0;
            bool has_baseline = false;
        };

        int current_interval_ms = interval_ms;

        std::unordered_map<const TransformerLayer*, LayerRebalanceState> layer_states;

        while (!rebalance_stop_.load()) {
            if (rebalance_disabled_.load()) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(current_interval_ms));
            if (rebalance_stop_.load()) break;

            std::vector<std::pair<const TransformerLayer*, std::shared_ptr<MoELayerRegistry>>> registries;
            {
                std::lock_guard<std::mutex> lock(registry_mutex_);
                registries.reserve(moe_registries_.size());
                for (const auto& entry : moe_registries_) {
                    registries.emplace_back(entry.first, entry.second);
                }
            }

            bool any_high = false;
            bool all_low = !registries.empty();

            for (const auto& entry : registries) {
                const TransformerLayer* layer_key = entry.first;
                const auto& registry = entry.second;
                if (!registry) {
                    continue;
                }

                std::shared_ptr<moe::ExpertProfiler> profiler;
                std::vector<ExpertWeights> weights;
                {
                    std::lock_guard<std::mutex> lock(registry->mutex);
                    profiler = registry->profiler;
                    weights = registry->experts;
                }

                if (!profiler) {
                    all_low = false;
                    continue;
                }

                profiler->ApplyDecay();
                auto hot_experts = profiler->GetHotExperts(top_k);
                auto now = std::chrono::steady_clock::now();
                uint64_t total_hits = profiler->GetTotalHits();

                LayerRebalanceState& state = layer_states[layer_key];
                if (!state.has_baseline) {
                    state.previous_total_hits = total_hits;
                    state.previous_time = now;
                }

                double elapsed_sec =
                    std::chrono::duration_cast<std::chrono::duration<double>>(now - state.previous_time).count();
                if (elapsed_sec <= 0.0) {
                    elapsed_sec = static_cast<double>(current_interval_ms) / 1000.0;
                }

                uint64_t delta_hits =
                    (total_hits >= state.previous_total_hits) ? (total_hits - state.previous_total_hits) : 0;
                double hit_rate = (elapsed_sec > 0.0) ? (delta_hits / elapsed_sec) : 0.0;

                if (!state.has_baseline) {
                    state.hit_rate_ema = hit_rate;
                    state.has_baseline = true;
                } else {
                    state.hit_rate_ema = 0.8 * state.hit_rate_ema + 0.2 * hit_rate;
                }

                const float churn = ComputeHotSetChurn(state.previous_hot, hot_experts);

                const float churn_high = 0.35f;
                const float churn_low = 0.10f;
                const double high_rate_threshold = std::max(1000.0, 512.0 * static_cast<double>(top_k));
                bool high_churn = churn >= churn_high;
                bool low_churn = churn <= churn_low;
                bool high_rate =
                    (state.hit_rate_ema > 0.0 && hit_rate > state.hit_rate_ema * 1.5) || hit_rate > high_rate_threshold;

                state.migration_stable_cycles = NextMigrationStableCycles(
                    state.migration_stable_cycles, !state.previous_hot.empty(), churn, delta_hits);

                any_high = any_high || high_churn || high_rate;
                all_low = all_low && low_churn;

                if (enable_page_migration) {
                    if (state.migration_stable_cycles >= REBALANCE_MIGRATION_STABLE_CYCLES && !hot_experts.empty() &&
                        !weights.empty()) {
                        RebalanceExperts(layer_key, hot_experts, weights, profiler->GetNumExperts(), -1);
                    } else if (hot_experts.empty() || weights.empty()) {
                        all_low = false;
                    }
                }

                state.previous_hot = std::move(hot_experts);
                state.previous_total_hits = total_hits;
                state.previous_time = now;
            }

#ifdef DENSECORE_TEST_BUILD
            const int previous_interval_ms = current_interval_ms;
#endif
            current_interval_ms = NextRebalanceIntervalMs(current_interval_ms, interval_ms, any_high, all_low);
#ifdef DENSECORE_TEST_BUILD
            rebalance_test_cycles_.fetch_add(1);
            if (current_interval_ms < previous_interval_ms) {
                rebalance_test_interval_decreases_.fetch_add(1);
            } else if (current_interval_ms > previous_interval_ms) {
                rebalance_test_interval_increases_.fetch_add(1);
            }
            rebalance_test_current_interval_ms_.store(current_interval_ms);
#endif
        }
        rebalance_running_.store(false);
    });
}

void CpuBackend::StopRebalanceThread() {
    rebalance_stop_.store(true);

    if (rebalance_thread_.joinable()) {
        rebalance_thread_.join();
    }

    rebalance_running_.store(false);
}

#ifdef DENSECORE_TEST_BUILD
void CpuBackend::SetNumaRebalanceTestHooks(MovePagesTestHook move_pages_hook,
                                           QueryNumaNodeRangeTestHook query_range_hook) {
    std::lock_guard<std::mutex> lock(rebalance_mutex_);
    move_pages_test_hook_ = std::move(move_pages_hook);
    query_numa_node_range_test_hook_ = std::move(query_range_hook);
}

void CpuBackend::ClearNumaRebalanceTestHooks() {
    SetNumaRebalanceTestHooks({}, {});
}

std::vector<int> CpuBackend::GetLocalExpertIdsForTest(const TransformerLayer* layer_key) const {
    auto registry = GetMoELayerRegistry(layer_key);
    if (!registry) return {};
    std::lock_guard<std::mutex> lock(registry->mutex);
    return registry->local_expert_ids;
}

CpuBackend::NumaRebalanceTestStats CpuBackend::GetNumaRebalanceTestStats() const {
    return {
        rebalance_test_cycles_.load(),
        rebalance_test_interval_decreases_.load(),
        rebalance_test_interval_increases_.load(),
        rebalance_test_current_interval_ms_.load(),
    };
}
#endif

void CpuBackend::InitMoEProfiler(int n_experts, float ema_alpha) {
    InitMoEProfiler(nullptr, n_experts, ema_alpha);
}

void CpuBackend::InitMoEProfiler(const TransformerLayer* layer_key, int n_experts, float ema_alpha) {
    if (n_experts <= 0) {
        return;
    }

    auto registry = GetOrCreateMoELayerRegistry(layer_key, n_experts, ema_alpha);
    if (!registry) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(rebalance_mutex_);
        std::lock_guard<std::mutex> registry_lock(registry->mutex);
        const bool recreate = !registry->profiler || registry->profiler->GetNumExperts() != n_experts;
        registry->ema_alpha = ema_alpha;
        if (recreate) {
            registry->profiler = std::make_shared<moe::ExpertProfiler>(n_experts, ema_alpha);
            registry->local_expert_ids.clear();
        }
    }
}

moe::ExpertProfiler* CpuBackend::GetProfiler(const TransformerLayer* layer_key) {
    auto registry = GetMoELayerRegistry(layer_key);
    if (!registry) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(registry->mutex);
    return registry->profiler.get();
}

int CpuBackend::GetExpertNumaNode(const TransformerLayer* layer_key, int expert_id) const {
    auto registry = GetMoELayerRegistry(layer_key);
    if (!registry) {
        return -1;
    }
    std::shared_ptr<moe::ExpertProfiler> profiler;
    {
        std::lock_guard<std::mutex> lock(registry->mutex);
        profiler = registry->profiler;
    }
    return profiler ? profiler->GetExpertNumaNode(expert_id) : -1;
}

bool CpuBackend::CopyExpertNumaNodes(const TransformerLayer* layer_key, int n_experts, std::vector<int>* nodes) const {
    if (!nodes || n_experts <= 0) {
        return false;
    }
    auto registry = GetMoELayerRegistry(layer_key);
    if (!registry) {
        return false;
    }
    std::shared_ptr<moe::ExpertProfiler> profiler;
    {
        std::lock_guard<std::mutex> lock(registry->mutex);
        profiler = registry->profiler;
    }
    if (!profiler || profiler->GetNumExperts() < n_experts) {
        return false;
    }
    nodes->resize(static_cast<size_t>(n_experts));
    for (int expert_id = 0; expert_id < n_experts; ++expert_id) {
        (*nodes)[static_cast<size_t>(expert_id)] = profiler->GetExpertNumaNode(expert_id);
    }
    return true;
}

void CpuBackend::RegisterMoEExperts(const std::vector<ExpertWeights>& experts) {
    RegisterMoEExperts(nullptr, experts);
}

void CpuBackend::RegisterMoEExperts(const TransformerLayer* layer_key, const std::vector<ExpertWeights>& experts) {
    if (experts.empty()) {
        return;
    }

    auto registry = GetOrCreateMoELayerRegistry(layer_key, static_cast<int>(experts.size()), 0.1f);
    if (!registry) {
        return;
    }

    auto experts_match = [](const std::vector<ExpertWeights>& a, const std::vector<ExpertWeights>& b) {
        if (a.size() != b.size()) {
            return false;
        }
        for (size_t i = 0; i < a.size(); ++i) {
            const ExpertWeights& lhs = a[i];
            const ExpertWeights& rhs = b[i];
            if (lhs.w1.ptr != rhs.w1.ptr || lhs.w2.ptr != rhs.w2.ptr || lhs.w3.ptr != rhs.w3.ptr ||
                lhs.w1.size != rhs.w1.size || lhs.w2.size != rhs.w2.size || lhs.w3.size != rhs.w3.size ||
                lhs.hidden_dim != rhs.hidden_dim || lhs.intermediate_dim != rhs.intermediate_dim ||
                lhs.numa_node_hint != rhs.numa_node_hint) {
                return false;
            }
        }
        return true;
    };

    std::lock_guard<std::mutex> rebalance_lock(rebalance_mutex_);
    std::lock_guard<std::mutex> registry_lock(registry->mutex);
    if (experts_match(registry->experts, experts)) {
        return;
    }

    registry->experts = experts;
    registry->local_expert_ids.clear();
    registry->last_batch_experts.clear();
    registry->dequant_cache.clear();
    registry->dequant_cache_bytes = 0;
    registry->dequant_cache_use_counter = 0;

    if (registry->profiler) {
        int detected_count = 0;
        auto detect_expert_numa_node = [this](const ExpertWeights& expert) -> int {
            if (expert.numa_node_hint >= 0 && expert.numa_node_hint < GetNumaNodeCount()) {
                return expert.numa_node_hint;
            }
            int verified_node = -1;
            int verified_weights = 0;
            const auto verify_weight = [this, &verified_node, &verified_weights](const ExpertWeight& weight) {
                if (!weight.ptr || weight.size == 0) {
                    return true;
                }
                // Load-time expert partitioning can leave a small number of
                // shared or kernel-unmovable pages in a large tensor. Initial
                // routing needs the dominant residency, while rebalance commit
                // verification below deliberately remains all-pages strict.
                const int node = QueryDominantInitialNumaNode(this, weight);
                if (node < 0) {
                    return false;
                }
                if (verified_node >= 0 && node != verified_node) {
                    return false;
                }
                verified_node = node;
                ++verified_weights;
                return true;
            };

            if (!verify_weight(expert.w1) || !verify_weight(expert.w2) || !verify_weight(expert.w3)) {
                return -1;
            }
            return verified_weights > 0 ? verified_node : -1;
        };

        for (size_t i = 0; i < experts.size(); ++i) {
            int numa_node = detect_expert_numa_node(experts[i]);
            if (numa_node >= 0) {
                registry->profiler->SetExpertNumaNode(static_cast<int>(i), numa_node);
                detected_count++;
#if DENSECORE_NUMA_DEBUG
                std::cerr << "[NUMA] Expert " << i << " initially on node " << numa_node << std::endl;
#endif
            }
        }
#if DENSECORE_NUMA_DEBUG
        std::cerr << "[NUMA] Detected initial placement for " << detected_count << "/" << experts.size() << " experts"
                  << std::endl;
#endif
    }

    std::cerr << "[CpuBackend] Registered " << experts.size() << " MoE experts for layer " << layer_key
              << " NUMA rebalancing" << std::endl;
}

std::vector<CpuBackend::ExpertWeights> CpuBackend::GetRegisteredExperts() const {
    return GetRegisteredExperts(nullptr);
}

std::vector<CpuBackend::ExpertWeights> CpuBackend::GetRegisteredExperts(const TransformerLayer* layer_key) const {
    auto registry = GetMoELayerRegistry(layer_key);
    if (!registry) {
        return {};
    }
    std::lock_guard<std::mutex> lock(registry->mutex);
    return registry->experts;
}

bool CpuBackend::GetRegisteredExpertsView(const TransformerLayer* layer_key, const ExpertWeights** experts,
                                          int* count) const {
    if (experts) {
        *experts = nullptr;
    }
    if (count) {
        *count = 0;
    }

    auto registry = GetMoELayerRegistry(layer_key);
    if (!registry) {
        return false;
    }

    std::lock_guard<std::mutex> lock(registry->mutex);
    if (registry->experts.empty()) {
        return false;
    }

    if (experts) {
        *experts = registry->experts.data();
    }
    if (count) {
        *count = static_cast<int>(registry->experts.size());
    }
    return true;
}

CpuBackend::MoERuntimeStatsSnapshot CpuBackend::GetMoERuntimeStatsSnapshot() const {
    MoERuntimeStatsSnapshot snapshot;
    snapshot.batches = moe_stats_batches_.load(std::memory_order_relaxed);
    snapshot.total_active_experts = moe_stats_total_active_experts_.load(std::memory_order_relaxed);
    snapshot.total_assignments = moe_stats_total_assignments_.load(std::memory_order_relaxed);
    snapshot.total_local_hot_experts = moe_stats_total_local_hot_experts_.load(std::memory_order_relaxed);
    snapshot.total_reuse_intersection = moe_stats_total_reuse_intersection_.load(std::memory_order_relaxed);
    snapshot.total_reuse_union = moe_stats_total_reuse_union_.load(std::memory_order_relaxed);
    snapshot.total_max_expert_batch = moe_stats_total_max_expert_batch_.load(std::memory_order_relaxed);
    snapshot.total_ordering_considered = moe_stats_total_ordering_considered_.load(std::memory_order_relaxed);
    snapshot.total_ordering_applied = moe_stats_total_ordering_applied_.load(std::memory_order_relaxed);
    snapshot.total_ordering_skipped_small_batch =
        moe_stats_total_ordering_skipped_small_batch_.load(std::memory_order_relaxed);
    snapshot.total_ordering_skipped_low_reuse =
        moe_stats_total_ordering_skipped_low_reuse_.load(std::memory_order_relaxed);
    snapshot.total_ordering_numa_switches_before =
        moe_stats_total_ordering_numa_switches_before_.load(std::memory_order_relaxed);
    snapshot.total_ordering_numa_switches_after =
        moe_stats_total_ordering_numa_switches_after_.load(std::memory_order_relaxed);
    snapshot.total_prefetch_candidates = moe_stats_total_prefetch_candidates_.load(std::memory_order_relaxed);
    snapshot.total_prefetch_calls = moe_stats_total_prefetch_calls_.load(std::memory_order_relaxed);
    snapshot.total_prefetch_bytes = moe_stats_total_prefetch_bytes_.load(std::memory_order_relaxed);
    snapshot.total_prefetch_skipped_distance =
        moe_stats_total_prefetch_skipped_distance_.load(std::memory_order_relaxed);
    snapshot.total_prefetch_skipped_pressure =
        moe_stats_total_prefetch_skipped_pressure_.load(std::memory_order_relaxed);
    snapshot.total_prefetch_skipped_signal = moe_stats_total_prefetch_skipped_signal_.load(std::memory_order_relaxed);
    snapshot.total_cached_experts = moe_stats_total_cached_experts_.load(std::memory_order_relaxed);
    snapshot.total_dequantized_experts = moe_stats_total_dequantized_experts_.load(std::memory_order_relaxed);
    snapshot.total_dequantized_bytes = moe_stats_total_dequantized_bytes_.load(std::memory_order_relaxed);
    return snapshot;
}

}  // namespace densecore
