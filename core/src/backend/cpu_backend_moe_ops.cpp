#include "backend/cpu_backend_internal.h"

#include <algorithm>
#include <limits>
#include <unordered_set>

namespace densecore {

namespace {

struct AlignedScratch {
    float* ptr = nullptr;
    size_t capacity = 0;

    ~AlignedScratch() {
        if (ptr) {
            free(ptr);
        }
    }

    void Resize(CpuBackend* b, size_t required) {
        if (required > capacity) {
            if (ptr) free(ptr);
            ptr = static_cast<float*>(b->AllocateDevice(required * sizeof(float)));
            capacity = required;
        }
    }
};

}  // namespace

void CpuBackend::ApplyMultiLoRA(
    const Tensor& input, const std::string& layer_name,
    const std::unordered_map<std::shared_ptr<LoRAAdapter>, std::vector<int>>& adapter_token_map, Tensor* output) {
    if (!output || !input.IsValid() || !output->IsValid()) {
        return;
    }
    if (adapter_token_map.empty()) {
        return;
    }
    if (input.dtype != DType::F32 || output->dtype != DType::F32) {
        return;
    }
    if (input.ndim != 2 || output->ndim != 2) {
        return;
    }

    const int64_t total_tokens = input.shape[0];
    const int64_t input_dim = input.shape[1];
    const int64_t output_dim = output->shape[1];
    if (input_dim <= 0 || output_dim <= 0) {
        return;
    }

    const float* input_data = input.DataAs<float>();
    float* output_data = output->DataAs<float>();

    struct AdapterTask {
        const LoRAAdapter* adapter = nullptr;
        const std::vector<int>* indices = nullptr;
        int numa_node = -1;
    };

    std::unordered_map<int, std::vector<AdapterTask>> tasks_by_node;
    tasks_by_node.reserve(adapter_token_map.size());

    for (const auto& entry : adapter_token_map) {
        const LoRAAdapter* adapter = entry.first.get();
        const std::vector<int>& indices = entry.second;
        if (!adapter || indices.empty()) {
            continue;
        }

        int numa_node = -1;
        if (!adapter->weights.empty()) {
            const auto& weight = adapter->weights.begin()->second;
            if (weight.lora_a && weight.lora_a->data) {
                numa_node = QueryMemoryNumaNode(weight.lora_a->data);
            }
            if (numa_node < 0 && weight.lora_b && weight.lora_b->data) {
                numa_node = QueryMemoryNumaNode(weight.lora_b->data);
            }
        }

        tasks_by_node[numa_node].push_back(AdapterTask{adapter, &indices, numa_node});
    }

    for (auto& group : tasks_by_node) {
        auto& tasks = group.second;
        if (tasks.empty()) {
            continue;
        }

        auto& pool = GetThreadPool(group.first);
        const int task_count = static_cast<int>(tasks.size());

        pool.ParallelFor(task_count, [&](int start, int end, int) {
            static thread_local AlignedScratch input_scratch;
            static thread_local AlignedScratch down_scratch;
            static thread_local AlignedScratch out_scratch;
            static thread_local std::vector<float> lora_a_f32;
            static thread_local std::vector<float> lora_b_f32;

            for (int t = start; t < end; ++t) {
                const AdapterTask& task = tasks[t];
                const LoRAAdapter* adapter = task.adapter;
                const std::vector<int>& indices = *task.indices;
                if (!adapter || indices.empty()) {
                    continue;
                }

                auto it_w = adapter->weights.find(layer_name);
                if (it_w == adapter->weights.end()) {
                    continue;
                }
                const densecore::LoRALayerWeight* layer_weight = &it_w->second;
                if (!layer_weight || !layer_weight->lora_a || !layer_weight->lora_b) {
                    continue;
                }

                const ggml_tensor* lora_a = layer_weight->lora_a;
                const ggml_tensor* lora_b = layer_weight->lora_b;

                const int64_t lora_a_in = lora_a->ne[0];
                const int64_t lora_a_rank = lora_a->ne[1];
                const int64_t lora_b_rank = lora_b->ne[0];
                const int64_t lora_b_out = lora_b->ne[1];
                const int64_t rank = std::min<int64_t>({layer_weight->rank, lora_a_rank, lora_b_rank});

                if (lora_a_in != input_dim || lora_b_out != output_dim || rank <= 0) {
                    continue;
                }

                const float* lora_a_ptr = internal::GetLoRAWeightF32(lora_a, lora_a_f32);
                const float* lora_b_ptr = internal::GetLoRAWeightF32(lora_b, lora_b_f32);
                if (!lora_a_ptr || !lora_b_ptr) {
                    continue;
                }

                const int64_t token_count = static_cast<int64_t>(indices.size());
                if (token_count <= 0) {
                    continue;
                }

                input_scratch.Resize(this, static_cast<size_t>(token_count * input_dim));
                down_scratch.Resize(this, static_cast<size_t>(token_count * rank));
                out_scratch.Resize(this, static_cast<size_t>(token_count * output_dim));

                float* input_subset = input_scratch.ptr;
                float* down = down_scratch.ptr;
                float* out = out_scratch.ptr;

                for (int64_t i = 0; i < token_count; ++i) {
                    const int idx = indices[i];
                    if (idx < 0 || idx >= total_tokens) {
                        continue;
                    }
                    std::memcpy(input_subset + i * input_dim, input_data + static_cast<int64_t>(idx) * input_dim,
                                static_cast<size_t>(input_dim) * sizeof(float));
                }

                native::GemmF32(down, input_subset, lora_a_ptr, static_cast<int>(token_count), static_cast<int>(rank),
                                static_cast<int>(input_dim));
                native::GemmF32(out, down, lora_b_ptr, static_cast<int>(token_count), static_cast<int>(output_dim),
                                static_cast<int>(rank));

                const float scale = adapter->scale;
                for (int64_t i = 0; i < token_count; ++i) {
                    const int idx = indices[i];
                    if (idx < 0 || idx >= total_tokens) {
                        continue;
                    }
                    float* dst = output_data + static_cast<int64_t>(idx) * output_dim;
                    const float* src = out + i * output_dim;
                    if (scale == 1.0f) {
                        for (int64_t j = 0; j < output_dim; ++j) {
                            dst[j] += src[j];
                        }
                    } else {
                        for (int64_t j = 0; j < output_dim; ++j) {
                            dst[j] += src[j] * scale;
                        }
                    }
                }
            }
        });
    }
}

void CpuBackend::DispatchExpertFFN(int expert_id, const Tensor& input, const Tensor& w1, const Tensor& w2,
                                   const Tensor& w3, Tensor* output) {
    DispatchExpertFFN(nullptr, expert_id, input, w1, w2, w3, output);
}

void CpuBackend::DispatchExpertFFN(const TransformerLayer* layer_key, int expert_id, const Tensor& input,
                                   const Tensor& w1, const Tensor& w2, const Tensor& w3, Tensor* output) {
    int numa_node = -1;
    auto registry = GetMoELayerRegistry(layer_key);
    if (registry) {
        std::shared_ptr<moe::ExpertProfiler> profiler;
        {
            std::lock_guard<std::mutex> lock(registry->mutex);
            profiler = registry->profiler;
        }
        if (profiler) {
            numa_node = profiler->GetExpertNumaNode(expert_id);
        }
    }

    static thread_local AlignedScratch hidden_scratch;
    static thread_local AlignedScratch gate_scratch;

    const int64_t batch = input.shape[0];
    const int64_t intermediate_dim = w1.shape[0];
    const size_t hidden_size = static_cast<size_t>(batch * intermediate_dim);

    hidden_scratch.Resize(this, hidden_size);
    if (w3.IsValid()) {
        gate_scratch.Resize(this, hidden_size);
    }

    Tensor hidden = Tensor::Make2D(hidden_scratch.ptr, batch, intermediate_dim);
    MatMulTransB(input, w1, &hidden, numa_node);

    if (w3.IsValid()) {
        Tensor gate = Tensor::Make2D(gate_scratch.ptr, batch, intermediate_dim);
        MatMulTransB(input, w3, &gate, numa_node);

        auto& pool = GetThreadPool(numa_node);
        const int total = static_cast<int>(hidden_size);
        float* h_ptr = hidden_scratch.ptr;
        const float* g_ptr = gate_scratch.ptr;

        pool.ParallelFor(total, [=](int start, int end, int) {
            for (int i = start; i < end; i++) {
                float x = h_ptr[i];
                float silu = x / (1.0f + internal::FastExp(-x));
                h_ptr[i] = silu * g_ptr[i];
            }
        });
    }

    MatMulTransB(hidden, w2, output, numa_node);
}

void CpuBackend::ForwardMoE(const Tensor& input, const moe::MoERouteResult& routing,
                            const std::vector<ExpertWeights>& experts, Tensor* output) {
    ForwardMoE(nullptr, input, routing, experts.data(), static_cast<int>(experts.size()), output);
}

void CpuBackend::ForwardMoE(const TransformerLayer* layer_key, const Tensor& input, const moe::MoERouteResult& routing,
                            const std::vector<ExpertWeights>& experts, Tensor* output) {
    ForwardMoE(layer_key, input, routing, experts.data(), static_cast<int>(experts.size()), output);
}

void CpuBackend::ForwardMoE(const TransformerLayer* layer_key, const Tensor& input, const moe::MoERouteResult& routing,
                            const ExpertWeights* experts, int num_experts, Tensor* output) {
    const int batch_size = routing.batch_size;
    const int top_k = routing.top_k;
    const size_t hidden_dim = input.shape[0];
    const size_t assignment_count = routing.expert_ids.size();

    if (!experts || batch_size <= 0 || top_k <= 0 || num_experts <= 0 || assignment_count == 0) {
        return;
    }
    if (routing.weights.size() != assignment_count) {
        return;
    }
    if (!routing.token_indices.empty() && routing.token_indices.size() != assignment_count) {
        return;
    }
    if (routing.token_indices.empty() && assignment_count != static_cast<size_t>(batch_size * top_k)) {
        return;
    }

    auto registry = GetMoELayerRegistry(layer_key);
    if (registry) {
        std::shared_ptr<moe::ExpertProfiler> profiler;
        {
            std::lock_guard<std::mutex> lock(registry->mutex);
            profiler = registry->profiler;
        }
        if (profiler) {
            profiler->RecordHitBatch(routing.expert_ids.data(), static_cast<int>(routing.expert_ids.size()));
        }
    }

    float* out_data = output->DataAs<float>();
    std::memset(out_data, 0, batch_size * hidden_dim * sizeof(float));

    static thread_local AlignedScratch routing_scratch;
    static thread_local AlignedScratch expert_input_scratch;
    static thread_local AlignedScratch expert_output_scratch;
    static thread_local AlignedScratch w1_dequant;
    static thread_local AlignedScratch w2_dequant;
    static thread_local AlignedScratch w3_dequant;
    static thread_local AlignedScratch small_decode_output_scratch;

    const int total_assignments = static_cast<int>(assignment_count);
    if (total_assignments == 0 || num_experts == 0 || top_k <= 0) {
        return;
    }

    const bool has_token_indices = !routing.token_indices.empty();
    const float* input_data = input.DataAs<float>();

    std::unordered_set<int> small_step_local_hot_experts;
    std::vector<int> small_step_previous_batch_experts;
    if (registry) {
        std::lock_guard<std::mutex> lock(registry->mutex);
        small_step_local_hot_experts.insert(registry->local_expert_ids.begin(), registry->local_expert_ids.end());
        small_step_previous_batch_experts = registry->last_batch_experts;
    }

    const bool small_decode_candidate = batch_size <= 4 && total_assignments <= 8;
    std::vector<int> small_step_current_batch_experts;
    small_step_current_batch_experts.reserve(static_cast<size_t>(std::min(total_assignments, num_experts)));
    int small_step_max_expert_batch = 0;
    if (small_decode_candidate) {
        for (int i = 0; i < total_assignments; ++i) {
            const int expert_id = routing.expert_ids[static_cast<size_t>(i)];
            if (expert_id < 0 || expert_id >= num_experts) {
                continue;
            }
            bool seen = false;
            int expert_batch_count = 0;
            for (int j = 0; j < total_assignments; ++j) {
                if (routing.expert_ids[static_cast<size_t>(j)] == expert_id) {
                    ++expert_batch_count;
                }
            }
            small_step_max_expert_batch = std::max(small_step_max_expert_batch, expert_batch_count);
            for (int existing : small_step_current_batch_experts) {
                if (existing == expert_id) {
                    seen = true;
                    break;
                }
            }
            if (!seen) {
                small_step_current_batch_experts.push_back(expert_id);
            }
        }
    }

    if (small_decode_candidate && small_step_max_expert_batch <= 1 && !small_step_current_batch_experts.empty()) {
        std::unordered_set<int> previous_batch_set;
        int reuse_intersection = 0;
        if (!small_step_previous_batch_experts.empty()) {
            previous_batch_set.insert(small_step_previous_batch_experts.begin(), small_step_previous_batch_experts.end());
            for (int expert_id : small_step_current_batch_experts) {
                if (previous_batch_set.find(expert_id) != previous_batch_set.end()) {
                    ++reuse_intersection;
                }
            }
        }
        const int reuse_union = static_cast<int>(small_step_current_batch_experts.size() +
                                                 small_step_previous_batch_experts.size() - reuse_intersection);
        int local_hot_count = 0;
        for (int expert_id : small_step_current_batch_experts) {
            if (small_step_local_hot_experts.find(expert_id) != small_step_local_hot_experts.end()) {
                ++local_hot_count;
            }
        }

        moe_stats_batches_.fetch_add(1, std::memory_order_relaxed);
        moe_stats_total_active_experts_.fetch_add(static_cast<uint64_t>(small_step_current_batch_experts.size()),
                                                  std::memory_order_relaxed);
        moe_stats_total_assignments_.fetch_add(static_cast<uint64_t>(total_assignments), std::memory_order_relaxed);
        moe_stats_total_local_hot_experts_.fetch_add(static_cast<uint64_t>(local_hot_count), std::memory_order_relaxed);
        moe_stats_total_reuse_intersection_.fetch_add(static_cast<uint64_t>(reuse_intersection),
                                                      std::memory_order_relaxed);
        moe_stats_total_reuse_union_.fetch_add(static_cast<uint64_t>(std::max(0, reuse_union)),
                                               std::memory_order_relaxed);
        moe_stats_total_max_expert_batch_.fetch_add(static_cast<uint64_t>(small_step_max_expert_batch),
                                                    std::memory_order_relaxed);

        if (registry) {
            std::lock_guard<std::mutex> lock(registry->mutex);
            registry->last_batch_experts = small_step_current_batch_experts;
        }

        auto make_weight_f32_small = [&](void* ptr, int ggml_type_id, int64_t rows, int64_t cols, AlignedScratch& scratch,
                                         size_t* dequantized_bytes, bool* dequantized_any) -> Tensor {
            if (!ptr || rows <= 0 || cols <= 0) {
                return Tensor::Make2D(ptr, rows, cols);
            }
            const ggml_type wtype = static_cast<ggml_type>(ggml_type_id);
            if (wtype == GGML_TYPE_F32) {
                return Tensor::Make2D(ptr, rows, cols);
            }
            const struct ggml_type_traits* traits = ggml_get_type_traits(wtype);
            if (!traits || !traits->to_float) {
                return Tensor::Make2D(ptr, rows, cols);
            }
            const size_t row_bytes = ggml_row_size(wtype, cols);
            scratch.Resize(this, static_cast<size_t>(rows * cols));
            const char* src = static_cast<const char*>(ptr);
            for (int64_t r = 0; r < rows; ++r) {
                traits->to_float(src + r * static_cast<ptrdiff_t>(row_bytes), scratch.ptr + r * cols, cols);
            }
            if (dequantized_bytes) {
                *dequantized_bytes += static_cast<size_t>(rows * cols * sizeof(float));
            }
            if (dequantized_any) {
                *dequantized_any = true;
            }
            return Tensor::Make2D(scratch.ptr, rows, cols);
        };

        small_decode_output_scratch.Resize(this, hidden_dim);
        for (int i = 0; i < total_assignments; ++i) {
            const int expert_id = routing.expert_ids[static_cast<size_t>(i)];
            if (expert_id < 0 || expert_id >= num_experts) {
                continue;
            }
            const int token_idx = has_token_indices ? routing.token_indices[static_cast<size_t>(i)] : (i / top_k);
            if (token_idx < 0 || token_idx >= batch_size) {
                continue;
            }

            const float weight = routing.weights[static_cast<size_t>(i)];
            if (weight == 0.0f) {
                continue;
            }

            const ExpertWeights& exp = experts[static_cast<size_t>(expert_id)];
            size_t dequantized_bytes = 0;
            bool dequantized_any = false;
            Tensor w1 = make_weight_f32_small(exp.w1.ptr, exp.w1_type, static_cast<int64_t>(exp.intermediate_dim),
                                              static_cast<int64_t>(exp.hidden_dim), w1_dequant, &dequantized_bytes,
                                              &dequantized_any);
            Tensor w2 = make_weight_f32_small(exp.w2.ptr, exp.w2_type, static_cast<int64_t>(exp.hidden_dim),
                                              static_cast<int64_t>(exp.intermediate_dim), w2_dequant,
                                              &dequantized_bytes, &dequantized_any);
            Tensor w3;
            if (exp.w3.ptr != nullptr) {
                w3 = make_weight_f32_small(exp.w3.ptr, exp.w3_type, static_cast<int64_t>(exp.intermediate_dim),
                                           static_cast<int64_t>(exp.hidden_dim), w3_dequant, &dequantized_bytes,
                                           &dequantized_any);
            }
            if (dequantized_any) {
                moe_stats_total_dequantized_experts_.fetch_add(1, std::memory_order_relaxed);
                moe_stats_total_dequantized_bytes_.fetch_add(static_cast<uint64_t>(dequantized_bytes),
                                                             std::memory_order_relaxed);
            }

            Tensor expert_input =
                Tensor::Make2D(const_cast<float*>(input_data + static_cast<size_t>(token_idx) * hidden_dim), 1,
                               static_cast<int64_t>(hidden_dim));
            Tensor expert_out = Tensor::Make2D(small_decode_output_scratch.ptr, 1, static_cast<int64_t>(hidden_dim));
            DispatchExpertFFN(layer_key, expert_id, expert_input, w1, w2, w3, &expert_out);

            float* dst = out_data + static_cast<size_t>(token_idx) * hidden_dim;
            const float* src = small_decode_output_scratch.ptr;
            for (size_t d = 0; d < hidden_dim; ++d) {
                dst[d] += weight * src[d];
            }
        }
        return;
    }

    const size_t workspace_bytes = moe::GetMoERoutingWorkspaceSize(batch_size, num_experts, top_k);
    const size_t workspace_floats = (workspace_bytes + sizeof(float) - 1) / sizeof(float);
    routing_scratch.Resize(this, workspace_floats);

    moe::MoERoutingWorkspace ws;
    if (!moe::InitMoERoutingWorkspace(&ws, routing_scratch.ptr, routing_scratch.capacity * sizeof(float), batch_size,
                                      num_experts, top_k)) {
        return;
    }

    moe::MoEReorderMapView reorder_map;
    if (!moe::BuildMoEReorderMap(routing, num_experts, &reorder_map, &ws)) {
        return;
    }
    if (reorder_map.total_assignments == 0) {
        return;
    }

    const int hidden_dim_i = static_cast<int>(hidden_dim);
    const size_t packed_size = static_cast<size_t>(reorder_map.total_assignments) * hidden_dim;
    expert_input_scratch.Resize(this, packed_size);
    expert_output_scratch.Resize(this, packed_size);

    float* packed_input = expert_input_scratch.ptr;
    float* packed_output = expert_output_scratch.ptr;
    auto& reorder_pool = GetThreadPool(-1);
    moe::ReorderInputs(input_data, batch_size, hidden_dim_i, reorder_map, packed_input, &reorder_pool);

    std::shared_ptr<moe::ExpertProfiler> profiler;
    std::unordered_set<int> local_hot_experts;
    std::vector<int> previous_batch_experts;
    if (registry) {
        std::lock_guard<std::mutex> lock(registry->mutex);
        profiler = registry->profiler;
        local_hot_experts.insert(registry->local_expert_ids.begin(), registry->local_expert_ids.end());
        previous_batch_experts = registry->last_batch_experts;
    }

    struct ActiveExpertWork {
        int expert_id = -1;
        int start = 0;
        int count = 0;
        int numa_node = -1;
        float ema_load = 0.0f;
        bool local_hot = false;
    };

    std::vector<ActiveExpertWork> active_work;
    active_work.reserve(static_cast<size_t>(num_experts));
    std::vector<int> current_batch_experts;
    current_batch_experts.reserve(static_cast<size_t>(num_experts));

    for (int expert_id = 0; expert_id < num_experts; ++expert_id) {
        const int start = reorder_map.expert_offsets[static_cast<size_t>(expert_id)];
        const int end = reorder_map.expert_offsets[static_cast<size_t>(expert_id + 1)];
        const int count = end - start;
        if (count <= 0) {
            continue;
        }

        ActiveExpertWork work;
        work.expert_id = expert_id;
        work.start = start;
        work.count = count;
        work.local_hot = local_hot_experts.find(expert_id) != local_hot_experts.end();
        if (profiler) {
            work.numa_node = profiler->GetExpertNumaNode(expert_id);
            work.ema_load = profiler->GetEmaLoad(expert_id);
        }
        active_work.push_back(work);
        current_batch_experts.push_back(expert_id);
    }

    if (active_work.empty()) {
        return;
    }

    int reuse_intersection = 0;
    std::unordered_set<int> previous_batch_set;
    if (!previous_batch_experts.empty()) {
        previous_batch_set.insert(previous_batch_experts.begin(), previous_batch_experts.end());
        for (int expert_id : current_batch_experts) {
            if (previous_batch_set.find(expert_id) != previous_batch_set.end()) {
                ++reuse_intersection;
            }
        }
    }
    const int reuse_union =
        static_cast<int>(current_batch_experts.size() + previous_batch_experts.size() - reuse_intersection);
    const int max_expert_batch = reorder_map.max_expert_batch;
    const int local_hot_count =
        static_cast<int>(std::count_if(active_work.begin(), active_work.end(), [](const ActiveExpertWork& work) {
            return work.local_hot;
        }));
    const int worker_threads = std::max(1, reorder_pool.GetNumThreads());
    const bool small_decode_step = batch_size <= 4 && total_assignments <= 8 && max_expert_batch <= 1;

    moe_stats_batches_.fetch_add(1, std::memory_order_relaxed);
    moe_stats_total_active_experts_.fetch_add(static_cast<uint64_t>(active_work.size()), std::memory_order_relaxed);
    moe_stats_total_assignments_.fetch_add(static_cast<uint64_t>(reorder_map.total_assignments), std::memory_order_relaxed);
    moe_stats_total_local_hot_experts_.fetch_add(static_cast<uint64_t>(local_hot_count), std::memory_order_relaxed);
    moe_stats_total_reuse_intersection_.fetch_add(static_cast<uint64_t>(reuse_intersection), std::memory_order_relaxed);
    moe_stats_total_reuse_union_.fetch_add(static_cast<uint64_t>(std::max(0, reuse_union)), std::memory_order_relaxed);
    moe_stats_total_max_expert_batch_.fetch_add(static_cast<uint64_t>(std::max(0, max_expert_batch)),
                                                std::memory_order_relaxed);

    if (registry) {
        std::lock_guard<std::mutex> lock(registry->mutex);
        registry->last_batch_experts = current_batch_experts;
    }

    if (!small_decode_step && internal::IsMoELocalityOrderingEnabled()) {
        moe_stats_total_ordering_considered_.fetch_add(1, std::memory_order_relaxed);

        const bool enough_active_experts =
            static_cast<int>(active_work.size()) >= internal::GetMoELocalityOrderingMinActiveExperts();
        const bool enough_reuse_signal =
            reuse_intersection >= internal::GetMoELocalityOrderingMinReuseIntersection() || local_hot_count > 0;

        auto count_numa_switches = [](const std::vector<ActiveExpertWork>& work_items) -> uint64_t {
            uint64_t switches = 0;
            for (size_t i = 1; i < work_items.size(); ++i) {
                const int prev = work_items[i - 1].numa_node;
                const int cur = work_items[i].numa_node;
                if (prev >= 0 && cur >= 0 && prev != cur) {
                    ++switches;
                }
            }
            return switches;
        };

        if (!enough_active_experts) {
            moe_stats_total_ordering_skipped_small_batch_.fetch_add(1, std::memory_order_relaxed);
        } else if (!enough_reuse_signal) {
            moe_stats_total_ordering_skipped_low_reuse_.fetch_add(1, std::memory_order_relaxed);
        } else {
            const uint64_t switches_before = count_numa_switches(active_work);
            moe_stats_total_ordering_numa_switches_before_.fetch_add(switches_before, std::memory_order_relaxed);

            auto ordering_score = [&previous_batch_set](const ActiveExpertWork& work) -> int {
                const bool reused = previous_batch_set.find(work.expert_id) != previous_batch_set.end();
                int score = 0;
                if (work.local_hot) score += 32;
                if (reused) score += 24;
                if (work.numa_node >= 0) score += 4;
                score += std::min(work.count, 4) * 3;
                return score;
            };

            std::stable_sort(active_work.begin(), active_work.end(),
                             [&ordering_score](const ActiveExpertWork& lhs, const ActiveExpertWork& rhs) {
                                 const int lhs_score = ordering_score(lhs);
                                 const int rhs_score = ordering_score(rhs);
                                 if (lhs_score != rhs_score) return lhs_score > rhs_score;
                                 if (lhs.ema_load != rhs.ema_load) return lhs.ema_load < rhs.ema_load;
                                 if (lhs.count != rhs.count) return lhs.count < rhs.count;
                                 if (lhs.numa_node != rhs.numa_node) return lhs.numa_node < rhs.numa_node;
                                 return lhs.expert_id < rhs.expert_id;
                             });

            const uint64_t switches_after = count_numa_switches(active_work);
            moe_stats_total_ordering_applied_.fetch_add(1, std::memory_order_relaxed);
            moe_stats_total_ordering_numa_switches_after_.fetch_add(switches_after, std::memory_order_relaxed);
        }
    }

    const bool dequant_cache_enabled = !small_decode_step && internal::IsMoEDequantCacheEnabled() && registry != nullptr;
    const bool cache_all_active_experts = internal::ShouldCacheAllActiveExperts();
    const size_t dequant_cache_budget = internal::GetMoEDequantCacheBytes();

    auto make_weight_f32 = [&](void* ptr, int ggml_type_id, int64_t rows, int64_t cols, AlignedScratch& scratch,
                               size_t* dequantized_bytes, bool* dequantized_any) -> Tensor {
        if (!ptr || rows <= 0 || cols <= 0) {
            return Tensor::Make2D(ptr, rows, cols);
        }
        const ggml_type wtype = static_cast<ggml_type>(ggml_type_id);
        if (wtype == GGML_TYPE_F32) {
            return Tensor::Make2D(ptr, rows, cols);
        }
        const struct ggml_type_traits* traits = ggml_get_type_traits(wtype);
        if (!traits || !traits->to_float) {
            return Tensor::Make2D(ptr, rows, cols);
        }
        const size_t row_bytes = ggml_row_size(wtype, cols);
        scratch.Resize(this, static_cast<size_t>(rows * cols));
        const char* src = static_cast<const char*>(ptr);
        for (int64_t r = 0; r < rows; ++r) {
            traits->to_float(src + r * static_cast<ptrdiff_t>(row_bytes), scratch.ptr + r * cols, cols);
        }
        if (dequantized_bytes) {
            *dequantized_bytes += static_cast<size_t>(rows * cols * sizeof(float));
        }
        if (dequantized_any) {
            *dequantized_any = true;
        }
        return Tensor::Make2D(scratch.ptr, rows, cols);
    };

    auto dequantize_into_buffer = [](void* ptr, int ggml_type_id, int64_t rows, int64_t cols, std::vector<float>* dst) -> bool {
        if (!dst) return false;
        dst->clear();
        if (!ptr || rows <= 0 || cols <= 0) return false;
        const ggml_type wtype = static_cast<ggml_type>(ggml_type_id);
        if (wtype == GGML_TYPE_F32) return false;
        const struct ggml_type_traits* traits = ggml_get_type_traits(wtype);
        if (!traits || !traits->to_float) return false;
        const size_t total = static_cast<size_t>(rows * cols);
        dst->resize(total);
        const size_t row_bytes = ggml_row_size(wtype, cols);
        const char* src = static_cast<const char*>(ptr);
        float* out = dst->data();
        for (int64_t r = 0; r < rows; ++r) {
            traits->to_float(src + r * static_cast<ptrdiff_t>(row_bytes), out + r * cols, cols);
        }
        return true;
    };

    uint64_t cached_experts_this_step = 0;
    for (size_t idx = 0; idx < active_work.size(); ++idx) {
        const ActiveExpertWork& work = active_work[idx];
        const ExpertWeights& exp = experts[static_cast<size_t>(work.expert_id)];

        if (!small_decode_step && internal::IsMoENextExpertPrefetchEnabled() && idx + 1 < active_work.size()) {
            moe_stats_total_prefetch_candidates_.fetch_add(1, std::memory_order_relaxed);
            const ActiveExpertWork& next_work = active_work[idx + 1];
            const bool next_reused = previous_batch_set.find(next_work.expert_id) != previous_batch_set.end();
            const int reuse_signals =
                (next_work.local_hot ? 1 : 0) + (next_reused ? 1 : 0) + (next_work.count > 1 ? 1 : 0);
            const bool enough_compute_distance =
                work.count >= internal::GetMoEPrefetchMinCurrentExpertTokens();
            const bool under_pressure_budget =
                static_cast<int>(active_work.size()) <= internal::GetMoEPrefetchMaxActiveExperts() &&
                worker_threads <= internal::GetMoEPrefetchMaxThreadCount();

            if (!enough_compute_distance) {
                moe_stats_total_prefetch_skipped_distance_.fetch_add(1, std::memory_order_relaxed);
            } else if (!under_pressure_budget) {
                moe_stats_total_prefetch_skipped_pressure_.fetch_add(1, std::memory_order_relaxed);
            } else if (reuse_signals <= 0) {
                moe_stats_total_prefetch_skipped_signal_.fetch_add(1, std::memory_order_relaxed);
            } else {
                const ExpertWeights& next_exp = experts[static_cast<size_t>(next_work.expert_id)];
                size_t prefetch_budget = internal::GetMoEPrefetchBytes();
                if (reuse_signals == 1) {
                    prefetch_budget = std::max<size_t>(64, prefetch_budget / 2);
                }
                if (static_cast<int>(active_work.size()) >= std::max(2, internal::GetMoEPrefetchMaxActiveExperts() / 2)) {
                    prefetch_budget = std::max<size_t>(64, prefetch_budget / 2);
                }

                const int weight_count = next_exp.w3.ptr != nullptr ? 3 : 2;
                const size_t per_weight_budget = std::max<size_t>(64, prefetch_budget / static_cast<size_t>(weight_count));
                auto prefetch_weight = [per_weight_budget](const ExpertWeight& weight) {
                    if (!weight.ptr || weight.size == 0) return static_cast<size_t>(0);
                    const size_t bytes = std::min(per_weight_budget, weight.size);
                    densecore::simd::PrefetchRange(weight.ptr, bytes);
                    return bytes;
                };
                const size_t prefetched =
                    prefetch_weight(next_exp.w1) + prefetch_weight(next_exp.w2) + prefetch_weight(next_exp.w3);
                if (prefetched > 0) {
                    moe_stats_total_prefetch_calls_.fetch_add(1, std::memory_order_relaxed);
                    moe_stats_total_prefetch_bytes_.fetch_add(static_cast<uint64_t>(prefetched),
                                                              std::memory_order_relaxed);
                }
            }
        }

        std::shared_ptr<MoELayerRegistry::DequantizedExpertCacheEntry> cached_entry;
        const bool should_try_cache =
            dequant_cache_enabled &&
            (cache_all_active_experts || work.local_hot || previous_batch_set.find(work.expert_id) != previous_batch_set.end() ||
             work.count > 1);
        const size_t cacheable_bytes =
            (exp.w1_type == GGML_TYPE_F32 ? 0 : static_cast<size_t>(exp.intermediate_dim) * exp.hidden_dim * sizeof(float)) +
            (exp.w2_type == GGML_TYPE_F32 ? 0 : static_cast<size_t>(exp.hidden_dim) * exp.intermediate_dim * sizeof(float)) +
            ((exp.w3.ptr != nullptr && exp.w3_type != GGML_TYPE_F32)
                 ? static_cast<size_t>(exp.intermediate_dim) * exp.hidden_dim * sizeof(float)
                 : 0);
        if (should_try_cache && cacheable_bytes > 0 && cacheable_bytes <= dequant_cache_budget) {
            std::shared_ptr<MoELayerRegistry::DequantizedExpertCacheEntry> existing_entry;
            {
                std::lock_guard<std::mutex> lock(registry->mutex);
                auto it = registry->dequant_cache.find(work.expert_id);
                if (it != registry->dequant_cache.end()) {
                    existing_entry = it->second;
                    if (existing_entry) {
                        existing_entry->last_used = ++registry->dequant_cache_use_counter;
                    }
                }
            }

            if (!existing_entry) {
                auto candidate = std::make_shared<MoELayerRegistry::DequantizedExpertCacheEntry>();
                candidate->expert_id = work.expert_id;
                candidate->bytes = cacheable_bytes;
                dequantize_into_buffer(exp.w1.ptr, exp.w1_type, static_cast<int64_t>(exp.intermediate_dim),
                                       static_cast<int64_t>(exp.hidden_dim), &candidate->w1);
                dequantize_into_buffer(exp.w2.ptr, exp.w2_type, static_cast<int64_t>(exp.hidden_dim),
                                       static_cast<int64_t>(exp.intermediate_dim), &candidate->w2);
                if (exp.w3.ptr != nullptr) {
                    dequantize_into_buffer(exp.w3.ptr, exp.w3_type, static_cast<int64_t>(exp.intermediate_dim),
                                           static_cast<int64_t>(exp.hidden_dim), &candidate->w3);
                }

                std::lock_guard<std::mutex> lock(registry->mutex);
                auto it = registry->dequant_cache.find(work.expert_id);
                if (it != registry->dequant_cache.end()) {
                    cached_entry = it->second;
                    if (cached_entry) {
                        cached_entry->last_used = ++registry->dequant_cache_use_counter;
                    }
                } else if (candidate->bytes <= dequant_cache_budget) {
                    while (registry->dequant_cache_bytes + candidate->bytes > dequant_cache_budget &&
                           !registry->dequant_cache.empty()) {
                        auto evict_it = registry->dequant_cache.end();
                        uint64_t oldest_use = std::numeric_limits<uint64_t>::max();
                        for (auto it_cache = registry->dequant_cache.begin(); it_cache != registry->dequant_cache.end(); ++it_cache) {
                            if (!it_cache->second) {
                                evict_it = it_cache;
                                break;
                            }
                            if (it_cache->second->last_used < oldest_use) {
                                oldest_use = it_cache->second->last_used;
                                evict_it = it_cache;
                            }
                        }
                        if (evict_it == registry->dequant_cache.end()) {
                            break;
                        }
                        if (evict_it->second) {
                            registry->dequant_cache_bytes -=
                                std::min(registry->dequant_cache_bytes, evict_it->second->bytes);
                        }
                        registry->dequant_cache.erase(evict_it);
                    }
                    if (registry->dequant_cache_bytes + candidate->bytes <= dequant_cache_budget) {
                        candidate->last_used = ++registry->dequant_cache_use_counter;
                        registry->dequant_cache_bytes += candidate->bytes;
                        registry->dequant_cache.emplace(work.expert_id, candidate);
                        cached_entry = std::move(candidate);
                    }
                }
            } else {
                cached_entry = std::move(existing_entry);
            }
        }

        size_t dequantized_bytes = 0;
        bool dequantized_any = false;
        Tensor w1;
        Tensor w2;
        Tensor w3;
        if (cached_entry) {
            w1 = (exp.w1_type == GGML_TYPE_F32)
                     ? Tensor::Make2D(exp.w1.ptr, static_cast<int64_t>(exp.intermediate_dim), static_cast<int64_t>(exp.hidden_dim))
                     : Tensor::Make2D(cached_entry->w1.data(), static_cast<int64_t>(exp.intermediate_dim),
                                      static_cast<int64_t>(exp.hidden_dim));
            w2 = (exp.w2_type == GGML_TYPE_F32)
                     ? Tensor::Make2D(exp.w2.ptr, static_cast<int64_t>(exp.hidden_dim), static_cast<int64_t>(exp.intermediate_dim))
                     : Tensor::Make2D(cached_entry->w2.data(), static_cast<int64_t>(exp.hidden_dim),
                                      static_cast<int64_t>(exp.intermediate_dim));
            if (exp.w3.ptr != nullptr) {
                w3 = (exp.w3_type == GGML_TYPE_F32)
                         ? Tensor::Make2D(exp.w3.ptr, static_cast<int64_t>(exp.intermediate_dim), static_cast<int64_t>(exp.hidden_dim))
                         : Tensor::Make2D(cached_entry->w3.data(), static_cast<int64_t>(exp.intermediate_dim),
                                          static_cast<int64_t>(exp.hidden_dim));
            }
            ++cached_experts_this_step;
        } else {
            w1 = make_weight_f32(exp.w1.ptr, exp.w1_type, static_cast<int64_t>(exp.intermediate_dim),
                                 static_cast<int64_t>(exp.hidden_dim), w1_dequant, &dequantized_bytes, &dequantized_any);
            w2 = make_weight_f32(exp.w2.ptr, exp.w2_type, static_cast<int64_t>(exp.hidden_dim),
                                 static_cast<int64_t>(exp.intermediate_dim), w2_dequant, &dequantized_bytes,
                                 &dequantized_any);
            if (exp.w3.ptr != nullptr) {
                w3 = make_weight_f32(exp.w3.ptr, exp.w3_type, static_cast<int64_t>(exp.intermediate_dim),
                                     static_cast<int64_t>(exp.hidden_dim), w3_dequant, &dequantized_bytes,
                                     &dequantized_any);
            }
        }
        if (dequantized_any) {
            moe_stats_total_dequantized_experts_.fetch_add(1, std::memory_order_relaxed);
            moe_stats_total_dequantized_bytes_.fetch_add(static_cast<uint64_t>(dequantized_bytes),
                                                         std::memory_order_relaxed);
        }

        Tensor expert_input = Tensor::Make2D(packed_input + static_cast<size_t>(work.start) * hidden_dim,
                                             static_cast<int64_t>(work.count), static_cast<int64_t>(hidden_dim));
        Tensor expert_out = Tensor::Make2D(packed_output + static_cast<size_t>(work.start) * hidden_dim,
                                           static_cast<int64_t>(work.count), static_cast<int64_t>(hidden_dim));

        DispatchExpertFFN(layer_key, work.expert_id, expert_input, w1, w2, w3, &expert_out);
    }
    if (cached_experts_this_step > 0) {
        moe_stats_total_cached_experts_.fetch_add(cached_experts_this_step, std::memory_order_relaxed);
    }

    moe::ReorderOutputs(packed_output, hidden_dim_i, reorder_map, out_data, &reorder_pool);
}

}  // namespace densecore
