// Snapshot of MoE registry hot-expert state plus the per-call active-expert set,
// used to decide whether the tiny decode-specialized path can run. Extracted from
// CpuBackend::ForwardMoE so the gathering logic is a self-contained unit with
// explicit inputs; the caller unpacks these fields back into its locals.
struct MoESmallDecodeState {
    std::shared_ptr<moe::ExpertProfiler> profiler;
    // Hot / previous-batch expert ids captured from the registry. The fixed arrays
    // are the fast path; the *_overflow vectors hold the data when the registry set
    // exceeds the fixed-array capacity (snapshot_ok == false).
    std::array<int, kSmallDecodeMaxSnapshotExperts> local_hot_experts{};
    int local_hot_count = 0;
    std::array<int, kSmallDecodeMaxSnapshotExperts> previous_batch_experts{};
    int previous_batch_count = 0;
    bool snapshot_ok = true;
    std::vector<int> local_hot_experts_overflow;
    std::vector<int> previous_batch_experts_overflow;
    // Distinct experts touched by this call's routing, plus the largest per-expert
    // assignment count (>1 disqualifies the small-decode path).
    std::array<int, kSmallDecodeMaxAssignments> current_batch_experts{};
    int current_batch_expert_count = 0;
    int max_expert_batch = 0;
};

// Templated on the registry pointer type so this anonymous-namespace helper does
// not have to name CpuBackend::MoELayerRegistry (a private nested type). The only
// caller is CpuBackend::ForwardMoE, which has access at the point of instantiation.
template <typename RegistryPtr>
MoESmallDecodeState GatherMoESmallDecodeState(const RegistryPtr& registry, const moe::MoERouteResult& routing,
                                              int num_experts, int total_assignments, bool small_decode_candidate) {
    MoESmallDecodeState state;
    if (registry) {
        std::lock_guard<std::mutex> lock(registry->mutex);
        state.profiler = registry->profiler;
        if (small_decode_candidate) {
            state.snapshot_ok =
                CopyIntVectorToFixedArray(registry->local_expert_ids, &state.local_hot_experts,
                                          &state.local_hot_count) &&
                CopyIntVectorToFixedArray(registry->last_batch_experts, &state.previous_batch_experts,
                                          &state.previous_batch_count);
            if (!state.snapshot_ok) {
                state.local_hot_experts_overflow = registry->local_expert_ids;
                state.previous_batch_experts_overflow = registry->last_batch_experts;
            }
        } else {
            state.local_hot_experts_overflow = registry->local_expert_ids;
            state.previous_batch_experts_overflow = registry->last_batch_experts;
        }
    }

    if (small_decode_candidate && state.snapshot_ok) {
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
            state.max_expert_batch = std::max(state.max_expert_batch, expert_batch_count);
            for (int j = 0; j < state.current_batch_expert_count; ++j) {
                if (state.current_batch_experts[static_cast<size_t>(j)] == expert_id) {
                    seen = true;
                    break;
                }
            }
            if (!seen && state.current_batch_expert_count < kSmallDecodeMaxAssignments) {
                state.current_batch_experts[static_cast<size_t>(state.current_batch_expert_count++)] = expert_id;
            }
        }
    }
    return state;
}

// Handles the Gemma4 / Qwen3.6 "safe reference" fast paths that bypass the
// optimized MoE execution entirely. Returns true if the call was fully handled
// (the caller must then return); false to continue with the normal MoE path.
bool TryExecuteMoESafeReferenceFastPath(CpuBackend* backend, const TransformerModel* model, int layer_idx,
                                        const BatchSpec* batch, const float* input_data, int batch_size,
                                        int hidden_dim, const moe::MoERouteResult& routing,
                                        const CpuBackend::ExpertWeights* experts, int num_experts, float* out_data,
                                        bool qwen36_short_prefill_safe_reference, bool safe_reference_mode) {
    const bool gemma4_safe_reference_mode = model && model->arch_flags.is_gemma4 && safe_reference_mode;
    if (gemma4_safe_reference_mode) {
        if (!ExecuteMoEReferencePath(input_data, batch_size, hidden_dim, routing, experts, num_experts, out_data)) {
            std::fprintf(stderr, "[MoE_REF_EXEC] Gemma4 safe-reference execution failed; output left zeroed\n");
            return true;
        }
        RecordMoEReferencePathTrace(backend, layer_idx, batch, routing, num_experts);
        if (ShouldRunMoEReferenceCheck()) {
            RunMoEReferenceCheck(input_data, batch_size, hidden_dim, routing, experts, num_experts, out_data);
        }
        return true;
    }
    if (qwen36_short_prefill_safe_reference &&
        ExecuteMoEReferencePath(input_data, batch_size, hidden_dim, routing, experts, num_experts, out_data)) {
        if (ShouldRunMoEReferenceCheck()) {
            RunMoEReferenceCheck(input_data, batch_size, hidden_dim, routing, experts, num_experts, out_data);
        }
        return true;
    }
    return false;
}

struct MoESmallDecodeTileParallelRequest {
    CpuBackend* backend = nullptr;
    ThreadPool* pool = nullptr;
    const moe::MoERouteResult* routing = nullptr;
    const CpuBackend::ExpertWeights* experts = nullptr;
    moe::ExpertProfiler* profiler = nullptr;
    CpuBackend::MoEForwardProfile* profile = nullptr;
    AlignedScratch* hidden_scratch = nullptr;
    AlignedScratch* output_scratch = nullptr;
    const float* input_data = nullptr;
    float* out_data = nullptr;
    int batch_size = 0;
    int top_k = 0;
    int total_assignments = 0;
    int num_experts = 0;
    int worker_cap = 1;
    int64_t hidden_dim = 0;
    int64_t intermediate_dim = 0;
    bool has_token_indices = false;
};

bool MoESmallDecodeQuantizedTileAssignmentSupported(const CpuBackend::ExpertWeights& exp, int expert_id,
                                                    int64_t hidden_dim) {
    const bool down_scale_supported = exp.w2_scale_tensor == nullptr || IsScalarScaleSidecar(exp.w2_scale_tensor);
    const auto gate_type = static_cast<ggml_type>(exp.w1_type);
    const auto up_type = static_cast<ggml_type>(exp.w3_type);
    const auto* gate_traits_cpu = ggml_get_type_traits_cpu(gate_type);
    const bool gelu_gate_up_supported =
        exp.use_gelu_activation && gate_type == up_type && gate_traits_cpu && gate_traits_cpu->vec_dot &&
        IsKQuantRowPairGatedProjectionType(gate_type, gate_traits_cpu->vec_dot_type) &&
        (hidden_dim % ggml_blck_size(gate_type)) == 0;
    const bool supported = exp.w1.ptr && exp.w3.ptr && exp.w2.ptr && exp.w1_type != GGML_TYPE_F32 &&
                           exp.w2_type != GGML_TYPE_F32 && exp.w3_type != GGML_TYPE_F32 &&
                           ggml_is_quantized(static_cast<ggml_type>(exp.w1_type)) &&
                           ggml_is_quantized(static_cast<ggml_type>(exp.w2_type)) &&
                           (!exp.use_gelu_activation || gelu_gate_up_supported) &&
                           ggml_is_quantized(static_cast<ggml_type>(exp.w3_type)) && down_scale_supported;
    if (!supported && IsMoEMatmulPathDebugEnabled()) {
        std::fprintf(stderr,
                     "[MOE_TILE_DIAG] quantized_gated_ffn FAIL expert=%d w1=%d w3=%d w2=%d "
                     "gelu=%d down_scale=%d w1_ptr=%d w3_ptr=%d w2_ptr=%d\n",
                     expert_id, exp.w1_type, exp.w3_type, exp.w2_type, exp.use_gelu_activation ? 1 : 0,
                     down_scale_supported ? 1 : 0, exp.w1.ptr ? 1 : 0, exp.w3.ptr ? 1 : 0,
                     exp.w2.ptr ? 1 : 0);
    }
    return supported;
}

int MoESmallDecodeTokenIndex(const MoESmallDecodeTileParallelRequest& req, int assignment) {
    return req.has_token_indices ? req.routing->token_indices[static_cast<size_t>(assignment)]
                                 : (assignment / req.top_k);
}

bool PrepareMoESmallDecodeSharedInputCache(const MoESmallDecodeTileParallelRequest& req,
                                           QuantizedProjectionInputCache* cache,
                                           QuantizedProjectionInputCache** cache_ptr) {
    if (!cache || !cache_ptr || req.batch_size != 1) {
        return true;
    }
    const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const size_t q8_row_bytes = ggml_row_size(GGML_TYPE_Q8_K, req.hidden_dim);
    if (!q8_traits || !q8_traits->from_float || q8_row_bytes == 0) {
        return true;
    }
    cache->source = req.input_data;
    cache->rows = 1;
    cache->cols = req.hidden_dim;
    cache->type = GGML_TYPE_Q8_K;
    cache->row_bytes = q8_row_bytes;
    cache->bytes.resize(q8_row_bytes);
    q8_traits->from_float(req.input_data, cache->bytes.data(), req.hidden_dim);
    *cache_ptr = cache;
    return true;
}

// ============================================================================
// NUMA sticky decode dispatch
// ============================================================================
// On multi-node hosts the assignments are partitioned by the NUMA node that
// holds each expert's weights (as recorded by RegisterMoEExperts placement
// detection / load-time expert partitioning), and each group runs on that
// node's thread pool — concurrently across nodes via RunConcurrentNodeTasks.
// Single-node hosts (or DENSECORE_MOE_STICKY_DECODE=0) keep the legacy
// single-pool path unchanged.

bool IsMoEStickyDecodeDispatchEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_MOE_STICKY_DECODE");
        return !(env && env[0] != '\0' && std::strcmp(env, "0") == 0);  // default ON
    }();
    return enabled;
}

struct MoESmallDecodeNodePartition {
    bool active = false;
    int num_nodes = 0;
    std::vector<std::vector<int>> node_assignments;
};

void BuildMoESmallDecodeNodePartition(const MoESmallDecodeTileParallelRequest& req,
                                      MoESmallDecodeNodePartition* partition) {
    partition->active = false;
    if (!req.backend || !req.profiler || !IsMoEStickyDecodeDispatchEnabled()) {
        return;
    }
    const int num_nodes = req.backend->GetNumaNodeCount();
    if (num_nodes <= 1) {
        return;
    }
    partition->num_nodes = num_nodes;
    partition->node_assignments.assign(static_cast<size_t>(num_nodes), {});
    for (int assignment = 0; assignment < req.total_assignments; ++assignment) {
        const int expert_id = req.routing->expert_ids[static_cast<size_t>(assignment)];
        int node = (expert_id >= 0 && expert_id < req.num_experts) ? req.profiler->GetExpertNumaNode(expert_id) : -1;
        if (node < 0 || node >= num_nodes) {
            node = assignment % num_nodes;  // unknown placement: spread for aggregate bandwidth
        }
        partition->node_assignments[static_cast<size_t>(node)].push_back(assignment);
    }
    partition->active = true;
    static std::atomic<bool> logged{false};
    bool expected = false;
    if (logged.compare_exchange_strong(expected, true)) {
        std::fprintf(stderr, "[NUMA] MoE sticky decode dispatch active (nodes=%d)\n", num_nodes);
    }
}

inline void MoESmallDecodeGateUpUnit(const MoESmallDecodeTileParallelRequest& req, int assignment, int split,
                                     int gate_splits, float* assignment_hiddens,
                                     QuantizedProjectionInputCache* shared_input_cache,
                                     std::atomic<bool>* tile_ok) {
    const int expert_id = req.routing->expert_ids[static_cast<size_t>(assignment)];
    if (expert_id < 0 || expert_id >= req.num_experts) {
        return;
    }
    const int token_idx = MoESmallDecodeTokenIndex(req, assignment);
    if (token_idx < 0 || token_idx >= req.batch_size) {
        return;
    }

    const CpuBackend::ExpertWeights& exp = req.experts[static_cast<size_t>(expert_id)];
    const int64_t tile_start = (req.intermediate_dim * split) / gate_splits;
    const int64_t tile_end = (req.intermediate_dim * (split + 1)) / gate_splits;
    const int64_t tile_rows = tile_end - tile_start;
    if (tile_rows <= 0) {
        return;
    }
    const size_t gate_row_bytes = ggml_row_size(static_cast<ggml_type>(exp.w1_type), req.hidden_dim);
    const size_t up_row_bytes = ggml_row_size(static_cast<ggml_type>(exp.w3_type), req.hidden_dim);
    const char* gate_ptr = static_cast<const char*>(exp.w1.ptr) + static_cast<ptrdiff_t>(tile_start * gate_row_bytes);
    const char* up_ptr = static_cast<const char*>(exp.w3.ptr) + static_cast<ptrdiff_t>(tile_start * up_row_bytes);
    Tensor expert_input =
        Tensor::Make2D(const_cast<float*>(req.input_data + static_cast<size_t>(token_idx) * req.hidden_dim),
                       1, req.hidden_dim);
    float* hidden_tile = assignment_hiddens +
                         static_cast<size_t>(assignment) * static_cast<size_t>(req.intermediate_dim) +
                         static_cast<size_t>(tile_start);
    Tensor hidden_tensor = Tensor::Make2D(hidden_tile, 1, tile_rows);
    QuantizedProjectionInputCache local_input_projection_cache;
    QuantizedProjectionInputCache* unit_input_projection_cache =
        (shared_input_cache && CanUseSharedDecodeInputCacheForExpert(exp, shared_input_cache->type))
            ? shared_input_cache
            : &local_input_projection_cache;
    const int expert_numa_node = req.profiler ? req.profiler->GetExpertNumaNode(expert_id) : -1;
    const bool gate_up_ok =
        exp.use_gelu_activation
            ? TryRunGgmlQuantizedFusedGEGLUProjection(req.backend, gate_ptr, exp.w1_type, up_ptr, exp.w3_type,
                                                      expert_input, &hidden_tensor, tile_rows, req.hidden_dim,
                                                      expert_numa_node, /*allow_parallel=*/false,
                                                      unit_input_projection_cache)
            : TryRunGgmlQuantizedFusedSwiGLUProjection(req.backend, gate_ptr, exp.w1_type, up_ptr, exp.w3_type,
                                                       expert_input, &hidden_tensor, tile_rows, req.hidden_dim,
                                                       expert_numa_node, /*allow_parallel=*/false,
                                                       unit_input_projection_cache);
    if (!gate_up_ok) {
        tile_ok->store(false, std::memory_order_relaxed);
    }
}

void RunMoESmallDecodeTileGateUp(const MoESmallDecodeTileParallelRequest& req, int gate_splits,
                                 float* assignment_hiddens,
                                 QuantizedProjectionInputCache* shared_input_cache,
                                 std::atomic<bool>* tile_ok,
                                 const MoESmallDecodeNodePartition* partition = nullptr) {
    if (partition && partition->active) {
        req.backend->RunConcurrentNodeTasks(partition->num_nodes, [&](int node) {
            const std::vector<int>& group = partition->node_assignments[static_cast<size_t>(node)];
            if (group.empty()) {
                return;
            }
            const int group_units = static_cast<int>(group.size()) * gate_splits;
            req.backend->GetThreadPool(node).ParallelFor(group_units, [&](int unit_start, int unit_end,
                                                                          int /*thread_id*/) {
                for (int unit = unit_start; unit < unit_end && tile_ok->load(std::memory_order_relaxed); ++unit) {
                    MoESmallDecodeGateUpUnit(req, group[static_cast<size_t>(unit / gate_splits)], unit % gate_splits,
                                             gate_splits, assignment_hiddens, shared_input_cache, tile_ok);
                }
            });
        });
        return;
    }
    req.pool->ParallelFor(req.total_assignments * gate_splits, [&](int unit_start, int unit_end, int /*thread_id*/) {
        for (int unit = unit_start; unit < unit_end && tile_ok->load(std::memory_order_relaxed); ++unit) {
            const int assignment = unit / gate_splits;
            const int split = unit - assignment * gate_splits;
            MoESmallDecodeGateUpUnit(req, assignment, split, gate_splits, assignment_hiddens, shared_input_cache,
                                     tile_ok);
        }
    });
}

bool QuantizeMoESmallDecodeHiddenRows(const MoESmallDecodeTileParallelRequest& req, const float* assignment_hiddens,
                                      std::vector<QuantizedProjectionInputCache>* down_input_caches) {
    const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const size_t q8_row_bytes = ggml_row_size(GGML_TYPE_Q8_K, req.intermediate_dim);
    if (!q8_traits || !q8_traits->from_float || q8_row_bytes == 0 || !down_input_caches) {
        return false;
    }
    req.pool->ParallelFor(req.total_assignments, [&](int start, int end, int /*thread_id*/) {
        for (int assignment = start; assignment < end; ++assignment) {
            const int expert_id = req.routing->expert_ids[static_cast<size_t>(assignment)];
            if (expert_id < 0 || expert_id >= req.num_experts) {
                continue;
            }
            auto& cache = (*down_input_caches)[static_cast<size_t>(assignment)];
            const float* hidden_row =
                assignment_hiddens + static_cast<size_t>(assignment) * static_cast<size_t>(req.intermediate_dim);
            cache.source = hidden_row;
            cache.rows = 1;
            cache.cols = req.intermediate_dim;
            cache.type = GGML_TYPE_Q8_K;
            cache.row_bytes = q8_row_bytes;
            cache.bytes.resize(q8_row_bytes);
            q8_traits->from_float(hidden_row, cache.bytes.data(), req.intermediate_dim);
        }
    });
    return true;
}

inline void MoESmallDecodeDownUnit(const MoESmallDecodeTileParallelRequest& req, int assignment, int split,
                                   int down_splits, const float* assignment_hiddens, float* assignment_outputs,
                                   std::vector<QuantizedProjectionInputCache>* down_input_caches,
                                   std::atomic<bool>* tile_ok) {
    const int expert_id = req.routing->expert_ids[static_cast<size_t>(assignment)];
    if (expert_id < 0 || expert_id >= req.num_experts) {
        return;
    }
    const int token_idx = MoESmallDecodeTokenIndex(req, assignment);
    if (token_idx < 0 || token_idx >= req.batch_size) {
        return;
    }

    const CpuBackend::ExpertWeights& exp = req.experts[static_cast<size_t>(expert_id)];
    const int64_t tile_start = (req.hidden_dim * split) / down_splits;
    const int64_t tile_end = (req.hidden_dim * (split + 1)) / down_splits;
    const int64_t tile_rows = tile_end - tile_start;
    if (tile_rows <= 0) {
        return;
    }
    const size_t down_row_bytes = ggml_row_size(static_cast<ggml_type>(exp.w2_type), req.intermediate_dim);
    const char* down_ptr =
        static_cast<const char*>(exp.w2.ptr) + static_cast<ptrdiff_t>(tile_start * down_row_bytes);
    Tensor hidden_tensor = Tensor::Make2D(
        const_cast<float*>(assignment_hiddens +
                           static_cast<size_t>(assignment) * static_cast<size_t>(req.intermediate_dim)),
        1, req.intermediate_dim);
    float* output_tile = assignment_outputs + static_cast<size_t>(assignment) * static_cast<size_t>(req.hidden_dim) +
                         static_cast<size_t>(tile_start);
    Tensor output_tensor = Tensor::Make2D(output_tile, 1, tile_rows);
    QuantizedProjectionInputCache local_down_input_projection_cache;
    QuantizedProjectionInputCache* down_input_projection_cache =
        QuantizedProjectionInputTypeMatches(exp.w2_type, (*down_input_caches)[static_cast<size_t>(assignment)].type)
            ? &(*down_input_caches)[static_cast<size_t>(assignment)]
            : &local_down_input_projection_cache;
    const int expert_numa_node = req.profiler ? req.profiler->GetExpertNumaNode(expert_id) : -1;
    if (!TryRunGgmlQuantizedProjection(req.backend, down_ptr, exp.w2_type, hidden_tensor, &output_tensor,
                                       tile_rows, req.intermediate_dim, expert_numa_node,
                                       /*allow_parallel=*/false, down_input_projection_cache)) {
        tile_ok->store(false, std::memory_order_relaxed);
    } else if (exp.w2_scale_tensor != nullptr) {
        ApplyScalarScaleToTensor(&output_tensor, ReadScalarScaleSidecar(exp.w2_scale_tensor));
    }
}

void RunMoESmallDecodeTileDown(const MoESmallDecodeTileParallelRequest& req, int down_splits,
                               const float* assignment_hiddens, float* assignment_outputs,
                               std::vector<QuantizedProjectionInputCache>* down_input_caches,
                               std::atomic<bool>* tile_ok,
                               const MoESmallDecodeNodePartition* partition = nullptr) {
    if (partition && partition->active) {
        req.backend->RunConcurrentNodeTasks(partition->num_nodes, [&](int node) {
            const std::vector<int>& group = partition->node_assignments[static_cast<size_t>(node)];
            if (group.empty()) {
                return;
            }
            const int group_units = static_cast<int>(group.size()) * down_splits;
            req.backend->GetThreadPool(node).ParallelFor(group_units, [&](int unit_start, int unit_end,
                                                                          int /*thread_id*/) {
                for (int unit = unit_start; unit < unit_end && tile_ok->load(std::memory_order_relaxed); ++unit) {
                    MoESmallDecodeDownUnit(req, group[static_cast<size_t>(unit / down_splits)], unit % down_splits,
                                           down_splits, assignment_hiddens, assignment_outputs, down_input_caches,
                                           tile_ok);
                }
            });
        });
        return;
    }
    req.pool->ParallelFor(req.total_assignments * down_splits, [&](int unit_start, int unit_end, int /*thread_id*/) {
        for (int unit = unit_start; unit < unit_end && tile_ok->load(std::memory_order_relaxed); ++unit) {
            const int assignment = unit / down_splits;
            const int split = unit - assignment * down_splits;
            MoESmallDecodeDownUnit(req, assignment, split, down_splits, assignment_hiddens, assignment_outputs,
                                   down_input_caches, tile_ok);
        }
    });
}

void ScatterMoESmallDecodeTileOutputs(const MoESmallDecodeTileParallelRequest& req, const float* assignment_outputs) {
    for (int assignment = 0; assignment < req.total_assignments; ++assignment) {
        const int expert_id = req.routing->expert_ids[static_cast<size_t>(assignment)];
        if (expert_id < 0 || expert_id >= req.num_experts) {
            continue;
        }
        const int token_idx = MoESmallDecodeTokenIndex(req, assignment);
        if (token_idx < 0 || token_idx >= req.batch_size) {
            continue;
        }
        const float weight = req.routing->weights[static_cast<size_t>(assignment)];
        if (weight == 0.0f) {
            continue;
        }
        float* dst = req.out_data + static_cast<size_t>(token_idx) * static_cast<size_t>(req.hidden_dim);
        const float* src = assignment_outputs + static_cast<size_t>(assignment) * static_cast<size_t>(req.hidden_dim);
        for (int64_t d = 0; d < req.hidden_dim; ++d) {
            dst[d] += weight * src[static_cast<size_t>(d)];
        }
    }
}

bool TryExecuteMoESmallDecodeQuantizedTileParallel(const MoESmallDecodeTileParallelRequest& req) {
    if (!req.backend || !req.pool || !req.routing || !req.experts || !req.hidden_scratch || !req.output_scratch ||
        !req.input_data || !req.out_data || req.batch_size != 1 || req.total_assignments < 2 || req.num_experts <= 0 ||
        req.worker_cap < 16 || req.hidden_dim < 1024 || req.intermediate_dim < 256) {
        return false;
    }
    for (int assignment = 0; assignment < req.total_assignments; ++assignment) {
        const int expert_id = req.routing->expert_ids[static_cast<size_t>(assignment)];
        if (expert_id < 0 || expert_id >= req.num_experts) {
            continue;
        }
        if (!MoESmallDecodeQuantizedTileAssignmentSupported(req.experts[static_cast<size_t>(expert_id)], expert_id,
                                                           req.hidden_dim)) {
            return false;
        }
    }

    const int target_parallelism = std::max(1, req.worker_cap / std::max(1, req.total_assignments));
    const int max_gate_splits =
        std::min<int>(target_parallelism, std::max<int>(1, static_cast<int>(req.intermediate_dim / 64)));
    const int gate_splits = std::max<int>(1, std::min<int>(max_gate_splits, 8));
    const int down_splits =
        std::min<int>(std::max<int>(1, req.worker_cap / std::max(1, req.total_assignments)),
                      std::max<int>(1, static_cast<int>(req.hidden_dim / 256)));
    if (IsMoEMatmulPathDebugEnabled()) {
        std::fprintf(stderr,
                     "[MOE_TILE_DIAG] ENTERING tile_parallel gate_splits=%d down_splits=%d "
                     "target_par=%d workers=%d assignments=%d\n",
                     gate_splits, down_splits, target_parallelism, req.worker_cap, req.total_assignments);
    }

    const bool lfm2_decode_tile_fast_path =
        GetCurrentInferenceWorkContextModelVariant() == ModelVariant::LFM2MOE;
    const size_t assignment_hidden_elems =
        static_cast<size_t>(req.total_assignments) * static_cast<size_t>(req.intermediate_dim);
    const size_t assignment_output_elems =
        static_cast<size_t>(req.total_assignments) * static_cast<size_t>(req.hidden_dim);
    const bool reused_assignment_scratch =
        req.hidden_scratch->HasCapacity(assignment_hidden_elems) && req.output_scratch->HasCapacity(assignment_output_elems);
    req.hidden_scratch->Resize(req.backend, assignment_hidden_elems);
    req.output_scratch->Resize(req.backend, assignment_output_elems);
    float* assignment_hiddens = req.hidden_scratch->ptr;
    float* assignment_outputs = req.output_scratch->ptr;
    if (!lfm2_decode_tile_fast_path) {
        std::fill_n(assignment_hiddens, assignment_hidden_elems, 0.0f);
        std::fill_n(assignment_outputs, assignment_output_elems, 0.0f);
    }
    if (req.profile && reused_assignment_scratch) {
        req.profile->decode_scratch_reused += 2;
        req.profile->decode_allocations_avoided += 2;
    }

    static thread_local std::vector<QuantizedProjectionInputCache> lfm2_down_input_projection_caches;
    std::vector<QuantizedProjectionInputCache> down_input_projection_caches;
    std::vector<QuantizedProjectionInputCache>* down_input_projection_caches_ptr = nullptr;
    if (lfm2_decode_tile_fast_path) {
        lfm2_down_input_projection_caches.resize(static_cast<size_t>(req.total_assignments));
        down_input_projection_caches_ptr = &lfm2_down_input_projection_caches;
    } else {
        down_input_projection_caches.resize(static_cast<size_t>(req.total_assignments));
        down_input_projection_caches_ptr = &down_input_projection_caches;
    }
    QuantizedProjectionInputCache shared_decode_input_projection_cache;
    QuantizedProjectionInputCache* shared_decode_input_projection_cache_ptr = nullptr;
    PrepareMoESmallDecodeSharedInputCache(req, &shared_decode_input_projection_cache,
                                          &shared_decode_input_projection_cache_ptr);

    MoESmallDecodeNodePartition node_partition;
    BuildMoESmallDecodeNodePartition(req, &node_partition);

    std::atomic<bool> tile_ok{true};
    LogSmallDecodeExecutionPath("quantized_tile_parallel", req.total_assignments, req.worker_cap, req.batch_size);
    RunMoESmallDecodeTileGateUp(req, gate_splits, assignment_hiddens, shared_decode_input_projection_cache_ptr,
                                &tile_ok, &node_partition);
    if (tile_ok.load(std::memory_order_relaxed) &&
        !QuantizeMoESmallDecodeHiddenRows(req, assignment_hiddens, down_input_projection_caches_ptr)) {
        tile_ok.store(false, std::memory_order_relaxed);
    }
    if (tile_ok.load(std::memory_order_relaxed)) {
        RunMoESmallDecodeTileDown(req, down_splits, assignment_hiddens, assignment_outputs, down_input_projection_caches_ptr,
                                  &tile_ok, &node_partition);
    }
    if (!tile_ok.load(std::memory_order_relaxed)) {
        return false;
    }
    ScatterMoESmallDecodeTileOutputs(req, assignment_outputs);
    if (ShouldRunMoEReferenceCheck()) {
        RunMoEReferenceCheck(req.input_data, req.batch_size, static_cast<int>(req.hidden_dim), *req.routing, req.experts,
                             req.num_experts, req.out_data);
    }
    return true;
}
