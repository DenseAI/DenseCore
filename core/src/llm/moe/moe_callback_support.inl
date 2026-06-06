static std::atomic<uint64_t> g_moe_callback_entry_count{0};
static std::atomic<uint64_t> g_moe_callback_missing_userdata_count{0};
static std::atomic<uint64_t> g_moe_callback_missing_backend_count{0};
static std::atomic<uint64_t> g_moe_callback_missing_experts_count{0};
static std::atomic<uint64_t> g_moe_callback_routing_failure_count{0};
static std::atomic<uint64_t> g_moe_callback_empty_routing_count{0};
static std::atomic<uint64_t> g_moe_callback_fail_closed_count{0};
static std::atomic<bool> g_moe_strict_failure_pending{false};
static std::mutex g_moe_strict_failure_mu;
static std::string g_moe_strict_failure_message;

static void ResetMoECallbackEntryCount() {
    g_moe_callback_entry_count.store(0, std::memory_order_relaxed);
    g_moe_callback_missing_userdata_count.store(0, std::memory_order_relaxed);
    g_moe_callback_missing_backend_count.store(0, std::memory_order_relaxed);
    g_moe_callback_missing_experts_count.store(0, std::memory_order_relaxed);
    g_moe_callback_routing_failure_count.store(0, std::memory_order_relaxed);
    g_moe_callback_empty_routing_count.store(0, std::memory_order_relaxed);
    g_moe_callback_fail_closed_count.store(0, std::memory_order_relaxed);
    g_moe_strict_failure_pending.store(false, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(g_moe_strict_failure_mu);
    g_moe_strict_failure_message.clear();
}

static uint64_t GetMoECallbackEntryCount() {
    return g_moe_callback_entry_count.load(std::memory_order_relaxed);
}

static uint64_t GetMoECallbackMissingUserdataCount() {
    return g_moe_callback_missing_userdata_count.load(std::memory_order_relaxed);
}

static uint64_t GetMoECallbackMissingBackendCount() {
    return g_moe_callback_missing_backend_count.load(std::memory_order_relaxed);
}

static uint64_t GetMoECallbackMissingExpertsCount() {
    return g_moe_callback_missing_experts_count.load(std::memory_order_relaxed);
}

static uint64_t GetMoECallbackRoutingFailureCount() {
    return g_moe_callback_routing_failure_count.load(std::memory_order_relaxed);
}

static uint64_t GetMoECallbackEmptyRoutingCount() {
    return g_moe_callback_empty_routing_count.load(std::memory_order_relaxed);
}

static uint64_t GetMoECallbackFailClosedCount() {
    return g_moe_callback_fail_closed_count.load(std::memory_order_relaxed);
}

static bool IsMoEDebugLoggingEnabled() {
    return ParseTruthyEnv("DENSECORE_DEBUG_MOE", ParseTruthyEnv("DENSECORE_DEBUG_MOE_CALLBACKS", false));
}

static bool IsMoEStrictModeEnabled() {
    return ParseTruthyEnv("DENSECORE_MOE_STRICT", false);
}

static bool IsMoEStageTimingEnabled() {
    return ParseTruthyEnv("DENSECORE_MOE_STAGE_TIMING", ParseTruthyEnv("DENSECORE_DEBUG_MOE_STAGE_TIMING", false));
}

static bool IsMoEDetailedTraceEnabled() {
    return ParseTruthyEnv("DENSECORE_DEBUG_MOE_TRACE", false);
}

static void ResetMoEStrictFailureState() {
    g_moe_strict_failure_pending.store(false, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(g_moe_strict_failure_mu);
    g_moe_strict_failure_message.clear();
}

static bool ConsumeMoEStrictFailureState(std::string* message) {
    if (!g_moe_strict_failure_pending.exchange(false, std::memory_order_acq_rel)) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_moe_strict_failure_mu);
    if (message) {
        *message = g_moe_strict_failure_message;
    }
    g_moe_strict_failure_message.clear();
    return true;
}

static void RecordMoEStrictFailure(const std::string& message) {
    {
        std::lock_guard<std::mutex> lock(g_moe_strict_failure_mu);
        g_moe_strict_failure_message = message;
    }
    g_moe_strict_failure_pending.store(true, std::memory_order_release);
}

static void ZeroFillTensor(struct ggml_tensor* tensor) {
    if (!tensor || !tensor->data) {
        return;
    }
    std::memset(tensor->data, 0, ggml_nbytes(tensor));
}

static void FailClosedMoECallback(struct ggml_tensor* dst, std::atomic<uint64_t>* counter, const std::string& message) {
    if (counter) {
        counter->fetch_add(1, std::memory_order_relaxed);
    }
    g_moe_callback_fail_closed_count.fetch_add(1, std::memory_order_relaxed);
    ZeroFillTensor(dst);
    std::fprintf(stderr, "[DenseCore][MoE] fail-closed: %s\n", message.c_str());
    if (IsMoEStrictModeEnabled()) {
        RecordMoEStrictFailure(message);
    }
}

static int GetMoEDebugSelectedLayer() {
    static const int layer = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_MOE_LAYER");
        if (!env || env[0] == '\0') return -1;
        char* end = nullptr;
        const long parsed = std::strtol(env, &end, 10);
        return (end == env) ? -1 : static_cast<int>(parsed);
    }();
    return layer;
}

static int GetMoEDebugSelectedToken() {
    static const int token = std::max(0, ParsePositiveEnvInt("DENSECORE_DEBUG_MOE_TOKEN", 0));
    return token;
}

static bool ShouldTraceMoELayer(const MoEUserData* ud) {
    if (!IsMoEDetailedTraceEnabled() || !ud) {
        return false;
    }
    const int selected_layer = GetMoEDebugSelectedLayer();
    return selected_layer < 0 || ud->layer_idx == selected_layer;
}

static void DumpMoERouteTrace(const struct ggml_tensor* gate_logits, const MoEUserData* ud,
                              const densecore::moe::MoERouteResult& routing) {
    if (!ShouldTraceMoELayer(ud) || !gate_logits || !gate_logits->data || routing.batch_size <= 0) {
        return;
    }

    const int token_idx = std::min(GetMoEDebugSelectedToken(), routing.batch_size - 1);
    const int n_experts = static_cast<int>(gate_logits->ne[0]);
    const float* logits = reinterpret_cast<const float*>(gate_logits->data);
    const float* row = logits + static_cast<size_t>(token_idx) * n_experts;

    std::fprintf(stderr,
                 "[MOE_TRACE_ROUTE] layer=%d token=%d batch=%d experts=%d top_k=%d shared=%d scale=%g norm_topk=%d\n",
                 ud->layer_idx, token_idx, routing.batch_size, n_experts, routing.top_k,
                 ud->model ? ud->model->moe_n_shared_experts : 0,
                 ud->model ? static_cast<double>(ud->model->moe_routed_scaling_factor) : 0.0,
                 (ud->model && ud->model->moe_norm_topk_prob) ? 1 : 0);
    std::fprintf(stderr, "[MOE_TRACE_ROUTE] logits=");
    for (int e = 0; e < n_experts; ++e) {
        if (e != 0) std::fputc(',', stderr);
        std::fprintf(stderr, "%g", static_cast<double>(row[e]));
    }
    std::fputc('\n', stderr);
    std::fprintf(stderr, "[MOE_TRACE_ROUTE] selected=");
    for (int k = 0; k < routing.top_k; ++k) {
        const size_t idx = static_cast<size_t>(token_idx * routing.top_k + k);
        std::fprintf(stderr, "%s(%d,%g)", k == 0 ? "" : ",", routing.expert_ids[idx],
                     static_cast<double>(routing.weights[idx]));
    }
    std::fputc('\n', stderr);
    std::fprintf(stderr, "[MOE_TRACE_ROUTE] token_indices=");
    for (int k = 0; k < routing.top_k; ++k) {
        const size_t idx = static_cast<size_t>(token_idx * routing.top_k + k);
        std::fprintf(stderr, "%s%d", k == 0 ? "" : ",", routing.token_indices[idx]);
    }
    std::fputc('\n', stderr);
}

static void DumpMoEOutputTrace(const struct ggml_tensor* dst, const MoEUserData* ud) {
    if (!ShouldTraceMoELayer(ud) || !dst || !dst->data) {
        return;
    }
    const int64_t elems = ggml_nelements(dst);
    const float* values = reinterpret_cast<const float*>(dst->data);
    double sum = 0.0;
    double abs_sum = 0.0;
    double sq_sum = 0.0;
    for (int64_t i = 0; i < elems; ++i) {
        const double v = static_cast<double>(values[i]);
        sum += v;
        abs_sum += std::fabs(v);
        sq_sum += v * v;
    }
    std::fprintf(stderr, "[MOE_TRACE_OUT] layer=%d elems=%lld sum=%g abs_sum=%g sq_sum=%g\n", ud->layer_idx,
                 static_cast<long long>(elems), sum, abs_sum, sq_sum);
}

static void EnsureMoERebalanceThread(densecore::CpuBackend* backend) {
    if (IsBenchmarkMode() || !backend || backend->IsRebalanceThreadRunning()) {
        return;
    }
    backend->StartRebalanceThread(GetMoERebalanceIntervalMs(), GetMoERebalanceTopK(), IsMoEPageMigrationEnabled());
}

static void UpdateSchedulerExperts(const MoEUserData* ud, const densecore::moe::MoERouteResult& routing) {
    if (!ud) {
        return;
    }
    const BatchSpec* batch = GetCurrentBatch();
    if (!batch || !batch->scheduler || batch->seq_id.empty() || batch->scheduler_seq_ids.empty()) {
        return;
    }

    const int num_seqs = static_cast<int>(batch->scheduler_seq_ids.size());
    thread_local std::vector<std::vector<int>> per_seq_experts;
    per_seq_experts.resize(static_cast<size_t>(num_seqs));
    for (auto& experts : per_seq_experts) {
        experts.clear();
    }

    const bool has_token_indices = !routing.token_indices.empty();
    const int total_assignments = static_cast<int>(routing.expert_ids.size());
    const bool decode_single_token_per_seq = batch->seq_id.size() == batch->scheduler_seq_ids.size() &&
                                             static_cast<int>(batch->seq_id.size()) == routing.batch_size;

    if (decode_single_token_per_seq) {
        std::vector<int> experts;
        experts.reserve(static_cast<size_t>(std::max(1, routing.top_k)));

        const int token_count = static_cast<int>(batch->seq_id.size());
        for (int token_idx = 0; token_idx < token_count; ++token_idx) {
            const int seq_idx = batch->seq_id[static_cast<size_t>(token_idx)];
            if (seq_idx < 0 || seq_idx >= num_seqs) {
                continue;
            }

            experts.clear();
            for (int i = 0; i < total_assignments; ++i) {
                const int routed_token_idx = has_token_indices ? routing.token_indices[static_cast<size_t>(i)]
                                                               : (i / routing.top_k);
                if (routed_token_idx != token_idx) {
                    continue;
                }
                const int expert_id = routing.expert_ids[static_cast<size_t>(i)];
                if (expert_id < 0) {
                    continue;
                }
                if (std::find(experts.begin(), experts.end(), expert_id) == experts.end()) {
                    experts.push_back(expert_id);
                }
            }

            if (experts.empty()) {
                continue;
            }
            const int sched_seq_id = batch->scheduler_seq_ids[static_cast<size_t>(seq_idx)];
            if (sched_seq_id < 0) {
                continue;
            }
            batch->scheduler->SetPredictedExperts(sched_seq_id, experts);
        }
        return;
    }

    for (int i = 0; i < total_assignments; ++i) {
        const int token_idx = has_token_indices ? routing.token_indices[i] : (i / routing.top_k);
        if (token_idx < 0 || token_idx >= static_cast<int>(batch->seq_id.size())) {
            continue;
        }
        const int seq_idx = batch->seq_id[token_idx];
        if (seq_idx < 0 || seq_idx >= num_seqs) {
            continue;
        }
        const int expert_id = routing.expert_ids[i];
        if (expert_id < 0) {
            continue;
        }
        per_seq_experts[static_cast<size_t>(seq_idx)].push_back(expert_id);
    }

    for (int seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
        auto& experts = per_seq_experts[static_cast<size_t>(seq_idx)];
        if (experts.empty()) {
            continue;
        }
        const int sched_seq_id = batch->scheduler_seq_ids[static_cast<size_t>(seq_idx)];
        if (sched_seq_id < 0) {
            continue;
        }

        if (!decode_single_token_per_seq && experts.size() > 1) {
            std::sort(experts.begin(), experts.end());
            experts.erase(std::unique(experts.begin(), experts.end()), experts.end());
        }
        batch->scheduler->SetPredictedExperts(sched_seq_id, experts);
    }
}

static bool EnsureMoERouteStorage(densecore::moe::MoERouteResult* routing, int batch_size, int top_k) {
    if (!routing || batch_size <= 0 || top_k <= 0) {
        return false;
    }
    const size_t total = static_cast<size_t>(batch_size * top_k);
    routing->batch_size = batch_size;
    routing->top_k = top_k;
    routing->expert_ids.resize(total);
    routing->weights.resize(total);
    routing->token_indices.resize(total);
    std::fill(routing->expert_ids.begin(), routing->expert_ids.end(), -1);
    std::fill(routing->weights.begin(), routing->weights.end(), 0.0f);
    std::fill(routing->token_indices.begin(), routing->token_indices.end(), 0);
    return true;
}

static bool ForceMoESoftmaxRouting() {
    const char* env = std::getenv("DENSECORE_FORCE_MOE_SOFTMAX_ROUTING");
    return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

static bool ForceMoEGroupedSigmoidRouting() {
    const char* env = std::getenv("DENSECORE_FORCE_MOE_GROUPED_SIGMOID_ROUTING");
    return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

static bool RouteMoEGroupedSigmoid(const struct ggml_tensor* gate_logits, const MoEUserData* ud,
                                   densecore::moe::MoERouteResult* routing) {
    if (!gate_logits || !ud || !ud->model || !routing) {
        return false;
    }
    const int n_experts = static_cast<int>(gate_logits->ne[0]);
    const int batch_size = static_cast<int>(gate_logits->ne[1]);
    const int top_k = std::max(1, std::min(ud->k, n_experts));
    if (!EnsureMoERouteStorage(routing, batch_size, top_k)) {
        return false;
    }

    const float* logits = reinterpret_cast<const float*>(gate_logits->data);
    if (!logits) {
        return false;
    }

    const struct ggml_tensor* bias_t = ud->layer ? ud->layer->Get(model_keys::kMoeCorrectionBias) : nullptr;
    const float* correction_bias =
        (bias_t && bias_t->type == GGML_TYPE_F32) ? reinterpret_cast<const float*>(bias_t->data) : nullptr;

    const int n_group = std::max(1, ud->model->moe_n_group);
    const int group_size = std::max(1, n_experts / n_group);
    const int topk_group = std::max(1, std::min(ud->model->moe_topk_group, n_group));

    thread_local std::vector<float> probs;
    thread_local std::vector<float> choice_scores;
    thread_local std::vector<float> group_scores;
    thread_local std::vector<int> active_groups;
    thread_local std::vector<int> selected;
    thread_local std::vector<std::pair<float, int>> candidates;
    probs.assign(static_cast<size_t>(n_experts), 0.0f);
    choice_scores.assign(static_cast<size_t>(n_experts), 0.0f);
    group_scores.assign(static_cast<size_t>(n_group), -std::numeric_limits<float>::infinity());
    active_groups.resize(static_cast<size_t>(n_group));
    selected.assign(static_cast<size_t>(top_k), -1);
    const auto stable_sigmoid = [](float x) -> float {
        if (x >= 0.0f) {
            const float z = std::exp(-x);
            return 1.0f / (1.0f + z);
        }
        const float z = std::exp(x);
        return z / (1.0f + z);
    };

    for (int token_idx = 0; token_idx < batch_size; ++token_idx) {
        const float* row = logits + static_cast<size_t>(token_idx) * n_experts;
        for (int e = 0; e < n_experts; ++e) {
            const float p = stable_sigmoid(row[e]);
            probs[static_cast<size_t>(e)] = p;
            choice_scores[static_cast<size_t>(e)] = p + (correction_bias ? correction_bias[e] : 0.0f);
        }

        for (int g = 0; g < n_group; ++g) {
            const int begin = g * group_size;
            const int end = (g == n_group - 1) ? n_experts : std::min(n_experts, begin + group_size);
            float best = -std::numeric_limits<float>::infinity();
            float second = -std::numeric_limits<float>::infinity();
            for (int e = begin; e < end; ++e) {
                const float score = choice_scores[static_cast<size_t>(e)];
                if (score > best) {
                    second = best;
                    best = score;
                } else if (score > second) {
                    second = score;
                }
            }
            group_scores[static_cast<size_t>(g)] = best + ((end - begin) > 1 ? second : 0.0f);
        }

        std::iota(active_groups.begin(), active_groups.end(), 0);
        std::partial_sort(
            active_groups.begin(), active_groups.begin() + topk_group, active_groups.end(),
            [&](int a, int b) { return group_scores[static_cast<size_t>(a)] > group_scores[static_cast<size_t>(b)]; });

        candidates.clear();
        candidates.reserve(static_cast<size_t>(std::max(top_k, topk_group * group_size)));
        for (int group_rank = 0; group_rank < topk_group; ++group_rank) {
            const int g = active_groups[static_cast<size_t>(group_rank)];
            const int begin = g * group_size;
            const int end = (g == n_group - 1) ? n_experts : std::min(n_experts, begin + group_size);
            for (int e = begin; e < end; ++e) {
                candidates.emplace_back(choice_scores[static_cast<size_t>(e)], e);
            }
        }
        if (static_cast<int>(candidates.size()) < top_k) {
            candidates.clear();
            candidates.reserve(static_cast<size_t>(n_experts));
            for (int e = 0; e < n_experts; ++e) {
                candidates.emplace_back(choice_scores[static_cast<size_t>(e)], e);
            }
        }

        std::partial_sort(candidates.begin(), candidates.begin() + top_k, candidates.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });

        float weight_sum = 0.0f;
        for (int k = 0; k < top_k; ++k) {
            const int expert_id = candidates[static_cast<size_t>(k)].second;
            selected[static_cast<size_t>(k)] = expert_id;
            weight_sum += probs[static_cast<size_t>(expert_id)];
        }

        const float scale = ud->model->moe_routed_scaling_factor;
        for (int k = 0; k < top_k; ++k) {
            const int expert_id = selected[static_cast<size_t>(k)];
            float weight = probs[static_cast<size_t>(expert_id)];
            if (ud->model->moe_norm_topk_prob && weight_sum > 1e-20f) {
                weight /= weight_sum;
            }
            weight *= scale;
            routing->expert_ids[static_cast<size_t>(token_idx * top_k + k)] = expert_id;
            routing->weights[static_cast<size_t>(token_idx * top_k + k)] = weight;
            routing->token_indices[static_cast<size_t>(token_idx * top_k + k)] = token_idx;
        }
    }

    return true;
}

static bool RouteMoESoftmaxTopK(const struct ggml_tensor* gate_logits, const MoEUserData* ud,
                                densecore::moe::MoERouteResult* routing) {
    if (!gate_logits || !ud || !ud->model || !routing) {
        return false;
    }

    const int n_experts = static_cast<int>(gate_logits->ne[0]);
    const int batch_size = static_cast<int>(gate_logits->ne[1]);
    const int top_k = std::max(1, std::min(ud->k, n_experts));
    if (!EnsureMoERouteStorage(routing, batch_size, top_k)) {
        return false;
    }

    const float* logits = reinterpret_cast<const float*>(gate_logits->data);
    if (!logits) {
        return false;
    }

    thread_local std::vector<uint8_t> route_workspace;
    thread_local densecore::moe::MoERoutingWorkspace ws;
    const size_t workspace_bytes = densecore::moe::GetMoERoutingWorkspaceSize(batch_size, n_experts, top_k);
    route_workspace.resize(workspace_bytes + 64);
    if (!densecore::moe::InitMoERoutingWorkspace(&ws, route_workspace.data(), route_workspace.size(), batch_size,
                                                 n_experts, top_k)) {
        return false;
    }
    if (!densecore::moe::MoETopKRoute(logits, batch_size, n_experts, top_k, ud->model->moe_norm_topk_prob, routing,
                                      &ws)) {
        return false;
    }

    const float scale = ud->model->moe_routed_scaling_factor;
    for (float& weight : routing->weights) {
        weight *= scale;
    }
    return true;
}

static bool RouteMoEGemma4TopK(const struct ggml_tensor* gate_logits, const MoEUserData* ud,
                               densecore::moe::MoERouteResult* routing) {
    if (!gate_logits || !ud || !ud->model || !ud->layer || !routing) {
        return false;
    }

    const int n_experts = static_cast<int>(gate_logits->ne[0]);
    const int batch_size = static_cast<int>(gate_logits->ne[1]);
    const int top_k = std::max(1, std::min(ud->k, n_experts));
    if (!EnsureMoERouteStorage(routing, batch_size, top_k)) {
        return false;
    }

    const float* logits = reinterpret_cast<const float*>(gate_logits->data);
    if (!logits) {
        return false;
    }

    thread_local std::vector<uint8_t> route_workspace;
    thread_local densecore::moe::MoERoutingWorkspace ws;
    const size_t workspace_bytes = densecore::moe::GetMoERoutingWorkspaceSize(batch_size, n_experts, top_k);
    route_workspace.resize(workspace_bytes + 64);
    if (!densecore::moe::InitMoERoutingWorkspace(&ws, route_workspace.data(), route_workspace.size(), batch_size,
                                                 n_experts, top_k)) {
        return false;
    }
    // Gemma4/HF router uses softmax probabilities followed by top-k
    // renormalization. The per-expert scale is represented as a down-projection
    // sidecar in GGUF and is applied in the expert projection path, not as a
    // second router-probability multiplier.
    if (!densecore::moe::MoETopKRoute(logits, batch_size, n_experts, top_k, /*normalize_weights=*/true, routing,
                                      &ws)) {
        return false;
    }

    for (size_t i = 0; i < routing->weights.size(); ++i) {
        routing->token_indices[i] = static_cast<int>(i / static_cast<size_t>(top_k));
    }
    return true;
}
