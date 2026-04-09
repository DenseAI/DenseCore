// NOTE: MoEUserData is defined in densecore/inference_types_internal.h

MoEUserData* AllocateMoEUserData(struct ggml_context* ctx_c) {
    if (!ctx_c) {
        return nullptr;
    }
    struct ggml_tensor* storage = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I8, sizeof(MoEUserData));
    if (!storage || !storage->data) {
        return nullptr;
    }
    std::memset(storage->data, 0, sizeof(MoEUserData));
    return reinterpret_cast<MoEUserData*>(storage->data);
}

static GLMDSAPackUserData* AllocateGLMDSAPackUserData(struct ggml_context* ctx_c) {
    if (!ctx_c) {
        return nullptr;
    }
    struct ggml_tensor* storage = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I8, sizeof(GLMDSAPackUserData));
    if (!storage || !storage->data) {
        return nullptr;
    }
    return reinterpret_cast<GLMDSAPackUserData*>(storage->data);
}

static HiddenSnapshotUserData* AllocateHiddenSnapshotUserData(struct ggml_context* ctx_c) {
    if (!ctx_c) {
        return nullptr;
    }
    struct ggml_tensor* storage = ggml_new_tensor_1d(ctx_c, GGML_TYPE_I8, sizeof(HiddenSnapshotUserData));
    if (!storage || !storage->data) {
        return nullptr;
    }
    std::memset(storage->data, 0, sizeof(HiddenSnapshotUserData));
    return reinterpret_cast<HiddenSnapshotUserData*>(storage->data);
}

void cb_pack_glm_dsa_q(struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1, int ith,
                       int nth, void* userdata) {
    (void)src0;
    (void)nth;
    auto* ud = static_cast<GLMDSAPackUserData*>(userdata);
    if (ith != 0 || !ud || !dst || !src1 || !dst->data || !src1->data) {
        return;
    }

    const int n_tokens = static_cast<int>(src1->ne[1]);
    const int q_head_dim = ud->qk_nope_head_dim + ud->qk_rope_head_dim;
    const size_t src_row_stride = static_cast<size_t>(src1->nb[1] / sizeof(float));
    const size_t dst_row_stride = static_cast<size_t>(dst->nb[1] / sizeof(float));
    const float* src = reinterpret_cast<const float*>(src1->data);
    float* out = reinterpret_cast<float*>(dst->data);

    for (int t = 0; t < n_tokens; ++t) {
        const float* src_row = src + static_cast<size_t>(t) * src_row_stride;
        float* dst_row = out + static_cast<size_t>(t) * dst_row_stride;
        for (int h = 0; h < ud->n_heads; ++h) {
            const float* src_head = src_row + static_cast<size_t>(h) * q_head_dim;
            float* dst_head = dst_row + static_cast<size_t>(h) * q_head_dim;
            memcpy(dst_head, src_head + ud->qk_nope_head_dim,
                   static_cast<size_t>(ud->qk_rope_head_dim) * sizeof(float));
            memcpy(dst_head + ud->qk_rope_head_dim, src_head,
                   static_cast<size_t>(ud->qk_nope_head_dim) * sizeof(float));
        }
    }
}

void cb_pack_glm_dsa_k(struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1,
                       const struct ggml_tensor* src2, int ith, int nth, void* userdata) {
    (void)src0;
    (void)nth;
    auto* ud = static_cast<GLMDSAPackUserData*>(userdata);
    if (ith != 0 || !ud || !dst || !src1 || !src2 || !dst->data || !src1->data || !src2->data) {
        return;
    }

    const int n_tokens = static_cast<int>(src1->ne[1]);
    const int k_head_dim = ud->qk_nope_head_dim + ud->qk_rope_head_dim;
    const int kv_proj_head_dim = ud->qk_nope_head_dim + ud->v_head_dim;
    const size_t kv_row_stride = static_cast<size_t>(src1->nb[1] / sizeof(float));
    const size_t rope_row_stride = static_cast<size_t>(src2->nb[1] / sizeof(float));
    const size_t dst_row_stride = static_cast<size_t>(dst->nb[1] / sizeof(float));
    const float* kv_src = reinterpret_cast<const float*>(src1->data);
    const float* rope_src = reinterpret_cast<const float*>(src2->data);
    float* out = reinterpret_cast<float*>(dst->data);

    for (int t = 0; t < n_tokens; ++t) {
        const float* kv_row = kv_src + static_cast<size_t>(t) * kv_row_stride;
        const float* rope_row = rope_src + static_cast<size_t>(t) * rope_row_stride;
        float* dst_row = out + static_cast<size_t>(t) * dst_row_stride;
        for (int h = 0; h < ud->n_heads; ++h) {
            const float* kv_head = kv_row + static_cast<size_t>(h) * kv_proj_head_dim;
            float* dst_head = dst_row + static_cast<size_t>(h) * k_head_dim;
            memcpy(dst_head, rope_row, static_cast<size_t>(ud->qk_rope_head_dim) * sizeof(float));
            memcpy(dst_head + ud->qk_rope_head_dim, kv_head, static_cast<size_t>(ud->qk_nope_head_dim) * sizeof(float));
        }
    }
}

void cb_pack_glm_dsa_v(struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1, int ith,
                       int nth, void* userdata) {
    (void)src0;
    (void)nth;
    auto* ud = static_cast<GLMDSAPackUserData*>(userdata);
    if (ith != 0 || !ud || !dst || !src1 || !dst->data || !src1->data) {
        return;
    }

    const int n_tokens = static_cast<int>(src1->ne[1]);
    const int kv_proj_head_dim = ud->qk_nope_head_dim + ud->v_head_dim;
    const size_t src_row_stride = static_cast<size_t>(src1->nb[1] / sizeof(float));
    const size_t dst_row_stride = static_cast<size_t>(dst->nb[1] / sizeof(float));
    const float* src = reinterpret_cast<const float*>(src1->data);
    float* out = reinterpret_cast<float*>(dst->data);

    for (int t = 0; t < n_tokens; ++t) {
        const float* src_row = src + static_cast<size_t>(t) * src_row_stride;
        float* dst_row = out + static_cast<size_t>(t) * dst_row_stride;
        for (int h = 0; h < ud->n_heads; ++h) {
            const float* src_head = src_row + static_cast<size_t>(h) * kv_proj_head_dim;
            float* dst_head = dst_row + static_cast<size_t>(h) * ud->v_head_dim;
            memcpy(dst_head, src_head + ud->qk_nope_head_dim, static_cast<size_t>(ud->v_head_dim) * sizeof(float));
        }
    }
}

static std::vector<densecore::CpuBackend::ExpertWeights> BuildExpertWeights(const TransformerLayer* layer,
                                                                            const TransformerModel* model) {
    using ExpertWeights = densecore::CpuBackend::ExpertWeights;
    using ExpertPackedInt4Weight = densecore::CpuBackend::ExpertPackedInt4Weight;

    std::vector<ExpertWeights> experts;
    if (!layer) {
        return experts;
    }

    const auto make_int4_binding = [model](const ggml_tensor* tensor, int expected_k,
                                           int expected_n) -> ExpertPackedInt4Weight {
        ExpertPackedInt4Weight packed{};
        if (!model || !tensor || expected_k <= 0 || expected_n <= 0) {
            return packed;
        }
        const auto pack_direct = [&](const TransformerModel::Int4WeightBinding& binding,
                                     const ggml_tensor* bound_tensor, int logical_n) -> ExpertPackedInt4Weight {
            if (!bound_tensor || !bound_tensor->data || !binding.packed || !binding.scales || !binding.zeros ||
                !binding.packed->data || !binding.scales->data || !binding.zeros->data || binding.group_size <= 0 ||
                binding.k != expected_k || binding.n != logical_n) {
                return {};
            }
            ExpertPackedInt4Weight resolved{};
            resolved.packed_weights = reinterpret_cast<const uint8_t*>(bound_tensor->data);
            resolved.scales = reinterpret_cast<const float*>(binding.scales->data);
            resolved.zeros = reinterpret_cast<const float*>(binding.zeros->data);
            resolved.group_size = binding.group_size;
            resolved.K = static_cast<int>(binding.k);
            resolved.N = static_cast<int>(binding.n);
            return resolved;
        };

        const auto it = model->int4_weight_bindings.find(tensor);
        if (it != model->int4_weight_bindings.end()) {
            packed = pack_direct(it->second, tensor, expected_n);
            if (packed.IsValid()) {
                return packed;
            }
        }

        const ggml_tensor* root = tensor->view_src;
        if (!root) {
            return {};
        }
        const auto it_root = model->int4_weight_bindings.find(root);
        if (it_root == model->int4_weight_bindings.end()) {
            return {};
        }
        const auto& binding = it_root->second;
        if (!binding.packed || !binding.scales || !binding.zeros || !binding.packed->data || !binding.scales->data ||
            !binding.zeros->data || binding.group_size <= 0 || binding.k != expected_k || binding.n < expected_n ||
            !tensor->data) {
            return {};
        }

        const size_t row_stride_bytes = static_cast<size_t>(root->nb[1]);
        if (row_stride_bytes == 0) {
            return {};
        }
        const size_t plane_stride_bytes = static_cast<size_t>(root->nb[2]);
        const size_t total_offs = tensor->view_offs;
        size_t expert_idx = 0;
        size_t row_offs_bytes = total_offs;
        if (plane_stride_bytes > 0 && root->ne[2] > 1) {
            expert_idx = total_offs / plane_stride_bytes;
            row_offs_bytes = total_offs % plane_stride_bytes;
        }
        if ((row_offs_bytes % row_stride_bytes) != 0) {
            return {};
        }

        const size_t row_start = row_offs_bytes / row_stride_bytes;
        if (row_start + static_cast<size_t>(expected_n) > static_cast<size_t>(binding.n)) {
            return {};
        }

        const int64_t groups_per_row = binding.k / binding.group_size;
        if (groups_per_row <= 0) {
            return {};
        }

        const size_t scales_row_stride = static_cast<size_t>(binding.scales->nb[1] / sizeof(float));
        const size_t zeros_row_stride = static_cast<size_t>(binding.zeros->nb[1] / sizeof(float));
        if (scales_row_stride == 0 || zeros_row_stride == 0) {
            return {};
        }

        size_t scales_off = row_start * scales_row_stride;
        size_t zeros_off = row_start * zeros_row_stride;
        if (expert_idx > 0) {
            if (binding.scales->ne[2] <= 1 || binding.zeros->ne[2] <= 1 || binding.scales->nb[2] == 0 ||
                binding.zeros->nb[2] == 0) {
                return {};
            }
            scales_off += expert_idx * static_cast<size_t>(binding.scales->nb[2] / sizeof(float));
            zeros_off += expert_idx * static_cast<size_t>(binding.zeros->nb[2] / sizeof(float));
        }

        packed.packed_weights = reinterpret_cast<const uint8_t*>(tensor->data);
        packed.scales = reinterpret_cast<const float*>(binding.scales->data) + scales_off;
        packed.zeros = reinterpret_cast<const float*>(binding.zeros->data) + zeros_off;
        packed.group_size = binding.group_size;
        packed.K = expected_k;
        packed.N = expected_n;
        return packed;
    };

    size_t n_experts = layer->NumExperts();
    experts.reserve(n_experts);

    for (size_t i = 0; i < n_experts; ++i) {
        ExpertWeights w;
        w.w1 = {nullptr, 0};
        w.w2 = {nullptr, 0};
        w.w3 = {nullptr, 0};
        w.hidden_dim = 0;
        w.intermediate_dim = 0;
        w.use_gelu_activation = IsGemma4MoEModel(model, layer);

        auto* gw1 = layer->GetExpert(i, model_keys::kFfnGate);
        if (gw1) {
            w.w1.ptr = gw1->data;
            w.w1.size = ggml_nbytes(gw1);
            w.hidden_dim = static_cast<int>(gw1->ne[0]);
            w.intermediate_dim = static_cast<int>(gw1->ne[1]);
            w.w1_type = static_cast<int>(gw1->type);
            w.w1_tensor = gw1;
            w.w1_int4 = make_int4_binding(gw1, w.hidden_dim, w.intermediate_dim);
        }

        auto* gw2 = layer->GetExpert(i, model_keys::kFfnDown);
        if (gw2) {
            w.w2.ptr = gw2->data;
            w.w2.size = ggml_nbytes(gw2);
            w.w2_type = static_cast<int>(gw2->type);
            w.w2_tensor = gw2;
            w.w2_int4 = make_int4_binding(gw2, w.intermediate_dim, w.hidden_dim);
        }

        auto* gw3 = layer->GetExpert(i, model_keys::kFfnUp);
        if (gw3) {
            w.w3.ptr = gw3->data;
            w.w3.size = ggml_nbytes(gw3);
            w.w3_type = static_cast<int>(gw3->type);
            w.w3_tensor = gw3;
            w.w3_int4 = make_int4_binding(gw3, w.hidden_dim, w.intermediate_dim);
        }

        experts.push_back(w);
    }

    return experts;
}

static int GetMoERebalanceIntervalMs() {
    static const int interval_ms = std::max(250, ParsePositiveEnvInt("DENSECORE_MOE_REBALANCE_INTERVAL_MS", 5000));
    return interval_ms;
}

static int GetMoERebalanceTopK() {
    static const int top_k = std::max(1, ParsePositiveEnvInt("DENSECORE_MOE_REBALANCE_TOP_K", 4));
    return top_k;
}

static bool IsMoEPageMigrationEnabled() {
    static const bool enabled = ParseTruthyEnv("DENSECORE_MOE_ENABLE_PAGE_MIGRATION", false);
    return enabled;
}

static bool IsBenchmarkMode() {
    return ParseTruthyEnv("DENSECORE_BENCH_MODE", false);
}

static bool IsMoEDebugLoggingEnabled() {
    return ParseTruthyEnv("DENSECORE_DEBUG_MOE_CALLBACKS", false);
}

static bool IsMoEStageTimingEnabled() {
    return ParseTruthyEnv("DENSECORE_DEBUG_MOE_STAGE_TIMING", false);
}

static void EnsureMoERebalanceThread(densecore::CpuBackend* backend) {
    if (IsBenchmarkMode() || !backend || backend->IsRebalanceThreadRunning()) {
        return;
    }
    backend->StartRebalanceThread(GetMoERebalanceIntervalMs(), GetMoERebalanceTopK(), IsMoEPageMigrationEnabled());
}

static void UpdateSchedulerExperts(const MoEUserData* ud, const densecore::moe::MoERouteResult& routing) {
    if (!ud || !ud->scheduler || !ud->batch) {
        return;
    }
    const BatchSpec* batch = ud->batch;
    if (batch->seq_id.empty() || batch->scheduler_seq_ids.empty()) {
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
            ud->scheduler->SetPredictedExperts(sched_seq_id, experts);
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
        ud->scheduler->SetPredictedExperts(sched_seq_id, experts);
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

    thread_local std::vector<float> route_workspace;
    thread_local densecore::moe::MoERoutingWorkspace ws;
    const size_t workspace_bytes = densecore::moe::GetMoERoutingWorkspaceSize(batch_size, n_experts, top_k);
    const size_t workspace_floats = (workspace_bytes + sizeof(float) - 1) / sizeof(float);
    route_workspace.resize(workspace_floats);
    if (!densecore::moe::InitMoERoutingWorkspace(&ws, route_workspace.data(), route_workspace.size() * sizeof(float),
                                                 batch_size, n_experts, top_k)) {
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

    thread_local std::vector<float> route_workspace;
    thread_local densecore::moe::MoERoutingWorkspace ws;
    const size_t workspace_bytes = densecore::moe::GetMoERoutingWorkspaceSize(batch_size, n_experts, top_k);
    const size_t workspace_floats = (workspace_bytes + sizeof(float) - 1) / sizeof(float);
    route_workspace.resize(workspace_floats);
    if (!densecore::moe::InitMoERoutingWorkspace(&ws, route_workspace.data(), route_workspace.size() * sizeof(float),
                                                 batch_size, n_experts, top_k)) {
        return false;
    }
    if (!densecore::moe::MoETopKRoute(logits, batch_size, n_experts, top_k, true, routing, &ws)) {
        return false;
    }

    const struct ggml_tensor* per_expert_scale_t = ud->layer->Get(kGemma4RouterPerExpertScaleKey);
    const float* per_expert_scale =
        (per_expert_scale_t && per_expert_scale_t->type == GGML_TYPE_F32) ? reinterpret_cast<const float*>(per_expert_scale_t->data)
                                                                          : nullptr;
    const float scale = ud->model->moe_routed_scaling_factor;
    for (size_t i = 0; i < routing->weights.size(); ++i) {
        const int expert_id = routing->expert_ids[i];
        if (per_expert_scale && expert_id >= 0 && expert_id < n_experts) {
            routing->weights[i] *= per_expert_scale[expert_id];
        }
        routing->weights[i] *= scale;
        routing->token_indices[i] = static_cast<int>(i / static_cast<size_t>(top_k));
    }
    return true;
}

void cb_moe_forward(struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1, int ith,
                    int nth, void* userdata) {
    (void)nth;
    if (ith != 0) return;
    if (IsMoEDebugLoggingEnabled()) {
        static std::atomic<int> moe_call_count{0};
        int call_id = moe_call_count.fetch_add(1);
        fprintf(stderr, "[DBG] cb_moe_forward #%d src0=[%lld,%lld] src1=[%lld,%lld]\n", call_id, (long long)src0->ne[0],
                (long long)src0->ne[1], (long long)src1->ne[0], (long long)src1->ne[1]);
        if (call_id < 8) {
            DebugLogTensorFiniteStats("MOE_IN", src0);
            DebugLogTensorFiniteStats("MOE_GATE", src1);
        }
    }
    auto* ud = static_cast<MoEUserData*>(userdata);
    if (!ud || !ud->layer || !ud->backend) return;
    if (!ud->experts_registered || !ud->experts || ud->n_experts <= 0) return;

    // Routing (src1 = gate_logits)
    const bool debug_stage_timing = IsMoEStageTimingEnabled();
    const auto route_begin = debug_stage_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    thread_local densecore::moe::MoERouteResult routing;
    const bool routed = (ud->model && ud->model->arch_flags.is_glm_moe)
                            ? RouteMoEGroupedSigmoid(src1, ud, &routing)
                        : IsGemma4MoEModel(ud->model, ud->layer) ? RouteMoEGemma4TopK(src1, ud, &routing)
                                                                 : RouteMoESoftmaxTopK(src1, ud, &routing);
    if (!routed || routing.expert_ids.empty()) return;
    const auto route_end = debug_stage_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    UpdateSchedulerExperts(ud, routing);
    const auto scheduler_end =
        debug_stage_timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    // Forward (src0 = input, dst = output)
    densecore::Tensor t_input = GgmlToRowMajorTensor(src0);
    densecore::Tensor t_output = GgmlToRowMajorTensor(dst);

    ud->backend->ForwardMoE(ud->layer, t_input, routing, ud->experts, ud->n_experts, &t_output);
    if (debug_stage_timing) {
        const auto backend_end = std::chrono::steady_clock::now();
        const auto route_us =
            std::chrono::duration_cast<std::chrono::microseconds>(route_end - route_begin).count();
        const auto scheduler_us =
            std::chrono::duration_cast<std::chrono::microseconds>(scheduler_end - route_end).count();
        const auto backend_us =
            std::chrono::duration_cast<std::chrono::microseconds>(backend_end - scheduler_end).count();
        std::fprintf(stderr,
                     "[MOE_STAGE] layer=%d batch=%d top_k=%d assignments=%zu route_us=%lld scheduler_us=%lld "
                     "backend_us=%lld\n",
                     ud->layer_idx, routing.batch_size, routing.top_k, routing.expert_ids.size(),
                     static_cast<long long>(route_us), static_cast<long long>(scheduler_us),
                     static_cast<long long>(backend_us));
    }
    if (IsMoEDebugLoggingEnabled()) {
        static std::atomic<int> moe_out_count{0};
        const int call_id = moe_out_count.fetch_add(1);
        if (call_id < 8) {
            DebugLogTensorFiniteStats("MOE_OUT", dst);
        }
    }
}

// ============================================================================
// SSM (Mamba2) Callback Functions
// ============================================================================

static inline float SoftplusStable(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return std::exp(x);
    return std::log1p(std::exp(x));
}

static inline float SigmoidStable(float x) {
    if (x >= 0.0f) {
        const float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = std::exp(x);
    return z / (1.0f + z);
}

static bool IsSSMNonFiniteDebugEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_SSM_NONFINITE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

struct Qwen35SSMDebugScalars {
    float alpha = std::numeric_limits<float>::quiet_NaN();
    float beta = std::numeric_limits<float>::quiet_NaN();
    float softplus_alpha = std::numeric_limits<float>::quiet_NaN();
    float exp_a_log = std::numeric_limits<float>::quiet_NaN();
    float g = std::numeric_limits<float>::quiet_NaN();
    float decay = std::numeric_limits<float>::quiet_NaN();
    float beta_gate = std::numeric_limits<float>::quiet_NaN();
    float q_sum_sq = std::numeric_limits<float>::quiet_NaN();
    float k_sum_sq = std::numeric_limits<float>::quiet_NaN();
    float rms = std::numeric_limits<float>::quiet_NaN();
};

static void AbortOnFirstSSMNonFinite(int layer_idx, int token_idx, int seq_idx, int head_idx, const char* stage,
                                     const char* var_name, int elem_idx, float value,
                                     const Qwen35SSMDebugScalars& scalars) {
    static std::atomic<bool> fired{false};
    if (!IsSSMNonFiniteDebugEnabled()) return;
    if (fired.exchange(true, std::memory_order_relaxed)) return;

    std::fprintf(
        stderr,
        "[DenseCore][SSM_NONFINITE] layer=%d token=%d seq=%d head=%d stage=%s var=%s elem=%d value=%g "
        "alpha=%g beta=%g softplus=%g exp_a_log=%g g=%g decay=%g beta_gate=%g q_sum_sq=%g k_sum_sq=%g rms=%g\n",
        layer_idx, token_idx, seq_idx, head_idx, stage ? stage : "<unknown>", var_name ? var_name : "<unnamed>",
        elem_idx, static_cast<double>(value), static_cast<double>(scalars.alpha), static_cast<double>(scalars.beta),
        static_cast<double>(scalars.softplus_alpha), static_cast<double>(scalars.exp_a_log),
        static_cast<double>(scalars.g), static_cast<double>(scalars.decay), static_cast<double>(scalars.beta_gate),
        static_cast<double>(scalars.q_sum_sq), static_cast<double>(scalars.k_sum_sq), static_cast<double>(scalars.rms));
    std::fflush(stderr);
    std::abort();
}

static void CheckSSMFiniteScalar(int layer_idx, int token_idx, int seq_idx, int head_idx, const char* stage,
                                 const char* var_name, float value, const Qwen35SSMDebugScalars& scalars) {
    if (!IsSSMNonFiniteDebugEnabled() || std::isfinite(value)) return;
    AbortOnFirstSSMNonFinite(layer_idx, token_idx, seq_idx, head_idx, stage, var_name, -1, value, scalars);
}

static void CheckSSMFiniteVector(int layer_idx, int token_idx, int seq_idx, int head_idx, const char* stage,
                                 const char* var_name, const float* values, int n,
                                 const Qwen35SSMDebugScalars& scalars) {
    if (!IsSSMNonFiniteDebugEnabled() || !values || n <= 0) return;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(values[i])) {
            AbortOnFirstSSMNonFinite(layer_idx, token_idx, seq_idx, head_idx, stage, var_name, i, values[i], scalars);
        }
    }
}

static void CheckSSMFiniteTensor(int layer_idx, const int* token_seq_ids, const struct ggml_tensor* tensor,
                                 const char* stage, const char* var_name) {
    if (!IsSSMNonFiniteDebugEnabled() || !tensor || !tensor->data) return;
    const auto* values = reinterpret_cast<const float*>(tensor->data);
    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = std::max<int64_t>(1, tensor->ne[1]);
    const int64_t total = ggml_nelements(tensor);
    if (!values || ne0 <= 0 || ne1 <= 0 || total <= 0) return;
    const Qwen35SSMDebugScalars scalars{};
    for (int64_t idx = 0; idx < total; ++idx) {
        if (std::isfinite(values[idx])) continue;
        const int token_idx = static_cast<int>(idx / ne0);
        const int seq_idx = (token_seq_ids && token_idx >= 0 && token_idx < ne1) ? token_seq_ids[token_idx] : -1;
        const int elem_idx = static_cast<int>(idx % ne0);
        AbortOnFirstSSMNonFinite(layer_idx, token_idx, seq_idx, -1, stage, var_name, elem_idx, values[idx], scalars);
    }
}

[[noreturn]] static void FatalQwen35SSMRuntimeError(int layer_idx, int token_idx, int seq_idx, int head_idx,
                                                    const char* message) {
    std::fprintf(stderr, "[DenseCore][Qwen35SSM] FATAL layer=%d token=%d seq=%d head=%d: %s\n", layer_idx, token_idx,
                 seq_idx, head_idx, message ? message : "unknown error");
#if !defined(NDEBUG)
    std::abort();
#else
    std::terminate();
#endif
}

static void cb_projection_reference_probe(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                          void* userdata) {
    (void)nth;
    if (ith != 0) return;
    auto* ud = static_cast<ProjectionReferenceUserData*>(userdata);
    if (!dst || !src || !ud || !ud->weight_tensor || !ud->input_tensor) return;
    if (dst->data && src->data) {
        std::memcpy(dst->data, src->data, ggml_nbytes(src));
    }
    if (!src->data) {
        return;
    }

    std::vector<float> ref;
    ComputeMatmulReferenceF32(ud->weight_tensor, ud->input_tensor, &ref);
    if (ref.empty() && ud->int4_packed && ud->int4_scales && ud->int4_zeros) {
        ComputeMatmulReferenceInt4BindingF32(ud, &ref);
    }
    if (ref.empty() && ud->fp8_packed) {
        ComputeMatmulReferenceFp8BindingF32(ud, &ref);
    }
    if (ref.empty()) {
        std::fprintf(stderr,
                     "[PROJ_REF] layer=%d stage=%s var=%s status=skipped reason=reference_unavailable "
                     "weight=%s wtype=%d wne0=%lld wne1=%lld input_type=%d ine0=%lld ine1=%lld int4=%d fp8=%d\n",
                     ud->layer_idx, ud->stage ? ud->stage : "unknown", ud->var_name ? ud->var_name : "unknown",
                     ud->weight_tensor && ud->weight_tensor->name[0] ? ud->weight_tensor->name : "(unnamed)",
                     ud->weight_tensor ? static_cast<int>(ud->weight_tensor->type) : -1,
                     ud->weight_tensor ? static_cast<long long>(ud->weight_tensor->ne[0]) : -1LL,
                     ud->weight_tensor ? static_cast<long long>(ud->weight_tensor->ne[1]) : -1LL,
                     ud->input_tensor ? static_cast<int>(ud->input_tensor->type) : -1,
                     ud->input_tensor ? static_cast<long long>(ud->input_tensor->ne[0]) : -1LL,
                     ud->input_tensor ? static_cast<long long>(ud->input_tensor->ne[1]) : -1LL,
                     ud->int4_packed ? 1 : 0,
                     ud->fp8_packed ? 1 : 0);
        return;
    }

    const float* runtime = reinterpret_cast<const float*>(src->data);
    const int total = ggml_nelements(src);
    if (total <= 0 || static_cast<size_t>(total) != ref.size()) {
        std::fprintf(stderr,
                     "[PROJ_REF] layer=%d stage=%s var=%s status=skipped reason=shape_mismatch runtime=%d ref=%zu\n",
                     ud->layer_idx, ud->stage ? ud->stage : "unknown", ud->var_name ? ud->var_name : "unknown", total,
                     ref.size());
        return;
    }

    float max_abs_diff = 0.0f;
    int max_idx = -1;
    int first_bad_idx = -1;
    bool runtime_nonfinite = false;
    bool ref_nonfinite = false;
    for (int i = 0; i < total; ++i) {
        const bool runtime_finite = std::isfinite(runtime[i]);
        const bool ref_finite = std::isfinite(ref[static_cast<size_t>(i)]);
        if (!runtime_finite || !ref_finite) {
            if (first_bad_idx < 0) {
                first_bad_idx = i;
                runtime_nonfinite = !runtime_finite;
                ref_nonfinite = !ref_finite;
            }
            continue;
        }
        const float diff = std::fabs(runtime[i] - ref[static_cast<size_t>(i)]);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
            max_idx = i;
        }
    }

    const int row_dim = static_cast<int>(src->ne[0]);
    const int bad_token = (first_bad_idx >= 0 && row_dim > 0) ? first_bad_idx / row_dim : -1;
    const int bad_elem = (first_bad_idx >= 0 && row_dim > 0) ? first_bad_idx % row_dim : -1;
    const int bad_seq = (bad_token >= 0 && ud->token_seq_ids && bad_token < static_cast<int>(src->ne[1]))
                            ? ud->token_seq_ids[bad_token]
                            : -1;

    std::fprintf(stderr,
                 "[PROJ_REF] layer=%d stage=%s var=%s total=%d first_bad_idx=%d token=%d seq=%d elem=%d "
                 "runtime_nonfinite=%d ref_nonfinite=%d max_abs_diff=%.8g max_idx=%d\n",
                 ud->layer_idx, ud->stage ? ud->stage : "unknown", ud->var_name ? ud->var_name : "unknown", total,
                 first_bad_idx, bad_token, bad_seq, bad_elem, runtime_nonfinite ? 1 : 0, ref_nonfinite ? 1 : 0,
                 max_abs_diff, max_idx);
    if (first_bad_idx >= 0) {
        std::fprintf(stderr, "[PROJ_REF] first_bad runtime=%g ref=%g\n", static_cast<double>(runtime[first_bad_idx]),
                     static_cast<double>(ref[static_cast<size_t>(first_bad_idx)]));
    } else if (max_idx >= 0) {
        std::fprintf(stderr, "[PROJ_REF] worst_diff runtime=%g ref=%g\n", static_cast<double>(runtime[max_idx]),
                     static_cast<double>(ref[static_cast<size_t>(max_idx)]));
    }
}

static void cb_rmsnorm_reference_probe(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                       void* userdata) {
    (void)nth;
    if (ith != 0) return;

    auto* ud = static_cast<RmsNormReferenceUserData*>(userdata);
    if (!ud || !dst || !src || !src->data || !ud->input_tensor || !ud->input_tensor->data || !ud->norm_weight ||
        !ud->norm_weight->data || src->type != GGML_TYPE_F32 || ud->input_tensor->type != GGML_TYPE_F32 ||
        ud->norm_weight->type != GGML_TYPE_F32) {
        return;
    }
    if (dst->data && src->data) {
        std::memcpy(dst->data, src->data, ggml_nbytes(src));
    }

    const int n_embd = static_cast<int>(ud->input_tensor->ne[0]);
    const int n_tokens = static_cast<int>(ud->input_tensor->ne[1]);
    if (n_embd <= 0 || n_tokens <= 0 || static_cast<int>(src->ne[0]) != n_embd ||
        static_cast<int>(src->ne[1]) != n_tokens) {
        return;
    }

    const float eps = 1e-6f;
    const auto* input_base = reinterpret_cast<const char*>(ud->input_tensor->data);
    const auto* runtime_base = reinterpret_cast<const char*>(src->data);
    const auto* weight = reinterpret_cast<const float*>(ud->norm_weight->data);

    float max_abs_diff = 0.0f;
    int max_idx = -1;
    int max_token = -1;
    float max_actual = 0.0f;
    float max_ref = 0.0f;
    bool runtime_nonfinite = false;
    bool ref_nonfinite = false;

    for (int token_idx = 0; token_idx < n_tokens; ++token_idx) {
        const auto* input_row =
            reinterpret_cast<const float*>(input_base + static_cast<size_t>(token_idx) * ud->input_tensor->nb[1]);
        const auto* runtime_row =
            reinterpret_cast<const float*>(runtime_base + static_cast<size_t>(token_idx) * src->nb[1]);

        double sum_sq = 0.0;
        for (int i = 0; i < n_embd; ++i) {
            const float v = input_row[i];
            sum_sq += static_cast<double>(v) * static_cast<double>(v);
        }
        const float inv_rms = 1.0f / std::sqrt(static_cast<float>(sum_sq / std::max(1, n_embd)) + eps);

        for (int i = 0; i < n_embd; ++i) {
            const float ref = input_row[i] * inv_rms * weight[i];
            const float actual = runtime_row[i];
            runtime_nonfinite = runtime_nonfinite || !std::isfinite(actual);
            ref_nonfinite = ref_nonfinite || !std::isfinite(ref);
            const float diff = std::fabs(actual - ref);
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
                max_idx = i;
                max_token = token_idx;
                max_actual = actual;
                max_ref = ref;
            }
        }
    }

    const int seq_id = (ud->token_seq_ids && max_token >= 0) ? ud->token_seq_ids[max_token] : -1;
    std::fprintf(stderr,
                 "[RMS_REF] layer=%d stage=%s var=%s token=%d seq=%d runtime_nonfinite=%d ref_nonfinite=%d "
                 "max_abs_diff=%.8g max_idx=%d actual=%.8g ref=%.8g\n",
                 ud->layer_idx, ud->stage ? ud->stage : "unknown", ud->var_name ? ud->var_name : "unknown",
                 max_token, seq_id, runtime_nonfinite ? 1 : 0, ref_nonfinite ? 1 : 0, max_abs_diff, max_idx,
                 max_actual, max_ref);
}

static void cb_hidden_snapshot_probe(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                     void* userdata) {
    (void)nth;
    if (ith != 0) return;

    auto* ud = static_cast<HiddenSnapshotUserData*>(userdata);
    if (!ud || !dst || !src || !src->data || src->type != GGML_TYPE_F32) {
        return;
    }
    if (dst->data && src->data) {
        std::memcpy(dst->data, src->data, ggml_nbytes(src));
    }

    const int width = static_cast<int>(src->ne[0]);
    const int tokens = static_cast<int>(std::max<int64_t>(1, src->ne[1]));
    if (width <= 0 || tokens <= 0) {
        return;
    }

    int token_idx = ud->token_idx;
    if (token_idx < 0) {
        token_idx = tokens - 1;
    }
    token_idx = std::max(0, std::min(token_idx, tokens - 1));

    const auto* row = reinterpret_cast<const float*>(reinterpret_cast<const char*>(src->data) +
                                                     static_cast<size_t>(token_idx) * src->nb[1]);
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();
    float max_abs = 0.0f;
    double sum = 0.0;
    double sum_sq = 0.0;
    int finite_ct = 0;
    int nan_ct = 0;
    int inf_ct = 0;
    for (int i = 0; i < width; ++i) {
        const float v = row[i];
        if (std::isnan(v)) {
            nan_ct++;
            continue;
        }
        if (!std::isfinite(v)) {
            inf_ct++;
            continue;
        }
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
        max_abs = std::max(max_abs, std::fabs(v));
        sum += v;
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
        finite_ct++;
    }
    if (!std::isfinite(min_v)) min_v = 0.0f;
    if (!std::isfinite(max_v)) max_v = 0.0f;

    const int seq_id = (ud->token_seq_ids && token_idx >= 0) ? ud->token_seq_ids[token_idx] : -1;
    std::fprintf(stderr,
                 "[HIDDEN_SNAPSHOT] layer=%d stage=%s var=%s token=%d seq=%d width=%d nan=%d inf=%d min=%.8g "
                 "max=%.8g max_abs=%.8g mean=%.8g rms=%.8g\n",
                 ud->layer_idx, ud->stage ? ud->stage : "unknown", ud->var_name ? ud->var_name : "unknown",
                 token_idx, seq_id, width, nan_ct, inf_ct, min_v, max_v, max_abs,
                 finite_ct > 0 ? (sum / finite_ct) : 0.0, finite_ct > 0 ? std::sqrt(sum_sq / finite_ct) : 0.0);

    const int preview = std::min(width, 8);
    std::fprintf(stderr, "[HIDDEN_SNAPSHOT] values");
    for (int i = 0; i < preview; ++i) {
        std::fprintf(stderr, " %.8g", row[i]);
    }
    std::fprintf(stderr, "\n");
}

static void cb_shared_scalar_gate_reference_probe(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith,
                                                  int nth, void* userdata) {
    (void)nth;
    if (ith != 0) return;
    auto* ud = static_cast<SharedScalarGateReferenceUserData*>(userdata);
    if (!dst || !src || !ud || !ud->shared_ffn_pre_gate || !ud->shared_gate_logits_scalar) return;
    if (dst->data && src->data) {
        std::memcpy(dst->data, src->data, ggml_nbytes(src));
    }
    if (!src->data || !ud->shared_ffn_pre_gate->data || !ud->shared_gate_logits_scalar->data) return;

    const int hidden = static_cast<int>(src->ne[0]);
    const int tokens = static_cast<int>(std::max<int64_t>(1, src->ne[1]));
    const int scalar_elems = static_cast<int>(ggml_nelements(ud->shared_gate_logits_scalar));
    if (hidden <= 0 || tokens <= 0 || scalar_elems < tokens) {
        std::fprintf(stderr,
                     "[SHARED_GATE_REF] layer=%d stage=%s var=%s status=skipped reason=shape_mismatch hidden=%d "
                     "tokens=%d scalar=%d\n",
                     ud->layer_idx, ud->stage ? ud->stage : "unknown", ud->var_name ? ud->var_name : "unknown",
                     hidden, tokens, scalar_elems);
        return;
    }

    const float* runtime = reinterpret_cast<const float*>(src->data);
    const float* shared_pre = reinterpret_cast<const float*>(ud->shared_ffn_pre_gate->data);
    const float* gate_logits = reinterpret_cast<const float*>(ud->shared_gate_logits_scalar->data);

    float max_abs_diff = 0.0f;
    int max_idx = -1;
    for (int t = 0; t < tokens; ++t) {
        const float gate = SigmoidStable(gate_logits[t]);
        for (int d = 0; d < hidden; ++d) {
            const int idx = t * hidden + d;
            const float ref = shared_pre[idx] * gate;
            const float diff = std::fabs(runtime[idx] - ref);
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
                max_idx = idx;
            }
        }
    }

    const int token_idx = (max_idx >= 0 && hidden > 0) ? (max_idx / hidden) : -1;
    const int elem_idx = (max_idx >= 0 && hidden > 0) ? (max_idx % hidden) : -1;
    const int seq_idx = (ud->token_seq_ids && token_idx >= 0 && token_idx < tokens) ? ud->token_seq_ids[token_idx] : -1;
    const float gate = (token_idx >= 0 && token_idx < scalar_elems) ? SigmoidStable(gate_logits[token_idx]) : 0.0f;
    const float ref = (max_idx >= 0) ? shared_pre[max_idx] * gate : 0.0f;
    const float actual = (max_idx >= 0) ? runtime[max_idx] : 0.0f;
    std::fprintf(stderr,
                 "[SHARED_GATE_REF] layer=%d stage=%s var=%s token=%d seq=%d elem=%d max_abs_diff=%g runtime=%g ref=%g gate=%g\n",
                 ud->layer_idx, ud->stage ? ud->stage : "unknown", ud->var_name ? ud->var_name : "unknown",
                 token_idx, seq_idx, elem_idx, max_abs_diff, actual, ref, gate);
}

static void cb_attention_core_reference_probe(struct ggml_tensor* dst, const struct ggml_tensor* src,
                                              const struct ggml_tensor* q_tensor, const struct ggml_tensor* k_tensor,
                                              int ith, int nth, void* userdata) {
    (void)nth;
    if (ith != 0) return;
    auto* ud = static_cast<AttentionCoreReferenceUserData*>(userdata);
    if (!dst || !src || !q_tensor || !k_tensor || !ud || !ud->value_tensor) return;
    if (dst->data && src->data) {
        std::memcpy(dst->data, src->data, ggml_nbytes(src));
    }
    if (!src->data || !q_tensor->data || !k_tensor->data || !ud->value_tensor->data) {
        return;
    }
    if (ud->n_head <= 0 || ud->n_head_kv <= 0 || ud->head_dim_q <= 0 || ud->head_dim_k <= 0 || ud->head_dim_v <= 0) {
        return;
    }
    if (ud->n_head % ud->n_head_kv != 0) {
        std::fprintf(stderr,
                     "[ATTN_CORE_REF] layer=%d stage=%s status=skipped reason=invalid_gqa n_head=%d n_head_kv=%d\n",
                     ud->layer_idx, ud->stage ? ud->stage : "unknown", ud->n_head, ud->n_head_kv);
        return;
    }
    if (ud->head_dim_q != ud->head_dim_k) {
        std::fprintf(stderr, "[ATTN_CORE_REF] layer=%d stage=%s status=skipped reason=head_dim_mismatch q=%d k=%d\n",
                     ud->layer_idx, ud->stage ? ud->stage : "unknown", ud->head_dim_q, ud->head_dim_k);
        return;
    }

    const int n_tokens = static_cast<int>(src->ne[2]);
    const int n_total = static_cast<int>(k_tensor->ne[2]);
    if (n_tokens <= 0 || n_total <= 0) {
        return;
    }
    static const bool require_past = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_ATTN_CORE_REFERENCE_REQUIRE_PAST");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    if (require_past && ud->n_past <= 0) {
        return;
    }

    const int n_rep = ud->n_head / ud->n_head_kv;
    const float scale =
        ud->attention_scale > 0.0f ? ud->attention_scale : (1.0f / std::sqrt(static_cast<float>(ud->head_dim_q)));
    const bool causal = n_tokens > 1;

    std::vector<float> q_token(static_cast<size_t>(ud->n_head) * ud->head_dim_q, 0.0f);
    std::vector<float> runtime_token(static_cast<size_t>(ud->n_head) * ud->head_dim_v, 0.0f);
    std::vector<float> ref_token(static_cast<size_t>(ud->n_head) * ud->head_dim_v, 0.0f);
    std::vector<float> k_token(static_cast<size_t>(ud->n_head_kv) * ud->head_dim_k, 0.0f);
    std::vector<float> v_token(static_cast<size_t>(ud->n_head_kv) * ud->head_dim_v, 0.0f);
    std::vector<float> scores(static_cast<size_t>(n_total), -std::numeric_limits<float>::infinity());

    for (int token_idx = 0; token_idx < n_tokens; ++token_idx) {
        if (!ShouldRunAttentionCoreReferenceProbe(ud->layer_idx, token_idx)) {
            continue;
        }

        GatherTokenHeadContiguous(q_tensor, token_idx, ud->head_dim_q, ud->n_head, q_token.data());
        GatherTokenHeadContiguous(src, token_idx, ud->head_dim_v, ud->n_head, runtime_token.data());
        std::fill(ref_token.begin(), ref_token.end(), 0.0f);

        for (int h = 0; h < ud->n_head; ++h) {
            const int kv_head = h / n_rep;
            const float* q_head = q_token.data() + static_cast<size_t>(h) * ud->head_dim_q;
            float* out_head = ref_token.data() + static_cast<size_t>(h) * ud->head_dim_v;
            float max_score = -std::numeric_limits<float>::infinity();

            for (int k_idx = 0; k_idx < n_total; ++k_idx) {
                const int query_pos = ud->n_past + token_idx;
                if (causal && k_idx > query_pos) {
                    scores[static_cast<size_t>(k_idx)] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                if (ud->sliding_window >= 0 && k_idx < (query_pos - ud->sliding_window)) {
                    scores[static_cast<size_t>(k_idx)] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                GatherTokenHeadContiguous(k_tensor, k_idx, ud->head_dim_k, ud->n_head_kv, k_token.data());
                const float* k_head = k_token.data() + static_cast<size_t>(kv_head) * ud->head_dim_k;
                float dot = 0.0f;
                for (int d = 0; d < ud->head_dim_q; ++d) {
                    dot += q_head[d] * k_head[d];
                }
                float score = dot * scale;
                if (ud->logit_softcap > 0.0f) {
                    score = std::tanh(score / ud->logit_softcap) * ud->logit_softcap;
                }
                scores[static_cast<size_t>(k_idx)] = score;
                max_score = std::max(max_score, score);
            }

            if (!std::isfinite(max_score)) {
                continue;
            }

            float denom = 0.0f;
            for (int k_idx = 0; k_idx < n_total; ++k_idx) {
                float score = scores[static_cast<size_t>(k_idx)];
                if (!std::isfinite(score)) {
                    continue;
                }
                const float weight = std::exp(score - max_score);
                denom += weight;
                GatherTokenHeadContiguous(ud->value_tensor, k_idx, ud->head_dim_v, ud->n_head_kv, v_token.data());
                const float* v_head = v_token.data() + static_cast<size_t>(kv_head) * ud->head_dim_v;
                for (int d = 0; d < ud->head_dim_v; ++d) {
                    out_head[d] += weight * v_head[d];
                }
            }

            if (denom > 0.0f) {
                const float inv = 1.0f / denom;
                for (int d = 0; d < ud->head_dim_v; ++d) {
                    out_head[d] *= inv;
                }
            }
        }

        float max_abs_diff = 0.0f;
        int max_idx = -1;
        int first_bad_idx = -1;
        bool runtime_nonfinite = false;
        bool ref_nonfinite = false;
        const int total = static_cast<int>(runtime_token.size());
        for (int i = 0; i < total; ++i) {
            const bool runtime_finite = std::isfinite(runtime_token[static_cast<size_t>(i)]);
            const bool ref_finite = std::isfinite(ref_token[static_cast<size_t>(i)]);
            if (!runtime_finite || !ref_finite) {
                if (first_bad_idx < 0) {
                    first_bad_idx = i;
                    runtime_nonfinite = !runtime_finite;
                    ref_nonfinite = !ref_finite;
                }
                continue;
            }
            const float diff = std::fabs(runtime_token[static_cast<size_t>(i)] - ref_token[static_cast<size_t>(i)]);
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
                max_idx = i;
            }
        }

        const int seq_idx =
            (ud->token_seq_ids && token_idx >= 0 && token_idx < n_tokens) ? ud->token_seq_ids[token_idx] : -1;
        const int max_head = (max_idx >= 0 && ud->head_dim_v > 0) ? (max_idx / ud->head_dim_v) : -1;
        const int max_dim = (max_idx >= 0 && ud->head_dim_v > 0) ? (max_idx % ud->head_dim_v) : -1;
        std::fprintf(stderr,
                     "[ATTN_CORE_REF] layer=%d stage=%s token=%d seq=%d n_past=%d n_total=%d first_bad_idx=%d "
                     "runtime_nonfinite=%d ref_nonfinite=%d max_abs_diff=%.8g max_head=%d max_dim=%d\n",
                     ud->layer_idx, ud->stage ? ud->stage : "unknown", token_idx, seq_idx, ud->n_past, n_total,
                     first_bad_idx, runtime_nonfinite ? 1 : 0, ref_nonfinite ? 1 : 0, max_abs_diff, max_head, max_dim);
        if (first_bad_idx >= 0) {
            std::fprintf(stderr, "[ATTN_CORE_REF] first_bad runtime=%g ref=%g\n",
                         static_cast<double>(runtime_token[static_cast<size_t>(first_bad_idx)]),
                         static_cast<double>(ref_token[static_cast<size_t>(first_bad_idx)]));
        } else if (max_idx >= 0) {
            std::fprintf(stderr, "[ATTN_CORE_REF] worst_diff runtime=%g ref=%g\n",
                         static_cast<double>(runtime_token[static_cast<size_t>(max_idx)]),
                         static_cast<double>(ref_token[static_cast<size_t>(max_idx)]));
        }
    }
}

static void cb_attention_post_reference_probe(struct ggml_tensor* dst, const struct ggml_tensor* src,
                                              const struct ggml_tensor* kqv_tensor, int ith, int nth, void* userdata) {
    (void)nth;
    if (ith != 0) return;
    auto* ud = static_cast<AttentionCoreReferenceUserData*>(userdata);
    if (!dst || !src || !kqv_tensor || !ud) return;
    if (dst->data && src->data) {
        std::memcpy(dst->data, src->data, ggml_nbytes(src));
    }
    if (!src->data || !kqv_tensor->data || ud->n_head <= 0 || ud->head_dim_v <= 0) {
        return;
    }

    const int n_tokens = static_cast<int>(src->ne[1]);
    const int d_inner = static_cast<int>(src->ne[0]);
    if (n_tokens <= 0 || d_inner != ud->n_head * ud->head_dim_v) {
        return;
    }

    std::vector<float> kqv_token(static_cast<size_t>(ud->n_head) * ud->head_dim_v, 0.0f);
    std::vector<float> ref_token(static_cast<size_t>(d_inner), 0.0f);
    const float* runtime = reinterpret_cast<const float*>(src->data);
    const ptrdiff_t runtime_stride = static_cast<ptrdiff_t>(src->nb[1] / sizeof(float));

    for (int token_idx = 0; token_idx < n_tokens; ++token_idx) {
        if (!ShouldRunAttentionPostReferenceProbe(ud->layer_idx, token_idx)) {
            continue;
        }

        GatherTokenHeadContiguous(kqv_tensor, token_idx, ud->head_dim_v, ud->n_head, kqv_token.data());
        std::memcpy(ref_token.data(), kqv_token.data(), static_cast<size_t>(d_inner) * sizeof(float));
        if (ud->gate_tensor && ud->gate_tensor->data) {
            const float* gate_col =
                reinterpret_cast<const float*>(ud->gate_tensor->data) +
                static_cast<ptrdiff_t>(token_idx) * static_cast<ptrdiff_t>(ud->gate_tensor->nb[1] / sizeof(float));
            for (int i = 0; i < d_inner; ++i) {
                ref_token[static_cast<size_t>(i)] *= gate_col[i];
            }
        }

        const float* runtime_col = runtime + static_cast<ptrdiff_t>(token_idx) * runtime_stride;
        float max_abs_diff = 0.0f;
        int max_idx = -1;
        int first_bad_idx = -1;
        bool runtime_nonfinite = false;
        bool ref_nonfinite = false;
        for (int i = 0; i < d_inner; ++i) {
            const bool runtime_finite = std::isfinite(runtime_col[i]);
            const bool ref_finite = std::isfinite(ref_token[static_cast<size_t>(i)]);
            if (!runtime_finite || !ref_finite) {
                if (first_bad_idx < 0) {
                    first_bad_idx = i;
                    runtime_nonfinite = !runtime_finite;
                    ref_nonfinite = !ref_finite;
                }
                continue;
            }
            const float diff = std::fabs(runtime_col[i] - ref_token[static_cast<size_t>(i)]);
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
                max_idx = i;
            }
        }

        const int seq_idx =
            (ud->token_seq_ids && token_idx >= 0 && token_idx < n_tokens) ? ud->token_seq_ids[token_idx] : -1;
        std::fprintf(stderr,
                     "[ATTN_POST_REF] layer=%d stage=%s token=%d seq=%d first_bad_idx=%d runtime_nonfinite=%d "
                     "ref_nonfinite=%d max_abs_diff=%.8g max_idx=%d\n",
                     ud->layer_idx, ud->stage ? ud->stage : "unknown", token_idx, seq_idx, first_bad_idx,
                     runtime_nonfinite ? 1 : 0, ref_nonfinite ? 1 : 0, max_abs_diff, max_idx);
        if (first_bad_idx >= 0) {
            std::fprintf(stderr, "[ATTN_POST_REF] first_bad runtime=%g ref=%g\n",
                         static_cast<double>(runtime_col[first_bad_idx]),
                         static_cast<double>(ref_token[static_cast<size_t>(first_bad_idx)]));
        } else if (max_idx >= 0) {
            std::fprintf(stderr, "[ATTN_POST_REF] worst_diff runtime=%g ref=%g\n",
                         static_cast<double>(runtime_col[max_idx]),
                         static_cast<double>(ref_token[static_cast<size_t>(max_idx)]));
        }
    }
}

static void RunSSMConv1DReference(const float* conv_state, const float* input, const float* weight, float* output,
                                  int channels, int kernel_size) {
    if (!conv_state || !input || !weight || !output || channels <= 0 || kernel_size <= 0) return;
    const int hist = kernel_size - 1;
    for (int ch = 0; ch < channels; ++ch) {
        float sum = 0.0f;
        const float* state_row = conv_state + static_cast<size_t>(ch) * hist;
        const float* weight_row = weight + static_cast<size_t>(ch) * kernel_size;
        for (int k = 0; k < hist; ++k) {
            sum += state_row[k] * weight_row[k];
        }
        sum += input[ch] * weight_row[hist];
        output[ch] = sum;
    }
}

static void LogSSMCoreReferenceDiff(int layer_idx, int token_idx, int seq_idx, int head_idx, const char* stage,
                                    const char* var_name, const float* actual, const float* reference, int n) {
    if (!IsDebugSSMCoreReferenceEnabled() || !actual || !reference || n <= 0) return;
    float max_abs_diff = 0.0f;
    int max_idx = -1;
    int first_bad_idx = -1;
    bool actual_nonfinite = false;
    bool ref_nonfinite = false;
    for (int i = 0; i < n; ++i) {
        const bool a_fin = std::isfinite(actual[i]);
        const bool r_fin = std::isfinite(reference[i]);
        if (!a_fin || !r_fin) {
            if (first_bad_idx < 0) {
                first_bad_idx = i;
                actual_nonfinite = !a_fin;
                ref_nonfinite = !r_fin;
            }
            continue;
        }
        const float diff = std::fabs(actual[i] - reference[i]);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
            max_idx = i;
        }
    }
    if (first_bad_idx < 0 && max_idx < 0) return;
    std::fprintf(stderr,
                 "[DenseCore][SSM_CORE_REF] layer=%d token=%d seq=%d head=%d stage=%s var=%s n=%d first_bad=%d "
                 "actual_nonfinite=%d ref_nonfinite=%d max_abs_diff=%.8g max_idx=%d",
                 layer_idx, token_idx, seq_idx, head_idx, stage ? stage : "unknown", var_name ? var_name : "unknown", n,
                 first_bad_idx, actual_nonfinite ? 1 : 0, ref_nonfinite ? 1 : 0, max_abs_diff, max_idx);
    if (first_bad_idx >= 0) {
        std::fprintf(stderr, " actual=%g ref=%g\n", static_cast<double>(actual[first_bad_idx]),
                     static_cast<double>(reference[first_bad_idx]));
    } else {
        std::fprintf(stderr, " actual=%g ref=%g\n", static_cast<double>(actual[max_idx]),
                     static_cast<double>(reference[max_idx]));
    }
}

static bool RunQwen35ReferenceHeadStep(const Qwen35SSMHeadStepConfig& cfg, float* state_kv, float* y_head,
                                       Qwen35SSMHeadStepStats* stats) {
    if (!cfg.input_t || !cfg.q_head || !cfg.k_head || !cfg.v_head || !cfg.z_head || !cfg.alpha_row || !cfg.beta_row ||
        !state_kv || !y_head || cfg.n_embd <= 0 || cfg.head_dim_k <= 0 || cfg.head_dim_v <= 0) {
        return false;
    }

    std::vector<float> q_norm(static_cast<size_t>(cfg.head_dim_k), 0.0f);
    std::vector<float> k_norm(static_cast<size_t>(cfg.head_dim_k), 0.0f);
    std::vector<float> delta(static_cast<size_t>(cfg.head_dim_v), 0.0f);

    float alpha = cfg.dt_bias;
    float beta = 0.0f;
    for (int i = 0; i < cfg.n_embd; ++i) {
        alpha += cfg.alpha_row[i] * cfg.input_t[i];
        beta += cfg.beta_row[i] * cfg.input_t[i];
    }

    const float softplus_alpha = SoftplusStable(alpha);
    const float exp_a_log = std::exp(cfg.a_log);
    const float g = -exp_a_log * softplus_alpha;
    const float decay = std::exp(g);
    const float beta_gate = SigmoidStable(beta);

    float q_sum_sq = 0.0f;
    float k_sum_sq = 0.0f;
    for (int i = 0; i < cfg.head_dim_k; ++i) {
        q_sum_sq += cfg.q_head[i] * cfg.q_head[i];
        k_sum_sq += cfg.k_head[i] * cfg.k_head[i];
    }
    const float q_inv_norm =
        (1.0f / std::sqrt(static_cast<float>(cfg.head_dim_k))) / std::sqrt(q_sum_sq + cfg.norm_eps);
    const float k_inv_norm = 1.0f / std::sqrt(k_sum_sq + cfg.norm_eps);
    for (int i = 0; i < cfg.head_dim_k; ++i) {
        q_norm[static_cast<size_t>(i)] = cfg.q_head[i] * q_inv_norm;
        k_norm[static_cast<size_t>(i)] = cfg.k_head[i] * k_inv_norm;
    }

    const size_t state_elems = static_cast<size_t>(cfg.head_dim_k) * static_cast<size_t>(cfg.head_dim_v);
    for (size_t i = 0; i < state_elems; ++i) {
        state_kv[i] *= decay;
    }
    for (int v = 0; v < cfg.head_dim_v; ++v) {
        float kv_mem = 0.0f;
        for (int k = 0; k < cfg.head_dim_k; ++k) {
            kv_mem += state_kv[static_cast<size_t>(k) * cfg.head_dim_v + v] * k_norm[static_cast<size_t>(k)];
        }
        delta[static_cast<size_t>(v)] = (cfg.v_head[v] - kv_mem) * beta_gate;
    }
    for (int k = 0; k < cfg.head_dim_k; ++k) {
        float* row = state_kv + static_cast<size_t>(k) * cfg.head_dim_v;
        const float kval = k_norm[static_cast<size_t>(k)];
        for (int v = 0; v < cfg.head_dim_v; ++v) {
            row[v] += kval * delta[static_cast<size_t>(v)];
        }
    }

    float sum_sq = 0.0f;
    for (int v = 0; v < cfg.head_dim_v; ++v) {
        float sum = 0.0f;
        for (int k = 0; k < cfg.head_dim_k; ++k) {
            sum += state_kv[static_cast<size_t>(k) * cfg.head_dim_v + v] * q_norm[static_cast<size_t>(k)];
        }
        y_head[v] = sum;
        sum_sq += sum * sum;
    }
    const float rms = std::sqrt(sum_sq / cfg.head_dim_v + cfg.norm_eps);
    for (int v = 0; v < cfg.head_dim_v; ++v) {
        const float norm_w = cfg.norm_weight ? cfg.norm_weight[v] : 1.0f;
        y_head[v] = (y_head[v] / rms) * norm_w * (cfg.z_head[v] * SigmoidStable(cfg.z_head[v]));
    }

    if (stats) {
        stats->alpha = alpha;
        stats->beta = beta;
        stats->softplus_alpha = softplus_alpha;
        stats->exp_a_log = exp_a_log;
        stats->g = g;
        stats->decay = decay;
        stats->beta_gate = beta_gate;
        stats->q_sum_sq = q_sum_sq;
        stats->k_sum_sq = k_sum_sq;
        stats->rms = rms;
    }
    return true;
}

// Conv1D decode callback: processes N tokens sequentially through the ring buffer
static void cb_ssm_conv1d(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata) {
    (void)nth;
    if (ith != 0) return;
    auto* ud = static_cast<SSMConv1DUserData*>(userdata);
    const float* input = reinterpret_cast<const float*>(src->data);
    float* output = reinterpret_cast<float*>(dst->data);
    if (!ud || !input || !output) return;

    static const bool ssm_passthrough = (std::getenv("SSM_PASSTHROUGH") != nullptr);
    static const bool ssm_conv_passthrough = (std::getenv("DENSECORE_SSM_CONV_PASSTHROUGH") != nullptr);
    const int N = static_cast<int>(src->ne[1]);
    const ptrdiff_t input_stride = static_cast<ptrdiff_t>(src->nb[1] / sizeof(float));
    const ptrdiff_t output_stride = static_cast<ptrdiff_t>(dst->nb[1] / sizeof(float));
    const size_t conv_state_elems =
        static_cast<size_t>(ud->channels) * static_cast<size_t>(std::max(0, ud->kernel_size - 1));
    const bool debug_conv_ref = IsDebugSSMCoreReferenceEnabled();
    std::vector<float> conv_state_before;
    std::vector<float> conv_ref;
    if (debug_conv_ref) {
        conv_state_before.resize(conv_state_elems, 0.0f);
        conv_ref.resize(static_cast<size_t>(ud->channels), 0.0f);
    }
    if (ssm_passthrough || ssm_conv_passthrough) {
        for (int t = 0; t < N; ++t) {
            std::memcpy(output + static_cast<ptrdiff_t>(t) * output_stride,
                        input + static_cast<ptrdiff_t>(t) * input_stride,
                        static_cast<size_t>(ud->channels) * sizeof(float));
        }
        return;
    }
    for (int t = 0; t < N; ++t) {
        float* conv_state = ud->conv_state;
        if (ud->runtime_states && ud->token_seq_ids && ud->ssm_ordinal >= 0) {
            const int seq_idx = ud->token_seq_ids[t];
            if (seq_idx >= 0 && seq_idx < static_cast<int>(ud->runtime_states->size())) {
                auto* seq_states = (*ud->runtime_states)[static_cast<size_t>(seq_idx)];
                if (seq_states && ud->ssm_ordinal < static_cast<int>(seq_states->size())) {
                    auto& seq_state = (*seq_states)[static_cast<size_t>(ud->ssm_ordinal)].conv_state;
                    if (seq_state.size() >= conv_state_elems) {
                        conv_state = seq_state.data();
                    } else {
                        conv_state = nullptr;
                    }
                }
            }
        }
        if (!conv_state) {
            continue;
        }
        if (debug_conv_ref && !conv_state_before.empty()) {
            std::memcpy(conv_state_before.data(), conv_state, conv_state_elems * sizeof(float));
        }
        if (IsSSMNonFiniteDebugEnabled()) {
            const Qwen35SSMDebugScalars scalars{};
            const int seq_idx = (ud->token_seq_ids && t < N) ? ud->token_seq_ids[t] : -1;
            CheckSSMFiniteVector(ud->layer_idx, t, seq_idx, -1, "conv", "conv_input",
                                 input + static_cast<ptrdiff_t>(t) * input_stride, ud->channels, scalars);
            CheckSSMFiniteVector(ud->layer_idx, t, seq_idx, -1, "conv", "conv_state", conv_state,
                                 static_cast<int>(conv_state_elems), scalars);
            CheckSSMFiniteVector(ud->layer_idx, t, seq_idx, -1, "conv", "conv_weight", ud->weight,
                                 ud->channels * ud->kernel_size, scalars);
        }
        densecore::hwy_kernels::SSMConv1DDecode_Hwy(conv_state, input + static_cast<ptrdiff_t>(t) * input_stride,
                                                    ud->weight, output + static_cast<ptrdiff_t>(t) * output_stride,
                                                    ud->channels, ud->kernel_size);
        const int seq_idx = (ud->token_seq_ids && t < N) ? ud->token_seq_ids[t] : -1;
        if (debug_conv_ref && !conv_ref.empty()) {
            RunSSMConv1DReference(conv_state_before.data(), input + static_cast<ptrdiff_t>(t) * input_stride,
                                  ud->weight, conv_ref.data(), ud->channels, ud->kernel_size);
            LogSSMCoreReferenceDiff(ud->layer_idx, t, seq_idx, -1, "conv", "conv_output",
                                    output + static_cast<ptrdiff_t>(t) * output_stride, conv_ref.data(), ud->channels);
        }
        CheckSSMFiniteVector(ud->layer_idx, t, seq_idx, -1, "conv", "conv_output",
                             output + static_cast<ptrdiff_t>(t) * output_stride, ud->channels, {});
    }
}

// Qwen3.5 recurrent delta-net callback.
//
// Inputs:
//   a = convolved qkv_mixed after SiLU [conv_channels, N]
//   b = z projection before SiLU       [d_inner, N]
//   c = normalized layer input         [n_embd, N]
//
// Output:
//   dst = scratch buffer whose first d_inner rows contain the recurrent
//         linear-attention output after gated RMSNorm
void cb_ssm_qwen35_delta(struct ggml_tensor* dst, const struct ggml_tensor* a, const struct ggml_tensor* b,
                                const struct ggml_tensor* c, int ith, int nth, void* userdata) {
    (void)nth;
    if (ith != 0) return;
    if (IsMoEDebugLoggingEnabled()) {
        static std::atomic<int> ssm_call_count{0};
        int ssm_id = ssm_call_count.fetch_add(1);
        fprintf(stderr, "[DBG] cb_ssm_delta #%d a=[%lld,%lld]\n", ssm_id, (long long)a->ne[0], (long long)a->ne[1]);
    }
    auto* ud = static_cast<SSMQwen35DeltaUserData*>(userdata);
    const float* qkv_conv = reinterpret_cast<const float*>(a->data);
    const float* z_proj = reinterpret_cast<const float*>(b->data);
    const float* input = reinterpret_cast<const float*>(c->data);
    float* y_out = reinterpret_cast<float*>(dst->data);
    if (!ud || !qkv_conv || !z_proj || !input || !y_out) return;

    static const bool ssm_passthrough = (std::getenv("SSM_PASSTHROUGH") != nullptr);
    static const bool ssm_delta_passthrough = (std::getenv("DENSECORE_SSM_DELTA_PASSTHROUGH") != nullptr);
    const int N = static_cast<int>(a->ne[1]);
    const int out_elems = static_cast<int>(dst->ne[0]) * N;
    const ptrdiff_t qkv_stride = static_cast<ptrdiff_t>(a->nb[1] / sizeof(float));
    const ptrdiff_t z_stride = static_cast<ptrdiff_t>(b->nb[1] / sizeof(float));
    const ptrdiff_t input_stride = static_cast<ptrdiff_t>(c->nb[1] / sizeof(float));
    const ptrdiff_t out_stride = static_cast<ptrdiff_t>(dst->nb[1] / sizeof(float));
    if (ssm_passthrough || ssm_delta_passthrough) {
        std::memset(y_out, 0, static_cast<size_t>(out_elems) * sizeof(float));
        return;
    }

    static const bool ssm_debug = (std::getenv("SSM_DEBUG") != nullptr);
    const int num_k_heads = ud->n_groups;
    const int num_v_heads = ud->n_heads;
    const int head_k_dim = ud->head_dim_k;
    const int head_v_dim = ud->head_dim_v;
    const int qk_total = head_k_dim * num_k_heads;
    const int state_stride = head_k_dim * head_v_dim;
    const int heads_per_group = (num_k_heads > 0) ? std::max(1, num_v_heads / num_k_heads) : 1;

    // =========================================================================
    // LAYOUT ASSERTIONS & RUNTIME CONTRACT CHECK
    // =========================================================================
    if (!ud || !a || !b || !c || !dst || !a->data || !b->data || !c->data || !dst->data) {
        FatalQwen35SSMRuntimeError(ud ? ud->layer_idx : -1, 0, -1, -1, "SSM delta: null tensor or data");
    }

    const int expected_conv_channels = ud->d_inner + 2 * qk_total;
    
    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || c->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        FatalQwen35SSMRuntimeError(ud->layer_idx, 0, -1, -1, "SSM delta inputs/outputs must be F32");
    }

    if (b->nb[0] != sizeof(float) || c->nb[0] != sizeof(float) || dst->nb[0] != sizeof(float)) {
        FatalQwen35SSMRuntimeError(ud->layer_idx, 0, -1, -1, "SSM delta: non-contiguous element stride detected in b, c, or dst");
    }

    if (a->ne[0] != expected_conv_channels) {
        fprintf(stderr, "[DenseCore][Qwen35SSM] FATAL LAYOUT MISMATCH layer=%d: qkv_conv ne[0]=%lld expected=%d\n",
                ud->layer_idx, (long long)a->ne[0], expected_conv_channels);
        FatalQwen35SSMRuntimeError(ud->layer_idx, 0, -1, -1, "SSM QKV layout mismatch: ne[0] != d_inner + 2*n_groups*head_dim_k");
    }
    if (b->ne[0] != ud->d_inner) {
        FatalQwen35SSMRuntimeError(ud->layer_idx, 0, -1, -1, "SSM z tensor shape mismatch");
    }
    if (c->ne[0] != ud->n_embd) {
        FatalQwen35SSMRuntimeError(ud->layer_idx, 0, -1, -1, "SSM input tensor shape mismatch");
    }
    // ggml_map_custom3() allocates the output tensor with the same shape as `a`
    // (here, qkv_conv). The SSM block only writes the first d_inner values per row,
    // and the caller later views that prefix as [d_inner, N].
    if (dst->ne[0] < ud->d_inner) {
        FatalQwen35SSMRuntimeError(ud->layer_idx, 0, -1, -1, "SSM output tensor shape mismatch");
    }

    const bool requires_qkv_canonicalization = (a->nb[0] != sizeof(float));
    if (!requires_qkv_canonicalization && qkv_stride < expected_conv_channels) {
        FatalQwen35SSMRuntimeError(ud->layer_idx, 0, -1, -1, "SSM QKV row stride is smaller than element count");
    }
    if (z_stride < ud->d_inner) {
        FatalQwen35SSMRuntimeError(ud->layer_idx, 0, -1, -1, "SSM z row stride is smaller than element count");
    }
    if (input_stride < ud->n_embd) {
        FatalQwen35SSMRuntimeError(ud->layer_idx, 0, -1, -1, "SSM input row stride is smaller than element count");
    }



    // Allocate once instead of per-token to reduce memory allocation overhead
    // Stack allocation: Qwen3.5 max ~10KB (head_dim_k=128, head_dim_v=128, d_inner=2048)
    float q_norm_buf[256], k_norm_buf[256], kv_mem_buf[256], delta_buf[256], y_pre_norm_buf[256], y_buf[2048];
    float* const q_norm = q_norm_buf;
    float* const k_norm = k_norm_buf;
    float* const kv_mem = kv_mem_buf;
    float* const delta = delta_buf;
    float* const y_pre_norm = y_pre_norm_buf;
    float* const y = y_buf;
    const bool debug_core_ref = IsDebugSSMCoreReferenceEnabled();
    std::vector<float> q_expanded;
    std::vector<float> k_expanded;
    std::vector<float> ref_state;
    std::vector<float> ref_y_head;
    if (debug_core_ref) {
        q_expanded.resize(static_cast<size_t>(num_v_heads) * head_k_dim, 0.0f);
        k_expanded.resize(static_cast<size_t>(num_v_heads) * head_k_dim, 0.0f);
        ref_state.resize(static_cast<size_t>(state_stride), 0.0f);
        ref_y_head.resize(static_cast<size_t>(head_v_dim), 0.0f);
    }

    std::vector<float> canonical_qkv_row;
    if (requires_qkv_canonicalization) {
        canonical_qkv_row.resize(expected_conv_channels, 0.0f);
    }

    for (int t = 0; t < N; ++t) {
        std::fill(y, y + ud->d_inner, 0.0f);
        const float* input_t = input + static_cast<ptrdiff_t>(t) * input_stride;
        const float* z_t = z_proj + static_cast<ptrdiff_t>(t) * z_stride;
        
        const float* q_base;
        const float* k_base;
        const float* v_base;

        if (requires_qkv_canonicalization) {
            const char* row_bytes = reinterpret_cast<const char*>(a->data) + static_cast<size_t>(t) * a->nb[1];
            for (int i = 0; i < expected_conv_channels; ++i) {
                canonical_qkv_row[i] = *reinterpret_cast<const float*>(row_bytes + static_cast<size_t>(i) * a->nb[0]);
            }
            q_base = canonical_qkv_row.data();
            k_base = canonical_qkv_row.data() + qk_total;
            v_base = canonical_qkv_row.data() + 2 * qk_total;
        } else {
            const float* qkv_t = qkv_conv + static_cast<ptrdiff_t>(t) * qkv_stride;
            q_base = qkv_t;
            k_base = qkv_t + qk_total;
            v_base = qkv_t + 2 * qk_total;
        }
        if (debug_core_ref) {
            for (int h = 0; h < num_v_heads; ++h) {
                const int src_k_head =
                    (num_k_heads == num_v_heads) ? h : std::min(num_k_heads - 1, h / heads_per_group);
                std::memcpy(q_expanded.data() + static_cast<size_t>(h) * head_k_dim,
                            q_base + static_cast<size_t>(src_k_head) * head_k_dim,
                            static_cast<size_t>(head_k_dim) * sizeof(float));
                std::memcpy(k_expanded.data() + static_cast<size_t>(h) * head_k_dim,
                            k_base + static_cast<size_t>(src_k_head) * head_k_dim,
                            static_cast<size_t>(head_k_dim) * sizeof(float));
            }
        }

        for (int h = 0; h < num_v_heads; ++h) {
            const int seq_idx = (ud->token_seq_ids && t < N) ? ud->token_seq_ids[t] : -1;
            const int src_k_head = (num_k_heads == num_v_heads) ? h : std::min(num_k_heads - 1, h / heads_per_group);
            
            // Boundary check for QKV head slicing within the logical row
            if (src_k_head < 0 || src_k_head >= num_k_heads) {
                FatalQwen35SSMRuntimeError(ud->layer_idx, t, seq_idx, h, "SSM delta: K head index out of range");
            }
            if (h < 0 || h >= num_v_heads) {
                FatalQwen35SSMRuntimeError(ud->layer_idx, t, seq_idx, h, "SSM delta: V head index out of range");
            }
            
            const float* q_head = q_base + static_cast<size_t>(src_k_head) * head_k_dim;
            const float* k_head = k_base + static_cast<size_t>(src_k_head) * head_k_dim;
            const float* v_head = v_base + static_cast<size_t>(h) * head_v_dim;
            
            // Validate that slice pointers are within the expected bounds of the row
            const float* row_end = q_base + expected_conv_channels;
            if (q_head + head_k_dim > k_base) {
                FatalQwen35SSMRuntimeError(ud->layer_idx, t, seq_idx, h, "SSM delta: Q head slice out of bounds");
            }
            if (k_head + head_k_dim > v_base) {
                FatalQwen35SSMRuntimeError(ud->layer_idx, t, seq_idx, h, "SSM delta: K head slice out of bounds");
            }
            if (v_head + head_v_dim > row_end) {
                FatalQwen35SSMRuntimeError(ud->layer_idx, t, seq_idx, h, "SSM delta: V head slice out of bounds");
            }

            const float* alpha_row = ud->alpha_weight + static_cast<size_t>(h) * ud->n_embd;
            const float* beta_row = ud->beta_weight + static_cast<size_t>(h) * ud->n_embd;
            float* ssm_state_base = ud->ssm_state;
            if (ud->runtime_states && ud->token_seq_ids && ud->ssm_ordinal >= 0) {
                const int seq_idx = ud->token_seq_ids[t];
                if (seq_idx >= 0 && seq_idx < static_cast<int>(ud->runtime_states->size())) {
                    auto* seq_states = (*ud->runtime_states)[static_cast<size_t>(seq_idx)];
                    if (seq_states && ud->ssm_ordinal < static_cast<int>(seq_states->size())) {
                        auto& seq_state = (*seq_states)[static_cast<size_t>(ud->ssm_ordinal)].ssm_state;
                        const size_t required = static_cast<size_t>(ud->n_heads) * static_cast<size_t>(state_stride);
                        if (seq_state.size() >= required) {
                            ssm_state_base = seq_state.data();
                        } else {
                            ssm_state_base = nullptr;
                        }
                    }
                }
            }
            if (!ssm_state_base) {
                continue;
            }
            float* state = ssm_state_base + static_cast<size_t>(h) * state_stride;
            float* y_head = y + static_cast<size_t>(h) * head_v_dim;
            const float* norm_weight_head = ud->norm_weight;
            if (ud->norm_layout == Qwen35SSMNormLayout::FLATTENED_D_INNER) {
                norm_weight_head += static_cast<size_t>(h) * head_v_dim;
            } else if (ud->norm_layout != Qwen35SSMNormLayout::SHARED_HEAD_DIM) {
                FatalQwen35SSMRuntimeError(ud->layer_idx, t, seq_idx, h, "invalid hybrid SSM norm layout");
            }

            Qwen35SSMHeadStepConfig cfg{};
            cfg.input_t = input_t;
            cfg.q_head = q_head;
            cfg.k_head = k_head;
            cfg.v_head = v_head;
            cfg.z_head = z_t + static_cast<size_t>(h) * head_v_dim;
            cfg.alpha_row = alpha_row;
            cfg.beta_row = beta_row;
            cfg.norm_weight = norm_weight_head;
            cfg.n_embd = ud->n_embd;
            cfg.head_dim_k = head_k_dim;
            cfg.head_dim_v = head_v_dim;
            cfg.dt_bias = ud->dt_bias[h];
            cfg.a_log = ud->a_log[h];
            cfg.norm_eps = ud->norm_eps;

            Qwen35SSMHeadStepStats ref_stats{};
            const float* q_ref_head = q_head;
            const float* k_ref_head = k_head;
            if (debug_core_ref) {
                q_ref_head = q_expanded.data() + static_cast<size_t>(h) * head_k_dim;
                k_ref_head = k_expanded.data() + static_cast<size_t>(h) * head_k_dim;
                LogSSMCoreReferenceDiff(ud->layer_idx, t, seq_idx, h, "delta_mapping", "q_head", q_head, q_ref_head,
                                        head_k_dim);
                LogSSMCoreReferenceDiff(ud->layer_idx, t, seq_idx, h, "delta_mapping", "k_head", k_head, k_ref_head,
                                        head_k_dim);
                std::memcpy(ref_state.data(), state, static_cast<size_t>(state_stride) * sizeof(float));
                Qwen35SSMHeadStepConfig ref_cfg = cfg;
                ref_cfg.q_head = q_ref_head;
                ref_cfg.k_head = k_ref_head;
                if (!RunQwen35ReferenceHeadStep(ref_cfg, ref_state.data(), ref_y_head.data(), &ref_stats)) {
                    FatalQwen35SSMRuntimeError(ud->layer_idx, t, seq_idx, h,
                                               "RunQwen35ReferenceHeadStep rejected runtime inputs");
                }
            }

            Qwen35SSMHeadStepStats step_stats{};
            Qwen35SSMHeadStepDebugBuffers step_debug{};
            step_debug.q_norm = q_norm;
            step_debug.k_norm = k_norm;
            step_debug.kv_mem = kv_mem;
            step_debug.delta = delta;
            step_debug.y_pre_norm = y_pre_norm;
            if (!Qwen35RunGatedDeltaHeadStep(cfg, state, y_head, &step_stats, &step_debug)) {
                FatalQwen35SSMRuntimeError(ud->layer_idx, t, seq_idx, h,
                                           "Qwen35RunGatedDeltaHeadStep rejected runtime inputs");
            }
            if (debug_core_ref) {
                LogSSMCoreReferenceDiff(ud->layer_idx, t, seq_idx, h, "delta_reference", "state", state,
                                        ref_state.data(), state_stride);
                LogSSMCoreReferenceDiff(ud->layer_idx, t, seq_idx, h, "delta_reference", "y_head", y_head,
                                        ref_y_head.data(), head_v_dim);
            }

            Qwen35SSMDebugScalars dbg{};
            dbg.alpha = step_stats.alpha;
            dbg.beta = step_stats.beta;
            dbg.softplus_alpha = step_stats.softplus_alpha;
            dbg.exp_a_log = step_stats.exp_a_log;
            dbg.g = step_stats.g;
            dbg.decay = step_stats.decay;
            dbg.beta_gate = step_stats.beta_gate;
            dbg.q_sum_sq = step_stats.q_sum_sq;
            dbg.k_sum_sq = step_stats.k_sum_sq;
            dbg.rms = step_stats.rms;
            CheckSSMFiniteScalar(ud->layer_idx, t, seq_idx, h, "alpha_projection", "alpha", step_stats.alpha, dbg);
            CheckSSMFiniteScalar(ud->layer_idx, t, seq_idx, h, "beta_projection", "beta", step_stats.beta, dbg);
            CheckSSMFiniteScalar(ud->layer_idx, t, seq_idx, h, "decay", "softplus_alpha", step_stats.softplus_alpha,
                                 dbg);
            CheckSSMFiniteScalar(ud->layer_idx, t, seq_idx, h, "decay", "exp_a_log", step_stats.exp_a_log, dbg);
            CheckSSMFiniteScalar(ud->layer_idx, t, seq_idx, h, "decay", "g", step_stats.g, dbg);
            CheckSSMFiniteScalar(ud->layer_idx, t, seq_idx, h, "decay", "decay", step_stats.decay, dbg);
            CheckSSMFiniteScalar(ud->layer_idx, t, seq_idx, h, "qk_norm", "q_sum_sq", step_stats.q_sum_sq, dbg);
            CheckSSMFiniteScalar(ud->layer_idx, t, seq_idx, h, "qk_norm", "k_sum_sq", step_stats.k_sum_sq, dbg);
            CheckSSMFiniteVector(ud->layer_idx, t, seq_idx, h, "qk_norm", "q_norm", q_norm, head_k_dim, dbg);
            CheckSSMFiniteVector(ud->layer_idx, t, seq_idx, h, "qk_norm", "k_norm", k_norm, head_k_dim, dbg);
            CheckSSMFiniteVector(ud->layer_idx, t, seq_idx, h, "kv_mem", "kv_mem", kv_mem, head_v_dim, dbg);
            CheckSSMFiniteVector(ud->layer_idx, t, seq_idx, h, "delta", "delta", delta, head_v_dim, dbg);
            CheckSSMFiniteVector(ud->layer_idx, t, seq_idx, h, "state_update", "state", state, state_stride, dbg);
            CheckSSMFiniteVector(ud->layer_idx, t, seq_idx, h, "y_pre_norm", "y_head", y_pre_norm, head_v_dim,
                                 dbg);
            CheckSSMFiniteScalar(ud->layer_idx, t, seq_idx, h, "y_post_norm", "rms", step_stats.rms, dbg);
            CheckSSMFiniteVector(ud->layer_idx, t, seq_idx, h, "y_post_norm", "y_head", y_head, head_v_dim, dbg);

            if (ssm_debug && t == 0 && h == 0) {
                fprintf(stderr,
                        "[QWEN35_SSM] alpha=%.4f g=%.4f decay=%.4f beta=%.4f q0=%.4f k0=%.4f v0=%.4f z0=%.4f y0=%.4f\n",
                        step_stats.alpha, step_stats.g, step_stats.decay, step_stats.beta_gate, q_norm[0], k_norm[0],
                        v_head[0], cfg.z_head[0], y_head[0]);
            }
        }

        std::memcpy(y_out + static_cast<ptrdiff_t>(t) * out_stride, y,
                    static_cast<size_t>(ud->d_inner) * sizeof(float));
    }
}
