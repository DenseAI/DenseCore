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

    const bool force_qwen36_moe_safe_reference = []() {
        const char* env = std::getenv("DENSECORE_QWEN36_MOE_FORCE_SAFE_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    const densecore::models::DecoderLayerSpec* layer_spec =
        densecore::models::ResolveDecoderLayerSpecForLayer(model, layer);

    for (size_t i = 0; i < n_experts; ++i) {
        ExpertWeights w;
        w.w1 = {nullptr, 0};
        w.w2 = {nullptr, 0};
        w.w3 = {nullptr, 0};
        w.hidden_dim = 0;
        w.intermediate_dim = 0;
        w.use_gelu_activation =
            layer_spec ? layer_spec->ffn.activation == densecore::models::DecoderActivation::GeluPytorchTanh
                       : densecore::models::IsGemma4MoEModel(model, layer);
        w.force_safe_reference = false;
        w.gate_up_tensor = layer->GetExpert(i, model_keys::kGemma4PackedGateUpExpert);
        w.w2_scale_tensor = layer->GetExpert(i, model_keys::kGemma4PackedDownScale);
        w.uses_canonical_gemma4_packed_layout = (w.gate_up_tensor != nullptr);

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

        if (force_qwen36_moe_safe_reference && model && model->variant == ModelVariant::QWEN36 &&
            model->hparams.n_experts > 0) {
            w.force_safe_reference = true;
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
