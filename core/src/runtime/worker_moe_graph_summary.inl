// Graph-cache flags and MoE graph summary diagnostics.
bool IsGraphCacheReuseDisabled() {
    return GetWorkerRuntimeConfig().graph_cache_reuse_disabled;
}

bool IsMoETracePlumbingDisabled() {
    return GetWorkerRuntimeConfig().moe_trace_plumbing_disabled;
}

bool IsMoEGraphSummaryEnabled() {
    return GetWorkerRuntimeConfig().moe_graph_summary;
}

void MaybeLogMoEGraphSummary(struct ggml_cgraph* gf, const TransformerModel* model, bool is_prefill_batch) {
    if (!IsMoEGraphSummaryEnabled() || !gf || !model || model->hparams.n_layer == 0 || model->hparams.n_experts == 0) {
        return;
    }
    static std::atomic<bool> logged_prefill{false};
    static std::atomic<bool> logged_decode{false};
    if (is_prefill_batch) {
        if (logged_prefill.exchange(true, std::memory_order_relaxed)) return;
    } else {
        if (logged_decode.exchange(true, std::memory_order_relaxed)) return;
    }

    const int n_nodes = ggml_graph_n_nodes(gf);
    uint64_t moe_gating_ops = 0;
    uint64_t moe_scatter_ops = 0;
    uint64_t moe_forward_ops = 0;
    uint64_t moe_gather_ops = 0;
    uint64_t dense_ffn_matmul_ops = 0;
    std::vector<uint64_t> layer_dense_ffn_ops(model->hparams.n_layer, 0);
    std::vector<uint64_t> layer_moe_forward_ops(model->hparams.n_layer, 0);
    std::vector<uint64_t> layer_moe_gate_ops(model->hparams.n_layer, 0);

    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor* node = ggml_graph_node(gf, i);
        if (!node) continue;
        const char* name = node->name;
        if (!name || !name[0]) continue;

        int layer_idx = -1;
        if (std::sscanf(name, "blk.%d.", &layer_idx) != 1) {
            layer_idx = -1;
        }
        const bool layer_ok = layer_idx >= 0 && layer_idx < static_cast<int>(model->hparams.n_layer);

        if (std::strstr(name, ".moe_gate_logits")) {
            ++moe_gating_ops;
            if (layer_ok) {
                ++layer_moe_gate_ops[static_cast<size_t>(layer_idx)];
            }
        } else if (std::strstr(name, ".moe_forward")) {
            ++moe_forward_ops;
            if (layer_ok) {
                ++layer_moe_forward_ops[static_cast<size_t>(layer_idx)];
            }
        } else if (std::strstr(name, ".ffn_gate") || std::strstr(name, ".ffn_up") || std::strstr(name, ".ffn_down")) {
            ++dense_ffn_matmul_ops;
            if (layer_ok) {
                ++layer_dense_ffn_ops[static_cast<size_t>(layer_idx)];
            }
        }
    }

    std::fprintf(stderr,
                 "[MOE_GRAPH_SUMMARY] phase=%s nodes=%d ops{MoEGating:%llu,MoEScatter:%llu,MoEForward:%llu,"
                 "MoEGather:%llu,dense_ffn_matmul:%llu} layers=%u\n",
                 is_prefill_batch ? "prefill" : "decode", n_nodes, static_cast<unsigned long long>(moe_gating_ops),
                 static_cast<unsigned long long>(moe_scatter_ops), static_cast<unsigned long long>(moe_forward_ops),
                 static_cast<unsigned long long>(moe_gather_ops), static_cast<unsigned long long>(dense_ffn_matmul_ops),
                 model->hparams.n_layer);

    for (uint32_t layer = 0; layer < model->hparams.n_layer; ++layer) {
        const TransformerLayer& l = model->layers[layer];
        std::fprintf(stderr,
                     "[MOE_GRAPH_LAYER] phase=%s layer=%u is_moe=%d has_moe_gate=%d num_experts=%zu "
                     "ffn_ops{moe_gate:%llu,moe_forward:%llu,dense_ffn_matmul:%llu}\n",
                     is_prefill_batch ? "prefill" : "decode", layer, l.is_moe ? 1 : 0,
                     l.Get(model_keys::kMoeGate) ? 1 : 0, l.NumExperts(),
                     static_cast<unsigned long long>(layer_moe_gate_ops[layer]),
                     static_cast<unsigned long long>(layer_moe_forward_ops[layer]),
                     static_cast<unsigned long long>(layer_dense_ffn_ops[layer]));
    }
}
