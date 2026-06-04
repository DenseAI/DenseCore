#include "densecore/models/model_execution_contract.h"

bool RebindHybridSSMDecodeGraphRuntimeState(GgmlGraphHandle* graph, const BatchSpec& batch) {
    if (!graph) {
        return false;
    }
    if (batch.seq_id.empty() || batch.hybrid_ssm_runtime_states.empty()) {
        return false;
    }

    struct Custom1ParamsView {
        ggml_custom1_op_t fun;
        int n_tasks;
        void* userdata;
    };
    struct Custom3ParamsView {
        ggml_custom3_op_t fun;
        int n_tasks;
        void* userdata;
    };
    struct Custom2ParamsView {
        ggml_custom2_op_t fun;
        int n_tasks;
        void* userdata;
    };
    struct CustomParamsView {
        ggml_custom_op_t fun;
        int n_tasks;
        void* userdata;
    };
    static_assert(sizeof(Custom1ParamsView) <= GGML_MAX_OP_PARAMS, "Custom1ParamsView too large");
    static_assert(sizeof(Custom2ParamsView) <= GGML_MAX_OP_PARAMS, "Custom2ParamsView too large");
    static_assert(sizeof(Custom3ParamsView) <= GGML_MAX_OP_PARAMS, "Custom3ParamsView too large");
    static_assert(sizeof(CustomParamsView) <= GGML_MAX_OP_PARAMS, "CustomParamsView too large");

    int conv_rebinds = 0;
    int delta_rebinds = 0;
    const int* seq_ids = batch.seq_id.data();
    const auto* runtime_states = &batch.hybrid_ssm_runtime_states;
    auto* current_work_ctx = GetCurrentWorkContext();
    auto* profile = current_work_ctx ? &current_work_ctx->qwen36_profile : nullptr;

    const int n_nodes = ggml_graph_n_nodes(graph);
    for (int i = 0; i < n_nodes; ++i) {
        struct ggml_tensor* node = ggml_graph_node(graph, i);
        if (!node) {
            continue;
        }

        if (node->op == GGML_OP_MAP_CUSTOM1) {
            Custom1ParamsView params{};
            std::memcpy(&params, node->op_params, sizeof(params));
            if (params.fun == cb_ssm_conv1d) {
                auto* ud = static_cast<SSMConv1DUserData*>(params.userdata);
                if (!ud) {
                    return false;
                }
                ud->token_seq_ids = seq_ids;
                ud->runtime_states = runtime_states;
                ud->profile = profile;
                conv_rebinds++;
            }
            continue;
        }

        if (node->op == GGML_OP_MAP_CUSTOM3) {
            Custom3ParamsView params{};
            std::memcpy(&params, node->op_params, sizeof(params));
            if (params.fun == cb_ssm_qwen35_delta || params.fun == cb_ssm_qwen35_delta_z_qkv_alpha_beta) {
                auto* ud = static_cast<SSMQwen35DeltaUserData*>(params.userdata);
                if (!ud) {
                    return false;
                }
                ud->token_seq_ids = seq_ids;
                ud->runtime_states = runtime_states;
                ud->profile = profile;
                delta_rebinds++;
            }
            continue;
        }

        if (node->op == GGML_OP_MAP_CUSTOM2) {
            Custom2ParamsView params{};
            std::memcpy(&params, node->op_params, sizeof(params));
            if (params.fun == cb_ssm_qwen35_delta_z_qkv) {
                auto* ud = static_cast<SSMQwen35DeltaUserData*>(params.userdata);
                if (!ud) {
                    return false;
                }
                ud->token_seq_ids = seq_ids;
                ud->runtime_states = runtime_states;
                ud->profile = profile;
                delta_rebinds++;
            }
            continue;
        }

        if (node->op == GGML_OP_CUSTOM) {
            CustomParamsView params{};
            std::memcpy(&params, node->op_params, sizeof(params));
            if (params.fun == cb_ssm_qwen35_delta_custom) {
                auto* ud = static_cast<SSMQwen35DeltaUserData*>(params.userdata);
                if (!ud) {
                    return false;
                }
                ud->token_seq_ids = seq_ids;
                ud->runtime_states = runtime_states;
                ud->profile = profile;
                delta_rebinds++;
            }
        }
    }

    return conv_rebinds > 0 && delta_rebinds > 0 && conv_rebinds == delta_rebinds;
}

// LFM2 / LFM2.5: re-point the short-conv custom ops in a cached decode graph at
// the current batch's per-sequence conv state. Only the conv mixer carries
// cross-token state; MoE/attention nodes are stateless across replays.
bool RebindLFM2DecodeGraphRuntimeState(GgmlGraphHandle* graph, const BatchSpec& batch) {
    if (!graph) {
        return false;
    }
    if (batch.seq_id.empty() || batch.hybrid_ssm_runtime_states.empty()) {
        return false;
    }

    struct Custom2ParamsView {
        ggml_custom2_op_t fun;
        int n_tasks;
        void* userdata;
    };
    static_assert(sizeof(Custom2ParamsView) <= GGML_MAX_OP_PARAMS, "Custom2ParamsView too large");

    const int* seq_ids = batch.seq_id.data();
    const auto* runtime_states = &batch.hybrid_ssm_runtime_states;
    int conv_rebinds = 0;

    const int n_nodes = ggml_graph_n_nodes(graph);
    for (int i = 0; i < n_nodes; ++i) {
        struct ggml_tensor* node = ggml_graph_node(graph, i);
        if (!node || node->op != GGML_OP_MAP_CUSTOM2) {
            continue;
        }
        Custom2ParamsView params{};
        std::memcpy(&params, node->op_params, sizeof(params));
        if (params.fun == cb_lfm2_shortconv) {
            auto* ud = static_cast<LFM2ShortConvUserData*>(params.userdata);
            if (!ud) {
                return false;
            }
            ud->token_seq_ids = seq_ids;
            ud->runtime_states = runtime_states;
            conv_rebinds++;
        }
    }

    return conv_rebinds > 0;
}

bool RebindDecodeGraphRuntimeStateForModel(const TransformerModel* model, GgmlGraphHandle* graph,
                                           const BatchSpec& batch) {
    const auto contract = densecore::models::BuildModelExecutionContract(model);
    if (!densecore::models::ModelExecutionContractRequiresDecodeGraphRuntimeRebind(contract)) {
        return true;
    }

    bool needs_hybrid_ssm_rebind = false;
    bool needs_lfm2_rebind = false;
    for (const auto& descriptor : contract.rebind_descriptors) {
        switch (descriptor.op_kind) {
        case densecore::models::ExecutionCustomOpRebindKind::HybridSSMConv1D:
        case densecore::models::ExecutionCustomOpRebindKind::HybridSSMDelta:
            needs_hybrid_ssm_rebind = true;
            break;
        case densecore::models::ExecutionCustomOpRebindKind::LFM2ShortConv:
            needs_lfm2_rebind = true;
            break;
        case densecore::models::ExecutionCustomOpRebindKind::None:
            break;
        }
    }

    bool ok = true;
    if (needs_hybrid_ssm_rebind) {
        ok = RebindHybridSSMDecodeGraphRuntimeState(graph, batch) && ok;
    }
    if (needs_lfm2_rebind) {
        ok = RebindLFM2DecodeGraphRuntimeState(graph, batch) && ok;
    }
    return ok;
}
