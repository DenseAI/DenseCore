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
    static_assert(sizeof(Custom1ParamsView) <= GGML_MAX_OP_PARAMS, "Custom1ParamsView too large");
    static_assert(sizeof(Custom2ParamsView) <= GGML_MAX_OP_PARAMS, "Custom2ParamsView too large");
    static_assert(sizeof(Custom3ParamsView) <= GGML_MAX_OP_PARAMS, "Custom3ParamsView too large");

    int conv_rebinds = 0;
    int delta_rebinds = 0;
    const int* seq_ids = batch.seq_id.data();
    const auto* runtime_states = &batch.hybrid_ssm_runtime_states;

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
                delta_rebinds++;
            }
        }
    }

    return conv_rebinds > 0 && delta_rebinds > 0 && conv_rebinds == delta_rebinds;
}
