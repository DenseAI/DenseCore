void AccumulateQwen36SSMProjectionNodeTimes(InferenceWorkContext* work_ctx, ggml_cgraph* graph) {
    if (!work_ctx || !graph) {
        return;
    }
    uint64_t qkv_ns = 0;
    uint64_t gate_ns = 0;
    uint64_t out_ns = 0;
    const int n_nodes = ggml_graph_n_nodes(graph);
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor* node = ggml_graph_node(graph, i);
        if (!node) {
            continue;
        }
        const int64_t elapsed_us = ggml_cpu_get_last_node_perf_time_us(node);
        if (elapsed_us <= 0) {
            continue;
        }
        const uint64_t elapsed_ns = static_cast<uint64_t>(elapsed_us) * 1000ULL;
        if (HasNamePrefix(node, "qwen36_ssm_qkv_proj") || HasNamePrefix(node, "qwen35_ssm_qkv_proj") ||
            HasNamePrefix(node, "qwen35_ssm_qkv_gate_fused_proj") ||
            HasNamePrefix(node, "qwen36_ssm_qkv_gate_fused_proj")) {
            qkv_ns += elapsed_ns;
        } else if (HasNamePrefix(node, "qwen36_ssm_gate_proj") || HasNamePrefix(node, "qwen35_ssm_gate_proj")) {
            gate_ns += elapsed_ns;
        } else if (HasNamePrefix(node, "qwen36_ssm_out_proj") || HasNamePrefix(node, "qwen35_ssm_out_proj")) {
            out_ns += elapsed_ns;
        }
    }
    if (qkv_ns || gate_ns || out_ns) {
        AddQwen36SSMProjectionWallProfile(work_ctx, qkv_ns, gate_ns, out_ns);
    }
}

struct Qwen36PrefillBreakdown {
    uint64_t ssm_projection_ns = 0;
    uint64_t ssm_delta_state_ns = 0;
    uint64_t attention_ns = 0;
    uint64_t mlp_or_moe_ns = 0;
    std::vector<MatmulShapeCensusEntry> top_slow_ops;
};

struct HybridSSMGraphTimingBreakdown {
    uint64_t qkv_ns = 0;
    uint64_t gate_ns = 0;
    uint64_t out_ns = 0;
    uint64_t conv1d_ns = 0;
    uint64_t delta_ns = 0;
    uint64_t alpha_beta_qk_ns = 0;
    uint64_t total_ns = 0;
};

struct NativeMoEGraphTimingBreakdown {
    uint64_t route_ns = 0;
    uint64_t w1w3_ns = 0;
    uint64_t activation_ns = 0;
    uint64_t w2_ns = 0;
    uint64_t w1w3_fast_ns = 0;
    uint64_t w2_fast_ns = 0;
    uint64_t reduce_ns = 0;
    uint64_t total_ns = 0;
    uint64_t route_count = 0;
    uint64_t w1w3_count = 0;
    uint64_t activation_count = 0;
    uint64_t w2_count = 0;
    uint64_t w1w3_fast_count = 0;
    uint64_t w2_fast_count = 0;
    uint64_t reduce_count = 0;
    uint64_t native_node_count = 0;
    std::string node_hist;
    std::vector<MatmulShapeCensusEntry> top_slow_nodes;
};

struct DecodeGraphNodeTimingBreakdown {
    uint64_t measured_ns = 0;
    uint64_t custom_ns = 0;
    uint64_t custom_moe_ns = 0;
    uint64_t custom_ssm_ns = 0;
    uint64_t custom_projection_ns = 0;
    uint64_t custom_lm_head_ns = 0;
    uint64_t custom_paged_attention_ns = 0;
    uint64_t custom_other_ns = 0;
    uint64_t mul_mat_ns = 0;
    uint64_t mul_mat_id_ns = 0;
    uint64_t norm_ns = 0;
    uint64_t view_copy_ns = 0;
    uint64_t elementwise_ns = 0;
    uint64_t attention_ns = 0;
    uint64_t other_ns = 0;
    uint64_t custom_count = 0;
    uint64_t custom_moe_count = 0;
    uint64_t custom_ssm_count = 0;
    uint64_t custom_projection_count = 0;
    uint64_t custom_lm_head_count = 0;
    uint64_t custom_paged_attention_count = 0;
    uint64_t custom_other_count = 0;
    uint64_t mul_mat_count = 0;
    uint64_t mul_mat_id_count = 0;
    uint64_t norm_count = 0;
    uint64_t view_copy_count = 0;
    uint64_t elementwise_count = 0;
    uint64_t attention_count = 0;
    uint64_t other_count = 0;
    std::vector<MatmulShapeCensusEntry> top_slow_nodes;
};

static const char* DecodeGraphCustomBucketName(const ggml_tensor* node) {
    if (!node || node->op != GGML_OP_CUSTOM) {
        return "other";
    }
    const char* name = node->name[0] ? node->name : "";
    const char* src0 = node->src[0] && node->src[0]->name[0] ? node->src[0]->name : "";
    const char* src1 = node->src[1] && node->src[1]->name[0] ? node->src[1]->name : "";
    auto contains = [&](const char* needle) {
        return std::strstr(name, needle) || std::strstr(src0, needle) || std::strstr(src1, needle);
    };
    struct CustomOpParamsRawView {
        std::uintptr_t fun;
        int n_tasks;
        void* userdata;
    };
    static_assert(sizeof(CustomOpParamsRawView) <= GGML_MAX_OP_PARAMS, "custom op params view too large");
    CustomOpParamsRawView params{};
    std::memcpy(&params, node->op_params, sizeof(params));
    const auto same_fun = [&](auto* fun) { return params.fun == reinterpret_cast<std::uintptr_t>(fun); };
    if (same_fun(cb_paged_attention_decode) || same_fun(cb_flash_attention_hal_custom)) {
        return "paged_attention";
    }
    if (same_fun(cb_gemv_custom) || same_fun(cb_gemv_batched_custom)) {
        if (contains("token_embd.weight") || contains("output.weight") || contains("lm_head")) {
            return "lm_head";
        }
        return "projection";
    }
    if (contains("native_moe") || contains("qwen35_native_moe") || contains("lfm2_native_moe")) {
        return "moe";
    }
    if (contains("lfm2_shortconv") || contains("shortconv") || contains("ssm") || contains("conv1d") ||
        contains("delta")) {
        return "ssm_stateful";
    }
    if (contains("token_embd.weight") || contains("output.weight") || contains("lm_head")) {
        return "lm_head";
    }
    if (contains("paged_attention")) {
        return "paged_attention";
    }
    return "other";
}

struct Gemma4PrefillAttentionTimingBreakdown {
    uint64_t attention_ns = 0;
    uint64_t portable_flash_ns = 0;
    uint64_t hal_ns = 0;
    uint64_t moe_or_mlp_ns = 0;
    uint64_t mul_mat_id_ns = 0;
    uint64_t mul_mat_ns = 0;
    std::vector<MatmulShapeCensusEntry> top_slow_ops;
};

static const char* DecodeGraphNodeBucketName(const ggml_tensor* node) {
    if (!node) {
        return "other";
    }
    const char* name = node->name[0] ? node->name : "";
    if (std::strstr(name, "attn") || std::strstr(name, "paged") || std::strstr(name, "flash") ||
        std::strstr(name, "kv_") || std::strstr(name, "rope")) {
        return "attention";
    }
    switch (node->op) {
    case GGML_OP_CUSTOM: return "custom";
    case GGML_OP_MUL_MAT: return "mul_mat";
    case GGML_OP_MUL_MAT_ID: return "mul_mat_id";
    case GGML_OP_RMS_NORM:
    case GGML_OP_NORM:
    case GGML_OP_GROUP_NORM: return "norm";
    case GGML_OP_VIEW:
    case GGML_OP_RESHAPE:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE:
    case GGML_OP_CONT:
    case GGML_OP_DUP:
    case GGML_OP_CPY: return "view_copy";
    case GGML_OP_ADD:
    case GGML_OP_ADD1:
    case GGML_OP_SUB:
    case GGML_OP_MUL:
    case GGML_OP_DIV:
    case GGML_OP_SQR:
    case GGML_OP_SQRT:
    case GGML_OP_SCALE:
    case GGML_OP_UNARY:
    case GGML_OP_SOFT_MAX:
    case GGML_OP_SUM:
    case GGML_OP_SUM_ROWS:
    case GGML_OP_REPEAT:
    case GGML_OP_GET_ROWS: return "elementwise";
    default: return "other";
    }
}

const char* Gemma4WeightClass(const char* name);
const char* Gemma4CensusWeightType(ggml_type type);
bool Gemma4NodeHasCopyLikeInput(const ggml_tensor* node);
std::string Gemma4MatmulShapeBucket(const ggml_tensor* node, const ggml_tensor* weight);
std::string Gemma4CustomNodeClass(const ggml_tensor* node);

DecodeGraphNodeTimingBreakdown SummarizeDecodeGraphNodeTimes(const ggml_cgraph* graph,
                                                             bool collect_top_slow_nodes) {
    DecodeGraphNodeTimingBreakdown out;
    if (!graph) {
        return out;
    }
    const int n_nodes = ggml_graph_n_nodes(const_cast<ggml_cgraph*>(graph));
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor* node = ggml_graph_node(const_cast<ggml_cgraph*>(graph), i);
        if (!node) {
            continue;
        }
        const char* bucket = DecodeGraphNodeBucketName(node);
        if (std::strcmp(bucket, "custom") == 0) {
            out.custom_count += 1;
            const char* custom_bucket = DecodeGraphCustomBucketName(node);
            if (std::strcmp(custom_bucket, "moe") == 0) {
                out.custom_moe_count += 1;
            } else if (std::strcmp(custom_bucket, "ssm_stateful") == 0) {
                out.custom_ssm_count += 1;
            } else if (std::strcmp(custom_bucket, "projection") == 0) {
                out.custom_projection_count += 1;
            } else if (std::strcmp(custom_bucket, "lm_head") == 0) {
                out.custom_lm_head_count += 1;
            } else if (std::strcmp(custom_bucket, "paged_attention") == 0) {
                out.custom_paged_attention_count += 1;
            } else {
                out.custom_other_count += 1;
            }
        } else if (std::strcmp(bucket, "mul_mat") == 0) {
            out.mul_mat_count += 1;
        } else if (std::strcmp(bucket, "mul_mat_id") == 0) {
            out.mul_mat_id_count += 1;
        } else if (std::strcmp(bucket, "norm") == 0) {
            out.norm_count += 1;
        } else if (std::strcmp(bucket, "view_copy") == 0) {
            out.view_copy_count += 1;
        } else if (std::strcmp(bucket, "elementwise") == 0) {
            out.elementwise_count += 1;
        } else if (std::strcmp(bucket, "attention") == 0) {
            out.attention_count += 1;
        } else {
            out.other_count += 1;
        }
        const int64_t elapsed_us = ggml_cpu_get_last_node_perf_time_us(node);
        if (elapsed_us <= 0) {
            continue;
        }
        const uint64_t elapsed_ns = static_cast<uint64_t>(elapsed_us) * 1000ULL;
        if (std::strcmp(bucket, "custom") == 0) {
            out.custom_ns += elapsed_ns;
            const char* custom_bucket = DecodeGraphCustomBucketName(node);
            if (std::strcmp(custom_bucket, "moe") == 0) {
                out.custom_moe_ns += elapsed_ns;
            } else if (std::strcmp(custom_bucket, "ssm_stateful") == 0) {
                out.custom_ssm_ns += elapsed_ns;
            } else if (std::strcmp(custom_bucket, "projection") == 0) {
                out.custom_projection_ns += elapsed_ns;
            } else if (std::strcmp(custom_bucket, "lm_head") == 0) {
                out.custom_lm_head_ns += elapsed_ns;
            } else if (std::strcmp(custom_bucket, "paged_attention") == 0) {
                out.custom_paged_attention_ns += elapsed_ns;
            } else {
                out.custom_other_ns += elapsed_ns;
            }
        } else if (std::strcmp(bucket, "mul_mat") == 0) {
            out.mul_mat_ns += elapsed_ns;
        } else if (std::strcmp(bucket, "mul_mat_id") == 0) {
            out.mul_mat_id_ns += elapsed_ns;
        } else if (std::strcmp(bucket, "norm") == 0) {
            out.norm_ns += elapsed_ns;
        } else if (std::strcmp(bucket, "view_copy") == 0) {
            out.view_copy_ns += elapsed_ns;
        } else if (std::strcmp(bucket, "elementwise") == 0) {
            out.elementwise_ns += elapsed_ns;
        } else if (std::strcmp(bucket, "attention") == 0) {
            out.attention_ns += elapsed_ns;
        } else {
            out.other_ns += elapsed_ns;
        }
        out.measured_ns += elapsed_ns;

        // The op-bucket totals above are cheap and always collected for decode so
        // every profiling run shows where graph-execute time lands. The per-node
        // census below allocates a string per node, so it stays behind the debug
        // flag to keep the hot decode path free of that overhead.
        if (!collect_top_slow_nodes) {
            continue;
        }
        MatmulShapeCensusEntry entry;
        entry.phase = "decode";
        entry.op_type = ggml_op_name(node->op);
        const std::string custom_class = Gemma4CustomNodeClass(node);
        entry.dispatch_path = custom_class.empty() ? bucket : custom_class;
        entry.weight_type = ggml_op_name(node->op);
        std::ostringstream shape;
        shape << "M=" << node->ne[1] << ",N=" << node->ne[0] << ",K=" << (node->ne[2] > 1 ? node->ne[2] : 0);
        entry.shape_bucket = shape.str();
        entry.left_name = node->name[0] ? node->name : "unnamed";
        entry.right_name = "elapsed_us";
        entry.ops = static_cast<uint64_t>(elapsed_us);
        out.top_slow_nodes.push_back(std::move(entry));
    }
    std::sort(out.top_slow_nodes.begin(), out.top_slow_nodes.end(),
              [](const auto& a, const auto& b) { return a.ops != b.ops ? a.ops > b.ops : a.left_name < b.left_name; });
    if (out.top_slow_nodes.size() > kMatmulTopShapeCount) {
        out.top_slow_nodes.resize(kMatmulTopShapeCount);
    }
    return out;
}

Gemma4PrefillAttentionTimingBreakdown SummarizeGemma4PrefillAttentionNodeTimes(const TransformerModel* model,
                                                                               const ggml_cgraph* graph,
                                                                               bool collect_top_slow_ops) {
    Gemma4PrefillAttentionTimingBreakdown out;
    if (!model || !model->arch_flags.is_gemma4 || !graph) {
        return out;
    }
    const int n_nodes = ggml_graph_n_nodes(const_cast<ggml_cgraph*>(graph));
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor* node = ggml_graph_node(const_cast<ggml_cgraph*>(graph), i);
        if (!node) {
            continue;
        }
        const int64_t elapsed_us = ggml_cpu_get_last_node_perf_time_us(node);
        if (elapsed_us <= 0) {
            continue;
        }
        const uint64_t elapsed_ns = static_cast<uint64_t>(elapsed_us) * 1000ULL;
        const std::string custom_class = Gemma4CustomNodeClass(node);
        if (custom_class == "flash_attention_hal") {
            out.attention_ns += elapsed_ns;
            out.portable_flash_ns += elapsed_ns;
            out.hal_ns += elapsed_ns;
        }
        if (node->op == GGML_OP_MUL_MAT_ID) {
            out.mul_mat_id_ns += elapsed_ns;
        } else if (node->op == GGML_OP_MUL_MAT) {
            out.mul_mat_ns += elapsed_ns;
        }
        const char* node_name = node->name[0] ? node->name : "";
        if (std::strstr(node_name, "moe") || std::strstr(node_name, "ffn") || std::strstr(node_name, "shared_ffn")) {
            out.moe_or_mlp_ns += elapsed_ns;
        }
        const bool include_node = collect_top_slow_ops &&
            (node->op == GGML_OP_MUL_MAT_ID || node->op == GGML_OP_MUL_MAT ||
             custom_class == "flash_attention_hal");
        if (include_node) {
            const ggml_tensor* weight = node->src[0];
            const ggml_tensor* input = node->src[1];
            const char* weight_name = weight && weight->name[0] ? weight->name : "<src0>";
            const char* input_name = input && input->name[0] ? input->name : "<src1>";
            MatmulShapeCensusEntry entry;
            entry.phase = "prefill";
            if (node->op == GGML_OP_MUL_MAT_ID) {
                entry.op_type = "MUL_MAT_ID";
                entry.dispatch_path = "ggml_mul_mat_id";
            } else if (node->op == GGML_OP_MUL_MAT) {
                entry.op_type = "MUL_MAT";
                entry.dispatch_path = "ggml_mul_mat";
            } else {
                entry.op_type = "CUSTOM";
                entry.dispatch_path = custom_class.empty() ? "custom" : custom_class;
            }
            entry.weight_type = weight ? Gemma4CensusWeightType(weight->type) : "other";
            entry.weight_class = weight ? Gemma4WeightClass(weight_name) : "gemma4_other_dense";
            entry.shape_bucket = Gemma4MatmulShapeBucket(node, weight);
            entry.left_name = node_name[0] ? node_name : weight_name;
            entry.right_name = std::string("src0=") + weight_name + ",src1=" + input_name;
            entry.wall_ns = elapsed_ns;
            entry.ops = static_cast<uint64_t>(elapsed_us);
            entry.calls = 1;
            entry.active_threads = 0;
            entry.contiguous_or_copy_input = Gemma4NodeHasCopyLikeInput(node) ? 1 : 0;
            out.top_slow_ops.push_back(std::move(entry));
        }
    }
    if (collect_top_slow_ops) {
        std::sort(out.top_slow_ops.begin(), out.top_slow_ops.end(), [](const auto& a, const auto& b) {
            if (a.wall_ns != b.wall_ns) {
                return a.wall_ns > b.wall_ns;
            }
            return a.left_name < b.left_name;
        });
        constexpr std::size_t kGemma4PrefillTopSlowCount = 200;
        if (out.top_slow_ops.size() > kGemma4PrefillTopSlowCount) {
            out.top_slow_ops.resize(kGemma4PrefillTopSlowCount);
        }
    }
    return out;
}

HybridSSMGraphTimingBreakdown SummarizeHybridSSMGraphNodeTimes(const TransformerModel* model,
                                                               const ggml_cgraph* graph) {
    HybridSSMGraphTimingBreakdown out;
    if (!model || !model->arch_flags.is_hybrid_ssm || !graph) {
        return out;
    }
    const int n_nodes = ggml_graph_n_nodes(const_cast<ggml_cgraph*>(graph));
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor* node = ggml_graph_node(const_cast<ggml_cgraph*>(graph), i);
        if (!node) {
            continue;
        }
        const int64_t elapsed_us = ggml_cpu_get_last_node_perf_time_us(node);
        if (elapsed_us <= 0) {
            continue;
        }
        const uint64_t elapsed_ns = static_cast<uint64_t>(elapsed_us) * 1000ULL;
        if (HasNamePrefix(node, "qwen35_ssm_qkv_gate_fused_proj") ||
            HasNamePrefix(node, "qwen36_ssm_qkv_gate_fused_proj") ||
            HasNamePrefix(node, "qwen35_ssm_qkv_proj") || HasNamePrefix(node, "qwen36_ssm_qkv_proj")) {
            out.qkv_ns += elapsed_ns;
        } else if (HasNamePrefix(node, "qwen35_ssm_gate_proj") || HasNamePrefix(node, "qwen36_ssm_gate_proj")) {
            out.gate_ns += elapsed_ns;
        } else if (HasNamePrefix(node, "qwen35_ssm_out_proj") || HasNamePrefix(node, "qwen36_ssm_out_proj")) {
            out.out_ns += elapsed_ns;
        } else if (HasNamePrefix(node, "qwen35_ssm_conv1d") || HasNamePrefix(node, "qwen36_ssm_conv1d")) {
            out.conv1d_ns += elapsed_ns;
        } else if (HasNamePrefix(node, "qwen35_ssm_delta") || HasNamePrefix(node, "qwen36_ssm_delta")) {
            out.delta_ns += elapsed_ns;
        } else if (HasNamePrefix(node, "qwen35_ssm_alpha_beta_qk") || HasNamePrefix(node, "qwen36_ssm_alpha_beta_qk")) {
            out.alpha_beta_qk_ns += elapsed_ns;
        }
    }
    out.total_ns = out.qkv_ns + out.gate_ns + out.out_ns + out.conv1d_ns + out.delta_ns + out.alpha_beta_qk_ns;
    return out;
}

bool IsQwenHybridSSMModel(const TransformerModel* model);

Qwen36PrefillBreakdown SummarizeQwen36PrefillNodeTimes(const TransformerModel* model, const ggml_cgraph* graph) {
    Qwen36PrefillBreakdown out;
    if (!IsQwenHybridSSMModel(model) || !graph) {
        return out;
    }
    if (model->variant == ModelVariant::QWEN35 && model->hparams.n_experts > 0) {
        return out;
    }
    auto safe_op_name = [](enum ggml_op op) -> const char* {
        const int value = static_cast<int>(op);
        return value >= 0 && value < static_cast<int>(GGML_OP_COUNT) ? ggml_op_name(op) : "<invalid-op>";
    };
    const int n_nodes = ggml_graph_n_nodes(const_cast<ggml_cgraph*>(graph));
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor* node = ggml_graph_node(const_cast<ggml_cgraph*>(graph), i);
        if (!node) {
            continue;
        }
        const int64_t elapsed_us = ggml_cpu_get_last_node_perf_time_us(node);
        if (elapsed_us <= 0) {
            continue;
        }
        const uint64_t elapsed_ns = static_cast<uint64_t>(elapsed_us) * 1000ULL;
        const char* name = node->name[0] ? node->name : "unnamed";
        const bool is_ssm_proj = HasNamePrefix(node, "qwen35_ssm_qkv_gate_fused_proj") ||
                                 HasNamePrefix(node, "qwen36_ssm_qkv_gate_fused_proj") ||
                                 HasNamePrefix(node, "qwen35_ssm_qkv_proj") ||
                                 HasNamePrefix(node, "qwen35_ssm_gate_proj") ||
                                 HasNamePrefix(node, "qwen35_ssm_out_proj") ||
                                 HasNamePrefix(node, "qwen36_ssm_qkv_proj") ||
                                 HasNamePrefix(node, "qwen36_ssm_gate_proj") ||
                                 HasNamePrefix(node, "qwen36_ssm_out_proj");
        if (is_ssm_proj) {
            out.ssm_projection_ns += elapsed_ns;
        } else if (std::strstr(name, "delta") || std::strstr(name, "ssm_scan") || std::strstr(name, "conv1d")) {
            out.ssm_delta_state_ns += elapsed_ns;
        } else if (std::strstr(name, "attn") || std::strstr(name, "flash") || std::strstr(name, "kv_")) {
            out.attention_ns += elapsed_ns;
        } else if (std::strstr(name, "moe") || std::strstr(name, "ffn") || std::strstr(name, "mlp")) {
            out.mlp_or_moe_ns += elapsed_ns;
        }
        MatmulShapeCensusEntry entry;
        entry.phase = "prefill";
        const char* op_name = safe_op_name(node->op);
        entry.op_type = op_name;
        entry.dispatch_path =
            is_ssm_proj ? "ssm_projection" : (std::strstr(name, "moe") ? "mlp_or_moe" : op_name);
        const ggml_tensor* src0 = node->src[0];
        const ggml_tensor* src1 = node->src[1];
        entry.weight_type = src0 ? ggml_type_name(src0->type) : "node";
        std::ostringstream shape;
        shape << "M=" << node->ne[1] << ",N=" << node->ne[0] << ",K=" << (node->ne[2] > 1 ? node->ne[2] : 0);
        entry.shape_bucket = shape.str();
        entry.left_name = name;
        const char* src0_name = src0 && src0->name[0] ? src0->name : "<src0>";
        const char* src1_name = src1 && src1->name[0] ? src1->name : "<src1>";
        entry.right_name = std::string("src0=") + src0_name + ",src1=" + src1_name;
        entry.ops = static_cast<uint64_t>(elapsed_us);
        out.top_slow_ops.push_back(std::move(entry));
    }
    std::sort(out.top_slow_ops.begin(), out.top_slow_ops.end(),
              [](const auto& a, const auto& b) { return a.ops != b.ops ? a.ops > b.ops : a.left_name < b.left_name; });
    if (out.top_slow_ops.size() > kMatmulTopShapeCount) {
        out.top_slow_ops.resize(kMatmulTopShapeCount);
    }
    return out;
}

NativeMoEGraphTimingBreakdown SummarizeNativeQwenMoEGraphNodeTimes(const TransformerModel* model,
                                                                   const ggml_cgraph* graph) {
    NativeMoEGraphTimingBreakdown out;
    const bool native_qwen_moe = model && (model->variant == ModelVariant::QWEN35 || model->variant == ModelVariant::QWEN36);
    const bool native_lfm2_moe = model && model->variant == ModelVariant::LFM2MOE && model->arch_flags.is_lfm2_shortconv;
    if ((!native_qwen_moe && !native_lfm2_moe) || !graph) {
        return out;
    }
    struct Bucket {
        const char* name = "other";
        uint64_t ns = 0;
        uint64_t count = 0;
    };
    std::array<Bucket, 6> buckets = {
        {{"route", 0, 0}, {"w1w3", 0, 0}, {"activation", 0, 0}, {"w2", 0, 0}, {"reduce", 0, 0}, {"other", 0, 0}}};
    enum BucketIndex { Route = 0, W1W3 = 1, Activation = 2, W2 = 3, Reduce = 4, Other = 5 };
    const int n_nodes = ggml_graph_n_nodes(const_cast<ggml_cgraph*>(graph));
    bool has_native_qwen_moe = false;
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor* node = ggml_graph_node(const_cast<ggml_cgraph*>(graph), i);
        if (node && node->name[0] &&
            (std::strstr(node->name, "qwen35_native_moe") || std::strstr(node->name, "lfm2_native_moe"))) {
            has_native_qwen_moe = true;
            break;
        }
    }
    if (!has_native_qwen_moe) {
        return out;
    }
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor* node = ggml_graph_node(const_cast<ggml_cgraph*>(graph), i);
        if (!node) {
            continue;
        }
        const char* name = node->name[0] ? node->name : "unnamed";
        const bool native_moe_node = std::strstr(name, "qwen35_native_moe") ||
                                     std::strstr(name, "lfm2_native_moe") ||
                                     std::strstr(name, ".moe_gate_logits");
        if (!native_moe_node) {
            continue;
        }
        ++out.native_node_count;
        const int64_t elapsed_us = ggml_cpu_get_last_node_perf_time_us(node);
        if (elapsed_us <= 0) {
            continue;
        }
        const uint64_t elapsed_ns = static_cast<uint64_t>(elapsed_us) * 1000ULL;
        BucketIndex bucket = Other;
        if (std::strstr(name, ".moe_gate_logits") || std::strstr(name, "_topk") || std::strstr(name, "_norm_weights") ||
            std::strstr(name, "_probs") || std::strstr(name, "_weights") || std::strstr(name, "_scaled_weights")) {
            bucket = Route;
        } else if (std::strstr(name, "_gate_up") || std::strstr(name, "_gate") || std::strstr(name, "_up")) {
            bucket = W1W3;
        } else if (std::strstr(name, "_swiglu")) {
            bucket = Activation;
        } else if (std::strstr(name, "_down")) {
            bucket = W2;
        } else if (std::strstr(name, "_expert_sum") || std::strstr(name, "_moe_out")) {
            bucket = Reduce;
        }
        const bool fast_w1w3_node = bucket == W1W3 &&
                                     (std::strstr(name, "_gateup_raw_q4k_swiglu") ||
                                      std::strstr(name, "_gateup_raw_qxk_swiglu"));
        const bool fast_w2_node = bucket == W2 && std::strstr(name, "_down_q5k_fast");
        buckets[static_cast<std::size_t>(bucket)].ns += elapsed_ns;
        buckets[static_cast<std::size_t>(bucket)].count += 1;
        if (fast_w1w3_node) {
            out.w1w3_fast_ns += elapsed_ns;
            out.w1w3_fast_count += 1;
        }
        if (fast_w2_node) {
            out.w2_fast_ns += elapsed_ns;
            out.w2_fast_count += 1;
        }
        out.total_ns += elapsed_ns;

        MatmulShapeCensusEntry entry;
        entry.phase = native_lfm2_moe ? "lfm2" : (model->variant == ModelVariant::QWEN36 ? "qwen36" : "qwen35");
        entry.op_type = ggml_op_name(node->op);
        entry.dispatch_path = buckets[static_cast<std::size_t>(bucket)].name;
        entry.weight_type = ggml_op_name(node->op);
        std::ostringstream shape;
        shape << "M=" << node->ne[1] << ",N=" << node->ne[0] << ",K=" << (node->ne[2] > 1 ? node->ne[2] : 0);
        entry.shape_bucket = shape.str();
        entry.left_name = name;
        entry.right_name = "elapsed_us";
        entry.ops = static_cast<uint64_t>(elapsed_us);
        out.top_slow_nodes.push_back(std::move(entry));
    }
    out.route_ns = buckets[Route].ns;
    out.w1w3_ns = buckets[W1W3].ns;
    out.activation_ns = buckets[Activation].ns;
    out.w2_ns = buckets[W2].ns;
    out.reduce_ns = buckets[Reduce].ns;
    out.route_count = buckets[Route].count;
    out.w1w3_count = buckets[W1W3].count;
    out.activation_count = buckets[Activation].count;
    out.w2_count = buckets[W2].count;
    out.reduce_count = buckets[Reduce].count;
    std::ostringstream hist;
    for (std::size_t i = 0; i < buckets.size(); ++i) {
        if (i != 0) hist << ",";
        hist << buckets[i].name << ":count=" << buckets[i].count
             << ":ms=" << (static_cast<double>(buckets[i].ns) / 1.0e6);
    }
    out.node_hist = hist.str();
    std::sort(out.top_slow_nodes.begin(), out.top_slow_nodes.end(),
              [](const auto& a, const auto& b) { return a.ops != b.ops ? a.ops > b.ops : a.left_name < b.left_name; });
    if (out.top_slow_nodes.size() > kMatmulTopShapeCount) {
        out.top_slow_nodes.resize(kMatmulTopShapeCount);
    }
    return out;
}

void SynthesizeDecodeGraphNodeTimingFromProfiles(DecodeGraphNodeTimingBreakdown* timing, uint64_t graph_execute_ns,
                                                 const NativeMoEGraphTimingBreakdown& native_moe,
                                                 const HybridSSMGraphTimingBreakdown& hybrid_ssm,
                                                 const Qwen36ProfileSnapshot& profile) {
    if (!timing || graph_execute_ns == 0 || timing->measured_ns != 0) {
        return;
    }
    uint64_t ssm_ns = hybrid_ssm.total_ns;
    if (ssm_ns == 0) {
        ssm_ns = profile.ssm_qkv_wall_ns + profile.ssm_out_wall_ns + profile.ssm_delta_wall_ns;
    }
    uint64_t custom_ns = native_moe.total_ns + ssm_ns;
    const uint64_t attention_ns = profile.attention_ns;
    if (custom_ns > graph_execute_ns) {
        custom_ns = graph_execute_ns;
    }
    const uint64_t after_custom = graph_execute_ns - custom_ns;
    const uint64_t bounded_attention_ns = std::min(attention_ns, after_custom);
    const uint64_t other_ns = graph_execute_ns - custom_ns - bounded_attention_ns;

    timing->custom_ns = custom_ns;
    timing->custom_moe_ns = native_moe.total_ns;
    timing->custom_ssm_ns = ssm_ns;
    timing->custom_lm_head_ns = profile.lfm2_decode_lm_head_custom_gemv_ns;
    timing->custom_other_ns = custom_ns > timing->custom_moe_ns + timing->custom_ssm_ns +
                                              timing->custom_projection_ns + timing->custom_lm_head_ns
                                  ? custom_ns - timing->custom_moe_ns - timing->custom_ssm_ns -
                                        timing->custom_projection_ns - timing->custom_lm_head_ns
                                  : 0;
    timing->attention_ns = bounded_attention_ns;
    timing->other_ns = other_ns;
    timing->measured_ns = graph_execute_ns;

    if (timing->custom_count == 0) {
        timing->custom_count = native_moe.native_node_count + static_cast<uint64_t>(profile.ssm_conv1d_calls) +
                               static_cast<uint64_t>(profile.ssm_delta_calls);
    }
    if (timing->custom_moe_count == 0 && native_moe.native_node_count > 0) {
        timing->custom_moe_count = native_moe.native_node_count;
    }
    if (timing->custom_ssm_count == 0) {
        const uint64_t profile_ssm_count =
            static_cast<uint64_t>(profile.ssm_conv1d_calls + profile.ssm_delta_calls);
        timing->custom_ssm_count = profile_ssm_count > 0 ? profile_ssm_count : (ssm_ns > 0 ? 1 : 0);
    }
    if (timing->custom_lm_head_count == 0 && profile.lfm2_decode_lm_head_custom_gemv_used_ops > 0) {
        timing->custom_lm_head_count = profile.lfm2_decode_lm_head_custom_gemv_used_ops;
    }
    if (timing->custom_other_count == 0 && timing->custom_other_ns > 0) {
        timing->custom_other_count = 1;
    }
    if (timing->attention_count == 0 && bounded_attention_ns > 0) {
        timing->attention_count = 1;
    }
    if (timing->other_count == 0 && other_ns > 0) {
        timing->other_count = 1;
    }
}

bool IsGemma4NodeTimingDumpEnabled() {
    static const bool enabled = densecore::env::ParseNonZeroEnv("DENSECORE_DEBUG_GEMMA4_NODE_TIMES", false);
    return enabled;
}

bool IsLLMNodeTimingDumpEnabled() {
    static const bool enabled = densecore::env::ParseNonZeroEnv("DENSECORE_DEBUG_LLM_NODE_TIMES", false);
    return enabled;
}

int Gemma4NodeTimingDumpLimit() {
    return densecore::env::ParsePositiveEnvInt("DENSECORE_DEBUG_GEMMA4_NODE_TIMES_LIMIT", 24);
}

int LLMNodeTimingDumpLimit() {
    return densecore::env::ParsePositiveEnvInt("DENSECORE_DEBUG_LLM_NODE_TIMES_LIMIT", 24);
}

