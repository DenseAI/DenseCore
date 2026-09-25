// GGML graph node timing classification and debug dumps.
bool Gemma4NodeHasCopyLikeInput(const ggml_tensor* node) {
    if (!node || !node->src[1]) {
        return false;
    }
    const ggml_tensor* input = node->src[1];
    switch (input->op) {
    case GGML_OP_CONT:
    case GGML_OP_CPY:
    case GGML_OP_DUP:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE: return true;
    default: return false;
    }
}

std::string Gemma4MatmulShapeBucket(const ggml_tensor* node, const ggml_tensor* weight) {
    const int64_t out_rows = node ? node->ne[0] : 0;
    const int64_t tokens = node ? std::max<int64_t>(node->ne[1], node->ne[2]) : 0;
    const int64_t k = weight ? weight->ne[0] : 0;
    const int64_t experts = weight && weight->ne[2] > 1 ? weight->ne[2] : 0;
    const int64_t active_experts = (node && node->op == GGML_OP_MUL_MAT_ID && node->src[2]) ? node->src[2]->ne[0] : 0;
    std::ostringstream shape;
    shape << "M=" << tokens << ",N=" << out_rows << ",K=" << k << ",tokens=" << tokens << ",experts=" << experts
          << ",active_experts=" << active_experts << ",copy_in=" << (Gemma4NodeHasCopyLikeInput(node) ? 1 : 0);
    return shape.str();
}

std::string Gemma4CustomNodeClass(const ggml_tensor* node) {
    if (!node || node->op != GGML_OP_CUSTOM) {
        return {};
    }
    struct CustomOpParamsRawView {
        std::uintptr_t fun;
        int n_tasks;
        void* userdata;
    };
    static_assert(sizeof(CustomOpParamsRawView) <= GGML_MAX_OP_PARAMS, "custom op params view too large");
    CustomOpParamsRawView params{};
    std::memcpy(&params, node->op_params, sizeof(params));
    auto same_fun = [&](auto* fun) { return params.fun == reinterpret_cast<std::uintptr_t>(fun); };
    if (same_fun(cb_gemv_custom) || same_fun(cb_gemv_batched_custom)) {
        const ggml_tensor* weight = node->src[1];
        const char* weight_name = weight && weight->name[0] ? weight->name : "<unnamed_weight>";
        std::string label = same_fun(cb_gemv_custom) ? "gemv/" : "gemv_batched/";
        label += Gemma4WeightClass(weight_name);
        label += "/";
        label += weight_name;
        return label;
    }
    if (same_fun(cb_paged_attention_decode)) {
        return "paged_attention_decode";
    }
    if (same_fun(cb_flash_attention_hal_custom)) {
        return "flash_attention_hal";
    }
    if (same_fun(cb_kv_update_and_gather_custom)) {
        return "kv_update_and_gather";
    }
    if (same_fun(cb_rope_precomputed_custom)) {
        return "rope_precomputed";
    }
    if (same_fun(cb_matmul_hal_custom)) {
        return "matmul_hal_custom";
    }
    if (same_fun(cb_matmul_int4_custom)) {
        return "matmul_int4_custom";
    }
    if (same_fun(cb_matmul_fp8_custom)) {
        return "matmul_fp8_custom";
    }
    return "custom_other";
}

void DebugDumpGemma4NodeTimes(const TransformerModel* model, const ggml_cgraph* graph, const char* stage) {
    if (!model || !model->arch_flags.is_gemma4 || !graph || !IsGemma4NodeTimingDumpEnabled()) {
        return;
    }
    struct Entry {
        std::string key;
        uint64_t total_us = 0;
        int count = 0;
    };
    std::unordered_map<std::string, Entry> by_key;
    std::unordered_map<std::string, Entry> by_op;
    uint64_t measured_us = 0;
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
        char fallback_name[256];
        const char* name = node->name[0] ? node->name : nullptr;
        if (!name || std::strncmp(name, "node_", 5) == 0) {
            const char* src0_name = (node->src[0] && node->src[0]->name[0]) ? node->src[0]->name : "<src0>";
            const char* src1_name = (node->src[1] && node->src[1]->name[0]) ? node->src[1]->name : "<src1>";
            std::snprintf(fallback_name, sizeof(fallback_name), "node_%d/src0=%s/src1=%s", i, src0_name, src1_name);
            name = fallback_name;
        }
        const char* op = ggml_op_name(node->op);
        std::string op_key = op ? op : "<op>";
        const std::string custom_class = Gemma4CustomNodeClass(node);
        if (!custom_class.empty()) {
            op_key += "/";
            op_key += custom_class;
        }
        std::string key = op_key + ":" + name;
        Entry& entry = by_key[key];
        entry.key = std::move(key);
        entry.total_us += static_cast<uint64_t>(elapsed_us);
        entry.count += 1;
        Entry& op_entry = by_op[op_key];
        op_entry.key = std::move(op_key);
        op_entry.total_us += static_cast<uint64_t>(elapsed_us);
        op_entry.count += 1;
        measured_us += static_cast<uint64_t>(elapsed_us);
    }
    auto make_sorted_entries = [](std::unordered_map<std::string, Entry>& values) {
        std::vector<Entry> entries;
        entries.reserve(values.size());
        for (auto& kv : values) {
            entries.push_back(std::move(kv.second));
        }
        std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
            if (a.total_us != b.total_us) {
                return a.total_us > b.total_us;
            }
            return a.key < b.key;
        });
        return entries;
    };
    std::vector<Entry> entries = make_sorted_entries(by_key);
    std::vector<Entry> op_entries = make_sorted_entries(by_op);
    const int op_limit = std::min<int>(8, op_entries.size());
    std::cerr << "[Gemma4NodeTimesByOp] stage=" << (stage ? stage : "<unknown>") << " showing=" << op_limit;
    for (int i = 0; i < op_limit; ++i) {
        const Entry& entry = op_entries[static_cast<size_t>(i)];
        std::cerr << " op" << (i + 1) << "=" << entry.key << ":" << (static_cast<double>(entry.total_us) / 1000.0)
                  << "ms/" << entry.count;
    }
    std::cerr << std::endl;
    const int limit = std::min<int>(Gemma4NodeTimingDumpLimit(), entries.size());
    std::cerr << "[Gemma4NodeTimes] stage=" << (stage ? stage : "<unknown>") << " nodes=" << n_nodes
              << " measured_ms=" << (static_cast<double>(measured_us) / 1000.0) << " showing=" << limit << std::endl;
    for (int i = 0; i < limit; ++i) {
        const Entry& entry = entries[static_cast<size_t>(i)];
        std::cerr << "  rank=" << (i + 1) << " total_ms=" << (static_cast<double>(entry.total_us) / 1000.0)
                  << " count=" << entry.count << " key=" << entry.key << std::endl;
    }
}

void DebugDumpLLMNodeTimes(const TransformerModel* model, const ggml_cgraph* graph, const char* stage) {
    if (!model || !graph || !IsLLMNodeTimingDumpEnabled()) {
        return;
    }
    struct Entry {
        std::string key;
        uint64_t total_us = 0;
        int count = 0;
    };
    std::unordered_map<std::string, Entry> by_key;
    std::unordered_map<std::string, Entry> by_op;
    uint64_t measured_us = 0;
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
        char fallback_name[256];
        const char* name = node->name[0] ? node->name : nullptr;
        if (!name || std::strncmp(name, "node_", 5) == 0) {
            const char* src0_name = (node->src[0] && node->src[0]->name[0]) ? node->src[0]->name : "<src0>";
            const char* src1_name = (node->src[1] && node->src[1]->name[0]) ? node->src[1]->name : "<src1>";
            std::snprintf(fallback_name, sizeof(fallback_name), "node_%d/src0=%s/src1=%s", i, src0_name, src1_name);
            name = fallback_name;
        }
        const char* op = ggml_op_name(node->op);
        std::string op_key = op ? op : "<op>";
        const std::string custom_class = Gemma4CustomNodeClass(node);
        if (!custom_class.empty()) {
            op_key += "/";
            op_key += custom_class;
        }
        std::string key = op_key + ":" + name;
        Entry& entry = by_key[key];
        entry.key = std::move(key);
        entry.total_us += static_cast<uint64_t>(elapsed_us);
        entry.count += 1;
        Entry& op_entry = by_op[op_key];
        op_entry.key = std::move(op_key);
        op_entry.total_us += static_cast<uint64_t>(elapsed_us);
        op_entry.count += 1;
        measured_us += static_cast<uint64_t>(elapsed_us);
    }
    auto make_sorted_entries = [](std::unordered_map<std::string, Entry>& values) {
        std::vector<Entry> entries;
        entries.reserve(values.size());
        for (auto& kv : values) {
            entries.push_back(std::move(kv.second));
        }
        std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
            if (a.total_us != b.total_us) {
                return a.total_us > b.total_us;
            }
            return a.key < b.key;
        });
        return entries;
    };
    std::vector<Entry> entries = make_sorted_entries(by_key);
    std::vector<Entry> op_entries = make_sorted_entries(by_op);
    const int op_limit = std::min<int>(12, op_entries.size());
    std::cerr << "[LLMNodeTimesByOp] stage=" << (stage ? stage : "<unknown>") << " showing=" << op_limit;
    for (int i = 0; i < op_limit; ++i) {
        const Entry& entry = op_entries[static_cast<size_t>(i)];
        std::cerr << " op" << (i + 1) << "=" << entry.key << ":" << (static_cast<double>(entry.total_us) / 1000.0)
                  << "ms/" << entry.count;
    }
    std::cerr << std::endl;
    const int limit = std::min<int>(LLMNodeTimingDumpLimit(), entries.size());
    std::cerr << "[LLMNodeTimes] stage=" << (stage ? stage : "<unknown>") << " nodes=" << n_nodes
              << " measured_ms=" << (static_cast<double>(measured_us) / 1000.0) << " showing=" << limit << std::endl;
    for (int i = 0; i < limit; ++i) {
        const Entry& entry = entries[static_cast<size_t>(i)];
        std::cerr << "  rank=" << (i + 1) << " total_ms=" << (static_cast<double>(entry.total_us) / 1000.0)
                  << " count=" << entry.count << " key=" << entry.key << std::endl;
    }
}
