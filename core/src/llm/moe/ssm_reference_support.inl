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

static bool IsTraceQwen35SSMRuntimeContractEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_SSM_TRACE_RUNTIME_CONTRACT");
        return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static int TraceQwen35SSMMaxHeads() {
    static const int value = []() {
        const char* env = std::getenv("DENSECORE_SSM_TRACE_MAX_HEADS");
        if (!env || env[0] == '\0') {
            return 32;
        }
        char* end = nullptr;
        long parsed = std::strtol(env, &end, 10);
        if (end == env || (end && *end != '\0') || parsed <= 0) {
            return 32;
        }
        return static_cast<int>(parsed);
    }();
    return value;
}

static int TraceQwen35SSMMaxTokens() {
    static const int value = []() {
        const char* env = std::getenv("DENSECORE_SSM_TRACE_MAX_TOKENS");
        if (!env || env[0] == '\0') {
            return 1024;
        }
        char* end = nullptr;
        long parsed = std::strtol(env, &end, 10);
        if (end == env || (end && *end != '\0') || parsed <= 0) {
            return 1024;
        }
        return static_cast<int>(parsed);
    }();
    return value;
}

static uint64_t HashQwen35SSMFloatSpan(const float* data, size_t count) {
    if (!data) {
        return 0;
    }
    uint64_t hash = 1469598103934665603ull;
    hash ^= static_cast<uint64_t>(count);
    hash *= 1099511628211ull;
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, data + i, sizeof(bits));
        hash ^= static_cast<uint64_t>(bits);
        hash *= 1099511628211ull;
    }
    return hash;
}

static void LogQwen35SSMRuntimeContract(const char* mode, int layer_idx, int token_idx, int seq_idx, int head_idx,
                                        int src_k_head, int heads_per_group, uintptr_t token_state_base_ptr,
                                        size_t token_state_elems, size_t state_offset_bytes, size_t state_span_bytes,
                                        ptrdiff_t input_stride, ptrdiff_t output_stride, bool overlap_prev,
                                        bool grouped_src_reused, uint64_t q_hash, uint64_t k_hash, uint64_t v_hash,
                                        uint64_t state_in_hash, uint64_t state_out_hash, uint64_t y_hash) {
    if (!IsTraceQwen35SSMRuntimeContractEnabled()) {
        return;
    }
    std::fprintf(stderr,
                 "[SSM_RUNTIME_CONTRACT] mode=%s layer=%d token=%d seq=%d head=%d src_k_head=%d heads_per_group=%d "
                 "base_ptr=0x%llx total_elems=%zu state_off_bytes=%zu state_span_bytes=%zu input_stride=%lld "
                 "output_stride=%lld overlap_prev=%d grouped_src_reused=%d q_hash=0x%llx k_hash=0x%llx v_hash=0x%llx "
                 "state_in=0x%llx state_out=0x%llx y_hash=0x%llx\n",
                 mode ? mode : "unknown", layer_idx, token_idx, seq_idx, head_idx, src_k_head, heads_per_group,
                 static_cast<unsigned long long>(token_state_base_ptr), token_state_elems, state_offset_bytes,
                 state_span_bytes, static_cast<long long>(input_stride), static_cast<long long>(output_stride),
                 overlap_prev ? 1 : 0, grouped_src_reused ? 1 : 0, static_cast<unsigned long long>(q_hash),
                 static_cast<unsigned long long>(k_hash), static_cast<unsigned long long>(v_hash),
                 static_cast<unsigned long long>(state_in_hash), static_cast<unsigned long long>(state_out_hash),
                 static_cast<unsigned long long>(y_hash));
}

static void LogQwen35SSMLayerAggregate(const char* mode, int layer_idx, int token_idx, int seq_idx,
                                       uint64_t aggregate_hash) {
    if (!IsTraceQwen35SSMRuntimeContractEnabled()) {
        return;
    }
    std::fprintf(stderr, "[SSM_LAYER_AGG] mode=%s layer=%d token=%d seq=%d aggregate_state_hash=0x%llx\n",
                 mode ? mode : "unknown", layer_idx, token_idx, seq_idx,
                 static_cast<unsigned long long>(aggregate_hash));
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
    const float ssm_a = (cfg.a_log_prescaled && cfg.a_log <= 0.0f) ? cfg.a_log : -std::exp(cfg.a_log);
    const float exp_a_log = -ssm_a;
    const float g = ssm_a * softplus_alpha;
    const float decay = std::exp(g);
    const float beta_gate = SigmoidStable(beta);

    float q_sum_sq = 0.0f;
    float k_sum_sq = 0.0f;
    for (int i = 0; i < cfg.head_dim_k; ++i) {
        q_sum_sq += cfg.q_head[i] * cfg.q_head[i];
        k_sum_sq += cfg.k_head[i] * cfg.k_head[i];
    }
    const float q_inv_norm = 1.0f / std::max(std::sqrt(q_sum_sq), cfg.norm_eps);
    const float k_inv_norm = 1.0f / std::max(std::sqrt(k_sum_sq), cfg.norm_eps);
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
        y_head[v] = sum / std::sqrt(static_cast<float>(cfg.head_dim_k));
        sum_sq += y_head[v] * y_head[v];
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
