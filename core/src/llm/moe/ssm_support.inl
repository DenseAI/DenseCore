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

static inline void DotPairProducts(const float* lhs0, const float* lhs1, const float* rhs, int n, float* out0,
                                   float* out1) {
    float sum0 = 0.0f;
    float sum1 = 0.0f;
#if defined(__AVX512F__)
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 x = _mm512_loadu_ps(rhs + i);
        acc0 = _mm512_add_ps(acc0, _mm512_mul_ps(_mm512_loadu_ps(lhs0 + i), x));
        acc1 = _mm512_add_ps(acc1, _mm512_mul_ps(_mm512_loadu_ps(lhs1 + i), x));
    }
    sum0 = _mm512_reduce_add_ps(acc0);
    sum1 = _mm512_reduce_add_ps(acc1);
    for (; i < n; ++i) {
        const float x = rhs[i];
        sum0 += lhs0[i] * x;
        sum1 += lhs1[i] * x;
    }
#elif defined(__AVX2__)
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 x = _mm256_loadu_ps(rhs + i);
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_loadu_ps(lhs0 + i), x));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_loadu_ps(lhs1 + i), x));
    }
    alignas(32) float lanes0[8];
    alignas(32) float lanes1[8];
    _mm256_store_ps(lanes0, acc0);
    _mm256_store_ps(lanes1, acc1);
    for (int lane = 0; lane < 8; ++lane) {
        sum0 += lanes0[lane];
        sum1 += lanes1[lane];
    }
    for (; i < n; ++i) {
        const float x = rhs[i];
        sum0 += lhs0[i] * x;
        sum1 += lhs1[i] * x;
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t x = vld1q_f32(rhs + i);
        acc0 = vfmaq_f32(acc0, vld1q_f32(lhs0 + i), x);
        acc1 = vfmaq_f32(acc1, vld1q_f32(lhs1 + i), x);
    }
    sum0 = vaddvq_f32(acc0);
    sum1 = vaddvq_f32(acc1);
    for (; i < n; ++i) {
        const float x = rhs[i];
        sum0 += lhs0[i] * x;
        sum1 += lhs1[i] * x;
    }
#else
    for (int i = 0; i < n; ++i) {
        const float x = rhs[i];
        sum0 += lhs0[i] * x;
        sum1 += lhs1[i] * x;
    }
#endif
    *out0 = sum0;
    *out1 = sum1;
}

static inline void QKNormProducts(const float* q_head, const float* k_head, int n, float* q_sum_sq, float* k_sum_sq,
                                  float* qk_raw_dot) {
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    float qk_sum = 0.0f;
#if defined(__AVX512F__)
    __m512 q_acc = _mm512_setzero_ps();
    __m512 k_acc = _mm512_setzero_ps();
    __m512 qk_acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m512 qv = _mm512_loadu_ps(q_head + i);
        const __m512 kv = _mm512_loadu_ps(k_head + i);
        q_acc = _mm512_add_ps(q_acc, _mm512_mul_ps(qv, qv));
        k_acc = _mm512_add_ps(k_acc, _mm512_mul_ps(kv, kv));
        qk_acc = _mm512_add_ps(qk_acc, _mm512_mul_ps(qv, kv));
    }
    q_sum = _mm512_reduce_add_ps(q_acc);
    k_sum = _mm512_reduce_add_ps(k_acc);
    qk_sum = _mm512_reduce_add_ps(qk_acc);
    for (; i < n; ++i) {
        q_sum += q_head[i] * q_head[i];
        k_sum += k_head[i] * k_head[i];
        qk_sum += q_head[i] * k_head[i];
    }
#elif defined(__AVX2__)
    __m256 q_acc = _mm256_setzero_ps();
    __m256 k_acc = _mm256_setzero_ps();
    __m256 qk_acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 qv = _mm256_loadu_ps(q_head + i);
        const __m256 kv = _mm256_loadu_ps(k_head + i);
        q_acc = _mm256_add_ps(q_acc, _mm256_mul_ps(qv, qv));
        k_acc = _mm256_add_ps(k_acc, _mm256_mul_ps(kv, kv));
        qk_acc = _mm256_add_ps(qk_acc, _mm256_mul_ps(qv, kv));
    }
    alignas(32) float q_lanes[8];
    alignas(32) float k_lanes[8];
    alignas(32) float qk_lanes[8];
    _mm256_store_ps(q_lanes, q_acc);
    _mm256_store_ps(k_lanes, k_acc);
    _mm256_store_ps(qk_lanes, qk_acc);
    for (int lane = 0; lane < 8; ++lane) {
        q_sum += q_lanes[lane];
        k_sum += k_lanes[lane];
        qk_sum += qk_lanes[lane];
    }
    for (; i < n; ++i) {
        q_sum += q_head[i] * q_head[i];
        k_sum += k_head[i] * k_head[i];
        qk_sum += q_head[i] * k_head[i];
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
    float32x4_t q_acc = vdupq_n_f32(0.0f);
    float32x4_t k_acc = vdupq_n_f32(0.0f);
    float32x4_t qk_acc = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        const float32x4_t qv = vld1q_f32(q_head + i);
        const float32x4_t kv = vld1q_f32(k_head + i);
        q_acc = vfmaq_f32(q_acc, qv, qv);
        k_acc = vfmaq_f32(k_acc, kv, kv);
        qk_acc = vfmaq_f32(qk_acc, qv, kv);
    }
    q_sum = vaddvq_f32(q_acc);
    k_sum = vaddvq_f32(k_acc);
    qk_sum = vaddvq_f32(qk_acc);
    for (; i < n; ++i) {
        q_sum += q_head[i] * q_head[i];
        k_sum += k_head[i] * k_head[i];
        qk_sum += q_head[i] * k_head[i];
    }
#else
    for (int i = 0; i < n; ++i) {
        q_sum += q_head[i] * q_head[i];
        k_sum += k_head[i] * k_head[i];
        qk_sum += q_head[i] * k_head[i];
    }
#endif
    *q_sum_sq = q_sum;
    *k_sum_sq = k_sum;
    *qk_raw_dot = qk_sum;
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

static bool IsDebugSSMDeltaStatsEnabled() {
    static const bool enabled = ParseTruthyEnv("DENSECORE_DEBUG_SSM_DELTA_STATS", false);
    return enabled;
}

static int DebugSSMDeltaStatsLayer() {
    static const int layer = ParseIntEnv("DENSECORE_DEBUG_SSM_DELTA_STATS_LAYER", -1);
    return layer;
}

static int DebugSSMDeltaStatsToken() {
    static const int token = ParseIntEnv("DENSECORE_DEBUG_SSM_DELTA_STATS_TOKEN", -1);
    return token;
}

static bool ShouldLogSSMDeltaStats(int layer_idx, int token_idx) {
    if (!IsDebugSSMDeltaStatsEnabled()) {
        return false;
    }
    const int target_layer = DebugSSMDeltaStatsLayer();
    if (target_layer >= 0 && target_layer != layer_idx) {
        return false;
    }
    const int target_token = DebugSSMDeltaStatsToken();
    return target_token < 0 || target_token == token_idx;
}

static bool ConsumeSSMDeltaStatsBudget() {
    static std::atomic<int> remaining{ParsePositiveEnvInt("DENSECORE_DEBUG_SSM_DELTA_STATS_MAX_CALLS", 16)};
    int current = remaining.load(std::memory_order_relaxed);
    while (current > 0) {
        if (remaining.compare_exchange_weak(current, current - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
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
