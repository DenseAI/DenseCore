namespace {

bool IsDebugSharedExpertShapeEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_VALIDATE_MUL");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

void DebugLogSharedExpertTensor(const char* stage, int layer_idx, const struct ggml_tensor* tensor) {
    if (!IsDebugSharedExpertShapeEnabled() || !tensor) {
        return;
    }
    std::fprintf(stderr,
                 "[SharedExpertShape] layer=%d stage=%s name=%s type=%d ne=[%lld,%lld,%lld,%lld] "
                 "nb=[%lld,%lld,%lld,%lld]\n",
                 layer_idx, stage ? stage : "<unknown>", tensor->name[0] ? tensor->name : "<unnamed>",
                 static_cast<int>(tensor->type), static_cast<long long>(tensor->ne[0]),
                 static_cast<long long>(tensor->ne[1]), static_cast<long long>(tensor->ne[2]),
                 static_cast<long long>(tensor->ne[3]), static_cast<long long>(tensor->nb[0]),
                 static_cast<long long>(tensor->nb[1]), static_cast<long long>(tensor->nb[2]),
                 static_cast<long long>(tensor->nb[3]));
}

bool ShouldRunMoESharedDenseBranch(const TransformerModel* model, const densecore::models::DecoderLayerSpec* layer_spec,
                                   bool is_gemma4_moe, const struct ggml_tensor* ffn_gate,
                                   const struct ggml_tensor* ffn_up, const struct ggml_tensor* ffn_down) {
    if (!model || !ffn_gate || !ffn_up || !ffn_down) {
        return false;
    }
    if (layer_spec) {
        return layer_spec->ffn.has_shared_dense_branch;
    }
    if (model->moe_n_shared_experts > 0) {
        return true;
    }
    // Gemma4-26B-A4B carries a regular dense MLP branch alongside the sparse
    // MoE branch. Some GGUF exports do not advertise it via n_shared_experts,
    // so the presence of the shared FFN tensors is the load-bearing signal.
    return is_gemma4_moe;
}

using densecore::env::ParseIntEnv;
using densecore::env::ParsePositiveEnvInt;
using densecore::env::ParseTruthyEnv;
using densecore::env::RuntimeToggleMode;
using densecore::llm::config::DecodePagedAttentionMode;
using densecore::llm::config::DecodePagedAttentionPolicy;
using densecore::llm::config::KVRetentionPolicy;
using densecore::llm::config::KVRetentionSpan;
using densecore::llm::runtime::ResolveFastPathRuntimeConfig;

constexpr const char* kGemma4RouterScaleKey = "gemma4.router.scale";
constexpr const char* kGemma4RouterPerExpertScaleKey = "gemma4.router.per_expert_scale";
constexpr const char* kGemma4PreMoeNormKey = "gemma4.pre_feedforward_layernorm_2.weight";
constexpr const char* kGemma4PostSharedNormKey = "gemma4.post_feedforward_layernorm_1.weight";
constexpr const char* kGemma4PostMoeNormKey = "gemma4.post_feedforward_layernorm_2.weight";
constexpr const char* kGemma4PostFfnNormKey = "gemma4.post_feedforward_layernorm.weight";
constexpr const char* kGemma4PackedGateUpExpertsWeightKey = "ffn_gate_up_exps.weight";
constexpr const char* kGemma4PackedGateUpExpertsKey = "ffn_gate_up_exps";
constexpr const char* kGemma4PackedDownExpertsWeightKey = "ffn_down_exps.weight";
constexpr const char* kGemma4PackedDownExpertsKey = "ffn_down_exps";
constexpr const char* kGemma4PackedDownExpertsScaleKey = "ffn_down_exps.scale";
std::atomic<uint64_t> g_moe_graph_wiring_debug_counter{0};
constexpr int64_t kQwen36C4AmxGraphMaxTokens = 128;
constexpr int64_t kNativeMoEFastPathDefaultMaxDirectTokens = 4096;

bool IsQwen35NativeMoEDownQ5KDiagEnabled() {
    static const bool enabled = densecore::env::ParseTruthyEnv("DENSECORE_NATIVE_MOE_FAST_W2_Q5K_DIAG", false);
    return enabled;
}

bool ModelRequiresNativeMoEFastPath(const TransformerModel* model) {
    if (!model) {
        return false;
    }
    const auto contract = densecore::models::BuildModelExecutionContract(model);
    return densecore::models::ModelExecutionContractRequiresNativeMoEFastPath(contract);
}

int64_t NativeMoEFastPathMaxDirectTokens(const TransformerModel* model) {
    if (!model) {
        return kNativeMoEFastPathDefaultMaxDirectTokens;
    }
    const auto contract = densecore::models::BuildModelExecutionContract(model);
    const int64_t contract_limit = densecore::models::ModelExecutionContractNativeMoEMaxDirectTokens(contract);
    return contract_limit > 0 ? contract_limit : kNativeMoEFastPathDefaultMaxDirectTokens;
}

bool ShouldEnableNativeMoEFastPathByDefault(const TransformerModel* model, InferenceExecutionPhase phase,
                                            densecore::env::RuntimeToggleMode mode) {
    if (phase != InferenceExecutionPhase::Decode && phase != InferenceExecutionPhase::Prefill) {
        return false;
    }
    return ModelRequiresNativeMoEFastPath(model) && mode != densecore::env::RuntimeToggleMode::Off;
}

static int ResolveNativeMoEGraphCallbackTaskCount(const TransformerModel* model, const BatchSpec* batch,
                                                  InferenceExecutionPhase phase, int64_t n_tokens, int top_k);
struct Qwen35SharedQ8RowsUserData;
static Qwen35SharedQ8RowsUserData* AllocateQwen35SharedQ8RowsUserData(ggml_context* ctx, const ggml_tensor* src,
                                                                      int64_t max_assignments);
static void SetNativeMoECallbackRequestedTaskCount(Qwen35SharedQ8RowsUserData* ud, int requested_task_count);
static void cb_gemma4_native_moe_down_weighted_sum(struct ggml_tensor* dst, int ith, int nth, void* userdata);

bool IsDebugLFM2NativeMoEReferenceEnabled() {
    static const bool enabled = densecore::env::ParseTruthyEnv("DENSECORE_DEBUG_LFM2_NATIVE_MOE_REFERENCE", false);
    return enabled;
}

bool IsDebugQwen35NativeMoEReferenceEnabled() {
    static const bool enabled =
        densecore::env::ParseTruthyEnv("DENSECORE_DEBUG_QWEN35_NATIVE_MOE_REFERENCE", false);
    return enabled;
}

inline float NativeMoEFastExp(float x) {
    if (x < -50.0f) x = -50.0f;
    if (x > 50.0f) x = 50.0f;

    constexpr float kLog2E = 1.4426950408889634f;
    const float y = x * kLog2E;
    const int32_t i = static_cast<int32_t>(std::floor(y));
    const float f = y - static_cast<float>(i);
    const float p = 1.0f + f * (0.6960656421638072f + f * (0.224494337302845f + f * 0.07944023841053369f));

    const int32_t exp_bits = (i + 127) << 23;
    float two_i = 0.0f;
    std::memcpy(&two_i, &exp_bits, sizeof(two_i));
    return two_i * p;
}

inline float NativeMoESiLU(float x) {
    return x / (1.0f + NativeMoEFastExp(-x));
}

bool ShouldRunQwen35NativeMoEReferenceProbe(int layer_idx, const char* stage, int64_t token_idx) {
    if (!IsDebugQwen35NativeMoEReferenceEnabled()) {
        return false;
    }
    static const int target_layer = ParseIntEnv("DENSECORE_DEBUG_QWEN35_NATIVE_MOE_REFERENCE_LAYER", -1);
    static const int target_token = ParseIntEnv("DENSECORE_DEBUG_QWEN35_NATIVE_MOE_REFERENCE_TOKEN", -1);
    static const char* target_stage = std::getenv("DENSECORE_DEBUG_QWEN35_NATIVE_MOE_REFERENCE_STAGE");
    static std::atomic<int> remaining_budget{
        ParsePositiveEnvInt("DENSECORE_DEBUG_QWEN35_NATIVE_MOE_REFERENCE_MAX_CALLS", 16)};
    if (target_layer >= 0 && layer_idx != target_layer) {
        return false;
    }
    if (target_token >= 0 && token_idx != target_token) {
        return false;
    }
    if (target_stage && target_stage[0] != '\0' && stage && std::strcmp(target_stage, stage) != 0) {
        return false;
    }
    int remaining = remaining_budget.load(std::memory_order_relaxed);
    while (remaining > 0) {
        if (remaining_budget.compare_exchange_weak(remaining, remaining - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

float Qwen35NativeMoEReferenceTolerance() {
    static const float tol = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_QWEN35_NATIVE_MOE_REFERENCE_TOL");
        if (!env || env[0] == '\0') {
            return 1e-2f;
        }
        char* end = nullptr;
        const float parsed = std::strtof(env, &end);
        if (end == env || !std::isfinite(parsed) || parsed <= 0.0f) {
            return 1e-2f;
        }
        return parsed;
    }();
    return tol;
}

bool Qwen35NativeMoEReferenceDotQXK(ggml_type weight_type, const void* weight_row, const uint8_t* qrow, int64_t cols,
                                   float* out_value) {
    if (!weight_row || !qrow || !out_value || cols <= 0) {
        return false;
    }
    const ggml_type_traits_cpu* traits = ggml_get_type_traits_cpu(weight_type);
    if (!traits || !traits->vec_dot || traits->vec_dot_type != GGML_TYPE_Q8_K ||
        (cols % ggml_blck_size(weight_type)) != 0) {
        return false;
    }
    *out_value = 0.0f;
    traits->vec_dot(static_cast<int>(cols), out_value, 0, weight_row, 0, qrow, 0, 1);
    return std::isfinite(*out_value);
}

bool ShouldUsePrefillLastLogitsOnly(const TransformerModel* model, const BatchSpec& batch, int n_tokens) {
    return densecore::llm::decoder::ShouldUsePrefillLastLogitsOnly(model, batch, n_tokens);
}

enum class MoEWiringReasonCode : int {
    Wired = 0,
    ModelHasNoMoE = 1,
    LayerFlagFalse = 2,
    MissingMoeGate = 3,
    NoExperts = 4,
    DenseReplaceGate = 5,
    LayerFlagMismatch = 6,
};

bool IsMoEWiringDebugEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_MOE_WIRING_DEBUG");
        if (!env || env[0] == '\0') {
            env = std::getenv("DENSECORE_DEBUG_MOE_WIRING");
        }
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool IsDebugGemma4NativeMoEPrefillEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_GEMMA4_NATIVE_MOE_PREFILL");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

bool IsGemma4NativeMoEGraphEnabled() {
    return true;
}

bool Gemma4NativeMoEPrefillKernelSupported() {
    // C4 validation showed both candidate native prefill variants are not
    // promotable yet: full gate/up+down passes QA but regresses prefill, and
    // gate/up-only also regresses the default server path. Keep this fail-closed
    // until the prefill work moves to a batched/AMX-grade implementation.
    // This is deliberately not controlled by a benchmark env knob; once a faster
    // path passes QA and throughput gates it should be promoted as the default.
    return false;
}

bool IsGemma4NativeMoEPrefillEnabled() {
    return Gemma4NativeMoEPrefillKernelSupported();
}

struct Gemma4PackedMoERoots {
    ggml_tensor* gate_up = nullptr;
    ggml_tensor* down = nullptr;
    ggml_tensor* down_scale = nullptr;
    densecore::gemma4::PackedExpertLayout layout{};
};

struct Qwen35SharedQ8RowsUserData;
static Qwen35SharedQ8RowsUserData* AllocateQwen35SharedQ8RowsUserData(ggml_context* ctx, const ggml_tensor* src,
                                                                      int64_t max_assignments = 0);

ggml_tensor* GetLayerTensorAny(TransformerLayer* layer, std::initializer_list<const char*> keys) {
    if (!layer) {
        return nullptr;
    }
    for (const char* key : keys) {
        if (!key) {
            continue;
        }
        if (ggml_tensor* tensor = layer->Get(key)) {
            return tensor;
        }
    }
    return nullptr;
}

bool ResolveGemma4PackedMoERoots(TransformerLayer* layer, Gemma4PackedMoERoots* roots) {
    if (!layer || !roots) {
        return false;
    }
    roots->gate_up = GetLayerTensorAny(layer, {kGemma4PackedGateUpExpertsWeightKey, kGemma4PackedGateUpExpertsKey});
    roots->down = GetLayerTensorAny(layer, {kGemma4PackedDownExpertsWeightKey, kGemma4PackedDownExpertsKey});
    roots->down_scale = GetLayerTensorAny(layer, {kGemma4PackedDownExpertsScaleKey});
    if (!roots->gate_up || !roots->down || !roots->down_scale) {
        return false;
    }
    std::string reason;
    if (!densecore::gemma4::InferPackedExpertLayout(roots->gate_up, roots->down, &roots->layout, &reason)) {
        if (IsMoEWiringDebugEnabled()) {
            std::fprintf(stderr, "[Gemma4NativeMoE] packed root rejected: %s\n", reason.c_str());
        }
        return false;
    }
    return roots->layout.num_experts > 0 && roots->layout.hidden_dim > 0 && roots->layout.intermediate_dim > 0;
}

ggml_tensor* UseCpuRepackAliasIfAvailable(TransformerModel* model, ggml_tensor* tensor) {
    if (!model || !tensor) {
        return tensor;
    }
    auto it = model->cpu_repack_aliases.find(tensor);
    return it == model->cpu_repack_aliases.end() ? tensor : it->second;
}

bool CpuRepackAliasHasLayout(const TransformerModel* model, const ggml_tensor* tensor,
                             TransformerModel::CpuRepackAliasLayout layout) {
    if (!model || !tensor || layout == TransformerModel::CpuRepackAliasLayout::Unknown) {
        return false;
    }
    auto it = model->cpu_repack_alias_layouts.find(tensor);
    if (it != model->cpu_repack_alias_layouts.end()) {
        return it->second == layout;
    }
    const ggml_tensor* view_src = tensor->view_src;
    if (!view_src) {
        return false;
    }
    it = model->cpu_repack_alias_layouts.find(view_src);
    return it != model->cpu_repack_alias_layouts.end() && it->second == layout;
}

ggml_tensor* UseCpuRepackAliasForTokenCount(TransformerModel* model, ggml_tensor* tensor, int64_t n_tokens) {
    if (!model || !tensor) {
        return tensor;
    }
    auto it = model->cpu_repack_aliases.find(tensor);
    if (it == model->cpu_repack_aliases.end() || !it->second) {
        return tensor;
    }
    if (model->variant == ModelVariant::QWEN36 && n_tokens > 0 && n_tokens <= kQwen36C4AmxGraphMaxTokens &&
        model->cpu_amx_aliases.find(it->second) != model->cpu_amx_aliases.end()) {
        return tensor;
    }
    return it->second;
}

void cb_moe_expert_weighted_sum_with_weights(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    (void)userdata;
    if (!dst || !dst->src[0] || !dst->src[1] || !dst->data || !dst->src[0]->data || !dst->src[1]->data || nth <= 0) {
        return;
    }
    const ggml_tensor* experts = dst->src[0];
    const ggml_tensor* weights = dst->src[1];
    if (dst->type != GGML_TYPE_F32 || experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 ||
        experts->ne[0] != dst->ne[0] || experts->ne[2] != dst->ne[1] || experts->ne[1] <= 0 ||
        weights->ne[0] != 1 || weights->ne[1] != experts->ne[1] || weights->ne[2] != experts->ne[2]) {
        return;
    }

    const int64_t n_embd = dst->ne[0];
    const int64_t n_tokens = dst->ne[1];
    const int64_t n_expert_used = experts->ne[1];
    const int64_t total = n_embd * n_tokens;
    const int64_t start = (total * ith) / nth;
    const int64_t end = (total * (ith + 1)) / nth;

    const char* src_base = static_cast<const char*>(experts->data);
    const char* weight_base = static_cast<const char*>(weights->data);
    char* dst_base = static_cast<char*>(dst->data);
    const size_t dense_expert_stride = static_cast<size_t>(n_embd) * sizeof(float);
    const bool dense_strides =
        experts->nb[0] == static_cast<int64_t>(sizeof(float)) &&
        experts->nb[1] == dense_expert_stride &&
        weights->nb[0] == static_cast<int64_t>(sizeof(float)) &&
        dst->nb[0] == static_cast<int64_t>(sizeof(float));
    if (dense_strides) {
        int64_t idx = start;
        while (idx < end) {
            const int64_t tok = idx / n_embd;
            const int64_t embd0 = idx - tok * n_embd;
            const int64_t embd1 = std::min<int64_t>(n_embd, end - tok * n_embd);
            const int64_t count = embd1 - embd0;
            const float w0 =
                *reinterpret_cast<const float*>(weight_base + tok * weights->nb[2]);
            const float safe_w0 = std::isfinite(w0) ? w0 : 0.0f;
            const float* src0 = reinterpret_cast<const float*>(src_base + tok * experts->nb[2] + embd0 * experts->nb[0]);
            float* dst_ptr = reinterpret_cast<float*>(dst_base + tok * dst->nb[1] + embd0 * dst->nb[0]);
            if (n_expert_used == 8) {
                const float* src1 = reinterpret_cast<const float*>(
                    src_base + tok * experts->nb[2] + experts->nb[1] + embd0 * experts->nb[0]);
                const float* src2 = reinterpret_cast<const float*>(
                    src_base + tok * experts->nb[2] + 2 * experts->nb[1] + embd0 * experts->nb[0]);
                const float* src3 = reinterpret_cast<const float*>(
                    src_base + tok * experts->nb[2] + 3 * experts->nb[1] + embd0 * experts->nb[0]);
                const float* src4 = reinterpret_cast<const float*>(
                    src_base + tok * experts->nb[2] + 4 * experts->nb[1] + embd0 * experts->nb[0]);
                const float* src5 = reinterpret_cast<const float*>(
                    src_base + tok * experts->nb[2] + 5 * experts->nb[1] + embd0 * experts->nb[0]);
                const float* src6 = reinterpret_cast<const float*>(
                    src_base + tok * experts->nb[2] + 6 * experts->nb[1] + embd0 * experts->nb[0]);
                const float* src7 = reinterpret_cast<const float*>(
                    src_base + tok * experts->nb[2] + 7 * experts->nb[1] + embd0 * experts->nb[0]);
                const float w1 =
                    *reinterpret_cast<const float*>(weight_base + tok * weights->nb[2] + weights->nb[1]);
                const float w2 =
                    *reinterpret_cast<const float*>(weight_base + tok * weights->nb[2] + 2 * weights->nb[1]);
                const float w3 =
                    *reinterpret_cast<const float*>(weight_base + tok * weights->nb[2] + 3 * weights->nb[1]);
                const float w4 =
                    *reinterpret_cast<const float*>(weight_base + tok * weights->nb[2] + 4 * weights->nb[1]);
                const float w5 =
                    *reinterpret_cast<const float*>(weight_base + tok * weights->nb[2] + 5 * weights->nb[1]);
                const float w6 =
                    *reinterpret_cast<const float*>(weight_base + tok * weights->nb[2] + 6 * weights->nb[1]);
                const float w7 =
                    *reinterpret_cast<const float*>(weight_base + tok * weights->nb[2] + 7 * weights->nb[1]);
                const float safe_w1 = std::isfinite(w1) ? w1 : 0.0f;
                const float safe_w2 = std::isfinite(w2) ? w2 : 0.0f;
                const float safe_w3 = std::isfinite(w3) ? w3 : 0.0f;
                const float safe_w4 = std::isfinite(w4) ? w4 : 0.0f;
                const float safe_w5 = std::isfinite(w5) ? w5 : 0.0f;
                const float safe_w6 = std::isfinite(w6) ? w6 : 0.0f;
                const float safe_w7 = std::isfinite(w7) ? w7 : 0.0f;
                for (int64_t i = 0; i < count; ++i) {
                    float sum = src0[i] * safe_w0;
                    sum += src1[i] * safe_w1;
                    sum += src2[i] * safe_w2;
                    sum += src3[i] * safe_w3;
                    sum += src4[i] * safe_w4;
                    sum += src5[i] * safe_w5;
                    sum += src6[i] * safe_w6;
                    sum += src7[i] * safe_w7;
                    dst_ptr[i] = std::isfinite(sum) ? sum : 0.0f;
                }
                idx += count;
                continue;
            }
            for (int64_t i = 0; i < count; ++i) {
                const float sum = src0[i] * safe_w0;
                dst_ptr[i] = std::isfinite(sum) ? sum : 0.0f;
            }
            for (int64_t expert = 1; expert < n_expert_used; ++expert) {
                const float weight = *reinterpret_cast<const float*>(
                    weight_base + tok * weights->nb[2] + expert * weights->nb[1]);
                const float safe_weight = std::isfinite(weight) ? weight : 0.0f;
                const float* src_ptr = reinterpret_cast<const float*>(
                    src_base + tok * experts->nb[2] + expert * experts->nb[1] + embd0 * experts->nb[0]);
                for (int64_t i = 0; i < count; ++i) {
                    const float sum = dst_ptr[i] + src_ptr[i] * safe_weight;
                    dst_ptr[i] = std::isfinite(sum) ? sum : 0.0f;
                }
            }
            idx += count;
        }
        return;
    }

    for (int64_t idx = start; idx < end; ++idx) {
        const int64_t embd = idx % n_embd;
        const int64_t tok = idx / n_embd;
        float sum = 0.0f;
        for (int64_t expert = 0; expert < n_expert_used; ++expert) {
            const char* src_ptr = src_base + embd * experts->nb[0] + expert * experts->nb[1] + tok * experts->nb[2];
            const char* weight_ptr = weight_base + expert * weights->nb[1] + tok * weights->nb[2];
            const float weight = *reinterpret_cast<const float*>(weight_ptr);
            const float weighted = *reinterpret_cast<const float*>(src_ptr) * (std::isfinite(weight) ? weight : 0.0f);
            sum += weighted;
        }
        char* dst_ptr = dst_base + embd * dst->nb[0] + tok * dst->nb[1];
        *reinterpret_cast<float*>(dst_ptr) = std::isfinite(sum) ? sum : 0.0f;
    }
}

ggml_tensor* BuildMoeExpertWeightedSumWithWeights(struct ggml_context* ctx, ggml_tensor* experts, ggml_tensor* weights,
                                                  int64_t n_embd, int64_t n_tokens, const char* name) {
    if (!ctx || !experts || !weights || experts->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 ||
        n_embd <= 0 || n_tokens <= 0 || experts->ne[0] != n_embd || experts->ne[2] != n_tokens ||
        experts->ne[1] <= 0 || weights->ne[0] != 1 || weights->ne[1] != experts->ne[1] ||
        weights->ne[2] != n_tokens) {
        return nullptr;
    }
    ggml_tensor* args[] = {experts, weights};
    ggml_tensor* out = ggml_custom_4d(ctx, GGML_TYPE_F32, n_embd, n_tokens, 1, 1, args, 2,
                                      cb_moe_expert_weighted_sum_with_weights, GGML_N_TASKS_MAX, nullptr);
    ggml_set_name(out, name);
    return out;
}

void cb_moe_topk_weights_from_logits(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    (void)userdata;
    if (!dst || !dst->src[0] || !dst->src[1] || !dst->data || !dst->src[0]->data || !dst->src[1]->data || nth <= 0) {
        return;
    }
    const ggml_tensor* logits = dst->src[0];
    const ggml_tensor* selected = dst->src[1];
    if (dst->type != GGML_TYPE_F32 || logits->type != GGML_TYPE_F32 || selected->type != GGML_TYPE_I32 ||
        logits->ne[0] <= 0 || logits->ne[1] <= 0 || selected->ne[0] <= 0 || selected->ne[1] != logits->ne[1] ||
        dst->ne[0] != 1 || dst->ne[1] != selected->ne[0] || dst->ne[2] != logits->ne[1]) {
        return;
    }

    const int64_t n_experts = logits->ne[0];
    const int64_t n_tokens = logits->ne[1];
    const int64_t top_k = selected->ne[0];
    const int64_t start = (n_tokens * ith) / nth;
    const int64_t end = (n_tokens * (ith + 1)) / nth;
    const char* logits_base = static_cast<const char*>(logits->data);
    const char* selected_base = static_cast<const char*>(selected->data);
    char* dst_base = static_cast<char*>(dst->data);

    for (int64_t tok = start; tok < end; ++tok) {
        float max_logit = -INFINITY;
        for (int64_t k = 0; k < top_k; ++k) {
            const int32_t expert =
                *reinterpret_cast<const int32_t*>(selected_base + k * selected->nb[0] + tok * selected->nb[1]);
            if (expert >= 0 && expert < n_experts) {
                const float logit =
                    *reinterpret_cast<const float*>(logits_base + expert * logits->nb[0] + tok * logits->nb[1]);
                if (std::isfinite(logit)) {
                    max_logit = std::max(max_logit, logit);
                }
            }
        }

        float denom = 0.0f;
        int valid_selected = 0;
        for (int64_t k = 0; k < top_k; ++k) {
            const int32_t expert =
                *reinterpret_cast<const int32_t*>(selected_base + k * selected->nb[0] + tok * selected->nb[1]);
            if (expert >= 0 && expert < n_experts) {
                valid_selected++;
                const float logit =
                    *reinterpret_cast<const float*>(logits_base + expert * logits->nb[0] + tok * logits->nb[1]);
                if (std::isfinite(logit) && std::isfinite(max_logit)) {
                    denom += std::exp(logit - max_logit);
                }
            }
        }
        const bool use_uniform = !(denom > 0.0f) || !std::isfinite(denom);
        denom = use_uniform ? 1.0f : std::max(denom, 6.103515625e-5f);
        const float uniform = valid_selected > 0 ? 1.0f / static_cast<float>(valid_selected) : 0.0f;

        for (int64_t k = 0; k < top_k; ++k) {
            const int32_t expert =
                *reinterpret_cast<const int32_t*>(selected_base + k * selected->nb[0] + tok * selected->nb[1]);
            float value = 0.0f;
            if (expert >= 0 && expert < n_experts) {
                const float logit =
                    *reinterpret_cast<const float*>(logits_base + expert * logits->nb[0] + tok * logits->nb[1]);
                value = use_uniform || !std::isfinite(logit) || !std::isfinite(max_logit)
                            ? uniform
                            : std::exp(logit - max_logit) / denom;
            }
            *reinterpret_cast<float*>(dst_base + k * dst->nb[1] + tok * dst->nb[2]) = value;
        }
    }
}

ggml_tensor* BuildMoETopKWeightsFromLogits(struct ggml_context* ctx, ggml_tensor* logits, ggml_tensor* selected,
                                           const char* name) {
    if (!ctx || !logits || !selected || logits->type != GGML_TYPE_F32 || selected->type != GGML_TYPE_I32 ||
        logits->ne[0] <= 0 || logits->ne[1] <= 0 || selected->ne[0] <= 0 || selected->ne[1] != logits->ne[1]) {
        return nullptr;
    }
    const int n_tasks = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(GGML_N_TASKS_MAX, logits->ne[1])));
    ggml_tensor* args[] = {logits, selected};
    ggml_tensor* out =
        ggml_custom_4d(ctx, GGML_TYPE_F32, 1, selected->ne[0], logits->ne[1], 1, args, 2,
                       cb_moe_topk_weights_from_logits, n_tasks, nullptr);
    ggml_set_name(out, name);
    return out;
}

void cb_fused_gate_up_silu_mul(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    (void)userdata;
    if (!dst || !dst->src[0] || !dst->data || !dst->src[0]->data || nth <= 0) {
        return;
    }
    const ggml_tensor* gate_up = dst->src[0];
    if (dst->type != GGML_TYPE_F32 || gate_up->type != GGML_TYPE_F32 ||
        dst->nb[0] != static_cast<int64_t>(sizeof(float)) ||
        gate_up->nb[0] != static_cast<int64_t>(sizeof(float)) || gate_up->ne[0] != 2 * dst->ne[0] ||
        gate_up->ne[1] != dst->ne[1]) {
        return;
    }

    const int64_t n_ff = dst->ne[0];
    const int64_t n_tokens = dst->ne[1];
    const char* src_base = static_cast<const char*>(gate_up->data);
    char* dst_base = static_cast<char*>(dst->data);
    for (int64_t token = 0; token < n_tokens; ++token) {
        const float* gate =
            reinterpret_cast<const float*>(src_base + static_cast<size_t>(token) * static_cast<size_t>(gate_up->nb[1]));
        const float* up = gate + n_ff;
        float* out =
            reinterpret_cast<float*>(dst_base + static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1]));
        densecore::simd::SiLUMulParallel(out, gate, up, static_cast<size_t>(n_ff), ith, nth);
    }
}

ggml_tensor* BuildFusedGateUpSiluMul(struct ggml_context* ctx, ggml_tensor* gate_up, int64_t n_ff, int64_t n_tokens,
                                    const char* name) {
    if (!ctx || !gate_up || gate_up->type != GGML_TYPE_F32 || n_ff <= 0 || n_tokens <= 0 ||
        gate_up->ne[0] != 2 * n_ff || gate_up->ne[1] != n_tokens ||
        gate_up->nb[0] != static_cast<int64_t>(sizeof(float))) {
        return nullptr;
    }
    ggml_tensor* args[] = {gate_up};
    ggml_tensor* out = ggml_custom_4d(ctx, GGML_TYPE_F32, n_ff, n_tokens, 1, 1, args, 1,
                                      cb_fused_gate_up_silu_mul, GGML_N_TASKS_MAX, nullptr);
    ggml_set_name(out, name);
    return out;
}

ggml_tensor* BuildGemma4PackedGateOrUp3DView(ggml_context* ctx, const Gemma4PackedMoERoots& roots,
                                             bool up_projection) {
    ggml_tensor* root = roots.gate_up;
    const auto& layout = roots.layout;
    if (!ctx || !root) {
        return nullptr;
    }
    if (layout.gate_up_kind == densecore::gemma4::PackedGateUpLayoutKind::RowStacked3D) {
        const size_t offset = up_projection ? static_cast<size_t>(layout.intermediate_dim) *
                                                  static_cast<size_t>(root->nb[1])
                                            : 0;
        return ggml_view_3d(ctx, root, layout.hidden_dim, layout.intermediate_dim, layout.num_experts, root->nb[1],
                            root->nb[2], offset);
    }
    if (layout.gate_up_kind == densecore::gemma4::PackedGateUpLayoutKind::PlaneSeparated4D) {
        const size_t offset = up_projection ? static_cast<size_t>(root->nb[2]) : 0;
        return ggml_view_3d(ctx, root, layout.hidden_dim, layout.intermediate_dim, layout.num_experts, root->nb[1],
                            root->nb[3], offset);
    }
    return nullptr;
}

ggml_tensor* BuildGemma4PackedGateUpMerged3DView(ggml_context* ctx, const Gemma4PackedMoERoots& roots) {
    ggml_tensor* root = roots.gate_up;
    const auto& layout = roots.layout;
    if (!ctx || !root) {
        return nullptr;
    }
    const int64_t merged_rows = layout.intermediate_dim * 2;
    if (layout.gate_up_kind == densecore::gemma4::PackedGateUpLayoutKind::RowStacked3D) {
        return ggml_view_3d(ctx, root, layout.hidden_dim, merged_rows, layout.num_experts, root->nb[1], root->nb[2],
                            0);
    }
    if (layout.gate_up_kind == densecore::gemma4::PackedGateUpLayoutKind::PlaneSeparated4D &&
        root->nb[2] == root->nb[1] * layout.intermediate_dim) {
        return ggml_view_3d(ctx, root, layout.hidden_dim, merged_rows, layout.num_experts, root->nb[1], root->nb[3],
                            0);
    }
    return nullptr;
}

ggml_tensor* BuildGemma4PackedDown3DView(ggml_context* ctx, const Gemma4PackedMoERoots& roots) {
    ggml_tensor* root = roots.down;
    const auto& layout = roots.layout;
    if (!ctx || !root || layout.down_expert_axis < 0) {
        return nullptr;
    }
    return ggml_view_3d(ctx, root, layout.intermediate_dim, layout.hidden_dim, layout.num_experts, root->nb[1],
                        root->nb[layout.down_expert_axis], 0);
}

ggml_tensor* BuildGemma4PackedDownScaleRows(ggml_context* ctx, const Gemma4PackedMoERoots& roots) {
    ggml_tensor* scale = roots.down_scale;
    if (!ctx || !scale || scale->type != GGML_TYPE_F32 || scale->ne[0] != roots.layout.num_experts) {
        return nullptr;
    }
    return ggml_reshape_2d(ctx, scale, 1, roots.layout.num_experts);
}

struct Gemma4GateUpQ4KPrefillUserData {
    int64_t hidden_dim = 0;
    int64_t intermediate_dim = 0;
    int64_t top_k = 0;
    int64_t n_tokens = 0;
    int64_t n_experts = 0;
    int64_t max_assignments = 0;
    int64_t max_batches = 0;
    int32_t* expert_offsets = nullptr;
    int32_t* expert_cursors = nullptr;
    int32_t* assignment_tokens = nullptr;
    int32_t* assignment_slots = nullptr;
    int32_t* batch_experts = nullptr;
    int32_t* batch_starts = nullptr;
    std::atomic<int64_t> total_batches{0};
    std::atomic<int64_t> next_batch{0};
    std::atomic<uint64_t> epoch{0};
    std::atomic<uint64_t> ready_epoch{0};
    std::atomic<int> failure_count{0};
    std::atomic<const char*> first_failure_reason{nullptr};
    std::atomic<bool> used_recorded{false};
    static constexpr int kMaxTasks = 128;
    uint64_t task_epoch[kMaxTasks] = {};
};

constexpr int kGemma4GateUpPrefillRowsPerBatch = 16;

static float Gemma4GeluTanh(float x) {
    const float x3 = x * x * x;
    return 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x3)));
}

static inline void Gemma4ApplyGEGLU(float* out, const float* gate, const float* up, int cols) {
    for (int c = 0; c < cols; ++c) {
        const float value = Gemma4GeluTanh(gate[c]) * up[c];
        out[c] = std::isfinite(value) ? value : 0.0f;
    }
}

static bool Gemma4QuantizedRowDot(ggml_type weight_type, const void* weight_row, const uint8_t* quant_input,
                                  int64_t cols, float* out) {
    if (!weight_row || !quant_input || !out || cols <= 0) {
        return false;
    }
    const ggml_type_traits_cpu* traits = ggml_get_type_traits_cpu(weight_type);
    if (!traits || !traits->vec_dot || (cols % ggml_blck_size(weight_type)) != 0) {
        return false;
    }
    *out = 0.0f;
    traits->vec_dot(static_cast<int>(cols), out, 0, weight_row, 0, quant_input, 0, 1);
    if (!std::isfinite(*out)) {
        *out = 0.0f;
    }
    return true;
}

static bool Gemma4QuantizedRowDotInputType(ggml_type weight_type, int64_t cols, ggml_type* input_quant_type,
                                           size_t* input_quant_row_bytes) {
    if (!input_quant_type || !input_quant_row_bytes || cols <= 0) {
        return false;
    }
    const ggml_type_traits_cpu* traits = ggml_get_type_traits_cpu(weight_type);
    if (!traits || !traits->vec_dot || (cols % ggml_blck_size(weight_type)) != 0) {
        return false;
    }
    const ggml_type quant_type = traits->vec_dot_type;
    const ggml_type_traits_cpu* input_traits = ggml_get_type_traits_cpu(quant_type);
    const size_t row_bytes = ggml_row_size(quant_type, cols);
    if (!input_traits || !input_traits->from_float || row_bytes == 0) {
        return false;
    }
    *input_quant_type = quant_type;
    *input_quant_row_bytes = row_bytes;
    return true;
}

static bool Gemma4CanUseDequantizedF32RowDot(ggml_type weight_type, int64_t cols) {
    if (cols <= 0 || (cols % ggml_blck_size(weight_type)) != 0) {
        return false;
    }
    const ggml_type_traits* traits = ggml_get_type_traits(weight_type);
    return traits && traits->to_float && ggml_row_size(weight_type, cols) > 0;
}

static bool Gemma4DequantizedF32RowDot(ggml_type weight_type, const void* weight_row, const float* input,
                                       int64_t cols, float* out) {
    if (!weight_row || !input || !out || !Gemma4CanUseDequantizedF32RowDot(weight_type, cols)) {
        return false;
    }
    const ggml_type_traits* traits = ggml_get_type_traits(weight_type);
    thread_local std::vector<float> weight_f32;
    weight_f32.resize(static_cast<size_t>(cols));
    traits->to_float(weight_row, weight_f32.data(), cols);
    double acc = 0.0;
    for (int64_t i = 0; i < cols; ++i) {
        const float w = weight_f32[static_cast<size_t>(i)];
        const float x = input[static_cast<size_t>(i)];
        if (std::isfinite(w) && std::isfinite(x)) {
            acc += static_cast<double>(w) * static_cast<double>(x);
        }
    }
    *out = std::isfinite(acc) ? static_cast<float>(acc) : 0.0f;
    return true;
}

static void ZeroGemma4PrefillF32Tensor(ggml_tensor* tensor) {
    if (!tensor || !tensor->data || tensor->type != GGML_TYPE_F32 || tensor->ne[0] <= 0 || tensor->ne[1] <= 0 ||
        tensor->ne[2] <= 0) {
        return;
    }
    char* base = static_cast<char*>(tensor->data);
    const size_t row_bytes = static_cast<size_t>(tensor->ne[0]) * sizeof(float);
    for (int64_t z = 0; z < tensor->ne[2]; ++z) {
        for (int64_t y = 0; y < tensor->ne[1]; ++y) {
            std::memset(base + static_cast<size_t>(y) * tensor->nb[1] + static_cast<size_t>(z) * tensor->nb[2], 0,
                        row_bytes);
        }
    }
}

static Gemma4GateUpQ4KPrefillUserData* AllocateGemma4GateUpQ4KPrefillUserData(ggml_context* ctx, int64_t hidden_dim,
                                                                               int64_t intermediate_dim,
                                                                               int64_t top_k, int64_t n_tokens,
                                                                               int64_t n_experts) {
    if (!ctx || hidden_dim <= 0 || intermediate_dim <= 0 || top_k <= 0 || n_tokens <= 0 || n_experts <= 0) {
        return nullptr;
    }
    const int64_t max_assignments = top_k * n_tokens;
    const int64_t max_batches = max_assignments;
    if (max_assignments <= 0 || max_assignments > std::numeric_limits<int32_t>::max() ||
        n_experts + 1 > std::numeric_limits<int32_t>::max()) {
        return nullptr;
    }
    if (ggml_get_no_alloc(ctx)) {
        // ggml no-alloc graph construction stores the userdata pointer in the
        // custom op. A thread-local scratch object is unsafe because later graph
        // builds (for example decode n_tokens=1) overwrite it before the prefill
        // graph executes. The native prefill path is diagnostic/opt-in, so use a
        // stable process-lifetime sidecar here until this path is promoted into a
        // graph-owned allocation contract.
        auto* ud = new Gemma4GateUpQ4KPrefillUserData();
        ud->expert_offsets = new int32_t[static_cast<size_t>(n_experts + 1)]();
        ud->expert_cursors = new int32_t[static_cast<size_t>(n_experts)]();
        ud->assignment_tokens = new int32_t[static_cast<size_t>(max_assignments)]();
        ud->assignment_slots = new int32_t[static_cast<size_t>(max_assignments)]();
        ud->batch_experts = new int32_t[static_cast<size_t>(max_batches)]();
        ud->batch_starts = new int32_t[static_cast<size_t>(max_batches)]();
        ud->hidden_dim = hidden_dim;
        ud->intermediate_dim = intermediate_dim;
        ud->top_k = top_k;
        ud->n_tokens = n_tokens;
        ud->n_experts = n_experts;
        ud->max_assignments = max_assignments;
        ud->max_batches = max_batches;
        return ud;
    }

    ggml_tensor* ud_storage = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, sizeof(Gemma4GateUpQ4KPrefillUserData));
    ggml_tensor* offsets_storage = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_experts + 1);
    ggml_tensor* cursors_storage = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_experts);
    ggml_tensor* tokens_storage = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, max_assignments);
    ggml_tensor* slots_storage = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, max_assignments);
    ggml_tensor* batch_experts_storage = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, max_batches);
    ggml_tensor* batch_starts_storage = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, max_batches);
    if (!ud_storage || !ud_storage->data || !offsets_storage || !offsets_storage->data || !cursors_storage ||
        !cursors_storage->data || !tokens_storage ||
        !tokens_storage->data || !slots_storage || !slots_storage->data || !batch_experts_storage ||
        !batch_experts_storage->data || !batch_starts_storage ||
        !batch_starts_storage->data) {
        return nullptr;
    }

    auto* ud = new (ud_storage->data) Gemma4GateUpQ4KPrefillUserData();
    ud->hidden_dim = hidden_dim;
    ud->intermediate_dim = intermediate_dim;
    ud->top_k = top_k;
    ud->n_tokens = n_tokens;
    ud->n_experts = n_experts;
    ud->max_assignments = max_assignments;
    ud->max_batches = max_batches;
    ud->expert_offsets = static_cast<int32_t*>(offsets_storage->data);
    ud->expert_cursors = static_cast<int32_t*>(cursors_storage->data);
    ud->assignment_tokens = static_cast<int32_t*>(tokens_storage->data);
    ud->assignment_slots = static_cast<int32_t*>(slots_storage->data);
    ud->batch_experts = static_cast<int32_t*>(batch_experts_storage->data);
    ud->batch_starts = static_cast<int32_t*>(batch_starts_storage->data);
    return ud;
}

static void MarkGemma4NativeMoEPrefillFailure(Gemma4GateUpQ4KPrefillUserData* ud, const char* reason,
                                              const char* stage, int ith) {
    if (!ud) {
        return;
    }
    ud->failure_count.fetch_add(1, std::memory_order_relaxed);
    const char* expected = nullptr;
    ud->first_failure_reason.compare_exchange_strong(expected, reason, std::memory_order_relaxed);
    if (ith == 0 && IsDebugGemma4NativeMoEPrefillEnabled()) {
        std::fprintf(stderr, "[Gemma4NativeMoEPrefillDebug] %s failed reason=%s\n", stage ? stage : "native_prefill",
                     reason ? reason : "unknown");
    }
}

static bool Gemma4PrefillInputShapeOk(const ggml_tensor* input, int64_t hidden_dim, int64_t top_k, int64_t n_tokens) {
    if (!input || input->type != GGML_TYPE_F32 || input->ne[0] != hidden_dim || top_k <= 0 || n_tokens <= 0) {
        return false;
    }
    if (input->ne[1] == n_tokens) {
        return true;
    }
    if (input->ne[2] == n_tokens && (input->ne[1] == top_k || input->ne[1] == 1)) {
        return true;
    }
    return false;
}

static bool Gemma4PrefillAssignmentShapeOk(const ggml_tensor* input, int64_t top_k, int64_t n_tokens) {
    if (!input || input->type != GGML_TYPE_F32 || top_k <= 0 || n_tokens <= 0) {
        return false;
    }
    if (input->ne[1] == n_tokens) {
        return true;
    }
    if (input->ne[2] == n_tokens && (input->ne[1] == top_k || input->ne[1] == 1)) {
        return true;
    }
    return false;
}

static const float* Gemma4PrefillInputRow(const ggml_tensor* input, int32_t token, int32_t slot,
                                          const Gemma4GateUpQ4KPrefillUserData* ud) {
    if (!input || !input->data || !ud || token < 0 || token >= ud->n_tokens || slot < 0 || slot >= ud->top_k ||
        !Gemma4PrefillInputShapeOk(input, ud->hidden_dim, ud->top_k, ud->n_tokens)) {
        return nullptr;
    }
    const char* base = static_cast<const char*>(input->data);
    if (input->ne[1] == ud->n_tokens) {
        return reinterpret_cast<const float*>(base + static_cast<size_t>(token) * input->nb[1]);
    }
    const int32_t source_slot = input->ne[1] == 1 ? 0 : slot;
    return reinterpret_cast<const float*>(base + static_cast<size_t>(source_slot) * input->nb[1] +
                                          static_cast<size_t>(token) * input->nb[2]);
}

static bool PrepareGemma4GateUpQ4KPrefillBatches(Gemma4GateUpQ4KPrefillUserData* ud,
                                                  const ggml_tensor* input, const ggml_tensor* selected_experts, int ith,
                                                  int nth) {
    if (!ud || !input || !selected_experts || !input->data || !selected_experts->data ||
        input->type != GGML_TYPE_F32 || selected_experts->type != GGML_TYPE_I32 || nth <= 0 || ith < 0 ||
        ith >= nth || nth > Gemma4GateUpQ4KPrefillUserData::kMaxTasks ||
        !Gemma4PrefillAssignmentShapeOk(input, ud->top_k, ud->n_tokens) ||
        selected_experts->ne[0] != ud->top_k ||
        selected_experts->ne[1] != ud->n_tokens || !ud->expert_offsets || !ud->expert_cursors ||
        !ud->assignment_tokens || !ud->assignment_slots || !ud->batch_experts || !ud->batch_starts) {
        return false;
    }

    if (ith == 0) {
        const uint64_t epoch = ud->epoch.fetch_add(1, std::memory_order_acq_rel) + 1;
        ud->task_epoch[0] = epoch;
        std::fill_n(ud->expert_offsets, static_cast<size_t>(ud->n_experts + 1), int32_t{0});
        for (int64_t token = 0; token < ud->n_tokens; ++token) {
            for (int64_t k = 0; k < ud->top_k; ++k) {
                const int32_t expert = *reinterpret_cast<const int32_t*>(
                    static_cast<const char*>(selected_experts->data) + static_cast<size_t>(k) * selected_experts->nb[0] +
                    static_cast<size_t>(token) * selected_experts->nb[1]);
                if (expert >= 0 && expert < ud->n_experts) {
                    ++ud->expert_offsets[expert + 1];
                }
            }
        }
        for (int64_t expert = 0; expert < ud->n_experts; ++expert) {
            ud->expert_offsets[expert + 1] += ud->expert_offsets[expert];
            ud->expert_cursors[expert] = ud->expert_offsets[expert];
        }
        for (int64_t token = 0; token < ud->n_tokens; ++token) {
            for (int64_t k = 0; k < ud->top_k; ++k) {
                const int32_t expert = *reinterpret_cast<const int32_t*>(
                    static_cast<const char*>(selected_experts->data) + static_cast<size_t>(k) * selected_experts->nb[0] +
                    static_cast<size_t>(token) * selected_experts->nb[1]);
                if (expert >= 0 && expert < ud->n_experts) {
                    const int32_t pos = ud->expert_cursors[expert]++;
                    if (pos >= 0 && pos < ud->max_assignments) {
                        ud->assignment_tokens[pos] = static_cast<int32_t>(token);
                        ud->assignment_slots[pos] = static_cast<int32_t>(k);
                    }
                }
            }
        }
        int64_t batches = 0;
        for (int64_t expert = 0; expert < ud->n_experts; ++expert) {
            for (int32_t start = ud->expert_offsets[expert]; start < ud->expert_offsets[expert + 1];
                 start += kGemma4GateUpPrefillRowsPerBatch) {
                if (batches >= ud->max_batches) {
                    break;
                }
                ud->batch_experts[batches] = static_cast<int32_t>(expert);
                ud->batch_starts[batches] = start;
                ++batches;
            }
        }
        ud->total_batches.store(batches, std::memory_order_release);
        ud->next_batch.store(0, std::memory_order_release);
        ud->ready_epoch.store(epoch, std::memory_order_release);
        if (IsDebugGemma4NativeMoEPrefillEnabled()) {
            int64_t assignments = 0;
            int64_t active_experts = 0;
            for (int64_t expert = 0; expert < ud->n_experts; ++expert) {
                const int32_t count = ud->expert_offsets[expert + 1] - ud->expert_offsets[expert];
                assignments += count;
                active_experts += count > 0 ? 1 : 0;
            }
            std::fprintf(stderr,
                         "[Gemma4NativeMoEPrefillDebug] prepare tokens=%lld top_k=%lld experts=%lld assignments=%lld "
                         "active_experts=%lld batches=%lld selected_ne=[%lld,%lld,%lld] selected_nb=[%lld,%lld,%lld]\n",
                         static_cast<long long>(ud->n_tokens), static_cast<long long>(ud->top_k),
                         static_cast<long long>(ud->n_experts), static_cast<long long>(assignments),
                         static_cast<long long>(active_experts), static_cast<long long>(batches),
                         static_cast<long long>(selected_experts->ne[0]),
                         static_cast<long long>(selected_experts->ne[1]),
                         static_cast<long long>(selected_experts->ne[2]),
                         static_cast<long long>(selected_experts->nb[0]),
                         static_cast<long long>(selected_experts->nb[1]),
                         static_cast<long long>(selected_experts->nb[2]));
        }
        return true;
    }

    const uint64_t last_epoch = ud->task_epoch[ith];
    uint64_t epoch = ud->epoch.load(std::memory_order_acquire);
    while (epoch == last_epoch) {
        std::this_thread::yield();
        epoch = ud->epoch.load(std::memory_order_acquire);
    }
    while (ud->ready_epoch.load(std::memory_order_acquire) != epoch) {
        std::this_thread::yield();
    }
    ud->task_epoch[ith] = epoch;
    return true;
}

static bool RunGemma4GateUpQ4KPrefillFusedGEGLU(ggml_tensor* dst, const ggml_tensor* gate_up_exps,
                                                const ggml_tensor* input, const ggml_tensor* selected_experts,
                                                int ith, int nth, Gemma4GateUpQ4KPrefillUserData* ud) {
    if (!dst || !gate_up_exps || !input || !selected_experts || !ud || !dst->data || !gate_up_exps->data ||
        !input->data || nth <= 0) {
        MarkGemma4NativeMoEPrefillFailure(ud, "gateup_missing_runtime_data", "gateup", ith);
        return false;
    }
    if (dst->type != GGML_TYPE_F32 || gate_up_exps->type != GGML_TYPE_Q4_K || input->type != GGML_TYPE_F32 ||
        selected_experts->type != GGML_TYPE_I32 || dst->ne[0] != ud->intermediate_dim || dst->ne[1] != ud->top_k ||
        dst->ne[2] != ud->n_tokens ||
        !Gemma4PrefillInputShapeOk(input, ud->hidden_dim, ud->top_k, ud->n_tokens) ||
        gate_up_exps->ne[0] != ud->hidden_dim || gate_up_exps->ne[1] != 2 * ud->intermediate_dim ||
        gate_up_exps->ne[2] != ud->n_experts || (ud->hidden_dim % QK_K) != 0 || (ud->intermediate_dim % 8) != 0 ||
        dst->nb[0] != static_cast<int64_t>(sizeof(float))) {
        if (ith == 0 && IsDebugGemma4NativeMoEPrefillEnabled()) {
            std::fprintf(stderr,
                         "[Gemma4NativeMoEPrefillDebug] gateup shape_mismatch dst_ne=[%lld,%lld,%lld] "
                         "input_ne=[%lld,%lld,%lld] gate_ne=[%lld,%lld,%lld] selected_ne=[%lld,%lld,%lld] "
                         "ud=[hidden=%lld,intermediate=%lld,top_k=%lld,tokens=%lld,experts=%lld]\n",
                         static_cast<long long>(dst->ne[0]), static_cast<long long>(dst->ne[1]),
                         static_cast<long long>(dst->ne[2]), static_cast<long long>(input->ne[0]),
                         static_cast<long long>(input->ne[1]), static_cast<long long>(input->ne[2]),
                         static_cast<long long>(gate_up_exps->ne[0]), static_cast<long long>(gate_up_exps->ne[1]),
                         static_cast<long long>(gate_up_exps->ne[2]), static_cast<long long>(selected_experts->ne[0]),
                         static_cast<long long>(selected_experts->ne[1]),
                         static_cast<long long>(selected_experts->ne[2]), static_cast<long long>(ud->hidden_dim),
                         static_cast<long long>(ud->intermediate_dim), static_cast<long long>(ud->top_k),
                         static_cast<long long>(ud->n_tokens), static_cast<long long>(ud->n_experts));
        }
        MarkGemma4NativeMoEPrefillFailure(ud, "gateup_shape_or_type_mismatch", "gateup", ith);
        return false;
    }
    if (ith == 0) {
        ZeroGemma4PrefillF32Tensor(dst);
    }
    if (!PrepareGemma4GateUpQ4KPrefillBatches(ud, input, selected_experts, ith, nth)) {
        if (ith == 0 && IsDebugGemma4NativeMoEPrefillEnabled()) {
            std::fprintf(stderr,
                         "[Gemma4NativeMoEPrefillDebug] gateup prepare_failed dst_data=%p gate_data=%p input_data=%p "
                         "selected_data=%p dst_ne=[%lld,%lld,%lld] gate_ne=[%lld,%lld,%lld] input_ne=[%lld,%lld,%lld] "
                         "selected_ne=[%lld,%lld,%lld]\n",
                         dst ? dst->data : nullptr, gate_up_exps ? gate_up_exps->data : nullptr,
                         input ? input->data : nullptr, selected_experts ? selected_experts->data : nullptr,
                         dst ? static_cast<long long>(dst->ne[0]) : -1LL,
                         dst ? static_cast<long long>(dst->ne[1]) : -1LL,
                         dst ? static_cast<long long>(dst->ne[2]) : -1LL,
                         gate_up_exps ? static_cast<long long>(gate_up_exps->ne[0]) : -1LL,
                         gate_up_exps ? static_cast<long long>(gate_up_exps->ne[1]) : -1LL,
                         gate_up_exps ? static_cast<long long>(gate_up_exps->ne[2]) : -1LL,
                         input ? static_cast<long long>(input->ne[0]) : -1LL,
                         input ? static_cast<long long>(input->ne[1]) : -1LL,
                         input ? static_cast<long long>(input->ne[2]) : -1LL,
                         selected_experts ? static_cast<long long>(selected_experts->ne[0]) : -1LL,
                         selected_experts ? static_cast<long long>(selected_experts->ne[1]) : -1LL,
                         selected_experts ? static_cast<long long>(selected_experts->ne[2]) : -1LL);
        }
        MarkGemma4NativeMoEPrefillFailure(ud, "gateup_prepare_failed", "gateup", ith);
        return false;
    }
    if (ith == 0 && IsDebugGemma4NativeMoEPrefillEnabled()) {
        static std::atomic<int> gateup_debug_count{0};
        const int debug_idx = gateup_debug_count.fetch_add(1, std::memory_order_relaxed);
        if (debug_idx < 96) {
            std::fprintf(stderr,
                         "[Gemma4NativeMoEPrefillDebug] gateup_execute[%d] dst=%s tokens=%lld top_k=%lld "
                         "total_batches=%lld nth=%d\n",
                         debug_idx, dst->name, static_cast<long long>(ud->n_tokens),
                         static_cast<long long>(ud->top_k),
                         static_cast<long long>(ud->total_batches.load(std::memory_order_acquire)), nth);
        }
    }

    const int tile_count = static_cast<int>(ud->intermediate_dim / 8);
    const int64_t total_batches = ud->total_batches.load(std::memory_order_acquire);
    const char* weight_base = static_cast<const char*>(gate_up_exps->data);
    char* dst_base = static_cast<char*>(dst->data);
    const size_t q4_row_bytes = ggml_row_size(GGML_TYPE_Q4_K, ud->hidden_dim);
    if (q4_row_bytes == 0 || static_cast<size_t>(gate_up_exps->nb[1]) < q4_row_bytes ||
        static_cast<size_t>(gate_up_exps->nb[2]) <
            static_cast<size_t>(2 * ud->intermediate_dim - 1) * static_cast<size_t>(gate_up_exps->nb[1]) +
                q4_row_bytes) {
        MarkGemma4NativeMoEPrefillFailure(ud, "gateup_row_stride_mismatch", "gateup", ith);
        return false;
    }
    const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const auto* q4_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K);
    if (!q8_traits || !q8_traits->from_float || !q4_traits || q4_traits->vec_dot_type != GGML_TYPE_Q8_K) {
        MarkGemma4NativeMoEPrefillFailure(ud, "gateup_q4k_q8k_traits_unavailable", "gateup", ith);
        return false;
    }

    thread_local std::vector<uint8_t> q8_tail_buf;
    q8_tail_buf.resize(ggml_row_size(GGML_TYPE_Q8_K, ud->hidden_dim));

    for (;;) {
        const int64_t batch = ud->next_batch.fetch_add(1, std::memory_order_relaxed);
        if (batch >= total_batches) {
            break;
        }
        const int32_t expert = ud->batch_experts[batch];
        const int32_t start = ud->batch_starts[batch];
        if (expert < 0 || expert >= ud->n_experts || start < 0 || start >= ud->max_assignments) {
            continue;
        }
        const int32_t end = ud->expert_offsets[expert + 1];
        const int rows = static_cast<int>(std::min<int32_t>(kGemma4GateUpPrefillRowsPerBatch, end - start));
        if (rows <= 0) {
            continue;
        }
        const char* expert_base =
            weight_base + static_cast<size_t>(expert) * static_cast<size_t>(gate_up_exps->nb[2]);

        std::array<float, 8> gate_tail{};
        std::array<float, 8> up_tail{};
        for (int r = 0; r < rows; ++r) {
            const int32_t token = ud->assignment_tokens[start + r];
            const int32_t slot = ud->assignment_slots[start + r];
            if (token < 0 || token >= ud->n_tokens || slot < 0 || slot >= ud->top_k) {
                continue;
            }
            const float* src = Gemma4PrefillInputRow(input, token, slot, ud);
            if (!src) {
                MarkGemma4NativeMoEPrefillFailure(ud, "gateup_input_row_unavailable", "gateup", ith);
                return false;
            }
            q8_traits->from_float(src, q8_tail_buf.data(), ud->hidden_dim);
            for (int tile = 0; tile < tile_count; ++tile) {
                for (int lane = 0; lane < 8; ++lane) {
                    const int64_t row = static_cast<int64_t>(tile) * 8 + lane;
                    const void* gate_row =
                        expert_base + static_cast<size_t>(row) * static_cast<size_t>(gate_up_exps->nb[1]);
                    const void* up_row =
                        expert_base + static_cast<size_t>(ud->intermediate_dim + row) *
                                          static_cast<size_t>(gate_up_exps->nb[1]);
                    if (!Gemma4QuantizedRowDot(GGML_TYPE_Q4_K, gate_row, q8_tail_buf.data(), ud->hidden_dim,
                                               &gate_tail[lane]) ||
                        !Gemma4QuantizedRowDot(GGML_TYPE_Q4_K, up_row, q8_tail_buf.data(), ud->hidden_dim,
                                               &up_tail[lane])) {
                        MarkGemma4NativeMoEPrefillFailure(ud, "gateup_row_dot_failed", "gateup", ith);
                        return false;
                    }
                }
                float* out = reinterpret_cast<float*>(dst_base + static_cast<size_t>(tile) * 8 * dst->nb[0] +
                                                      static_cast<size_t>(slot) * dst->nb[1] +
                                                      static_cast<size_t>(token) * dst->nb[2]);
                Gemma4ApplyGEGLU(out, gate_tail.data(), up_tail.data(), 8);
                if (ith == 0 && batch == 0 && r == 0 && tile == 0 && IsDebugGemma4NativeMoEPrefillEnabled()) {
                    float max_abs = 0.0f;
                    for (float v : gate_tail) max_abs = std::max(max_abs, std::fabs(v));
                    for (float v : up_tail) max_abs = std::max(max_abs, std::fabs(v));
                    std::fprintf(stderr,
                                 "[Gemma4NativeMoEPrefillDebug] gateup_first token=%d slot=%d expert=%d max_abs=%g "
                                 "out0=%g out1=%g\n",
                                 token, slot, expert, max_abs, out[0], out[1]);
                }
            }
        }
    }
    return true;
}

static void cb_gemma4_gateup_q4k_prefill_geglu(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    const auto begin = std::chrono::steady_clock::now();
    auto* ud = static_cast<Gemma4GateUpQ4KPrefillUserData*>(userdata);
    const bool ok = RunGemma4GateUpQ4KPrefillFusedGEGLU(dst, dst ? dst->src[0] : nullptr,
                                                        dst ? dst->src[1] : nullptr, dst ? dst->src[2] : nullptr,
                                                        ith, nth, ud);
    const uint64_t ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count());
    if (GetCurrentExecutionPhase() == InferenceExecutionPhase::Prefill) {
        RecordGemma4NativeMoEPrefillTiming(GetCurrentWorkContext(), ns, 0, ns);
        if (!ok && ud) {
            RecordGemma4NativeMoEPrefillDecision(GetCurrentWorkContext(), /*candidate=*/false, /*used=*/false,
                                                 ud->first_failure_reason.load(std::memory_order_relaxed), 0, false);
        }
    }
    if (ith == 0 && GetCurrentExecutionPhase() == InferenceExecutionPhase::Decode) {
        RecordGemma4NativeFusedGateUpUsed(GetCurrentWorkContext());
        RecordGemma4DecodeNativeDecision(GetCurrentWorkContext(), /*candidate=*/true, /*used=*/true, nullptr,
                                         /*moe_used=*/true, /*dense_used=*/false, /*lm_head_used=*/false, ns,
                                         /*replaced_mul_mat_ops=*/0, /*replaced_mul_mat_id_ops=*/1,
                                         /*duplicate_work_detected=*/false);
    }
}

static bool RunGemma4DownQuantPrefill(ggml_tensor* dst, const ggml_tensor* down_exps, const ggml_tensor* hidden,
                                      const ggml_tensor* selected_experts, int ith, int nth,
                                      Gemma4GateUpQ4KPrefillUserData* ud, ggml_type weight_type,
                                      const char* debug_label) {
    if (!dst || !down_exps || !hidden || !selected_experts || !ud || !dst->data || !down_exps->data ||
        !hidden->data || nth <= 0) {
        MarkGemma4NativeMoEPrefillFailure(ud, "down_quant_missing_runtime_data", debug_label, ith);
        return false;
    }
    if (dst->type != GGML_TYPE_F32 || down_exps->type != weight_type || hidden->type != GGML_TYPE_F32 ||
        selected_experts->type != GGML_TYPE_I32 || dst->ne[0] != ud->hidden_dim || dst->ne[1] != ud->top_k ||
        dst->ne[2] != ud->n_tokens || hidden->ne[0] != ud->intermediate_dim || hidden->ne[1] != ud->top_k ||
        hidden->ne[2] != ud->n_tokens || down_exps->ne[0] != ud->intermediate_dim ||
        down_exps->ne[1] != ud->hidden_dim || down_exps->ne[2] != ud->n_experts ||
        dst->nb[0] != static_cast<int64_t>(sizeof(float))) {
        MarkGemma4NativeMoEPrefillFailure(ud, "down_quant_shape_or_type_mismatch", debug_label, ith);
        return false;
    }
    ggml_type input_quant_type = GGML_TYPE_COUNT;
    size_t input_quant_row_bytes = 0;
    const bool use_vec_dot = Gemma4QuantizedRowDotInputType(weight_type, ud->intermediate_dim, &input_quant_type,
                                                           &input_quant_row_bytes);
    const bool use_dequant_dot = !use_vec_dot && Gemma4CanUseDequantizedF32RowDot(weight_type, ud->intermediate_dim);
    if (!use_vec_dot && !use_dequant_dot) {
        MarkGemma4NativeMoEPrefillFailure(ud, "down_quant_dot_traits_unavailable", debug_label, ith);
        return false;
    }
    const size_t weight_row_bytes = ggml_row_size(weight_type, ud->intermediate_dim);
    if (weight_row_bytes == 0 || static_cast<size_t>(down_exps->nb[1]) < weight_row_bytes ||
        static_cast<size_t>(down_exps->nb[2]) <
            static_cast<size_t>(ud->hidden_dim - 1) * static_cast<size_t>(down_exps->nb[1]) + weight_row_bytes) {
        MarkGemma4NativeMoEPrefillFailure(ud, "down_quant_row_stride_mismatch", debug_label, ith);
        return false;
    }
    const ggml_type_traits_cpu* input_traits = use_vec_dot ? ggml_get_type_traits_cpu(input_quant_type) : nullptr;
    if (use_vec_dot && (!input_traits || !input_traits->from_float)) {
        MarkGemma4NativeMoEPrefillFailure(ud, "down_quant_input_traits_unavailable", debug_label, ith);
        return false;
    }
    if (ith == 0) {
        ZeroGemma4PrefillF32Tensor(dst);
    }
    if (!PrepareGemma4GateUpQ4KPrefillBatches(ud, hidden, selected_experts, ith, nth)) {
        if (ith == 0 && IsDebugGemma4NativeMoEPrefillEnabled()) {
            std::fprintf(stderr, "[Gemma4NativeMoEPrefillDebug] %s prepare_failed\n",
                         debug_label ? debug_label : "down_quant");
        }
        MarkGemma4NativeMoEPrefillFailure(ud, "down_quant_prepare_failed", debug_label, ith);
        return false;
    }
    if (ith == 0 && IsDebugGemma4NativeMoEPrefillEnabled()) {
        static std::atomic<int> down_quant_debug_count{0};
        const int debug_idx = down_quant_debug_count.fetch_add(1, std::memory_order_relaxed);
        if (debug_idx < 96) {
            std::fprintf(stderr,
                         "[Gemma4NativeMoEPrefillDebug] %s_execute[%d] dst=%s type=%s input_quant=%s tokens=%lld "
                         "top_k=%lld total_batches=%lld nth=%d\n",
                         debug_label ? debug_label : "down_quant", debug_idx, dst->name, ggml_type_name(weight_type),
                         use_vec_dot ? ggml_type_name(input_quant_type) : "f32_dequant_row_dot",
                         static_cast<long long>(ud->n_tokens),
                         static_cast<long long>(ud->top_k),
                         static_cast<long long>(ud->total_batches.load(std::memory_order_acquire)), nth);
        }
    }

    const int64_t total_batches = ud->total_batches.load(std::memory_order_acquire);
    const char* weight_base = static_cast<const char*>(down_exps->data);
    const char* hidden_base = static_cast<const char*>(hidden->data);
    char* dst_base = static_cast<char*>(dst->data);
    thread_local std::vector<uint8_t> qrow;
    if (use_vec_dot) {
        qrow.resize(input_quant_row_bytes);
    }

    for (;;) {
        const int64_t batch = ud->next_batch.fetch_add(1, std::memory_order_relaxed);
        if (batch >= total_batches) {
            break;
        }
        const int32_t expert = ud->batch_experts[batch];
        const int32_t start = ud->batch_starts[batch];
        if (expert < 0 || expert >= ud->n_experts || start < 0 || start >= ud->max_assignments) {
            continue;
        }
        const int32_t end = ud->expert_offsets[expert + 1];
        const int rows = static_cast<int>(std::min<int32_t>(kGemma4GateUpPrefillRowsPerBatch, end - start));
        if (rows <= 0) {
            continue;
        }
        const char* expert_base =
            weight_base + static_cast<size_t>(expert) * static_cast<size_t>(down_exps->nb[2]);
        for (int r = 0; r < rows; ++r) {
            const int32_t token = ud->assignment_tokens[start + r];
            const int32_t slot = ud->assignment_slots[start + r];
            if (token < 0 || token >= ud->n_tokens || slot < 0 || slot >= ud->top_k) {
                continue;
            }
            const float* src = reinterpret_cast<const float*>(hidden_base + static_cast<size_t>(slot) * hidden->nb[1] +
                                                              static_cast<size_t>(token) * hidden->nb[2]);
            float* out = reinterpret_cast<float*>(dst_base + static_cast<size_t>(slot) * dst->nb[1] +
                                                  static_cast<size_t>(token) * dst->nb[2]);
            if (use_vec_dot) {
                input_traits->from_float(src, qrow.data(), ud->intermediate_dim);
            }
            for (int64_t row = 0; row < ud->hidden_dim; ++row) {
                const void* down_row =
                    expert_base + static_cast<size_t>(row) * static_cast<size_t>(down_exps->nb[1]);
                if (use_vec_dot) {
                    if (!Gemma4QuantizedRowDot(weight_type, down_row, qrow.data(), ud->intermediate_dim, out + row)) {
                        MarkGemma4NativeMoEPrefillFailure(ud, "down_quant_row_dot_failed", debug_label, ith);
                        return false;
                    }
                } else if (!Gemma4DequantizedF32RowDot(weight_type, down_row, src, ud->intermediate_dim, out + row)) {
                    MarkGemma4NativeMoEPrefillFailure(ud, "down_quant_dequant_row_dot_failed", debug_label, ith);
                    return false;
                }
            }
            if (ith == 0 && batch == 0 && r == 0 && IsDebugGemma4NativeMoEPrefillEnabled()) {
                std::fprintf(stderr,
                             "[Gemma4NativeMoEPrefillDebug] %s_first token=%d slot=%d expert=%d out0=%g out1=%g\n",
                             debug_label ? debug_label : "down_quant", token, slot, expert, out[0], out[1]);
            }
        }
    }
    return true;
}

static bool RunGemma4DownQ4KPrefill(ggml_tensor* dst, const ggml_tensor* down_exps, const ggml_tensor* hidden,
                                    const ggml_tensor* selected_experts, int ith, int nth,
                                    Gemma4GateUpQ4KPrefillUserData* ud) {
    if (!dst || !down_exps || !hidden || !selected_experts || !ud || !dst->data || !down_exps->data ||
        !hidden->data || nth <= 0) {
        MarkGemma4NativeMoEPrefillFailure(ud, "down_q4k_missing_runtime_data", "down_q4k", ith);
        return false;
    }
    if (dst->type != GGML_TYPE_F32 || down_exps->type != GGML_TYPE_Q4_K || hidden->type != GGML_TYPE_F32 ||
        selected_experts->type != GGML_TYPE_I32 || dst->ne[0] != ud->hidden_dim || dst->ne[1] != ud->top_k ||
        dst->ne[2] != ud->n_tokens || hidden->ne[0] != ud->intermediate_dim || hidden->ne[1] != ud->top_k ||
        hidden->ne[2] != ud->n_tokens || down_exps->ne[0] != ud->intermediate_dim ||
        down_exps->ne[1] != ud->hidden_dim || down_exps->ne[2] != ud->n_experts ||
        (ud->intermediate_dim % QK_K) != 0 || (ud->hidden_dim % 8) != 0 ||
        dst->nb[0] != static_cast<int64_t>(sizeof(float))) {
        MarkGemma4NativeMoEPrefillFailure(ud, "down_q4k_shape_or_type_mismatch", "down_q4k", ith);
        return false;
    }
    if (ith == 0) {
        ZeroGemma4PrefillF32Tensor(dst);
    }
    if (!PrepareGemma4GateUpQ4KPrefillBatches(ud, hidden, selected_experts, ith, nth)) {
        if (ith == 0 && IsDebugGemma4NativeMoEPrefillEnabled()) {
            std::fprintf(stderr, "[Gemma4NativeMoEPrefillDebug] down_q4k prepare_failed\n");
        }
        MarkGemma4NativeMoEPrefillFailure(ud, "down_q4k_prepare_failed", "down_q4k", ith);
        return false;
    }
    if (ith == 0 && IsDebugGemma4NativeMoEPrefillEnabled()) {
        static std::atomic<int> down_q4k_debug_count{0};
        const int debug_idx = down_q4k_debug_count.fetch_add(1, std::memory_order_relaxed);
        if (debug_idx < 96) {
            std::fprintf(stderr,
                         "[Gemma4NativeMoEPrefillDebug] down_q4k_execute[%d] dst=%s tokens=%lld top_k=%lld "
                         "total_batches=%lld nth=%d\n",
                         debug_idx, dst->name, static_cast<long long>(ud->n_tokens),
                         static_cast<long long>(ud->top_k),
                         static_cast<long long>(ud->total_batches.load(std::memory_order_acquire)), nth);
        }
    }

    const int tile_count = static_cast<int>(ud->hidden_dim / 8);
    const int64_t total_batches = ud->total_batches.load(std::memory_order_acquire);
    const char* weight_base = static_cast<const char*>(down_exps->data);
    const char* hidden_base = static_cast<const char*>(hidden->data);
    char* dst_base = static_cast<char*>(dst->data);
    const size_t q4_row_bytes = ggml_row_size(GGML_TYPE_Q4_K, ud->intermediate_dim);
    if (q4_row_bytes == 0 || static_cast<size_t>(down_exps->nb[1]) < q4_row_bytes ||
        static_cast<size_t>(down_exps->nb[2]) <
            static_cast<size_t>(ud->hidden_dim - 1) * static_cast<size_t>(down_exps->nb[1]) + q4_row_bytes) {
        MarkGemma4NativeMoEPrefillFailure(ud, "down_q4k_row_stride_mismatch", "down_q4k", ith);
        return false;
    }
    const auto* q8_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    const auto* q4_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K);
    if (!q8_traits || !q8_traits->from_float || !q4_traits || q4_traits->vec_dot_type != GGML_TYPE_Q8_K) {
        MarkGemma4NativeMoEPrefillFailure(ud, "down_q4k_q8k_traits_unavailable", "down_q4k", ith);
        return false;
    }

    thread_local std::vector<uint8_t> q8_tail_buf;
    q8_tail_buf.resize(ggml_row_size(GGML_TYPE_Q8_K, ud->intermediate_dim));

    for (;;) {
        const int64_t batch = ud->next_batch.fetch_add(1, std::memory_order_relaxed);
        if (batch >= total_batches) {
            break;
        }
        const int32_t expert = ud->batch_experts[batch];
        const int32_t start = ud->batch_starts[batch];
        if (expert < 0 || expert >= ud->n_experts || start < 0 || start >= ud->max_assignments) {
            continue;
        }
        const int32_t end = ud->expert_offsets[expert + 1];
        const int rows = static_cast<int>(std::min<int32_t>(kGemma4GateUpPrefillRowsPerBatch, end - start));
        if (rows <= 0) {
            continue;
        }
        const char* expert_base =
            weight_base + static_cast<size_t>(expert) * static_cast<size_t>(down_exps->nb[2]);

        std::array<float, 8> down_tail{};
        for (int r = 0; r < rows; ++r) {
            const int32_t token = ud->assignment_tokens[start + r];
            const int32_t slot = ud->assignment_slots[start + r];
            if (token < 0 || token >= ud->n_tokens || slot < 0 || slot >= ud->top_k) {
                continue;
            }
            const float* src = reinterpret_cast<const float*>(hidden_base + static_cast<size_t>(slot) * hidden->nb[1] +
                                                              static_cast<size_t>(token) * hidden->nb[2]);
            q8_traits->from_float(src, q8_tail_buf.data(), ud->intermediate_dim);
            for (int tile = 0; tile < tile_count; ++tile) {
                for (int lane = 0; lane < 8; ++lane) {
                    const int64_t row = static_cast<int64_t>(tile) * 8 + lane;
                    const void* down_row =
                        expert_base + static_cast<size_t>(row) * static_cast<size_t>(down_exps->nb[1]);
                    if (!Gemma4QuantizedRowDot(GGML_TYPE_Q4_K, down_row, q8_tail_buf.data(), ud->intermediate_dim,
                                               &down_tail[lane])) {
                        MarkGemma4NativeMoEPrefillFailure(ud, "down_q4k_row_dot_failed", "down_q4k", ith);
                        return false;
                    }
                }
                float* out = reinterpret_cast<float*>(dst_base + static_cast<size_t>(tile) * 8 * dst->nb[0] +
                                                      static_cast<size_t>(slot) * dst->nb[1] +
                                                      static_cast<size_t>(token) * dst->nb[2]);
                std::memcpy(out, down_tail.data(), sizeof(float) * 8);
                if (ith == 0 && batch == 0 && r == 0 && tile == 0 && IsDebugGemma4NativeMoEPrefillEnabled()) {
                    std::fprintf(stderr,
                                 "[Gemma4NativeMoEPrefillDebug] down_q4k_first token=%d slot=%d expert=%d out0=%g "
                                 "out1=%g\n",
                                 token, slot, expert, out[0], out[1]);
                }
            }
        }
    }
    return true;
}

static void cb_gemma4_down_q4k_prefill(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    const auto begin = std::chrono::steady_clock::now();
    auto* ud = static_cast<Gemma4GateUpQ4KPrefillUserData*>(userdata);
    const bool ok = RunGemma4DownQ4KPrefill(dst, dst ? dst->src[0] : nullptr, dst ? dst->src[1] : nullptr,
                                            dst ? dst->src[2] : nullptr, ith, nth, ud);
    const uint64_t ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count());
    RecordGemma4NativeMoEPrefillTiming(GetCurrentWorkContext(), 0, ns, ns);
    if (ud) {
        if (ok && !ud->used_recorded.exchange(true, std::memory_order_relaxed)) {
            RecordGemma4NativeMoEPrefillDecision(GetCurrentWorkContext(), /*candidate=*/false, /*used=*/true, nullptr,
                                                 /*replaced_mul_mat_id_ops=*/2, false);
        } else if (!ok) {
            RecordGemma4NativeMoEPrefillDecision(GetCurrentWorkContext(), /*candidate=*/false, /*used=*/false,
                                                 ud->first_failure_reason.load(std::memory_order_relaxed), 0, false);
        }
    }
}

static bool RunGemma4DownQ8_0Prefill(ggml_tensor* dst, const ggml_tensor* down_exps, const ggml_tensor* hidden,
                                     const ggml_tensor* selected_experts, int ith, int nth,
                                     Gemma4GateUpQ4KPrefillUserData* ud) {
    if (!dst || !down_exps || !hidden || !selected_experts || !ud || !dst->data || !down_exps->data ||
        !hidden->data || nth <= 0) {
        MarkGemma4NativeMoEPrefillFailure(ud, "down_q8_0_missing_runtime_data", "down_q8_0", ith);
        return false;
    }
    if (dst->type != GGML_TYPE_F32 || down_exps->type != GGML_TYPE_Q8_0 || hidden->type != GGML_TYPE_F32 ||
        selected_experts->type != GGML_TYPE_I32 || dst->ne[0] != ud->hidden_dim || dst->ne[1] != ud->top_k ||
        dst->ne[2] != ud->n_tokens || hidden->ne[0] != ud->intermediate_dim || hidden->ne[1] != ud->top_k ||
        hidden->ne[2] != ud->n_tokens || down_exps->ne[0] != ud->intermediate_dim ||
        down_exps->ne[1] != ud->hidden_dim || down_exps->ne[2] != ud->n_experts ||
        (ud->intermediate_dim % QK8_0) != 0 || (ud->hidden_dim % 4) != 0 ||
        dst->nb[0] != static_cast<int64_t>(sizeof(float))) {
        MarkGemma4NativeMoEPrefillFailure(ud, "down_q8_0_shape_or_type_mismatch", "down_q8_0", ith);
        return false;
    }
    if (ith == 0) {
        ZeroGemma4PrefillF32Tensor(dst);
    }
    if (!PrepareGemma4GateUpQ4KPrefillBatches(ud, hidden, selected_experts, ith, nth)) {
        if (ith == 0 && IsDebugGemma4NativeMoEPrefillEnabled()) {
            std::fprintf(stderr, "[Gemma4NativeMoEPrefillDebug] down_q8_0 prepare_failed\n");
        }
        MarkGemma4NativeMoEPrefillFailure(ud, "down_q8_0_prepare_failed", "down_q8_0", ith);
        return false;
    }
    if (ith == 0 && IsDebugGemma4NativeMoEPrefillEnabled()) {
        static std::atomic<int> down_q8_debug_count{0};
        const int debug_idx = down_q8_debug_count.fetch_add(1, std::memory_order_relaxed);
        if (debug_idx < 96) {
            std::fprintf(stderr,
                         "[Gemma4NativeMoEPrefillDebug] down_q8_0_execute[%d] dst=%s tokens=%lld top_k=%lld "
                         "total_batches=%lld nth=%d\n",
                         debug_idx, dst->name, static_cast<long long>(ud->n_tokens),
                         static_cast<long long>(ud->top_k),
                         static_cast<long long>(ud->total_batches.load(std::memory_order_acquire)), nth);
        }
    }

    const size_t q8_row_bytes = ggml_row_size(GGML_TYPE_Q8_0, ud->intermediate_dim);
    if (q8_row_bytes == 0 || static_cast<size_t>(down_exps->nb[1]) < q8_row_bytes ||
        static_cast<size_t>(down_exps->nb[2]) <
            static_cast<size_t>(ud->hidden_dim - 1) * static_cast<size_t>(down_exps->nb[1]) + q8_row_bytes) {
        MarkGemma4NativeMoEPrefillFailure(ud, "down_q8_0_row_stride_mismatch", "down_q8_0", ith);
        return false;
    }
    const int64_t total_batches = ud->total_batches.load(std::memory_order_acquire);
    const char* weight_base = static_cast<const char*>(down_exps->data);
    const char* hidden_base = static_cast<const char*>(hidden->data);
    char* dst_base = static_cast<char*>(dst->data);
    const auto* weight_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    if (!weight_traits || !weight_traits->vec_dot) {
        MarkGemma4NativeMoEPrefillFailure(ud, "down_q8_0_weight_traits_unavailable", "down_q8_0", ith);
        return false;
    }
    const ggml_type input_quant_type = weight_traits->vec_dot_type;
    const auto* input_traits = ggml_get_type_traits_cpu(input_quant_type);
    if (!input_traits || !input_traits->from_float) {
        MarkGemma4NativeMoEPrefillFailure(ud, "down_q8_0_input_traits_unavailable", "down_q8_0", ith);
        return false;
    }

    thread_local std::vector<uint8_t> q8_row;
    q8_row.resize(ggml_row_size(input_quant_type, ud->intermediate_dim));

    for (;;) {
        const int64_t batch = ud->next_batch.fetch_add(1, std::memory_order_relaxed);
        if (batch >= total_batches) {
            break;
        }
        const int32_t expert = ud->batch_experts[batch];
        const int32_t start = ud->batch_starts[batch];
        if (expert < 0 || expert >= ud->n_experts || start < 0 || start >= ud->max_assignments) {
            continue;
        }
        const int32_t end = ud->expert_offsets[expert + 1];
        const int rows = static_cast<int>(std::min<int32_t>(kGemma4GateUpPrefillRowsPerBatch, end - start));
        if (rows <= 0) {
            continue;
        }
        const char* expert_base =
            weight_base + static_cast<size_t>(expert) * static_cast<size_t>(down_exps->nb[2]);
        for (int r = 0; r < rows; ++r) {
            const int32_t token = ud->assignment_tokens[start + r];
            const int32_t slot = ud->assignment_slots[start + r];
            if (token < 0 || token >= ud->n_tokens || slot < 0 || slot >= ud->top_k) {
                continue;
            }
            const float* src = reinterpret_cast<const float*>(hidden_base + static_cast<size_t>(slot) * hidden->nb[1] +
                                                              static_cast<size_t>(token) * hidden->nb[2]);
            input_traits->from_float(src, q8_row.data(), ud->intermediate_dim);
            float* out = reinterpret_cast<float*>(dst_base + static_cast<size_t>(slot) * dst->nb[1] +
                                                  static_cast<size_t>(token) * dst->nb[2]);
            for (int64_t row = 0; row < ud->hidden_dim; ++row) {
                const void* down_row =
                    expert_base + static_cast<size_t>(row) * static_cast<size_t>(down_exps->nb[1]);
                if (!Gemma4QuantizedRowDot(GGML_TYPE_Q8_0, down_row, q8_row.data(), ud->intermediate_dim, out + row)) {
                    MarkGemma4NativeMoEPrefillFailure(ud, "down_q8_0_row_dot_failed", "down_q8_0", ith);
                    return false;
                }
            }
            if (ith == 0 && batch == 0 && r == 0 && IsDebugGemma4NativeMoEPrefillEnabled()) {
                std::fprintf(stderr,
                             "[Gemma4NativeMoEPrefillDebug] down_q8_0_first token=%d slot=%d expert=%d out0=%g "
                             "out1=%g\n",
                             token, slot, expert, out[0], out[1]);
            }
        }
    }
    return true;
}

static void cb_gemma4_down_q8_0_prefill(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    const auto begin = std::chrono::steady_clock::now();
    auto* ud = static_cast<Gemma4GateUpQ4KPrefillUserData*>(userdata);
    const bool ok = RunGemma4DownQ8_0Prefill(dst, dst ? dst->src[0] : nullptr, dst ? dst->src[1] : nullptr,
                                             dst ? dst->src[2] : nullptr, ith, nth, ud);
    const uint64_t ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count());
    RecordGemma4NativeMoEPrefillTiming(GetCurrentWorkContext(), 0, ns, ns);
    if (ud && ok && !ud->used_recorded.exchange(true, std::memory_order_relaxed)) {
        RecordGemma4NativeMoEPrefillDecision(GetCurrentWorkContext(), /*candidate=*/false, /*used=*/true, nullptr,
                                             /*replaced_mul_mat_id_ops=*/2, false);
    } else if (ud && !ok) {
        RecordGemma4NativeMoEPrefillDecision(GetCurrentWorkContext(), /*candidate=*/false, /*used=*/false,
                                             ud->first_failure_reason.load(std::memory_order_relaxed), 0, false);
    }
}

static void cb_gemma4_down_q5_1_prefill(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    const auto begin = std::chrono::steady_clock::now();
    auto* ud = static_cast<Gemma4GateUpQ4KPrefillUserData*>(userdata);
    const bool ok = RunGemma4DownQuantPrefill(dst, dst ? dst->src[0] : nullptr, dst ? dst->src[1] : nullptr,
                                              dst ? dst->src[2] : nullptr, ith, nth, ud, GGML_TYPE_Q5_1,
                                              "down_q5_1");
    const uint64_t ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count());
    RecordGemma4NativeMoEPrefillTiming(GetCurrentWorkContext(), 0, ns, ns);
    if (ud && ok && !ud->used_recorded.exchange(true, std::memory_order_relaxed)) {
        RecordGemma4NativeMoEPrefillDecision(GetCurrentWorkContext(), /*candidate=*/false, /*used=*/true, nullptr,
                                             /*replaced_mul_mat_id_ops=*/2, false);
    } else if (ud && !ok) {
        RecordGemma4NativeMoEPrefillDecision(GetCurrentWorkContext(), /*candidate=*/false, /*used=*/false,
                                             ud->first_failure_reason.load(std::memory_order_relaxed), 0, false);
    }
}

static bool CanUseGemma4GateUpQ4KFusedGEGLU(const TransformerModel* model, const ggml_tensor* gate_up_exps,
                                            const ggml_tensor* input, const ggml_tensor* selected_experts,
                                            int64_t intermediate_dim, int64_t n_experts) {
    if (!model || !gate_up_exps || !input || !selected_experts || !model->arch_flags.is_gemma4) return false;
    const InferenceExecutionPhase phase = GetCurrentExecutionPhase();
    if (phase != InferenceExecutionPhase::Prefill) return false;
    const BatchSpec* current_batch = GetCurrentBatch();
    if (current_batch && !current_batch->lora_map.empty()) return false;
    if (gate_up_exps->type != GGML_TYPE_Q4_K || input->type != GGML_TYPE_F32 || selected_experts->type != GGML_TYPE_I32) {
        return false;
    }
    const int64_t n_tokens = selected_experts->ne[1];
    const int64_t top_k = selected_experts->ne[0];
    if (n_tokens <= 0 || top_k <= 0 || !Gemma4PrefillInputShapeOk(input, gate_up_exps->ne[0], top_k, n_tokens) ||
        gate_up_exps->ne[1] != 2 * intermediate_dim || gate_up_exps->ne[2] != n_experts ||
        selected_experts->ne[0] <= 0) {
        return false;
    }
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_K, input->ne[0]);
    if (row_bytes == 0 || gate_up_exps->nb[1] < static_cast<int64_t>(row_bytes)) return false;
    if ((input->ne[0] % QK_K) != 0 || (intermediate_dim % 8) != 0) return false;
    return Gemma4NativeMoEPrefillKernelSupported();
}

static bool CanUseGemma4DownQuantPrefill(const TransformerModel* model, const ggml_tensor* down_exps,
                                         const ggml_tensor* hidden, const ggml_tensor* selected_experts,
                                         int64_t hidden_dim, int64_t intermediate_dim, int64_t n_experts,
                                         ggml_type weight_type) {
    if (!model || !down_exps || !selected_experts || !model->arch_flags.is_gemma4) return false;
    if (GetCurrentExecutionPhase() != InferenceExecutionPhase::Prefill) return false;
    const BatchSpec* current_batch = GetCurrentBatch();
    if (current_batch && !current_batch->lora_map.empty()) return false;
    if (down_exps->type != weight_type || selected_experts->type != GGML_TYPE_I32 ||
        (hidden && hidden->type != GGML_TYPE_F32)) {
        return false;
    }
    if (down_exps->ne[0] != intermediate_dim || down_exps->ne[1] != hidden_dim ||
        down_exps->ne[2] != n_experts || selected_experts->ne[0] <= 0 || selected_experts->ne[1] <= 0) {
        return false;
    }
    if (hidden && (hidden->ne[1] <= 0 || hidden->ne[2] <= 0 || hidden->ne[0] != intermediate_dim ||
                   selected_experts->ne[1] != hidden->ne[2] || selected_experts->ne[0] != hidden->ne[1])) {
        return false;
    }
    ggml_type input_quant_type = GGML_TYPE_COUNT;
    size_t input_quant_row_bytes = 0;
    if (!Gemma4QuantizedRowDotInputType(weight_type, intermediate_dim, &input_quant_type, &input_quant_row_bytes) &&
        !Gemma4CanUseDequantizedF32RowDot(weight_type, intermediate_dim)) {
        return false;
    }
    (void)input_quant_type;
    (void)input_quant_row_bytes;
    const size_t row_bytes = ggml_row_size(weight_type, intermediate_dim);
    if (row_bytes == 0 || down_exps->nb[1] < static_cast<int64_t>(row_bytes)) return false;
    return Gemma4NativeMoEPrefillKernelSupported();
}

static bool CanUseGemma4DownQ4KPrefill(const TransformerModel* model, const ggml_tensor* down_exps,
                                       const ggml_tensor* hidden, const ggml_tensor* selected_experts,
                                       int64_t hidden_dim, int64_t intermediate_dim, int64_t n_experts) {
    if (!model || !down_exps || !selected_experts || !model->arch_flags.is_gemma4) return false;
    if (GetCurrentExecutionPhase() != InferenceExecutionPhase::Prefill) return false;
    const BatchSpec* current_batch = GetCurrentBatch();
    if (current_batch && !current_batch->lora_map.empty()) return false;
    if (down_exps->type != GGML_TYPE_Q4_K || selected_experts->type != GGML_TYPE_I32 ||
        (hidden && hidden->type != GGML_TYPE_F32)) {
        return false;
    }
    if (down_exps->ne[0] != intermediate_dim || down_exps->ne[1] != hidden_dim ||
        down_exps->ne[2] != n_experts || selected_experts->ne[0] <= 0 || selected_experts->ne[1] <= 0) {
        return false;
    }
    if (hidden && (hidden->ne[1] <= 0 || hidden->ne[2] <= 0 || hidden->ne[0] != intermediate_dim ||
                   selected_experts->ne[1] != hidden->ne[2] || selected_experts->ne[0] != hidden->ne[1])) {
        return false;
    }
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_K, intermediate_dim);
    if (row_bytes == 0 || down_exps->nb[1] < static_cast<int64_t>(row_bytes)) return false;
    if ((intermediate_dim % QK_K) != 0 || (hidden_dim % 8) != 0) return false;
    return Gemma4NativeMoEPrefillKernelSupported();
}

static bool CanUseGemma4DownQ8_0Prefill(const TransformerModel* model, const ggml_tensor* down_exps,
                                        const ggml_tensor* hidden, const ggml_tensor* selected_experts,
                                        int64_t hidden_dim, int64_t intermediate_dim, int64_t n_experts) {
    if (!model || !down_exps || !selected_experts || !model->arch_flags.is_gemma4) return false;
    if (GetCurrentExecutionPhase() != InferenceExecutionPhase::Prefill) return false;
    const BatchSpec* current_batch = GetCurrentBatch();
    if (current_batch && !current_batch->lora_map.empty()) return false;
    if (down_exps->type != GGML_TYPE_Q8_0 || selected_experts->type != GGML_TYPE_I32 ||
        (hidden && hidden->type != GGML_TYPE_F32)) {
        return false;
    }
    if (down_exps->ne[0] != intermediate_dim || down_exps->ne[1] != hidden_dim ||
        down_exps->ne[2] != n_experts || selected_experts->ne[0] <= 0 || selected_experts->ne[1] <= 0) {
        return false;
    }
    if (hidden && (hidden->ne[1] <= 0 || hidden->ne[2] <= 0 || hidden->ne[0] != intermediate_dim ||
                   selected_experts->ne[1] != hidden->ne[2] || selected_experts->ne[0] != hidden->ne[1])) {
        return false;
    }
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q8_0, intermediate_dim);
    if (row_bytes == 0 || down_exps->nb[1] != row_bytes) return false;
    if ((intermediate_dim % QK8_0) != 0 || (hidden_dim % 4) != 0) return false;
    return Gemma4NativeMoEPrefillKernelSupported();
}

static bool CanUseGemma4DownNativePrefill(const TransformerModel* model, const ggml_tensor* down_exps,
                                          const ggml_tensor* hidden, const ggml_tensor* selected_experts,
                                          int64_t hidden_dim, int64_t intermediate_dim, int64_t n_experts) {
    return CanUseGemma4DownQ4KPrefill(model, down_exps, hidden, selected_experts, hidden_dim, intermediate_dim,
                                      n_experts) ||
           CanUseGemma4DownQ8_0Prefill(model, down_exps, hidden, selected_experts, hidden_dim, intermediate_dim,
                                       n_experts) ||
           CanUseGemma4DownQuantPrefill(model, down_exps, hidden, selected_experts, hidden_dim, intermediate_dim,
                                        n_experts, GGML_TYPE_Q5_1);
}

static bool CanUseGemma4DownWeightedSumDecode(const TransformerModel* model, const ggml_tensor* down_exps,
                                              const ggml_tensor* hidden, const ggml_tensor* selected_experts,
                                              const ggml_tensor* weights, const char** reject_reason) {
    auto reject = [&](const char* reason) {
        if (reject_reason) {
            *reject_reason = reason;
        }
        return false;
    };
    if (reject_reason) {
        *reject_reason = nullptr;
    }
    if (!model || !down_exps || !hidden || !selected_experts || !weights) return reject("null_ptr");
    if (!model->arch_flags.is_gemma4) return reject("wrong_variant");
    const BatchSpec* current_batch = GetCurrentBatch();
    if (current_batch && !current_batch->lora_map.empty()) return reject("dynamic_lora");
    if (down_exps->type != GGML_TYPE_Q8_0) return reject("unsupported_down_quant");
    if (hidden->type != GGML_TYPE_F32 || selected_experts->type != GGML_TYPE_I32 ||
        weights->type != GGML_TYPE_F32) {
        return reject("bad_tensor_type");
    }
    if (selected_experts->ne[0] <= 0 || selected_experts->ne[1] != 1 ||
        selected_experts->ne[1] > NativeMoEFastPathMaxDirectTokens(model)) {
        return reject("unsupported_token_count");
    }
    if (hidden->ne[1] != selected_experts->ne[0] || hidden->ne[2] != selected_experts->ne[1]) {
        return reject("hidden_shape");
    }
    if (weights->ne[0] != 1 || weights->ne[1] != selected_experts->ne[0] ||
        weights->ne[2] != selected_experts->ne[1]) {
        return reject("weights_shape");
    }
    if (down_exps->ne[0] != hidden->ne[0] || down_exps->ne[2] <= 0) return reject("down_shape");
    const int64_t q8_block = ggml_blck_size(GGML_TYPE_Q8_0);
    if (q8_block <= 0 || (down_exps->ne[0] % q8_block) != 0) return reject("bad_q8_0_alignment");
    const size_t q8_row_bytes = ggml_row_size(GGML_TYPE_Q8_0, down_exps->ne[0]);
    if (q8_row_bytes == 0 || static_cast<size_t>(down_exps->nb[1]) != q8_row_bytes) {
        return reject("bad_q8_0_stride");
    }
    return true;
}

ggml_tensor* TryBuildGemma4NativeMoEGraph(ggml_context* ctx, ggml_cgraph* gf, TransformerModel* model,
                                          TransformerLayer* layer, int layer_idx, ggml_tensor* routed_input,
                                          ggml_tensor* gate_logits, int top_k) {
    if (!ctx || !gf || !model || !layer || !routed_input || !gate_logits || !model->arch_flags.is_gemma4 ||
        !IsGemma4NativeMoEGraphEnabled()) {
        return nullptr;
    }
    Gemma4PackedMoERoots roots;
    if (!ResolveGemma4PackedMoERoots(layer, &roots)) {
        return nullptr;
    }
    if (GetCurrentExecutionPhase() != InferenceExecutionPhase::Prefill) {
        ggml_tensor* gate_up_repack = UseCpuRepackAliasIfAvailable(model, roots.gate_up);
        if (!gate_up_repack || gate_up_repack->type == roots.gate_up->type) {
            roots.gate_up = gate_up_repack;
        }
    }
    std::string repack_reason;
    if (!densecore::gemma4::InferPackedExpertLayout(roots.gate_up, roots.down, &roots.layout, &repack_reason)) {
        if (IsMoEWiringDebugEnabled()) {
            std::fprintf(stderr, "[Gemma4NativeMoE] CPU_REPACK alias rejected: %s\n", repack_reason.c_str());
        }
        return nullptr;
    }
    const int64_t n_tokens = routed_input->ne[1];
    const int64_t n_embd = roots.layout.hidden_dim;
    const int64_t n_expert_used = std::max<int64_t>(1, std::min<int64_t>(top_k, roots.layout.num_experts));
    if (n_tokens <= 0 || routed_input->ne[0] != n_embd || gate_logits->ne[0] != roots.layout.num_experts ||
        gate_logits->ne[1] != n_tokens) {
        return nullptr;
    }
    ggml_tensor* gate_up_exps = BuildGemma4PackedGateUpMerged3DView(ctx, roots);
    if (gate_up_exps && gate_up_exps->view_src && gate_up_exps->view_src->buffer && !gate_up_exps->buffer) {
        ggml_backend_view_init(gate_up_exps);
    }
    ggml_tensor* gate_exps = nullptr;
    ggml_tensor* up_exps = nullptr;
    if (!gate_up_exps) {
        gate_exps = BuildGemma4PackedGateOrUp3DView(ctx, roots, /*up_projection=*/false);
        up_exps = BuildGemma4PackedGateOrUp3DView(ctx, roots, /*up_projection=*/true);
        if (gate_exps && gate_exps->view_src && gate_exps->view_src->buffer && !gate_exps->buffer) {
            ggml_backend_view_init(gate_exps);
        }
        if (up_exps && up_exps->view_src && up_exps->view_src->buffer && !up_exps->buffer) {
            ggml_backend_view_init(up_exps);
        }
    }
    ggml_tensor* down_exps = BuildGemma4PackedDown3DView(ctx, roots);
    if (down_exps && down_exps->view_src && down_exps->view_src->buffer && !down_exps->buffer) {
        ggml_backend_view_init(down_exps);
    }
    ggml_tensor* scale_rows = BuildGemma4PackedDownScaleRows(ctx, roots);
    if ((!gate_up_exps && (!gate_exps || !up_exps)) || !down_exps || !scale_rows) {
        return nullptr;
    }

    ggml_tensor* selected_experts = ggml_argsort_top_k(ctx, gate_logits, static_cast<int>(n_expert_used));
    ggml_set_name(selected_experts, "gemma4_native_moe_topk");
    ggml_tensor* weights = BuildMoETopKWeightsFromLogits(ctx, gate_logits, selected_experts,
                                                         "gemma4_native_moe_topk_weights");
    if (!weights) {
        return nullptr;
    }

    scale_rows = ggml_repeat_4d(ctx, scale_rows, 1, roots.layout.num_experts, n_tokens, 1);
    ggml_tensor* selected_scales = ggml_get_rows(ctx, scale_rows, selected_experts);
    weights = ggml_mul(ctx, weights, selected_scales);
    ggml_set_name(weights, "gemma4_native_moe_weights");
    ggml_build_forward_expand(gf, weights);

    ggml_tensor* hidden = nullptr;
    // Do not build the full native gate/up+down prefill graph yet. The C4
    // shape-correct run passed QA but regressed prefill because the native
    // down-projection still performs per-row scalar dot work. Keep the proven
    // gate/up custom op below and leave down/weighted-sum on the maintained
    // graph path until a batched down kernel is available.

    ggml_tensor* cur3 = ggml_reshape_3d(ctx, routed_input, n_embd, 1, n_tokens);
    ggml_tensor* gate = nullptr;
    ggml_tensor* up = nullptr;
    if (gate_up_exps) {
        if (CanUseGemma4GateUpQ4KFusedGEGLU(model, gate_up_exps, routed_input, selected_experts,
                                            roots.layout.intermediate_dim, roots.layout.num_experts)) {
            Gemma4GateUpQ4KPrefillUserData* gateup_ud = AllocateGemma4GateUpQ4KPrefillUserData(
                ctx, n_embd, roots.layout.intermediate_dim, n_expert_used, n_tokens, roots.layout.num_experts);
            if (gateup_ud) {
                ggml_tensor* args[] = {gate_up_exps, routed_input, selected_experts};
                hidden = ggml_custom_4d(ctx, GGML_TYPE_F32, roots.layout.intermediate_dim, n_expert_used, n_tokens, 1,
                                        args, 3, cb_gemma4_gateup_q4k_prefill_geglu, GGML_N_TASKS_MAX, gateup_ud);
                ggml_set_name(hidden, "gemma4_native_moe_gateup_q4k_prefill_geglu");
            }
        }
        if (!hidden) {
            ggml_tensor* gate_up = ggml_mul_mat_id(ctx, gate_up_exps, cur3, selected_experts);
            ggml_set_name(gate_up, "gemma4_native_moe_gate_up");
            gate = ggml_view_3d(ctx, gate_up, roots.layout.intermediate_dim, n_expert_used, n_tokens, gate_up->nb[1],
                                gate_up->nb[2], 0);
            ggml_set_name(gate, "gemma4_native_moe_gate");
            up = ggml_view_3d(ctx, gate_up, roots.layout.intermediate_dim, n_expert_used, n_tokens, gate_up->nb[1],
                              gate_up->nb[2], static_cast<size_t>(roots.layout.intermediate_dim) *
                                                  static_cast<size_t>(gate_up->nb[0]));
            ggml_set_name(up, "gemma4_native_moe_up");
        }
    } else {
        up = ggml_mul_mat_id(ctx, up_exps, cur3, selected_experts);
        ggml_set_name(up, "gemma4_native_moe_up");
        gate = ggml_mul_mat_id(ctx, gate_exps, cur3, selected_experts);
        ggml_set_name(gate, "gemma4_native_moe_gate");
    }
    if (!hidden) {
        hidden = ggml_geglu_split(ctx, gate, up);
        ggml_set_name(hidden, "gemma4_native_moe_geglu");
    }
    const char* weighted_down_reject = nullptr;
    if (CanUseGemma4DownWeightedSumDecode(model, down_exps, hidden, selected_experts, weights,
                                          &weighted_down_reject)) {
        Qwen35SharedQ8RowsUserData* hidden_q8_ud =
            AllocateQwen35SharedQ8RowsUserData(ctx, hidden, n_expert_used * n_tokens);
        if (hidden_q8_ud) {
            const int native_moe_callback_tasks = ResolveNativeMoEGraphCallbackTaskCount(
                model, GetCurrentBatch(), GetCurrentExecutionPhase(), n_tokens, static_cast<int>(n_expert_used));
            SetNativeMoECallbackRequestedTaskCount(hidden_q8_ud, native_moe_callback_tasks);
            ggml_tensor* down_args[] = {down_exps, hidden, selected_experts, weights};
            ggml_tensor* out = ggml_custom_4d(ctx, GGML_TYPE_F32, n_embd, n_tokens, 1, 1, down_args, 4,
                                              cb_gemma4_native_moe_down_weighted_sum, native_moe_callback_tasks,
                                              hidden_q8_ud);
            char native_name[80];
            std::snprintf(native_name, sizeof(native_name), "blk.%d.gemma4_native_moe_down_weighted_sum", layer_idx);
            ggml_set_name(out, native_name);
            return out;
        }
        weighted_down_reject = "userdata_allocation_failed";
    }
    if (n_tokens == 1) {
        RecordGemma4DecodeNativeDecision(GetCurrentWorkContext(), /*candidate=*/true, /*used=*/false,
                                         weighted_down_reject ? weighted_down_reject : "unsupported_down_weighted_sum",
                                         /*moe_used=*/true, /*dense_used=*/false, /*lm_head_used=*/false, 0,
                                         /*replaced_mul_mat_ops=*/0, /*replaced_mul_mat_id_ops=*/0,
                                         /*duplicate_work_detected=*/false);
    }
    ggml_tensor* experts = ggml_mul_mat_id(ctx, down_exps, hidden, selected_experts);
    experts = ggml_mul(ctx, experts, weights);
    ggml_set_name(experts, "gemma4_native_moe_weighted_down");

    ggml_tensor* expert_views[32] = {nullptr};
    if (n_expert_used > static_cast<int64_t>(std::size(expert_views))) {
        return nullptr;
    }
    for (int64_t i = 0; i < n_expert_used; ++i) {
        expert_views[i] = ggml_view_2d(ctx, experts, n_embd, n_tokens, experts->nb[2],
                                       static_cast<size_t>(i) * static_cast<size_t>(experts->nb[1]));
        ggml_build_forward_expand(gf, expert_views[i]);
    }
    ggml_tensor* out = expert_views[0];
    for (int64_t i = 1; i < n_expert_used; ++i) {
        out = ggml_add(ctx, out, expert_views[i]);
    }
    if (n_expert_used == 1) {
        out = ggml_cont(ctx, out);
    }
    char name[80];
    std::snprintf(name, sizeof(name), "blk.%d.gemma4_native_moe_out", layer_idx);
    ggml_set_name(out, name);
    return out;
}

// ============================================================================
// Qwen3.5 W2 Q5_K fast down projection — replaces ggml_mul_mat_id with a
// direct custom callback so all ggml threads participate in the GEMV without
// ggml_mul_mat_id dispatch overhead. The row dot and Q8_K activation
// quantization are DenseCore HWY kernels, not GGML quant vec_dot.
// ============================================================================

constexpr int kQwen35SharedQ8MaxTasks = 128;
constexpr int64_t kQwen35NativeMoEBatchedQ4KMaxAssignments = 256;
constexpr int64_t kQwen35NativeMoEGateUpBatchTile = 128;

struct Qwen35MoEAssignment {
    int32_t expert = -1;
    int32_t token = -1;
    int32_t topk_index = -1;
    float weight = 1.0f;
};

struct Qwen35SharedQ8RowsUserData {
    int64_t cols = 0;
    int64_t ne1 = 0;
    int64_t ne2 = 0;
    int debug_layer_idx = -1;
    size_t row_bytes = 0;
    uint8_t* rows = nullptr;
    InferenceWorkContext* work_ctx = nullptr;
    std::atomic<uint64_t> epoch{0};
    std::atomic<uint64_t> ready_epoch{0};
    std::atomic<int> remaining_tasks{0};
    std::atomic<int> failed{0};
    bool prefer_q4k_repacked_swiglu = false;
    bool q5k_gateup_8x8_single_copy = false;
    bool q5k_gateup_8x8_single_copy_required = false;
    bool record_lfm2_w1w3_kernel = false;
    std::atomic<int> repacked_swiglu_failed{0};
    bool weighted_logits_lfm2_sigmoid = false;
    bool weighted_logits_norm_topk = true;
    float weighted_logits_scale = 1.0f;
    int requested_task_count = 0;
    std::atomic<uint64_t> assignments_ready_epoch{0};
    std::atomic<int> assignments_failed{0};
    Qwen35MoEAssignment* assignments = nullptr;
    size_t assignment_capacity = 0;
    size_t assignment_count = 0;
    uint64_t task_epoch[kQwen35SharedQ8MaxTasks] = {};
};

static Qwen35SharedQ8RowsUserData* AllocateQwen35SharedQ8RowsUserData(ggml_context* ctx, const ggml_tensor* src,
                                                                      int64_t max_assignments) {
    if (!ctx || !src || src->type != GGML_TYPE_F32 || src->ne[0] <= 0 || src->ne[1] <= 0 || src->ne[2] <= 0) {
        return nullptr;
    }
    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q8_K, src->ne[0]);
    const int64_t rows = src->ne[1] * src->ne[2];
    if (row_bytes == 0 || rows <= 0) {
        return nullptr;
    }
    const size_t rows_bytes = row_bytes * static_cast<size_t>(rows);
    const size_t assignment_capacity =
        max_assignments > 0 ? static_cast<size_t>(std::min<int64_t>(max_assignments, 4096 * 64)) : 0;
    const size_t assignments_bytes = assignment_capacity * sizeof(Qwen35MoEAssignment);
    if (ggml_get_no_alloc(ctx)) {
        thread_local Qwen35SharedQ8RowsUserData dry_run_ud;
        thread_local std::vector<uint8_t> dry_run_rows;
        thread_local std::vector<Qwen35MoEAssignment> dry_run_assignments;
        dry_run_rows.resize(rows_bytes);
        dry_run_assignments.resize(assignment_capacity);
        dry_run_ud.cols = src->ne[0];
        dry_run_ud.ne1 = src->ne[1];
        dry_run_ud.ne2 = src->ne[2];
        dry_run_ud.debug_layer_idx = -1;
        dry_run_ud.row_bytes = row_bytes;
        dry_run_ud.rows = dry_run_rows.data();
        dry_run_ud.work_ctx = nullptr;
        dry_run_ud.epoch.store(0, std::memory_order_relaxed);
        dry_run_ud.ready_epoch.store(0, std::memory_order_relaxed);
        dry_run_ud.remaining_tasks.store(0, std::memory_order_relaxed);
        dry_run_ud.failed.store(0, std::memory_order_relaxed);
        dry_run_ud.prefer_q4k_repacked_swiglu = false;
        dry_run_ud.q5k_gateup_8x8_single_copy = false;
        dry_run_ud.q5k_gateup_8x8_single_copy_required = false;
        dry_run_ud.record_lfm2_w1w3_kernel = false;
        dry_run_ud.repacked_swiglu_failed.store(0, std::memory_order_relaxed);
        dry_run_ud.weighted_logits_lfm2_sigmoid = false;
        dry_run_ud.weighted_logits_norm_topk = true;
        dry_run_ud.weighted_logits_scale = 1.0f;
        dry_run_ud.requested_task_count = 0;
        dry_run_ud.assignments_ready_epoch.store(0, std::memory_order_relaxed);
        dry_run_ud.assignments_failed.store(0, std::memory_order_relaxed);
        dry_run_ud.assignments = dry_run_assignments.empty() ? nullptr : dry_run_assignments.data();
        dry_run_ud.assignment_capacity = dry_run_assignments.size();
        dry_run_ud.assignment_count = 0;
        std::memset(dry_run_ud.task_epoch, 0, sizeof(dry_run_ud.task_epoch));
        return &dry_run_ud;
    }
    ggml_tensor* ud_storage = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, sizeof(Qwen35SharedQ8RowsUserData));
    ggml_tensor* rows_storage = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, static_cast<int64_t>(rows_bytes));
    ggml_tensor* assignments_storage =
        assignments_bytes > 0 ? ggml_new_tensor_1d(ctx, GGML_TYPE_I8, static_cast<int64_t>(assignments_bytes))
                              : nullptr;
    if (!ud_storage || !ud_storage->data || !rows_storage || !rows_storage->data) {
        return nullptr;
    }
    auto* ud = new (ud_storage->data) Qwen35SharedQ8RowsUserData();
    ud->cols = src->ne[0];
    ud->ne1 = src->ne[1];
    ud->ne2 = src->ne[2];
    ud->debug_layer_idx = -1;
    ud->row_bytes = row_bytes;
    ud->rows = static_cast<uint8_t*>(rows_storage->data);
    ud->prefer_q4k_repacked_swiglu = false;
    ud->q5k_gateup_8x8_single_copy = false;
    ud->q5k_gateup_8x8_single_copy_required = false;
    ud->record_lfm2_w1w3_kernel = false;
    ud->repacked_swiglu_failed.store(0, std::memory_order_relaxed);
    ud->weighted_logits_lfm2_sigmoid = false;
    ud->weighted_logits_norm_topk = true;
    ud->weighted_logits_scale = 1.0f;
    ud->requested_task_count = 0;
    ud->assignments_ready_epoch.store(0, std::memory_order_relaxed);
    ud->assignments_failed.store(0, std::memory_order_relaxed);
    if (assignments_storage && assignments_storage->data) {
        ud->assignments = static_cast<Qwen35MoEAssignment*>(assignments_storage->data);
        ud->assignment_capacity = assignment_capacity;
    }
    ud->assignment_count = 0;
    return ud;
}

static bool RemapNativeMoECallbackTask(Qwen35SharedQ8RowsUserData* ud, int ith, int nth, int* effective_ith,
                                       int* effective_nth) {
    if (!effective_ith || !effective_nth || ith < 0 || nth <= 0 || ith >= nth) {
        return false;
    }
    int task_count = ud ? ud->requested_task_count : 0;
    if (task_count <= 0 || task_count >= nth) {
        *effective_ith = ith;
        *effective_nth = nth;
        return true;
    }
    task_count = std::max(1, task_count);
    if (ith >= task_count) {
        return false;
    }
    *effective_ith = ith;
    *effective_nth = task_count;
    return true;
}

static void SetNativeMoECallbackRequestedTaskCount(Qwen35SharedQ8RowsUserData* ud, int requested_task_count) {
    if (ud) {
        ud->requested_task_count = requested_task_count;
    }
}

static bool Qwen35NativeQuantizeRowQ8K(const float* src, uint8_t* dst, int64_t cols) {
    return densecore::hwy_kernels::QuantizeRowQ8K_Hwy(src, dst, cols);
}

static bool Qwen35QuantizeSharedQ8RowsRange(Qwen35SharedQ8RowsUserData* ud, const ggml_tensor* src, int ith, int nth) {
    const int64_t rows = ud->ne1 * ud->ne2;
    const int64_t row_start = (static_cast<int64_t>(ith) * rows) / nth;
    const int64_t row_end = (static_cast<int64_t>(ith + 1) * rows) / nth;
    bool ok = true;
    for (int64_t linear = row_start; linear < row_end; ++linear) {
        const int64_t row1 = linear % ud->ne1;
        const int64_t row2 = linear / ud->ne1;
        const float* src_row =
            reinterpret_cast<const float*>(static_cast<const char*>(src->data) +
                                           static_cast<size_t>(row1) * static_cast<size_t>(src->nb[1]) +
                                           static_cast<size_t>(row2) * static_cast<size_t>(src->nb[2]));
        if (!Qwen35NativeQuantizeRowQ8K(src_row, ud->rows + static_cast<size_t>(linear) * ud->row_bytes,
                                        ud->cols)) {
            ok = false;
            break;
        }
    }
    return ok;
}

static bool PrepareQwen35SharedQ8Rows(Qwen35SharedQ8RowsUserData* ud, const ggml_tensor* src, int ith, int nth) {
    if (!ud || !src || !src->data || src->type != GGML_TYPE_F32 || ith < 0 || ith >= nth || nth <= 0 ||
        nth > kQwen35SharedQ8MaxTasks || src->ne[0] != ud->cols || src->ne[1] != ud->ne1 || src->ne[2] != ud->ne2 ||
        !ud->rows) {
        return false;
    }
    uint64_t epoch = 0;
    if (ith == 0) {
        epoch = ud->epoch.load(std::memory_order_relaxed) + 1;
        ud->failed.store(0, std::memory_order_relaxed);
        ud->remaining_tasks.store(nth, std::memory_order_relaxed);
        ud->task_epoch[0] = epoch;
        ud->epoch.store(epoch, std::memory_order_release);
    } else {
        const uint64_t last_epoch = ud->task_epoch[ith];
        epoch = ud->epoch.load(std::memory_order_acquire);
        while (epoch == last_epoch) {
            std::this_thread::yield();
            epoch = ud->epoch.load(std::memory_order_acquire);
        }
    }

    const bool ok = Qwen35QuantizeSharedQ8RowsRange(ud, src, ith, nth);
    if (!ok) {
        ud->failed.store(1, std::memory_order_relaxed);
    }
    if (ud->remaining_tasks.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        ud->ready_epoch.store(epoch, std::memory_order_release);
    }
    while (ud->ready_epoch.load(std::memory_order_acquire) != epoch) {
        std::this_thread::yield();
    }
    ud->task_epoch[ith] = epoch;
    return ud->failed.load(std::memory_order_acquire) == 0;
}

static const uint8_t* Qwen35SharedQ8RowPtr(const Qwen35SharedQ8RowsUserData* ud, int64_t row1, int64_t row2 = 0) {
    if (!ud || !ud->rows || row1 < 0 || row1 >= ud->ne1 || row2 < 0 || row2 >= ud->ne2) {
        return nullptr;
    }
    const int64_t linear = row2 * ud->ne1 + row1;
    return ud->rows + static_cast<size_t>(linear) * ud->row_bytes;
}

static void Qwen35NativeMoEZeroDst2D(ggml_tensor* dst, int64_t n_rows, int64_t n_tokens) {
    if (!dst || !dst->data) return;
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t row = 0; row < n_rows; ++row) {
            *reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                      static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                                      static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1])) = 0.0f;
        }
    }
}

static void Qwen35NativeMoEZeroDst2DRange(ggml_tensor* dst, int64_t row_start, int64_t row_end, int64_t n_tokens) {
    if (!dst || !dst->data) return;
    row_start = std::max<int64_t>(0, row_start);
    row_end = std::min<int64_t>(dst->ne[0], row_end);
    if (row_start >= row_end) return;
    for (int64_t token = 0; token < n_tokens; ++token) {
        char* dst_col = static_cast<char*>(dst->data) + static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1]);
        for (int64_t row = row_start; row < row_end; ++row) {
            *reinterpret_cast<float*>(dst_col + static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0])) = 0.0f;
        }
    }
}

static bool Qwen35BuildMoEAssignments(const ggml_tensor* selected_experts, int64_t n_experts,
                                      std::vector<Qwen35MoEAssignment>* assignments) {
    if (!selected_experts || !selected_experts->data || selected_experts->type != GGML_TYPE_I32 ||
        n_experts <= 0 || !assignments) {
        return false;
    }
    const int64_t top_k = selected_experts->ne[0];
    const int64_t n_tokens = selected_experts->ne[1];
    if (top_k <= 0 || n_tokens <= 0) {
        return false;
    }
    assignments->clear();
    assignments->reserve(static_cast<size_t>(top_k) * static_cast<size_t>(n_tokens));
    const char* selected_base = static_cast<const char*>(selected_experts->data);
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t k = 0; k < top_k; ++k) {
            const int32_t expert = *reinterpret_cast<const int32_t*>(
                selected_base + static_cast<size_t>(k) * static_cast<size_t>(selected_experts->nb[0]) +
                static_cast<size_t>(token) * static_cast<size_t>(selected_experts->nb[1]));
            if (expert < 0 || expert >= n_experts) {
                continue;
            }
            assignments->push_back({expert, static_cast<int32_t>(token), static_cast<int32_t>(k), 1.0f});
        }
    }
    std::sort(assignments->begin(), assignments->end(), [](const Qwen35MoEAssignment& a,
                                                           const Qwen35MoEAssignment& b) {
        if (a.expert != b.expert) return a.expert < b.expert;
        if (a.token != b.token) return a.token < b.token;
        return a.topk_index < b.topk_index;
    });
    return !assignments->empty();
}

static bool Qwen35BuildMoEAssignmentsToBuffer(const ggml_tensor* selected_experts, int64_t n_experts,
                                              Qwen35MoEAssignment* assignments, size_t assignment_capacity,
                                              size_t* assignment_count_out) {
    if (assignment_count_out) {
        *assignment_count_out = 0;
    }
    if (!selected_experts || !selected_experts->data || selected_experts->type != GGML_TYPE_I32 ||
        n_experts <= 0 || !assignments || assignment_capacity == 0 || !assignment_count_out) {
        return false;
    }
    const int64_t top_k = selected_experts->ne[0];
    const int64_t n_tokens = selected_experts->ne[1];
    if (top_k <= 0 || n_tokens <= 0 ||
        static_cast<uint64_t>(top_k) * static_cast<uint64_t>(n_tokens) > assignment_capacity) {
        return false;
    }
    size_t count = 0;
    const char* selected_base = static_cast<const char*>(selected_experts->data);
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t k = 0; k < top_k; ++k) {
            const int32_t expert = *reinterpret_cast<const int32_t*>(
                selected_base + static_cast<size_t>(k) * static_cast<size_t>(selected_experts->nb[0]) +
                static_cast<size_t>(token) * static_cast<size_t>(selected_experts->nb[1]));
            if (expert < 0 || expert >= n_experts) {
                continue;
            }
            assignments[count++] = {expert, static_cast<int32_t>(token), static_cast<int32_t>(k), 1.0f};
        }
    }
    std::sort(assignments, assignments + count, [](const Qwen35MoEAssignment& a, const Qwen35MoEAssignment& b) {
        if (a.expert != b.expert) return a.expert < b.expert;
        if (a.token != b.token) return a.token < b.token;
        return a.topk_index < b.topk_index;
    });
    *assignment_count_out = count;
    return count > 0;
}

static const Qwen35MoEAssignment* PrepareQwen35SharedMoEAssignments(
    Qwen35SharedQ8RowsUserData* ud, const ggml_tensor* selected_experts, int64_t n_experts, int ith,
    size_t* assignment_count_out) {
    if (assignment_count_out) {
        *assignment_count_out = 0;
    }
    if (!ud || !selected_experts || ith < 0 || !assignment_count_out) {
        return nullptr;
    }
    const uint64_t epoch = ud->epoch.load(std::memory_order_acquire);
    if (epoch == 0) {
        return nullptr;
    }
    if (ith == 0) {
        size_t count = 0;
        const bool ok = Qwen35BuildMoEAssignmentsToBuffer(selected_experts, n_experts, ud->assignments,
                                                          ud->assignment_capacity, &count);
        ud->assignment_count = ok ? count : 0;
        ud->assignments_failed.store(ok ? 0 : 1, std::memory_order_relaxed);
        ud->assignments_ready_epoch.store(epoch, std::memory_order_release);
    } else {
        while (ud->assignments_ready_epoch.load(std::memory_order_acquire) != epoch) {
            std::this_thread::yield();
        }
    }
    if (ud->assignments_failed.load(std::memory_order_acquire) != 0) {
        return nullptr;
    }
    *assignment_count_out = ud->assignment_count;
    return ud->assignments;
}

static const float* Qwen35NativeMoEDownHiddenRowPtr(const ggml_tensor* hidden, int64_t token, int64_t topk_index) {
    if (!hidden || !hidden->data) return nullptr;
    return reinterpret_cast<const float*>(static_cast<const char*>(hidden->data) +
                                          static_cast<size_t>(topk_index) * static_cast<size_t>(hidden->nb[1]) +
                                          static_cast<size_t>(token) * static_cast<size_t>(hidden->nb[2]));
}

// Phase 0: capability decisions now live in kernels/kernel_caps.h (single source
// of truth). These thin wrappers preserve the call-site names.
static bool PreferGgmlQ4KVecDotForNativeMoE() { return densecore::kernels::PreferGgmlKQuantVecDot(); }

static bool CanUseGgmlQ4KVecDotRowPairForNativeMoE() {
    return densecore::kernels::KQuantVecDotRowPairSupported();
}

static bool Qwen35NativeMoEDownQ5KQuantizeHidden(const ggml_tensor* hidden, int64_t token, int64_t topk_index,
                                                 std::vector<uint8_t>& qbuf) {
    if (!hidden || !hidden->data || hidden->type != GGML_TYPE_F32) return false;
    const float* hidden_row = Qwen35NativeMoEDownHiddenRowPtr(hidden, token, topk_index);
    if (!hidden_row) return false;
    const int64_t cols = hidden->ne[0];
    const size_t qrow_bytes = ggml_row_size(GGML_TYPE_Q8_K, cols);
    if (qrow_bytes == 0) return false;
    qbuf.resize(qrow_bytes);
    return Qwen35NativeQuantizeRowQ8K(hidden_row, qbuf.data(), cols);
}

static bool Qwen35NativeMoEDownQuantizeHiddenForWeight(const ggml_tensor* hidden, int64_t token, int64_t topk_index,
                                                       ggml_type weight_type, std::vector<uint8_t>& qbuf) {
    if (weight_type == GGML_TYPE_Q8_0) {
        if (!hidden || !hidden->data || hidden->type != GGML_TYPE_F32) return false;
        const float* hidden_row = Qwen35NativeMoEDownHiddenRowPtr(hidden, token, topk_index);
        if (!hidden_row) return false;
        const int64_t cols = hidden->ne[0];
        const ggml_type_traits_cpu* q8_0_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
        const size_t qrow_bytes = ggml_row_size(GGML_TYPE_Q8_0, cols);
        if (!q8_0_traits || !q8_0_traits->from_float || qrow_bytes == 0) return false;
        qbuf.resize(qrow_bytes);
        q8_0_traits->from_float(hidden_row, qbuf.data(), cols);
        return true;
    }
    return Qwen35NativeMoEDownQ5KQuantizeHidden(hidden, token, topk_index, qbuf);
}

static bool Qwen35NativeMoEDownQ8_0ReferenceDotRowForExpertWithHidden(const ggml_tensor* down_exps, int32_t expert,
                                                                       int64_t row, const float* hidden_row,
                                                                       float* out_value) {
    if (!down_exps || !hidden_row || !out_value || down_exps->type != GGML_TYPE_Q8_0) return false;
    if (expert < 0 || expert >= down_exps->ne[2] || row < 0 || row >= down_exps->ne[1]) return false;
    const ggml_type_traits* q8_0_traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
    if (!q8_0_traits || !q8_0_traits->to_float) return false;
    const int64_t cols = down_exps->ne[0];
    const char* weight_row = static_cast<const char*>(down_exps->data) +
                             static_cast<size_t>(expert) * static_cast<size_t>(down_exps->nb[2]) +
                             static_cast<size_t>(row) * static_cast<size_t>(down_exps->nb[1]);
    thread_local std::vector<float> wbuf;
    wbuf.resize(static_cast<size_t>(cols));
    q8_0_traits->to_float(weight_row, wbuf.data(), cols);
    double acc = 0.0;
    for (int64_t i = 0; i < cols; ++i) {
        acc += static_cast<double>(hidden_row[static_cast<size_t>(i)]) * wbuf[static_cast<size_t>(i)];
    }
    *out_value = static_cast<float>(acc);
    return true;
}

static bool Qwen35NativeMoEDownQ8_0DotRowForExpertWithQbuf(const ggml_tensor* down_exps, int32_t expert,
                                                            int64_t row, const uint8_t* qbuf, float* out_value) {
    if (!down_exps || !qbuf || !out_value || down_exps->type != GGML_TYPE_Q8_0) return false;
    if (expert < 0 || expert >= down_exps->ne[2] || row < 0 || row >= down_exps->ne[1]) return false;
    const int64_t cols = down_exps->ne[0];
    const ggml_type_traits_cpu* q8_0_traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    if (!q8_0_traits || !q8_0_traits->vec_dot || q8_0_traits->vec_dot_type != GGML_TYPE_Q8_0 ||
        (cols % ggml_blck_size(GGML_TYPE_Q8_0)) != 0) {
        return false;
    }
    const char* weight_row = static_cast<const char*>(down_exps->data) +
                             static_cast<size_t>(expert) * static_cast<size_t>(down_exps->nb[2]) +
                             static_cast<size_t>(row) * static_cast<size_t>(down_exps->nb[1]);
    q8_0_traits->vec_dot(static_cast<int>(cols), out_value, 0, weight_row, 0, qbuf, 0, 1);
    return true;
}

static bool Qwen35NativeMoEDownQ5KDotRowForExpertWithQbuf(const ggml_tensor* down_exps, int32_t expert, int64_t row,
                                                           const uint8_t* qbuf, const float* hidden_row,
                                                           float* out_value) {
    if (!down_exps || !out_value) return false;
    if (expert < 0 || expert >= down_exps->ne[2] || row < 0 || row >= down_exps->ne[1]) return false;
    const int64_t cols = down_exps->ne[0];
    const char* weight_row = static_cast<const char*>(down_exps->data) +
                             static_cast<size_t>(expert) * static_cast<size_t>(down_exps->nb[2]) +
                             static_cast<size_t>(row) * static_cast<size_t>(down_exps->nb[1]);
    (void)hidden_row;
    if (down_exps->type == GGML_TYPE_Q8_0) {
        return Qwen35NativeMoEDownQ8_0DotRowForExpertWithQbuf(down_exps, expert, row, qbuf, out_value);
    }
    if (!qbuf) return false;
    if (down_exps->type == GGML_TYPE_Q5_K) {
        if (PreferGgmlQ4KVecDotForNativeMoE()) {
            const ggml_type_traits_cpu* traits = ggml_get_type_traits_cpu(GGML_TYPE_Q5_K);
            if (traits && traits->vec_dot && traits->vec_dot_type == GGML_TYPE_Q8_K &&
                (cols % ggml_blck_size(GGML_TYPE_Q5_K)) == 0) {
                traits->vec_dot(static_cast<int>(cols), out_value, 0, weight_row, 0, qbuf, 0, 1);
                return true;
            }
        }
        return densecore::hwy_kernels::DotQ5KQ8K_Hwy(weight_row, qbuf, cols, out_value);
    }
    if (down_exps->type == GGML_TYPE_Q4_K) {
        if (PreferGgmlQ4KVecDotForNativeMoE()) {
            const ggml_type_traits_cpu* traits = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K);
            if (traits && traits->vec_dot && traits->vec_dot_type == GGML_TYPE_Q8_K &&
                (cols % ggml_blck_size(GGML_TYPE_Q4_K)) == 0) {
                traits->vec_dot(static_cast<int>(cols), out_value, 0, weight_row, 0, qbuf, 0, 1);
                return true;
            }
        }
        return densecore::hwy_kernels::DotQ4KQ8K_Hwy(weight_row, qbuf, cols, out_value);
    }
    if (down_exps->type == GGML_TYPE_Q6_K) {
        const ggml_type_traits_cpu* traits = ggml_get_type_traits_cpu(GGML_TYPE_Q6_K);
        if (!traits || !traits->vec_dot || traits->vec_dot_type != GGML_TYPE_Q8_K) {
            return false;
        }
        traits->vec_dot(static_cast<int>(cols), out_value, 0, weight_row, 0, qbuf, 0, 1);
        return true;
    }
    return false;
}

static bool Qwen35NativeMoEDownQ5KReadExpert(const ggml_tensor* selected_experts, const ggml_tensor* down_exps,
                                              int64_t token, int64_t topk_index, int32_t* expert_out) {
    if (!selected_experts || !down_exps || !selected_experts->data || !expert_out) return false;
    const int32_t expert = *reinterpret_cast<const int32_t*>(
        static_cast<const char*>(selected_experts->data) +
        static_cast<size_t>(topk_index) * static_cast<size_t>(selected_experts->nb[0]) +
        static_cast<size_t>(token) * static_cast<size_t>(selected_experts->nb[1]));
    if (expert < 0 || expert >= down_exps->ne[2]) return false;
    *expert_out = expert;
    return true;
}

static bool Qwen35NativeMoEDownQ5KDotRowPairForExpertWithQbuf(const ggml_tensor* down_exps, int32_t expert,
                                                               int64_t row, const uint8_t* qbuf,
                                                               const float* hidden_row, float* out0, float* out1) {
    if (!down_exps || !out0 || !out1) return false;
    if (expert < 0 || expert >= down_exps->ne[2] || row < 0 || row + 1 >= down_exps->ne[1]) return false;
    (void)hidden_row;
    if (down_exps->type == GGML_TYPE_Q8_0) {
        if (!qbuf) return false;
        const ggml_type_traits_cpu* traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
        if (traits && traits->vec_dot && traits->vec_dot_type == GGML_TYPE_Q8_0 &&
            (down_exps->ne[0] % ggml_blck_size(GGML_TYPE_Q8_0)) == 0) {
            const char* weight_row = static_cast<const char*>(down_exps->data) +
                                     static_cast<size_t>(expert) * static_cast<size_t>(down_exps->nb[2]) +
                                     static_cast<size_t>(row) * static_cast<size_t>(down_exps->nb[1]);
            float sums[32] = {};
            traits->vec_dot(static_cast<int>(down_exps->ne[0]), sums, 2, weight_row,
                            static_cast<size_t>(down_exps->nb[1]), qbuf, 0, 2);
            *out0 = sums[0];
            *out1 = sums[1];
            return true;
        }
        return Qwen35NativeMoEDownQ8_0DotRowForExpertWithQbuf(down_exps, expert, row, qbuf, out0) &&
               Qwen35NativeMoEDownQ8_0DotRowForExpertWithQbuf(down_exps, expert, row + 1, qbuf, out1);
    }
    if (!qbuf) return false;
    if ((down_exps->type == GGML_TYPE_Q4_K || down_exps->type == GGML_TYPE_Q5_K ||
         down_exps->type == GGML_TYPE_Q6_K) &&
        CanUseGgmlQ4KVecDotRowPairForNativeMoE()) {
        const ggml_type_traits_cpu* traits = ggml_get_type_traits_cpu(down_exps->type);
        if (traits && traits->vec_dot && traits->vec_dot_type == GGML_TYPE_Q8_K &&
            (down_exps->ne[0] % ggml_blck_size(down_exps->type)) == 0) {
            const char* weight_row = static_cast<const char*>(down_exps->data) +
                                     static_cast<size_t>(expert) * static_cast<size_t>(down_exps->nb[2]) +
                                     static_cast<size_t>(row) * static_cast<size_t>(down_exps->nb[1]);
            float sums[32] = {};
            traits->vec_dot(static_cast<int>(down_exps->ne[0]), sums, 2, weight_row,
                            static_cast<size_t>(down_exps->nb[1]), qbuf, 0, 2);
            *out0 = sums[0];
            *out1 = sums[1];
            return true;
        }
    }
    return Qwen35NativeMoEDownQ5KDotRowForExpertWithQbuf(down_exps, expert, row, qbuf, hidden_row, out0) &&
           Qwen35NativeMoEDownQ5KDotRowForExpertWithQbuf(down_exps, expert, row + 1, qbuf, hidden_row, out1);
}

static bool Qwen35NativeMoEDownQXKAccumulateAssignmentsForRowRange(
    ggml_tensor* dst, const ggml_tensor* down_exps, const uint8_t* qtile, size_t qrow_bytes,
    const std::vector<Qwen35MoEAssignment>& tile_assignments, int32_t expert, int64_t row_start, int64_t row_end) {
    if (!dst || !down_exps || !qtile || !dst->data || !down_exps->data || tile_assignments.empty()) {
        return false;
    }
    const ggml_type wtype = down_exps->type;
    const auto* traits = ggml_get_type_traits_cpu(wtype);
    if (!traits || !traits->vec_dot || traits->vec_dot_type != GGML_TYPE_Q8_K) {
        return false;
    }
    const int64_t K = down_exps->ne[0];
    const int64_t n_embd = down_exps->ne[1];
    if (K <= 0 || n_embd <= 0 || row_start < 0 || row_end > n_embd || row_start > row_end ||
        (K % ggml_blck_size(wtype)) != 0 || qrow_bytes < ggml_row_size(GGML_TYPE_Q8_K, K)) {
        return false;
    }
    const size_t w_row_bytes = static_cast<size_t>(down_exps->nb[1]);
    const char* expert_base = static_cast<const char*>(down_exps->data) +
                              static_cast<size_t>(expert) * static_cast<size_t>(down_exps->nb[2]);
    if (wtype == GGML_TYPE_Q5_K && row_start < row_end) {
        const int64_t row_count = row_end - row_start;
        thread_local std::vector<float> projection_tile;
        projection_tile.resize(tile_assignments.size() * static_cast<size_t>(row_count));
        densecore::CpuBackend& backend = densecore::GetCpuBackend();
        const void* weight_start = expert_base + static_cast<size_t>(row_start) * w_row_bytes;
        const auto raw_batched_begin = std::chrono::steady_clock::now();
        const bool projected = densecore::RunMoEKQuantRawBatchedProjection(
            &backend, static_cast<int>(wtype), weight_start, qtile, qrow_bytes, projection_tile.data(),
            static_cast<int64_t>(tile_assignments.size()), row_count, K, /*numa_node=*/0, /*allow_parallel=*/false);
        if (projected) {
            const auto elapsed = std::chrono::steady_clock::now() - raw_batched_begin;
            RecordMoEKQuantRawBatchedUse(GetCurrentWorkContext(), wtype,
                                         static_cast<uint64_t>(
                                             std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()),
                                         /*qwen_native_w2_q5k=*/true);
            for (size_t m = 0; m < tile_assignments.size(); ++m) {
                const Qwen35MoEAssignment& assignment = tile_assignments[m];
                if (assignment.weight == 0.0f) {
                    continue;
                }
                char* dst_col = static_cast<char*>(dst->data) +
                                static_cast<size_t>(assignment.token) * static_cast<size_t>(dst->nb[1]);
                const float* src = projection_tile.data() + m * static_cast<size_t>(row_count);
                for (int64_t row = 0; row < row_count; ++row) {
                    *reinterpret_cast<float*>(dst_col + static_cast<size_t>(row_start + row) *
                                                            static_cast<size_t>(dst->nb[0])) +=
                        src[row] * assignment.weight;
                }
            }
            return true;
        }
    }
    if (wtype == GGML_TYPE_Q5_K && !PreferGgmlQ4KVecDotForNativeMoE()) {
        for (int64_t row = row_start; row < row_end; ++row) {
            const void* w_row = expert_base + static_cast<size_t>(row) * w_row_bytes;
            for (size_t m = 0; m < tile_assignments.size(); ++m) {
                const Qwen35MoEAssignment& assignment = tile_assignments[m];
                const uint8_t* qrow = qtile + m * qrow_bytes;
                float sum = 0.0f;
                if (!densecore::hwy_kernels::DotQ5KQ8K_Hwy(w_row, qrow, K, &sum)) {
                    return false;
                }
                char* dst_col = static_cast<char*>(dst->data) +
                                static_cast<size_t>(assignment.token) * static_cast<size_t>(dst->nb[1]);
                *reinterpret_cast<float*>(dst_col + static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0])) +=
                    sum * assignment.weight;
            }
        }
        return true;
    }

    const InferenceExecutionPhase phase = GetCurrentExecutionPhase();
    const bool prefill_rows = phase == InferenceExecutionPhase::Prefill;

    if (wtype == GGML_TYPE_Q4_K && row_start < row_end && (row_start % 8) == 0 && ((row_end - row_start) % 8) == 0) {
        const int64_t row_count = row_end - row_start;
        thread_local std::vector<float> projection_tile;
        projection_tile.resize(tile_assignments.size() * static_cast<size_t>(row_count));
        densecore::CpuBackend& backend = densecore::GetCpuBackend();
        const void* weight_start = expert_base + static_cast<size_t>(row_start) * w_row_bytes;
        const bool projected = prefill_rows
                                   ? densecore::RunMoEKQuantRawBatchedProjection(
                                         &backend, static_cast<int>(wtype), weight_start, qtile, qrow_bytes,
                                         projection_tile.data(), static_cast<int64_t>(tile_assignments.size()),
                                         row_count, K, /*numa_node=*/0, /*allow_parallel=*/false)
                                   : densecore::RunQ4KRepackedMoEProjection(
                                         &backend, weight_start, qtile, qrow_bytes, projection_tile.data(),
                                         static_cast<int64_t>(tile_assignments.size()), row_count, K,
                                         /*numa_node=*/0, /*allow_parallel=*/false);
        if (projected) {
            for (size_t m = 0; m < tile_assignments.size(); ++m) {
                const Qwen35MoEAssignment& assignment = tile_assignments[m];
                if (assignment.weight == 0.0f) {
                    continue;
                }
                char* dst_col = static_cast<char*>(dst->data) +
                                static_cast<size_t>(assignment.token) * static_cast<size_t>(dst->nb[1]);
                const float* src = projection_tile.data() + m * static_cast<size_t>(row_count);
                for (int64_t row = 0; row < row_count; ++row) {
                    *reinterpret_cast<float*>(dst_col + static_cast<size_t>(row_start + row) *
                                                            static_cast<size_t>(dst->nb[0])) +=
                        src[row] * assignment.weight;
                }
            }
            return true;
        }
    }
    if (wtype == GGML_TYPE_Q6_K && row_start < row_end && (row_start % 8) == 0 && ((row_end - row_start) % 8) == 0) {
        const int64_t row_count = row_end - row_start;
        thread_local std::vector<float> projection_tile;
        projection_tile.resize(tile_assignments.size() * static_cast<size_t>(row_count));
        densecore::CpuBackend& backend = densecore::GetCpuBackend();
        const void* weight_start = expert_base + static_cast<size_t>(row_start) * w_row_bytes;
        const bool lfm2_decode_q6k_raw_batched =
            GetCurrentInferenceWorkContextModelVariant() == ModelVariant::LFM2MOE &&
            phase == InferenceExecutionPhase::Decode;
        const bool projected =
            lfm2_decode_q6k_raw_batched
                ? densecore::RunMoEKQuantRawBatchedProjection(
                      &backend, static_cast<int>(wtype), weight_start, qtile, qrow_bytes, projection_tile.data(),
                      static_cast<int64_t>(tile_assignments.size()), row_count, K, /*numa_node=*/0,
                      /*allow_parallel=*/false)
                : densecore::RunQ6KRepackedMoEProjection(
                      &backend, weight_start, qtile, qrow_bytes, projection_tile.data(),
                      static_cast<int64_t>(tile_assignments.size()), row_count, K, /*numa_node=*/0,
                      /*allow_parallel=*/false);
        if (projected) {
            for (size_t m = 0; m < tile_assignments.size(); ++m) {
                const Qwen35MoEAssignment& assignment = tile_assignments[m];
                if (assignment.weight == 0.0f) {
                    continue;
                }
                char* dst_col = static_cast<char*>(dst->data) +
                                static_cast<size_t>(assignment.token) * static_cast<size_t>(dst->nb[1]);
                const float* src = projection_tile.data() + m * static_cast<size_t>(row_count);
                for (int64_t row = 0; row < row_count; ++row) {
                    *reinterpret_cast<float*>(dst_col + static_cast<size_t>(row_start + row) *
                                                            static_cast<size_t>(dst->nb[0])) +=
                        src[row] * assignment.weight;
                }
            }
            return true;
        }
    }

    const bool kquant_row_pair =
        (wtype == GGML_TYPE_Q4_K || wtype == GGML_TYPE_Q5_K || wtype == GGML_TYPE_Q6_K) &&
        CanUseGgmlQ4KVecDotRowPairForNativeMoE();
    const bool supports_row_pair = (wtype == GGML_TYPE_Q8_0 && traits->nrows >= 2) || kquant_row_pair;
    const int64_t pair_start = supports_row_pair ? (row_start + 1) / 2 : 0;
    const int64_t pair_end = supports_row_pair ? (row_end / 2) : 0;
    if ((row_start & 1) != 0) {
        const int64_t row = row_start;
        const void* w_row = expert_base + static_cast<size_t>(row) * w_row_bytes;
        for (size_t m = 0; m < tile_assignments.size(); ++m) {
            const Qwen35MoEAssignment& assignment = tile_assignments[m];
            const uint8_t* qrow = qtile + m * qrow_bytes;
            float sum = 0.0f;
            traits->vec_dot(static_cast<int>(K), &sum, 0, w_row, 0, qrow, 0, 1);
            char* dst_col = static_cast<char*>(dst->data) +
                            static_cast<size_t>(assignment.token) * static_cast<size_t>(dst->nb[1]);
            *reinterpret_cast<float*>(dst_col + static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0])) +=
                sum * assignment.weight;
        }
    }
    if (supports_row_pair) {
        for (int64_t pair = pair_start; pair < pair_end; ++pair) {
            const int64_t row = pair * 2;
            const void* w_row = expert_base + static_cast<size_t>(row) * w_row_bytes;
            for (size_t m = 0; m < tile_assignments.size(); ++m) {
                const Qwen35MoEAssignment& assignment = tile_assignments[m];
                const uint8_t* qrow = qtile + m * qrow_bytes;
                float sums[4] = {};
                traits->vec_dot(static_cast<int>(K), sums, 2, w_row, w_row_bytes, qrow, 0, 2);
                char* dst_col = static_cast<char*>(dst->data) +
                                static_cast<size_t>(assignment.token) * static_cast<size_t>(dst->nb[1]);
                *reinterpret_cast<float*>(dst_col + static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0])) +=
                    sums[0] * assignment.weight;
                *reinterpret_cast<float*>(dst_col + static_cast<size_t>(row + 1) * static_cast<size_t>(dst->nb[0])) +=
                    sums[1] * assignment.weight;
            }
        }
    } else {
        for (int64_t row = row_start; row < row_end; ++row) {
            const void* w_row = expert_base + static_cast<size_t>(row) * w_row_bytes;
            for (size_t m = 0; m < tile_assignments.size(); ++m) {
                const Qwen35MoEAssignment& assignment = tile_assignments[m];
                const uint8_t* qrow = qtile + m * qrow_bytes;
                float sum = 0.0f;
                traits->vec_dot(static_cast<int>(K), &sum, 0, w_row, 0, qrow, 0, 1);
                char* dst_col = static_cast<char*>(dst->data) +
                                static_cast<size_t>(assignment.token) * static_cast<size_t>(dst->nb[1]);
                *reinterpret_cast<float*>(dst_col + static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0])) +=
                    sum * assignment.weight;
            }
        }
    }
    if ((row_end & 1) != 0 && row_end - 1 >= row_start) {
        const int64_t row = row_end - 1;
        const void* w_row = expert_base + static_cast<size_t>(row) * w_row_bytes;
        for (size_t m = 0; m < tile_assignments.size(); ++m) {
            const Qwen35MoEAssignment& assignment = tile_assignments[m];
            const uint8_t* qrow = qtile + m * qrow_bytes;
            float sum = 0.0f;
            traits->vec_dot(static_cast<int>(K), &sum, 0, w_row, 0, qrow, 0, 1);
            char* dst_col = static_cast<char*>(dst->data) +
                            static_cast<size_t>(assignment.token) * static_cast<size_t>(dst->nb[1]);
            *reinterpret_cast<float*>(dst_col + static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0])) +=
                sum * assignment.weight;
        }
    }
    return true;
}

static void RunQwen35NativeMoEDownQ5KFastPath(ggml_tensor* dst, const ggml_tensor* down_exps,
                                               const ggml_tensor* hidden, const ggml_tensor* selected_experts,
                                               int ith, int nth, Qwen35SharedQ8RowsUserData* shared_q8) {
    if (!dst || !down_exps || !hidden || !selected_experts || !dst->data || !down_exps->data ||
        !hidden->data || !selected_experts->data || nth <= 0) return;
    const int64_t n_embd = down_exps->ne[1];
    const int64_t top_k = selected_experts->ne[0];
    const int64_t n_tokens = selected_experts->ne[1];
    const bool use_shared_q8 =
        down_exps->type != GGML_TYPE_Q8_0 && PrepareQwen35SharedQ8Rows(shared_q8, hidden, ith, nth);
    thread_local std::vector<uint8_t> qbuf;
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t k = 0; k < top_k; ++k) {
            int32_t expert = -1;
            if (!Qwen35NativeMoEDownQ5KReadExpert(selected_experts, down_exps, token, k, &expert)) continue;
            const float* hidden_row = Qwen35NativeMoEDownHiddenRowPtr(hidden, token, k);
            const uint8_t* qrow =
                (down_exps->type != GGML_TYPE_Q8_0 && use_shared_q8) ? Qwen35SharedQ8RowPtr(shared_q8, k, token)
                                                                     : nullptr;
            if (!qrow) {
                if (!Qwen35NativeMoEDownQuantizeHiddenForWeight(hidden, token, k, down_exps->type, qbuf)) continue;
                qrow = qbuf.data();
            }
            const int64_t pair_count = n_embd / 2;
            const int64_t pair_start = (static_cast<int64_t>(ith) * pair_count) / nth;
            const int64_t pair_end = (static_cast<int64_t>(ith + 1) * pair_count) / nth;
            for (int64_t pair = pair_start; pair < pair_end; ++pair) {
                const int64_t row = pair * 2;
                float value0 = 0.0f;
                float value1 = 0.0f;
                if (Qwen35NativeMoEDownQ5KDotRowPairForExpertWithQbuf(down_exps, expert, row, qrow, hidden_row,
                                                                       &value0, &value1)) {
                    *reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                              static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                                              static_cast<size_t>(k) * static_cast<size_t>(dst->nb[1]) +
                                              static_cast<size_t>(token) * static_cast<size_t>(dst->nb[2])) = value0;
                    *reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                              static_cast<size_t>(row + 1) * static_cast<size_t>(dst->nb[0]) +
                                              static_cast<size_t>(k) * static_cast<size_t>(dst->nb[1]) +
                                              static_cast<size_t>(token) * static_cast<size_t>(dst->nb[2])) = value1;
                }
            }
            if ((n_embd & 1) != 0 && ith == nth - 1) {
                const int64_t row = n_embd - 1;
                float value = 0.0f;
                if (Qwen35NativeMoEDownQ5KDotRowForExpertWithQbuf(down_exps, expert, row, qrow, hidden_row,
                                                                   &value)) {
                    *reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                              static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                                              static_cast<size_t>(k) * static_cast<size_t>(dst->nb[1]) +
                                              static_cast<size_t>(token) * static_cast<size_t>(dst->nb[2])) = value;
                }
            }
        }
    }
}

static const float* Qwen35NativeMoEDownWeightPtr(const ggml_tensor* weights, int64_t token, int64_t topk_index) {
    if (!weights || !weights->data || weights->type != GGML_TYPE_F32) return nullptr;
    return reinterpret_cast<const float*>(static_cast<const char*>(weights->data) +
                                          static_cast<size_t>(topk_index) * static_cast<size_t>(weights->nb[1]) +
                                          static_cast<size_t>(token) * static_cast<size_t>(weights->nb[2]));
}

static bool Qwen35NativeMoEComputeTopKWeightsFromLogits(const ggml_tensor* gate_logits,
                                                         const ggml_tensor* selected_experts,
                                                         int64_t token, float* weights,
                                                         int64_t weights_capacity) {
    if (!gate_logits || !selected_experts || !gate_logits->data || !selected_experts->data || !weights ||
        gate_logits->type != GGML_TYPE_F32 || selected_experts->type != GGML_TYPE_I32 ||
        token < 0 || token >= gate_logits->ne[1] || selected_experts->ne[1] != gate_logits->ne[1]) {
        return false;
    }
    const int64_t n_experts = gate_logits->ne[0];
    const int64_t top_k = selected_experts->ne[0];
    if (top_k <= 0 || top_k > weights_capacity) {
        return false;
    }

    const char* logits_base = static_cast<const char*>(gate_logits->data);
    const char* selected_base = static_cast<const char*>(selected_experts->data);
    float max_logit = -INFINITY;
    for (int64_t k = 0; k < top_k; ++k) {
        const int32_t expert =
            *reinterpret_cast<const int32_t*>(selected_base + static_cast<size_t>(k) *
                                                                  static_cast<size_t>(selected_experts->nb[0]) +
                                              static_cast<size_t>(token) *
                                                  static_cast<size_t>(selected_experts->nb[1]));
        if (expert >= 0 && expert < n_experts) {
            const float logit =
                *reinterpret_cast<const float*>(logits_base + static_cast<size_t>(expert) *
                                                                  static_cast<size_t>(gate_logits->nb[0]) +
                                                static_cast<size_t>(token) *
                                                    static_cast<size_t>(gate_logits->nb[1]));
            max_logit = std::max(max_logit, logit);
        }
    }
    if (!std::isfinite(max_logit)) {
        max_logit = 0.0f;
    }

    float denom = 0.0f;
    for (int64_t k = 0; k < top_k; ++k) {
        const int32_t expert =
            *reinterpret_cast<const int32_t*>(selected_base + static_cast<size_t>(k) *
                                                                  static_cast<size_t>(selected_experts->nb[0]) +
                                              static_cast<size_t>(token) *
                                                  static_cast<size_t>(selected_experts->nb[1]));
        float value = 0.0f;
        if (expert >= 0 && expert < n_experts) {
            const float logit =
                *reinterpret_cast<const float*>(logits_base + static_cast<size_t>(expert) *
                                                                  static_cast<size_t>(gate_logits->nb[0]) +
                                                static_cast<size_t>(token) *
                                                    static_cast<size_t>(gate_logits->nb[1]));
            value = std::exp(logit - max_logit);
            denom += value;
        }
        weights[k] = value;
    }
    denom = std::max(denom, 6.103515625e-5f);
    for (int64_t k = 0; k < top_k; ++k) {
        weights[k] /= denom;
    }
    return true;
}

static bool LFM2NativeMoEComputeTopKWeightsFromLogits(const ggml_tensor* gate_logits,
                                                       const ggml_tensor* selected_experts,
                                                       int64_t token, float* weights,
                                                       int64_t weights_capacity, bool normalize_weights,
                                                       float routed_scale) {
    if (!gate_logits || !selected_experts || !gate_logits->data || !selected_experts->data || !weights ||
        gate_logits->type != GGML_TYPE_F32 || selected_experts->type != GGML_TYPE_I32 ||
        token < 0 || token >= gate_logits->ne[1] || selected_experts->ne[1] != gate_logits->ne[1]) {
        return false;
    }
    const int64_t n_experts = gate_logits->ne[0];
    const int64_t top_k = selected_experts->ne[0];
    if (top_k <= 0 || top_k > weights_capacity) {
        return false;
    }

    const char* logits_base = static_cast<const char*>(gate_logits->data);
    const char* selected_base = static_cast<const char*>(selected_experts->data);
    float denom = 0.0f;
    for (int64_t k = 0; k < top_k; ++k) {
        const int32_t expert =
            *reinterpret_cast<const int32_t*>(selected_base + static_cast<size_t>(k) *
                                                                  static_cast<size_t>(selected_experts->nb[0]) +
                                              static_cast<size_t>(token) *
                                                  static_cast<size_t>(selected_experts->nb[1]));
        float value = 0.0f;
        if (expert >= 0 && expert < n_experts) {
            const float logit =
                *reinterpret_cast<const float*>(logits_base + static_cast<size_t>(expert) *
                                                                  static_cast<size_t>(gate_logits->nb[0]) +
                                                static_cast<size_t>(token) *
                                                    static_cast<size_t>(gate_logits->nb[1]));
            value = 1.0f / (1.0f + std::exp(-logit));
            denom += value;
        }
        weights[k] = value;
    }
    const float scale = (routed_scale != 0.0f && routed_scale != 1.0f) ? routed_scale : 1.0f;
    if (normalize_weights) {
        denom = std::max(denom, 6.103515625e-5f);
        for (int64_t k = 0; k < top_k; ++k) {
            weights[k] = (weights[k] / denom) * scale;
        }
    } else if (scale != 1.0f) {
        for (int64_t k = 0; k < top_k; ++k) {
            weights[k] *= scale;
        }
    }
    return true;
}

static void ProbeQwen35NativeMoEDownWeightedLogitsReference(const ggml_tensor* dst, const ggml_tensor* down_exps,
                                                            const ggml_tensor* hidden,
                                                            const ggml_tensor* selected_experts,
                                                            const ggml_tensor* gate_logits, int64_t row_start,
                                                            int64_t row_end,
                                                            const Qwen35SharedQ8RowsUserData* shared_q8) {
    if (!dst || !down_exps || !hidden || !selected_experts || !gate_logits || !dst->data || !down_exps->data ||
        !hidden->data || !selected_experts->data || !gate_logits->data || row_start >= row_end) {
        return;
    }
    const int layer_idx = shared_q8 ? shared_q8->debug_layer_idx : -1;
    const int64_t n_tokens = selected_experts->ne[1];
    const int64_t target_token_env = ParseIntEnv("DENSECORE_DEBUG_QWEN35_NATIVE_MOE_REFERENCE_TOKEN", -1);
    const int64_t token = target_token_env >= 0 ? target_token_env : n_tokens - 1;
    if (token < 0 || token >= n_tokens ||
        !ShouldRunQwen35NativeMoEReferenceProbe(layer_idx, "down_weighted_logits", token)) {
        return;
    }
    const int64_t top_k = selected_experts->ne[0];
    if (top_k <= 0 || top_k > 64) {
        return;
    }
    float topk_weights[64];
    const bool weights_ok =
        shared_q8 && shared_q8->weighted_logits_lfm2_sigmoid
            ? LFM2NativeMoEComputeTopKWeightsFromLogits(gate_logits, selected_experts, token, topk_weights, 64,
                                                        shared_q8->weighted_logits_norm_topk,
                                                        shared_q8->weighted_logits_scale)
            : Qwen35NativeMoEComputeTopKWeightsFromLogits(gate_logits, selected_experts, token, topk_weights, 64);
    if (!weights_ok) {
        return;
    }
    const int64_t K = down_exps->ne[0];
    const size_t w_row_bytes = static_cast<size_t>(down_exps->nb[1]);
    thread_local std::vector<uint8_t> qbuf;
    float max_abs_diff = 0.0f;
    int64_t max_row = -1;
    float max_actual = 0.0f;
    float max_ref = 0.0f;
    for (int64_t row = row_start; row < row_end; ++row) {
        float ref = 0.0f;
        for (int64_t k = 0; k < top_k; ++k) {
            int32_t expert = -1;
            if (!Qwen35NativeMoEDownQ5KReadExpert(selected_experts, down_exps, token, k, &expert)) {
                continue;
            }
            const float gate_weight = topk_weights[k];
            if (gate_weight == 0.0f) {
                continue;
            }
            const float* hidden_row = Qwen35NativeMoEDownHiddenRowPtr(hidden, token, k);
            const uint8_t* qrow =
                (down_exps->type != GGML_TYPE_Q8_0 && shared_q8) ? Qwen35SharedQ8RowPtr(shared_q8, k, token)
                                                                 : nullptr;
            if (!qrow) {
                if (!Qwen35NativeMoEDownQuantizeHiddenForWeight(hidden, token, k, down_exps->type, qbuf)) continue;
                qrow = qbuf.data();
            }
            float sum = 0.0f;
            bool ok = false;
            if (down_exps->type == GGML_TYPE_Q8_0 && hidden_row) {
                ok = Qwen35NativeMoEDownQ8_0ReferenceDotRowForExpertWithHidden(down_exps, expert, row, hidden_row,
                                                                                &sum);
            } else {
                const char* expert_base = static_cast<const char*>(down_exps->data) +
                                          static_cast<size_t>(expert) * static_cast<size_t>(down_exps->nb[2]);
                const void* w_row = expert_base + static_cast<size_t>(row) * w_row_bytes;
                ok = Qwen35NativeMoEReferenceDotQXK(down_exps->type, w_row, qrow, K, &sum);
            }
            if (ok) {
                ref += sum * gate_weight;
            }
        }
        const auto* actual_ptr = reinterpret_cast<const float*>(
            static_cast<const char*>(dst->data) + static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
            static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1]));
        const float actual = *actual_ptr;
        const float diff = std::fabs(actual - ref);
        if (diff > max_abs_diff) {
            max_abs_diff = diff;
            max_row = row;
            max_actual = actual;
            max_ref = ref;
        }
    }
    std::fprintf(stderr,
                 "[QWEN35_NATIVE_MOE_REF] layer=%d stage=down_weighted_logits token=%lld rows=[%lld,%lld) "
                 "max_abs_diff=%.8g row=%lld actual=%.8g ref=%.8g tol=%.8g\n",
                 layer_idx, static_cast<long long>(token), static_cast<long long>(row_start),
                 static_cast<long long>(row_end), max_abs_diff, static_cast<long long>(max_row), max_actual, max_ref,
                 Qwen35NativeMoEReferenceTolerance());
}

static void RunQwen35NativeMoEDownQ5KWeightedSumFastPath(ggml_tensor* dst, const ggml_tensor* down_exps,
                                                          const ggml_tensor* hidden,
                                                          const ggml_tensor* selected_experts,
                                                          const ggml_tensor* weights, int ith, int nth,
                                                          Qwen35SharedQ8RowsUserData* shared_q8) {
    if (!dst || !down_exps || !hidden || !selected_experts || !weights || !dst->data || !down_exps->data ||
        !hidden->data || !selected_experts->data || !weights->data || nth <= 0) {
        return;
    }
    const int64_t n_embd = down_exps->ne[1];
    const int64_t top_k = selected_experts->ne[0];
    const int64_t n_tokens = selected_experts->ne[1];
    if (dst->type != GGML_TYPE_F32 || dst->ne[0] != n_embd || dst->ne[1] != n_tokens ||
        hidden->ne[1] != top_k || hidden->ne[2] != n_tokens || weights->type != GGML_TYPE_F32 ||
        weights->ne[0] != 1 || weights->ne[1] != top_k || weights->ne[2] != n_tokens) {
        return;
    }

    const int64_t pair_count = n_embd / 2;
    const int64_t pair_start = (static_cast<int64_t>(ith) * pair_count) / nth;
    const int64_t pair_end = (static_cast<int64_t>(ith + 1) * pair_count) / nth;
    const bool owns_odd_tail = (n_embd & 1) != 0 && ith == nth - 1;
    const bool use_shared_q8 =
        down_exps->type != GGML_TYPE_Q8_0 && PrepareQwen35SharedQ8Rows(shared_q8, hidden, ith, nth);
    thread_local std::vector<uint8_t> qbuf;
    thread_local std::vector<Qwen35MoEAssignment> assignments;

    const bool batched_qxk_down =
        down_exps->type == GGML_TYPE_Q4_K || down_exps->type == GGML_TYPE_Q5_K || down_exps->type == GGML_TYPE_Q6_K;
    if (n_tokens > 4 && batched_qxk_down && use_shared_q8) {
        size_t shared_assignment_count = 0;
        const Qwen35MoEAssignment* shared_assignments =
            PrepareQwen35SharedMoEAssignments(shared_q8, selected_experts, down_exps->ne[2], ith,
                                              &shared_assignment_count);
        if (shared_assignments && shared_assignment_count > 0) {
            assignments.assign(shared_assignments, shared_assignments + shared_assignment_count);
        } else if (!Qwen35BuildMoEAssignments(selected_experts, down_exps->ne[2], &assignments)) {
            assignments.clear();
        }
    }
    if (n_tokens > 4 && batched_qxk_down && use_shared_q8 && !assignments.empty()) {
        for (Qwen35MoEAssignment& assignment : assignments) {
            const float* weight_ptr = Qwen35NativeMoEDownWeightPtr(weights, assignment.token, assignment.topk_index);
            assignment.weight = weight_ptr ? *weight_ptr : 0.0f;
        }
        const int64_t row_start = (static_cast<int64_t>(ith) * n_embd) / nth;
        const int64_t row_end = (static_cast<int64_t>(ith + 1) * n_embd) / nth;
        if (row_start >= row_end) return;
        Qwen35NativeMoEZeroDst2DRange(dst, row_start, row_end, n_tokens);
        const size_t qrow_bytes = shared_q8->row_bytes;
        thread_local std::vector<uint8_t> qtile;
        thread_local std::vector<Qwen35MoEAssignment> tile_assignments;
        qtile.resize(static_cast<size_t>(kQwen35NativeMoEBatchedQ4KMaxAssignments) * qrow_bytes);
        tile_assignments.reserve(static_cast<size_t>(kQwen35NativeMoEBatchedQ4KMaxAssignments));
        for (size_t group_start = 0; group_start < assignments.size();) {
            const int32_t expert = assignments[group_start].expert;
            size_t group_end = group_start + 1;
            while (group_end < assignments.size() && assignments[group_end].expert == expert) {
                ++group_end;
            }
            for (size_t tile_start = group_start; tile_start < group_end;) {
                tile_assignments.clear();
                while (tile_start < group_end &&
                       tile_assignments.size() < static_cast<size_t>(kQwen35NativeMoEBatchedQ4KMaxAssignments)) {
                    const Qwen35MoEAssignment& assignment = assignments[tile_start++];
                    if (assignment.weight == 0.0f) {
                        continue;
                    }
                    const uint8_t* qrow = Qwen35SharedQ8RowPtr(shared_q8, assignment.topk_index, assignment.token);
                    if (!qrow) {
                        continue;
                    }
                    std::memcpy(qtile.data() + tile_assignments.size() * qrow_bytes, qrow, qrow_bytes);
                    tile_assignments.push_back(assignment);
                }
                if (tile_assignments.empty()) {
                    continue;
                }
                const bool ok = Qwen35NativeMoEDownQXKAccumulateAssignmentsForRowRange(
                    dst, down_exps, qtile.data(), qrow_bytes, tile_assignments, expert, row_start, row_end);
                if (!ok) {
                    return;
                }
            }
            group_start = group_end;
        }
        return;
    }
    if (n_tokens > 4 && !assignments.empty()) {
        for (Qwen35MoEAssignment& assignment : assignments) {
            const float* weight_ptr = Qwen35NativeMoEDownWeightPtr(weights, assignment.token, assignment.topk_index);
            assignment.weight = weight_ptr ? *weight_ptr : 0.0f;
        }
        for (int64_t token = 0; token < n_tokens; ++token) {
            for (int64_t pair = pair_start; pair < pair_end; ++pair) {
                const int64_t row = pair * 2;
                *reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                          static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                                          static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1])) = 0.0f;
                *reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                          static_cast<size_t>(row + 1) * static_cast<size_t>(dst->nb[0]) +
                                          static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1])) = 0.0f;
            }
            if (owns_odd_tail) {
                const int64_t row = n_embd - 1;
                *reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                          static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                                          static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1])) = 0.0f;
            }
        }
        for (size_t group_start = 0; group_start < assignments.size();) {
            const int32_t expert = assignments[group_start].expert;
            size_t group_end = group_start + 1;
            while (group_end < assignments.size() && assignments[group_end].expert == expert) {
                ++group_end;
            }
            for (int64_t pair = pair_start; pair < pair_end; ++pair) {
                const int64_t row = pair * 2;
                for (size_t ai = group_start; ai < group_end; ++ai) {
                    const Qwen35MoEAssignment& assignment = assignments[ai];
                    if (assignment.weight == 0.0f) continue;
                    const float* hidden_row =
                        Qwen35NativeMoEDownHiddenRowPtr(hidden, assignment.token, assignment.topk_index);
                    const uint8_t* qrow =
                        (down_exps->type != GGML_TYPE_Q8_0 && use_shared_q8)
                            ? Qwen35SharedQ8RowPtr(shared_q8, assignment.topk_index, assignment.token)
                            : nullptr;
                    if (!qrow) {
                        if (!Qwen35NativeMoEDownQuantizeHiddenForWeight(hidden, assignment.token,
                                                                        assignment.topk_index, down_exps->type,
                                                                        qbuf)) {
                            continue;
                        }
                        qrow = qbuf.data();
                    }
                    float value0 = 0.0f;
                    float value1 = 0.0f;
                    if (!Qwen35NativeMoEDownQ5KDotRowPairForExpertWithQbuf(down_exps, expert, row, qrow, hidden_row,
                                                                           &value0, &value1)) {
                        continue;
                    }
                    float* dst0 = reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                                           static_cast<size_t>(row) *
                                                               static_cast<size_t>(dst->nb[0]) +
                                                           static_cast<size_t>(assignment.token) *
                                                               static_cast<size_t>(dst->nb[1]));
                    float* dst1 = reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                                           static_cast<size_t>(row + 1) *
                                                               static_cast<size_t>(dst->nb[0]) +
                                                           static_cast<size_t>(assignment.token) *
                                                               static_cast<size_t>(dst->nb[1]));
                    *dst0 += value0 * assignment.weight;
                    *dst1 += value1 * assignment.weight;
                }
            }
            if (owns_odd_tail) {
                const int64_t row = n_embd - 1;
                for (size_t ai = group_start; ai < group_end; ++ai) {
                    const Qwen35MoEAssignment& assignment = assignments[ai];
                    if (assignment.weight == 0.0f) continue;
                    const float* hidden_row =
                        Qwen35NativeMoEDownHiddenRowPtr(hidden, assignment.token, assignment.topk_index);
                    const uint8_t* qrow =
                        (down_exps->type != GGML_TYPE_Q8_0 && use_shared_q8)
                            ? Qwen35SharedQ8RowPtr(shared_q8, assignment.topk_index, assignment.token)
                            : nullptr;
                    if (!qrow) {
                        if (!Qwen35NativeMoEDownQuantizeHiddenForWeight(hidden, assignment.token,
                                                                        assignment.topk_index, down_exps->type,
                                                                        qbuf)) {
                            continue;
                        }
                        qrow = qbuf.data();
                    }
                    float value = 0.0f;
                    if (Qwen35NativeMoEDownQ5KDotRowForExpertWithQbuf(down_exps, expert, row, qrow, hidden_row,
                                                                       &value)) {
                        float* dst_ptr =
                            reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                                     static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                                                     static_cast<size_t>(assignment.token) *
                                                         static_cast<size_t>(dst->nb[1]));
                        *dst_ptr += value * assignment.weight;
                    }
                }
            }
            group_start = group_end;
        }
        return;
    }

    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t pair = pair_start; pair < pair_end; ++pair) {
            const int64_t row = pair * 2;
            *reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                      static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                                      static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1])) = 0.0f;
            *reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                      static_cast<size_t>(row + 1) * static_cast<size_t>(dst->nb[0]) +
                                      static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1])) = 0.0f;
        }
        if (owns_odd_tail) {
            const int64_t row = n_embd - 1;
            *reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                      static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                                      static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1])) = 0.0f;
        }

        for (int64_t k = 0; k < top_k; ++k) {
            int32_t expert = -1;
            if (!Qwen35NativeMoEDownQ5KReadExpert(selected_experts, down_exps, token, k, &expert)) continue;
            const float* weight_ptr = Qwen35NativeMoEDownWeightPtr(weights, token, k);
            if (!weight_ptr) continue;
            const float gate_weight = *weight_ptr;
            if (gate_weight == 0.0f) continue;
            const float* hidden_row = Qwen35NativeMoEDownHiddenRowPtr(hidden, token, k);
            const uint8_t* qrow =
                (down_exps->type != GGML_TYPE_Q8_0 && use_shared_q8) ? Qwen35SharedQ8RowPtr(shared_q8, k, token)
                                                                     : nullptr;
            if (!qrow) {
                if (!Qwen35NativeMoEDownQuantizeHiddenForWeight(hidden, token, k, down_exps->type, qbuf)) continue;
                qrow = qbuf.data();
            }
            for (int64_t pair = pair_start; pair < pair_end; ++pair) {
                const int64_t row = pair * 2;
                float value0 = 0.0f;
                float value1 = 0.0f;
                if (!Qwen35NativeMoEDownQ5KDotRowPairForExpertWithQbuf(down_exps, expert, row, qrow, hidden_row,
                                                                       &value0, &value1)) {
                    continue;
                }
                float* dst0 = reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                                       static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                                                       static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1]));
                float* dst1 = reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                                       static_cast<size_t>(row + 1) * static_cast<size_t>(dst->nb[0]) +
                                                       static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1]));
                *dst0 += value0 * gate_weight;
                *dst1 += value1 * gate_weight;
            }
            if (owns_odd_tail) {
                const int64_t row = n_embd - 1;
                float value = 0.0f;
                if (Qwen35NativeMoEDownQ5KDotRowForExpertWithQbuf(down_exps, expert, row, qrow, hidden_row,
                                                                   &value)) {
                    float* dst_ptr =
                        reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                                 static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                                                 static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1]));
                    *dst_ptr += value * gate_weight;
                }
            }
        }
    }
}

static void RunQwen35NativeMoEDownQ5KWeightedLogitsFastPath(ggml_tensor* dst, const ggml_tensor* down_exps,
                                                             const ggml_tensor* hidden,
                                                             const ggml_tensor* selected_experts,
                                                             const ggml_tensor* gate_logits, int ith, int nth,
                                                             Qwen35SharedQ8RowsUserData* shared_q8) {
    if (!dst || !down_exps || !hidden || !selected_experts || !gate_logits || !dst->data || !down_exps->data ||
        !hidden->data || !selected_experts->data || !gate_logits->data || nth <= 0) {
        return;
    }
    const int64_t n_embd = down_exps->ne[1];
    const int64_t top_k = selected_experts->ne[0];
    const int64_t n_tokens = selected_experts->ne[1];
    if (dst->type != GGML_TYPE_F32 || dst->ne[0] != n_embd || dst->ne[1] != n_tokens ||
        hidden->ne[1] != top_k || hidden->ne[2] != n_tokens || gate_logits->type != GGML_TYPE_F32 ||
        gate_logits->ne[1] != n_tokens || gate_logits->ne[0] != down_exps->ne[2] || top_k > 64) {
        return;
    }

    thread_local std::vector<uint8_t> qbuf;
    const int64_t pair_count = n_embd / 2;
    const int64_t pair_start = (static_cast<int64_t>(ith) * pair_count) / nth;
    const int64_t pair_end = (static_cast<int64_t>(ith + 1) * pair_count) / nth;
    const bool owns_odd_tail = (n_embd & 1) != 0 && ith == nth - 1;
    const bool use_shared_q8 =
        down_exps->type != GGML_TYPE_Q8_0 && PrepareQwen35SharedQ8Rows(shared_q8, hidden, ith, nth);
    thread_local std::vector<Qwen35MoEAssignment> assignments;

    const bool batched_qxk_down =
        down_exps->type == GGML_TYPE_Q4_K || down_exps->type == GGML_TYPE_Q5_K || down_exps->type == GGML_TYPE_Q6_K;
    if (n_tokens > 4 && batched_qxk_down && use_shared_q8) {
        size_t shared_assignment_count = 0;
        const Qwen35MoEAssignment* shared_assignments =
            PrepareQwen35SharedMoEAssignments(shared_q8, selected_experts, down_exps->ne[2], ith,
                                              &shared_assignment_count);
        if (shared_assignments && shared_assignment_count > 0) {
            assignments.assign(shared_assignments, shared_assignments + shared_assignment_count);
        } else if (!Qwen35BuildMoEAssignments(selected_experts, down_exps->ne[2], &assignments)) {
            assignments.clear();
        }
    }
    if (n_tokens > 4 && batched_qxk_down && use_shared_q8 && !assignments.empty()) {
        const int64_t row_start = (static_cast<int64_t>(ith) * n_embd) / nth;
        const int64_t row_end = (static_cast<int64_t>(ith + 1) * n_embd) / nth;
        if (row_start >= row_end) return;
        thread_local std::vector<float> topk_weight_matrix;
        topk_weight_matrix.assign(static_cast<size_t>(n_tokens) * static_cast<size_t>(top_k), 0.0f);
        for (int64_t token = 0; token < n_tokens; ++token) {
            float topk_weights[64];
            const bool weights_ok =
                shared_q8 && shared_q8->weighted_logits_lfm2_sigmoid
                    ? LFM2NativeMoEComputeTopKWeightsFromLogits(
                          gate_logits, selected_experts, token, topk_weights, 64,
                          shared_q8->weighted_logits_norm_topk, shared_q8->weighted_logits_scale)
                    : Qwen35NativeMoEComputeTopKWeightsFromLogits(gate_logits, selected_experts, token, topk_weights,
                                                                  64);
            if (!weights_ok) {
                continue;
            }
            float* token_weights =
                topk_weight_matrix.data() + static_cast<size_t>(token) * static_cast<size_t>(top_k);
            for (int64_t k = 0; k < top_k; ++k) {
                token_weights[static_cast<size_t>(k)] = topk_weights[k];
            }
        }
        for (Qwen35MoEAssignment& assignment : assignments) {
            assignment.weight = 0.0f;
            if (assignment.token >= 0 && assignment.token < n_tokens && assignment.topk_index >= 0 &&
                assignment.topk_index < top_k) {
                assignment.weight =
                    topk_weight_matrix[static_cast<size_t>(assignment.token) * static_cast<size_t>(top_k) +
                                       static_cast<size_t>(assignment.topk_index)];
            }
        }
        Qwen35NativeMoEZeroDst2DRange(dst, row_start, row_end, n_tokens);
        const size_t qrow_bytes = shared_q8->row_bytes;
        thread_local std::vector<uint8_t> qtile;
        thread_local std::vector<Qwen35MoEAssignment> tile_assignments;
        qtile.resize(static_cast<size_t>(kQwen35NativeMoEBatchedQ4KMaxAssignments) * qrow_bytes);
        tile_assignments.reserve(static_cast<size_t>(kQwen35NativeMoEBatchedQ4KMaxAssignments));
        for (size_t group_start = 0; group_start < assignments.size();) {
            const int32_t expert = assignments[group_start].expert;
            size_t group_end = group_start + 1;
            while (group_end < assignments.size() && assignments[group_end].expert == expert) {
                ++group_end;
            }
            for (size_t tile_start = group_start; tile_start < group_end;) {
                tile_assignments.clear();
                while (tile_start < group_end &&
                       tile_assignments.size() < static_cast<size_t>(kQwen35NativeMoEBatchedQ4KMaxAssignments)) {
                    const Qwen35MoEAssignment& assignment = assignments[tile_start++];
                    if (assignment.weight == 0.0f) {
                        continue;
                    }
                    const uint8_t* qrow = Qwen35SharedQ8RowPtr(shared_q8, assignment.topk_index, assignment.token);
                    if (!qrow) {
                        continue;
                    }
                    std::memcpy(qtile.data() + tile_assignments.size() * qrow_bytes, qrow, qrow_bytes);
                    tile_assignments.push_back(assignment);
                }
                if (tile_assignments.empty()) {
                    continue;
                }
                const bool ok = Qwen35NativeMoEDownQXKAccumulateAssignmentsForRowRange(
                    dst, down_exps, qtile.data(), qrow_bytes, tile_assignments, expert, row_start, row_end);
                if (!ok) {
                    return;
                }
            }
            group_start = group_end;
        }
        ProbeQwen35NativeMoEDownWeightedLogitsReference(dst, down_exps, hidden, selected_experts, gate_logits,
                                                        row_start, row_end, shared_q8);
        return;
    }

    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t pair = pair_start; pair < pair_end; ++pair) {
            const int64_t row = pair * 2;
            *reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                      static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                                      static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1])) = 0.0f;
            *reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                      static_cast<size_t>(row + 1) * static_cast<size_t>(dst->nb[0]) +
                                      static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1])) = 0.0f;
        }
        if (owns_odd_tail) {
            const int64_t row = n_embd - 1;
            *reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                      static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                                      static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1])) = 0.0f;
        }

        float topk_weights[64];
        const bool weights_ok =
            shared_q8 && shared_q8->weighted_logits_lfm2_sigmoid
                ? LFM2NativeMoEComputeTopKWeightsFromLogits(gate_logits, selected_experts, token, topk_weights, 64,
                                                            shared_q8->weighted_logits_norm_topk,
                                                            shared_q8->weighted_logits_scale)
                : Qwen35NativeMoEComputeTopKWeightsFromLogits(gate_logits, selected_experts, token, topk_weights, 64);
        if (!weights_ok) {
            continue;
        }
        for (int64_t k = 0; k < top_k; ++k) {
            int32_t expert = -1;
            if (!Qwen35NativeMoEDownQ5KReadExpert(selected_experts, down_exps, token, k, &expert)) continue;
            const float gate_weight = topk_weights[k];
            if (gate_weight == 0.0f) continue;
            const float* hidden_row = Qwen35NativeMoEDownHiddenRowPtr(hidden, token, k);
            const uint8_t* qrow =
                (down_exps->type != GGML_TYPE_Q8_0 && use_shared_q8) ? Qwen35SharedQ8RowPtr(shared_q8, k, token)
                                                                     : nullptr;
            if (!qrow) {
                if (!Qwen35NativeMoEDownQuantizeHiddenForWeight(hidden, token, k, down_exps->type, qbuf)) continue;
                qrow = qbuf.data();
            }
            for (int64_t pair = pair_start; pair < pair_end; ++pair) {
                const int64_t row = pair * 2;
                float value0 = 0.0f;
                float value1 = 0.0f;
                if (!Qwen35NativeMoEDownQ5KDotRowPairForExpertWithQbuf(down_exps, expert, row, qrow, hidden_row,
                                                                       &value0, &value1)) {
                    continue;
                }
                float* dst0 = reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                                       static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                                                       static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1]));
                float* dst1 = reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                                       static_cast<size_t>(row + 1) * static_cast<size_t>(dst->nb[0]) +
                                                       static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1]));
                *dst0 += value0 * gate_weight;
                *dst1 += value1 * gate_weight;
            }
            if (owns_odd_tail) {
                const int64_t row = n_embd - 1;
                float value = 0.0f;
                if (Qwen35NativeMoEDownQ5KDotRowForExpertWithQbuf(down_exps, expert, row, qrow, hidden_row,
                                                                   &value)) {
                    float* dst_ptr =
                        reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                                 static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                                                 static_cast<size_t>(token) * static_cast<size_t>(dst->nb[1]));
                    *dst_ptr += value * gate_weight;
                }
            }
        }
    }
    const int64_t row_start = pair_start * 2;
    const int64_t row_end = pair_end * 2 + (owns_odd_tail ? 1 : 0);
    ProbeQwen35NativeMoEDownWeightedLogitsReference(dst, down_exps, hidden, selected_experts, gate_logits,
                                                    row_start, std::min<int64_t>(row_end, n_embd), shared_q8);
}

static void cb_qwen35_native_moe_down_q5k(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    const auto start = std::chrono::steady_clock::now();
    auto* shared_q8 = static_cast<Qwen35SharedQ8RowsUserData*>(userdata);
    int effective_ith = ith;
    int effective_nth = nth;
    if (!RemapNativeMoECallbackTask(shared_q8, ith, nth, &effective_ith, &effective_nth)) {
        return;
    }
    if (ith == 0 && IsQwen35NativeMoEDownQ5KDiagEnabled()) {
        static std::atomic<int> exec_count{0};
        int c = exec_count.fetch_add(1, std::memory_order_relaxed);
        if (c < 5 || (c % 200 == 0)) {
            std::fprintf(stderr, "[W2_Q5K_DIAG] CALLBACK EXECUTING: count=%d nth=%d\n", c + 1, nth);
        }
    }
    RunQwen35NativeMoEDownQ5KFastPath(dst, dst ? dst->src[0] : nullptr,
                                      dst ? dst->src[1] : nullptr,
                                      dst ? dst->src[2] : nullptr, effective_ith, effective_nth,
                                      shared_q8);
    if (ith == 0) {
        const ggml_tensor* selected = dst ? dst->src[2] : nullptr;
        const int selected_experts = selected ? static_cast<int>(std::max<int64_t>(0, selected->ne[0])) : 0;
        InferenceWorkContext* work_ctx =
            shared_q8 && shared_q8->work_ctx ? shared_q8->work_ctx : GetCurrentWorkContext();
        if (GetCurrentExecutionPhase() == InferenceExecutionPhase::Decode) {
            RecordNativeMoEGraphCallbackExecution(work_ctx, selected_experts, effective_nth);
        }
        const auto end = std::chrono::steady_clock::now();
        const auto wall_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        RecordNativeMoEFastDecodeDecision(work_ctx, /*candidate=*/true, /*used=*/true, nullptr,
                                         /*w1w3_used=*/false, /*w2_used=*/true, wall_ns);
        RecordNativeMoEFastW2Q5KDecision(work_ctx, /*candidate=*/true, /*used=*/true, nullptr, wall_ns);
    }
}

static void cb_qwen35_native_moe_down_q5k_weighted_sum(struct ggml_tensor* dst, int ith, int nth,
                                                        void* userdata) {
    const auto start = std::chrono::steady_clock::now();
    auto* shared_q8 = static_cast<Qwen35SharedQ8RowsUserData*>(userdata);
    int effective_ith = ith;
    int effective_nth = nth;
    if (!RemapNativeMoECallbackTask(shared_q8, ith, nth, &effective_ith, &effective_nth)) {
        return;
    }
    if (ith == 0 && IsQwen35NativeMoEDownQ5KDiagEnabled()) {
        static std::atomic<int> exec_count{0};
        int c = exec_count.fetch_add(1, std::memory_order_relaxed);
        if (c < 5 || (c % 200 == 0)) {
            std::fprintf(stderr, "[W2_Q5K_DIAG] WEIGHTED CALLBACK EXECUTING: count=%d nth=%d\n", c + 1, nth);
        }
    }
    RunQwen35NativeMoEDownQ5KWeightedSumFastPath(dst, dst ? dst->src[0] : nullptr,
                                                dst ? dst->src[1] : nullptr,
                                                dst ? dst->src[2] : nullptr,
                                                dst ? dst->src[3] : nullptr, effective_ith, effective_nth,
                                                shared_q8);
    if (ith == 0) {
        const ggml_tensor* selected = dst ? dst->src[2] : nullptr;
        const int selected_experts = selected ? static_cast<int>(std::max<int64_t>(0, selected->ne[0])) : 0;
        InferenceWorkContext* work_ctx =
            shared_q8 && shared_q8->work_ctx ? shared_q8->work_ctx : GetCurrentWorkContext();
        if (GetCurrentExecutionPhase() == InferenceExecutionPhase::Decode) {
            RecordNativeMoEGraphCallbackExecution(work_ctx, selected_experts, effective_nth);
        }
        const auto end = std::chrono::steady_clock::now();
        const auto wall_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        RecordNativeMoEFastDecodeDecision(work_ctx, /*candidate=*/true, /*used=*/true, nullptr,
                                         /*w1w3_used=*/false, /*w2_used=*/true, wall_ns);
        RecordNativeMoEFastW2Q5KDecision(work_ctx, /*candidate=*/true, /*used=*/true, nullptr, wall_ns);
    }
}

static void cb_gemma4_native_moe_down_weighted_sum(struct ggml_tensor* dst, int ith, int nth, void* userdata) {
    const auto start = std::chrono::steady_clock::now();
    auto* shared_q8 = static_cast<Qwen35SharedQ8RowsUserData*>(userdata);
    int effective_ith = ith;
    int effective_nth = nth;
    if (!RemapNativeMoECallbackTask(shared_q8, ith, nth, &effective_ith, &effective_nth)) {
        return;
    }
    RunQwen35NativeMoEDownQ5KWeightedSumFastPath(dst, dst ? dst->src[0] : nullptr,
                                                dst ? dst->src[1] : nullptr,
                                                dst ? dst->src[2] : nullptr,
                                                dst ? dst->src[3] : nullptr, effective_ith, effective_nth,
                                                shared_q8);
    if (ith == 0) {
        InferenceWorkContext* work_ctx =
            shared_q8 && shared_q8->work_ctx ? shared_q8->work_ctx : GetCurrentWorkContext();
        const auto end = std::chrono::steady_clock::now();
        const auto wall_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        RecordGemma4DecodeNativeDecision(work_ctx, /*candidate=*/true, /*used=*/true, nullptr,
                                         /*moe_used=*/true, /*dense_used=*/false, /*lm_head_used=*/false, wall_ns,
                                         /*replaced_mul_mat_ops=*/0, /*replaced_mul_mat_id_ops=*/1,
                                         /*duplicate_work_detected=*/false);
    }
}

static void cb_qwen35_native_moe_down_q5k_weighted_logits(struct ggml_tensor* dst, int ith, int nth,
                                                           void* userdata) {
    const auto start = std::chrono::steady_clock::now();
    auto* shared_q8 = static_cast<Qwen35SharedQ8RowsUserData*>(userdata);
    int effective_ith = ith;
    int effective_nth = nth;
    if (!RemapNativeMoECallbackTask(shared_q8, ith, nth, &effective_ith, &effective_nth)) {
        return;
    }
    if (ith == 0 && IsQwen35NativeMoEDownQ5KDiagEnabled()) {
        static std::atomic<int> exec_count{0};
        int c = exec_count.fetch_add(1, std::memory_order_relaxed);
        if (c < 5 || (c % 200 == 0)) {
            std::fprintf(stderr, "[W2_Q5K_DIAG] WEIGHTED LOGITS CALLBACK EXECUTING: count=%d nth=%d\n", c + 1, nth);
        }
    }
    RunQwen35NativeMoEDownQ5KWeightedLogitsFastPath(dst, dst ? dst->src[0] : nullptr,
                                                   dst ? dst->src[1] : nullptr,
                                                   dst ? dst->src[2] : nullptr,
                                                   dst ? dst->src[3] : nullptr, effective_ith, effective_nth,
                                                   shared_q8);
    if (ith == 0) {
        const ggml_tensor* selected = dst ? dst->src[2] : nullptr;
        const int selected_experts = selected ? static_cast<int>(std::max<int64_t>(0, selected->ne[0])) : 0;
        InferenceWorkContext* work_ctx =
            shared_q8 && shared_q8->work_ctx ? shared_q8->work_ctx : GetCurrentWorkContext();
        if (GetCurrentExecutionPhase() == InferenceExecutionPhase::Decode) {
            RecordNativeMoEGraphCallbackExecution(work_ctx, selected_experts, effective_nth);
        }
        const auto end = std::chrono::steady_clock::now();
        const auto wall_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        RecordNativeMoEFastDecodeDecision(work_ctx, /*candidate=*/true, /*used=*/true, nullptr,
                                         /*w1w3_used=*/false, /*w2_used=*/true, wall_ns);
        RecordNativeMoEFastW2Q5KDecision(work_ctx, /*candidate=*/true, /*used=*/true, nullptr, wall_ns);
    }
}

static bool CanReplaceQwen35W2WithCustomCallback(const TransformerModel* model, const ggml_tensor* down_exps,
                                                  const ggml_tensor* hidden, const ggml_tensor* selected_experts) {
    auto reject = [](const char* reason) {
        if (IsQwen35NativeMoEDownQ5KDiagEnabled()) {
            std::fprintf(stderr, "[W2_Q5K_DIAG] rejected: %s\n", reason);
        }
        return false;
    };
    if (!model || !down_exps || !hidden || !selected_experts) return reject("null_ptr");
    const bool qwen_native_moe = (model->variant == ModelVariant::QWEN35 || model->variant == ModelVariant::QWEN36) &&
                                 model->arch_flags.is_hybrid_ssm;
    const bool lfm2_native_moe = model->variant == ModelVariant::LFM2MOE && model->arch_flags.is_lfm2_shortconv;
    if (!qwen_native_moe && !lfm2_native_moe) return reject("wrong_variant");
    if (model->hparams.n_experts <= 0) return reject("no_experts");
    const bool supported_down_quant =
        lfm2_native_moe ? (down_exps->type == GGML_TYPE_Q4_K || down_exps->type == GGML_TYPE_Q5_K ||
                           down_exps->type == GGML_TYPE_Q6_K || down_exps->type == GGML_TYPE_Q8_0)
                        : (down_exps->type == GGML_TYPE_Q4_K || down_exps->type == GGML_TYPE_Q5_K ||
                           down_exps->type == GGML_TYPE_Q6_K || down_exps->type == GGML_TYPE_Q8_0);
    if (!supported_down_quant) {
        return reject("unsupported_w2_quant");
    }
    if (down_exps->type == GGML_TYPE_Q6_K) {
        const ggml_type_traits_cpu* traits = ggml_get_type_traits_cpu(GGML_TYPE_Q6_K);
        if (!traits || !traits->vec_dot || traits->vec_dot_type != GGML_TYPE_Q8_K) {
            return reject("q6k_vec_dot_unavailable");
        }
    }
    if (hidden->type != GGML_TYPE_F32) return reject("hidden_not_f32");
    if (selected_experts->type != GGML_TYPE_I32) return reject("experts_not_i32");
    if (down_exps->ne[0] % ggml_blck_size(down_exps->type) != 0) return reject("bad_alignment");
    if ((hidden->ne[0] % QK_K) != 0) return reject("bad_q8k_alignment");
    const InferenceExecutionPhase phase = GetCurrentExecutionPhase();
    if (phase != InferenceExecutionPhase::Decode && phase != InferenceExecutionPhase::Prefill) {
        return reject("unsupported_phase");
    }
    const BatchSpec* current_batch = GetCurrentBatch();
    const auto& fast_config = ResolveFastPathRuntimeConfig(current_batch);
    const bool fast_moe_enabled =
        ShouldEnableNativeMoEFastPathByDefault(model, phase, fast_config.native_moe_fast_decode);
    if (!fast_moe_enabled) {
        return reject("native_moe_fast_decode_disabled");
    }
    if (current_batch && !current_batch->lora_map.empty()) {
        return reject("dynamic_lora");
    }
    if (selected_experts->ne[1] <= 0 || selected_experts->ne[1] > NativeMoEFastPathMaxDirectTokens(model)) {
        return reject("unsupported_token_count");
    }
    if (IsQwen35NativeMoEDownQ5KDiagEnabled()) {
        std::fprintf(stderr, "[W2_Q5K_DIAG] ACCEPTED — replacing ggml_mul_mat_id with custom Q5K callback\n");
    }
    return true;
}

static bool CanFuseQwen35W2NormWeightsFromLogitsWithCustomCallback(const TransformerModel* model,
                                                                    const ggml_tensor* down_exps,
                                                                    const ggml_tensor* hidden,
                                                                    const ggml_tensor* selected_experts,
                                                                    const ggml_tensor* gate_logits) {
    if (!CanReplaceQwen35W2WithCustomCallback(model, down_exps, hidden, selected_experts)) {
        return false;
    }
    auto reject = [](const char* reason) {
        if (IsQwen35NativeMoEDownQ5KDiagEnabled()) {
            std::fprintf(stderr, "[W2_Q5K_DIAG] weighted_logits rejected: %s\n", reason);
        }
        return false;
    };
    if (!gate_logits || gate_logits->type != GGML_TYPE_F32) return reject("logits_not_f32");
    if (gate_logits->ne[0] != down_exps->ne[2] || gate_logits->ne[1] != selected_experts->ne[1]) {
        return reject("logits_shape");
    }
    if (selected_experts->ne[0] <= 0 || selected_experts->ne[0] > 64) return reject("topk_too_large");
    const bool lfm2_native_moe = model->variant == ModelVariant::LFM2MOE && model->arch_flags.is_lfm2_shortconv;
    if (lfm2_native_moe) {
        return true;
    }
    if (!model->moe_norm_topk_prob) return reject("topk_not_normalized");
    if (model->moe_routed_scaling_factor != 0.0f && model->moe_routed_scaling_factor != 1.0f) {
        return reject("scaled_weights");
    }
    return true;
}

static bool CanFuseQwen35W2WeightedSumWithCustomCallback(const TransformerModel* model, const ggml_tensor* down_exps,
                                                          const ggml_tensor* hidden,
                                                          const ggml_tensor* selected_experts,
                                                          const ggml_tensor* weights) {
    if (!CanReplaceQwen35W2WithCustomCallback(model, down_exps, hidden, selected_experts)) {
        return false;
    }
    auto reject = [](const char* reason) {
        if (IsQwen35NativeMoEDownQ5KDiagEnabled()) {
            std::fprintf(stderr, "[W2_Q5K_DIAG] weighted_sum rejected: %s\n", reason);
        }
        return false;
    };
    if (!weights || weights->type != GGML_TYPE_F32) return reject("weights_not_f32");
    if (weights->ne[0] != 1 || weights->ne[1] != selected_experts->ne[0] ||
        weights->ne[2] != selected_experts->ne[1]) {
        return reject("weights_shape");
    }
    return true;
}

static const float* Qwen35NativeMoEGateUpInputRowPtr(const ggml_tensor* input, int64_t token) {
    if (!input || !input->data || input->type != GGML_TYPE_F32 || token < 0 || token >= input->ne[1]) {
        return nullptr;
    }
    return reinterpret_cast<const float*>(static_cast<const char*>(input->data) +
                                          static_cast<size_t>(token) * static_cast<size_t>(input->nb[1]));
}

static bool Qwen35NativeMoEGateUpQuantizeInput(const ggml_tensor* input, int64_t token, std::vector<uint8_t>& qbuf) {
    const float* input_row = Qwen35NativeMoEGateUpInputRowPtr(input, token);
    if (!input_row) return false;
    const int64_t cols = input->ne[0];
    const size_t qrow_bytes = ggml_row_size(GGML_TYPE_Q8_K, cols);
    if (qrow_bytes == 0) return false;
    qbuf.resize(qrow_bytes);
    return Qwen35NativeQuantizeRowQ8K(input_row, qbuf.data(), cols);
}

static bool Qwen35NativeMoEQ4KQ8KDotRow(const void* weight_row, const uint8_t* qrow, int64_t cols,
                                        float* out_value) {
    if (PreferGgmlQ4KVecDotForNativeMoE()) {
        const ggml_type_traits_cpu* traits = ggml_get_type_traits_cpu(GGML_TYPE_Q4_K);
        if (traits && traits->vec_dot && traits->vec_dot_type == GGML_TYPE_Q8_K &&
            (cols % ggml_blck_size(GGML_TYPE_Q4_K)) == 0) {
            traits->vec_dot(static_cast<int>(cols), out_value, 0, weight_row, 0, qrow, 0, 1);
            return true;
        }
    }
    return densecore::hwy_kernels::DotQ4KQ8K_Hwy(weight_row, qrow, cols, out_value);
}

static bool Qwen35NativeMoEKQ8KDotRow(ggml_type weight_type, const void* weight_row, const uint8_t* qrow,
                                      int64_t cols, float* out_value) {
    if (weight_type == GGML_TYPE_Q4_K) {
        return Qwen35NativeMoEQ4KQ8KDotRow(weight_row, qrow, cols, out_value);
    }
    if (weight_type == GGML_TYPE_Q5_K) {
        return densecore::hwy_kernels::DotQ5KQ8K_Hwy(weight_row, qrow, cols, out_value);
    }
    return false;
}

static bool Qwen35NativeMoEKQ8KFusedSwiGLURows(InferenceWorkContext* work_ctx, ggml_type weight_type,
                                               const void* gate_row_start, const void* up_row_start,
                                               const uint8_t* qrow, int64_t cols, int64_t row_count,
                                               size_t row_bytes, float* out_start) {
    if (!gate_row_start || !up_row_start || !qrow || !out_start || cols <= 0 || row_count <= 0 || row_bytes == 0) {
        return false;
    }
    if (weight_type != GGML_TYPE_Q4_K && weight_type != GGML_TYPE_Q5_K) {
        return false;
    }
    if (weight_type == GGML_TYPE_Q5_K) {
        RecordLFM2NativeMoEW1W3Kernel(work_ctx, "q5k_hwy");
        const auto* gate_base = static_cast<const char*>(gate_row_start);
        const auto* up_base = static_cast<const char*>(up_row_start);
        for (int64_t row = 0; row < row_count; ++row) {
            const void* gate_row = gate_base + static_cast<size_t>(row) * row_bytes;
            const void* up_row = up_base + static_cast<size_t>(row) * row_bytes;
            float gate = 0.0f;
            float up = 0.0f;
            if (!densecore::hwy_kernels::DotQ5KQ8K_Hwy(gate_row, qrow, cols, &gate) ||
                !densecore::hwy_kernels::DotQ5KQ8K_Hwy(up_row, qrow, cols, &up)) {
                return false;
            }
            out_start[row] = NativeMoESiLU(gate) * up;
        }
        return true;
    }
    // LFM2 W1/W3 (Q4_K gate/up) decode: routing this fused-SwiGLU through ggml's
    // K-quant 2-row vec_dot (rowpair, nrc=2) produced long-form repetition and
    // garbage on ARM. The C4A final-sweep showed lfm2_w1w3_q4k_vecdot_rowpair
    // dominating while QA failed (repetition_detected;garbage_output_detected),
    // whereas C4 (x86) used the validated Highway kernel (lfm2_w1w3_q4k_hwy) and
    // passed. The row-pair K-quant vec_dot is not parity-verified for this
    // fused-SwiGLU shape/input-layout, so force the known-good Highway path for
    // Q4_K on every ISA. This matches x86, where PreferGgmlQ4KVecDotForNativeMoE()
    // is already false and the ggml vec_dot branch was never taken. Q5_K above
    // already uses Highway unconditionally. Re-enabling the row-pair fast path
    // requires wiring it through ParityGate (kernel_caps.h) so a divergent kernel
    // auto-falls back to this Highway reference.
    RecordLFM2NativeMoEW1W3Kernel(work_ctx, "q4k_hwy");
    return densecore::hwy_kernels::FusedSwiGLUQ4KQ8KRows_Hwy(gate_row_start, up_row_start, qrow, cols, row_count,
                                                             row_bytes, out_start);
}

static void ProbeQwen35NativeMoEGateUpReference(const ggml_tensor* dst, const ggml_tensor* gate_exps,
                                                const ggml_tensor* up_exps, const ggml_tensor* input,
                                                const ggml_tensor* selected_experts, int64_t row_start,
                                                int64_t row_end, const Qwen35SharedQ8RowsUserData* shared_q8) {
    if (!dst || !gate_exps || !up_exps || !input || !selected_experts || !dst->data || !gate_exps->data ||
        !up_exps->data || !input->data || !selected_experts->data || row_start >= row_end) {
        return;
    }
    const int layer_idx = shared_q8 ? shared_q8->debug_layer_idx : -1;
    const int64_t n_tokens = selected_experts->ne[1];
    const int64_t target_token_env = ParseIntEnv("DENSECORE_DEBUG_QWEN35_NATIVE_MOE_REFERENCE_TOKEN", -1);
    const int64_t token = target_token_env >= 0 ? target_token_env : n_tokens - 1;
    if (token < 0 || token >= n_tokens ||
        !ShouldRunQwen35NativeMoEReferenceProbe(layer_idx, "gateup", token)) {
        return;
    }
    const int64_t top_k = selected_experts->ne[0];
    const int64_t cols = gate_exps->ne[0];
    const size_t row_bytes = static_cast<size_t>(gate_exps->nb[1]);
    thread_local std::vector<uint8_t> qbuf;
    const uint8_t* qrow = shared_q8 ? Qwen35SharedQ8RowPtr(shared_q8, token, 0) : nullptr;
    if (!qrow) {
        if (!Qwen35NativeMoEGateUpQuantizeInput(input, token, qbuf)) {
            return;
        }
        qrow = qbuf.data();
    }

    float max_abs_diff = 0.0f;
    int64_t max_topk = -1;
    int64_t max_row = -1;
    float max_actual = 0.0f;
    float max_ref = 0.0f;
    for (int64_t k = 0; k < top_k; ++k) {
        int32_t expert = -1;
        if (!Qwen35NativeMoEDownQ5KReadExpert(selected_experts, gate_exps, token, k, &expert)) {
            continue;
        }
        const char* gate_base = static_cast<const char*>(gate_exps->data) +
                                static_cast<size_t>(expert) * static_cast<size_t>(gate_exps->nb[2]);
        const char* up_base = static_cast<const char*>(up_exps->data) +
                              static_cast<size_t>(expert) * static_cast<size_t>(up_exps->nb[2]);
        for (int64_t row = row_start; row < row_end; ++row) {
            const void* gate_row = gate_base + static_cast<size_t>(row) * row_bytes;
            const void* up_row = up_base + static_cast<size_t>(row) * static_cast<size_t>(up_exps->nb[1]);
            float gate_ref = 0.0f;
            float up_ref = 0.0f;
            if (!Qwen35NativeMoEReferenceDotQXK(gate_exps->type, gate_row, qrow, cols, &gate_ref) ||
                !Qwen35NativeMoEReferenceDotQXK(up_exps->type, up_row, qrow, cols, &up_ref)) {
                continue;
            }
            const float ref = NativeMoESiLU(gate_ref) * up_ref;
            const auto* actual_ptr = reinterpret_cast<const float*>(
                static_cast<const char*>(dst->data) + static_cast<size_t>(row) * static_cast<size_t>(dst->nb[0]) +
                static_cast<size_t>(k) * static_cast<size_t>(dst->nb[1]) +
                static_cast<size_t>(token) * static_cast<size_t>(dst->nb[2]));
            const float actual = *actual_ptr;
            const float diff = std::fabs(actual - ref);
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
                max_topk = k;
                max_row = row;
                max_actual = actual;
                max_ref = ref;
            }
        }
    }
    std::fprintf(stderr,
                 "[QWEN35_NATIVE_MOE_REF] layer=%d stage=gateup token=%lld rows=[%lld,%lld) max_abs_diff=%.8g "
                 "topk=%lld row=%lld actual=%.8g ref=%.8g tol=%.8g\n",
                 layer_idx, static_cast<long long>(token), static_cast<long long>(row_start),
                 static_cast<long long>(row_end), max_abs_diff, static_cast<long long>(max_topk),
                 static_cast<long long>(max_row), max_actual, max_ref, Qwen35NativeMoEReferenceTolerance());
}

static bool Qwen35GateUpQ5KRepackKernelAvailable() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return ggml_cpu_has_neon() && ggml_cpu_has_dotprod();
#else
    return false;
#endif
}

static bool Qwen35GateUpQ4KRepackKernelAvailable() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return false;
#else
    return densecore::kernels::Q4KRealPackedGemvKernelAvailable() && ggml_cpu_has_avx2();
#endif
}

static void RunQwen35NativeMoEGateUpRawQXKSwiGLU(ggml_tensor* dst, const ggml_tensor* gate_exps,
                                                 const ggml_tensor* up_exps, const ggml_tensor* input,
                                                 const ggml_tensor* selected_experts, int ith, int nth,
                                                 Qwen35SharedQ8RowsUserData* shared_q8) {
    if (!dst || !gate_exps || !up_exps || !input || !selected_experts || !dst->data || nth <= 0) {
        return;
    }
    const int64_t n_ff = dst->ne[0];
    const int64_t top_k = selected_experts->ne[0];
    const int64_t n_tokens = selected_experts->ne[1];
    const bool q4_gateup = gate_exps->type == GGML_TYPE_Q4_K && up_exps->type == GGML_TYPE_Q4_K;
    const bool q5_gateup = gate_exps->type == GGML_TYPE_Q5_K && up_exps->type == GGML_TYPE_Q5_K;
    const bool q5_single_copy_8x8 = q5_gateup && shared_q8 && shared_q8->q5k_gateup_8x8_single_copy;
    const bool q5_single_copy_required =
        q5_gateup && shared_q8 && shared_q8->q5k_gateup_8x8_single_copy_required;
    InferenceWorkContext* work_ctx = shared_q8 && shared_q8->work_ctx ? shared_q8->work_ctx : GetCurrentWorkContext();
    InferenceWorkContext* lfm2_w1w3_work_ctx =
        shared_q8 && shared_q8->record_lfm2_w1w3_kernel ? work_ctx : nullptr;
    if (dst->type != GGML_TYPE_F32 || (!q4_gateup && !q5_gateup) || input->type != GGML_TYPE_F32 ||
        selected_experts->type != GGML_TYPE_I32 || gate_exps->ne[1] != n_ff || up_exps->ne[1] != n_ff ||
        dst->ne[1] != top_k || dst->ne[2] != n_tokens || gate_exps->ne[2] != up_exps->ne[2]) {
        return;
    }
    const int64_t block_size = ggml_blck_size(gate_exps->type);
    if (gate_exps->ne[0] != up_exps->ne[0] || block_size <= 0 || (gate_exps->ne[0] % block_size) != 0) {
        return;
    }

    const int64_t row_start = (static_cast<int64_t>(ith) * n_ff) / nth;
    const int64_t row_end = (static_cast<int64_t>(ith + 1) * n_ff) / nth;
    const int64_t row_count = row_end - row_start;
    const bool compact_gateup_rows = gate_exps->nb[1] == ggml_row_size(gate_exps->type, gate_exps->ne[0]) &&
                                     up_exps->nb[1] == ggml_row_size(up_exps->type, up_exps->ne[0]);
    const bool use_shared_q8 = PrepareQwen35SharedQ8Rows(shared_q8, input, ith, nth);
    thread_local std::vector<uint8_t> qbuf;
    thread_local std::vector<Qwen35MoEAssignment> assignments;
    thread_local std::vector<uint8_t> gateup_qtile;
    thread_local std::vector<float> gateup_input_tile;
    thread_local std::vector<float> gateup_tile_out;
    if (ith == 0 && shared_q8) {
        shared_q8->repacked_swiglu_failed.store(0, std::memory_order_relaxed);
    }
    const bool lfm2_q4k_repacked_prefill =
        shared_q8 && shared_q8->record_lfm2_w1w3_kernel && GetCurrentExecutionPhase() == InferenceExecutionPhase::Prefill;
    // Qwen Q5_K decode fast lane: run the loader-owned single-copy q5_K_8x8
    // layout through ggml's GEMV kernel. Do not build a runtime repack cache
    // here: keeping both raw Q5_K and q5_K_8x8 copies was the RAM wall for 35B
    // Q5. LFM2 Q5 does not use this Qwen-only layout and must fall through to
    // its validated raw vecdot lane instead of being rejected here.
    if (q5_gateup && (q5_single_copy_8x8 || q5_single_copy_required) && n_tokens == 1 && use_shared_q8 &&
        dst->nb[0] == static_cast<int64_t>(sizeof(float)) && (n_ff % 8) == 0) {
        if (!Qwen35GateUpQ5KRepackKernelAvailable()) {
            if (ith == 0) {
                RecordMoEQ5KRepackedDecision(work_ctx, /*candidate=*/true, /*used=*/false,
                                             "qwen35_gateup_q5k_repack_kernel_unavailable");
            }
            shared_q8->repacked_swiglu_failed.store(1, std::memory_order_relaxed);
            return;
        }
        if (!q5_single_copy_8x8) {
            if (ith == 0) {
                RecordMoEQ5KRepackedDecision(work_ctx, /*candidate=*/true, /*used=*/false,
                                             "qwen35_gateup_q5k_single_copy_missing");
            }
            shared_q8->repacked_swiglu_failed.store(1, std::memory_order_relaxed);
            return;
        }
        if (ith == 0) {
            RecordMoEQ5KRepackedDecision(work_ctx, /*candidate=*/true, /*used=*/true, nullptr);
        }
        const int64_t hidden_dim = gate_exps->ne[0];
        const size_t row_size = ggml_row_size(GGML_TYPE_Q5_K, hidden_dim);
        const int64_t tile_total = n_ff / 8;
        const int64_t tile_lo = (static_cast<int64_t>(ith) * tile_total) / nth;
        const int64_t tile_hi = (static_cast<int64_t>(ith + 1) * tile_total) / nth;
        const int64_t r0 = tile_lo * 8;
        const int64_t r1 = tile_hi * 8;
        if (r1 > r0) {
            const int nc = static_cast<int>(r1 - r0);
            const uint8_t* qrow = Qwen35SharedQ8RowPtr(shared_q8, 0, 0);
            if (!qrow) {
                shared_q8->repacked_swiglu_failed.store(1, std::memory_order_relaxed);
                return;
            }
            const size_t tile_byte_off = static_cast<size_t>(r0) * row_size;
            thread_local std::vector<float> q5k_gate_buf;
            thread_local std::vector<float> q5k_up_buf;
            if (static_cast<int>(q5k_gate_buf.size()) < nc) q5k_gate_buf.resize(static_cast<size_t>(nc));
            if (static_cast<int>(q5k_up_buf.size()) < nc) q5k_up_buf.resize(static_cast<size_t>(nc));
            for (int64_t k = 0; k < top_k; ++k) {
                int32_t expert = -1;
                if (!Qwen35NativeMoEDownQ5KReadExpert(selected_experts, gate_exps, 0, k, &expert)) {
                    continue;
                }
                const uint8_t* gate_vx = static_cast<const uint8_t*>(gate_exps->data) +
                                         static_cast<size_t>(expert) * static_cast<size_t>(gate_exps->nb[2]) +
                                         tile_byte_off;
                const uint8_t* up_vx = static_cast<const uint8_t*>(up_exps->data) +
                                       static_cast<size_t>(expert) * static_cast<size_t>(up_exps->nb[2]) +
                                       tile_byte_off;
                ggml_gemv_q5_K_8x8_q8_K(static_cast<int>(hidden_dim), q5k_gate_buf.data(), 0, gate_vx, qrow, 1,
                                        nc);
                ggml_gemv_q5_K_8x8_q8_K(static_cast<int>(hidden_dim), q5k_up_buf.data(), 0, up_vx, qrow, 1, nc);
                float* out_col = reinterpret_cast<float*>(
                    static_cast<char*>(dst->data) + static_cast<size_t>(k) * static_cast<size_t>(dst->nb[1]));
                for (int i = 0; i < nc; ++i) {
                    out_col[r0 + i] =
                        NativeMoESiLU(q5k_gate_buf[static_cast<size_t>(i)]) * q5k_up_buf[static_cast<size_t>(i)];
                }
            }
        }
        return;
    }
    size_t assignment_count = 0;
    const Qwen35MoEAssignment* assignment_data = nullptr;
    if ((q4_gateup || q5_gateup) && n_tokens > 4 && use_shared_q8) {
        assignment_data = PrepareQwen35SharedMoEAssignments(shared_q8, selected_experts, gate_exps->ne[2], ith,
                                                            &assignment_count);
        if (!assignment_data && Qwen35BuildMoEAssignments(selected_experts, gate_exps->ne[2], &assignments)) {
            assignment_data = assignments.data();
            assignment_count = assignments.size();
        }
    }
    if ((q4_gateup || q5_gateup) && n_tokens > 4 && use_shared_q8 && assignment_data && assignment_count > 0) {
        if (row_count <= 0) {
            return;
        }
        const size_t weight_row_bytes =
            q5_single_copy_8x8 ? ggml_row_size(GGML_TYPE_Q5_K, gate_exps->ne[0]) : static_cast<size_t>(gate_exps->nb[1]);
        for (size_t group_start = 0; group_start < assignment_count;) {
            const int32_t expert = assignment_data[group_start].expert;
            size_t group_end = group_start + 1;
            while (group_end < assignment_count && assignment_data[group_end].expert == expert) {
                ++group_end;
            }
            const char* gate_base = static_cast<const char*>(gate_exps->data) +
                                    static_cast<size_t>(expert) * static_cast<size_t>(gate_exps->nb[2]);
            const char* up_base = static_cast<const char*>(up_exps->data) +
                                  static_cast<size_t>(expert) * static_cast<size_t>(up_exps->nb[2]);
            const void* gate_row_start = gate_base + static_cast<size_t>(row_start) * weight_row_bytes;
            const void* up_row_start = up_base + static_cast<size_t>(row_start) * weight_row_bytes;
            const InferenceExecutionPhase phase = GetCurrentExecutionPhase();
            const bool q4_repacked_gateup =
                q4_gateup && shared_q8 && compact_gateup_rows && (row_count % 8) == 0 &&
                Qwen35GateUpQ4KRepackKernelAvailable() &&
                ((shared_q8->prefer_q4k_repacked_swiglu && phase == InferenceExecutionPhase::Decode) ||
                 lfm2_q4k_repacked_prefill);
            const bool repacked_gateup = q5_single_copy_8x8 || q4_repacked_gateup;
            const bool can_use_batched_gateup =
                (q4_gateup || q5_gateup) && (compact_gateup_rows || q5_single_copy_8x8) && row_count > 0 &&
                group_end - group_start >= 2 && shared_q8 && shared_q8->row_bytes > 0 &&
                (!q5_single_copy_8x8 || ((row_start % 8) == 0 && (row_count % 8) == 0));
            if (can_use_batched_gateup) {
                densecore::CpuBackend& backend = densecore::GetCpuBackend();
                const size_t qrow_bytes = shared_q8->row_bytes;
                gateup_qtile.resize(static_cast<size_t>(kQwen35NativeMoEGateUpBatchTile) * qrow_bytes);
                if (repacked_gateup) {
                    gateup_input_tile.resize(static_cast<size_t>(kQwen35NativeMoEGateUpBatchTile) *
                                             static_cast<size_t>(gate_exps->ne[0]));
                }
                gateup_tile_out.resize(static_cast<size_t>(kQwen35NativeMoEGateUpBatchTile) *
                                       static_cast<size_t>(row_count));
                for (size_t tile_start = group_start; tile_start < group_end;) {
                    size_t tile_count = 0;
                    while (tile_start < group_end && tile_count < static_cast<size_t>(kQwen35NativeMoEGateUpBatchTile)) {
                        const Qwen35MoEAssignment& assignment = assignment_data[tile_start++];
                        const uint8_t* qrow = Qwen35SharedQ8RowPtr(shared_q8, assignment.token, 0);
                        const float* input_row =
                            repacked_gateup ? Qwen35NativeMoEGateUpInputRowPtr(input, assignment.token) : nullptr;
                        if (!qrow || (repacked_gateup && !input_row)) {
                            shared_q8->repacked_swiglu_failed.store(1, std::memory_order_relaxed);
                            return;
                        }
                        std::memcpy(gateup_qtile.data() + tile_count * qrow_bytes, qrow, qrow_bytes);
                        if (repacked_gateup) {
                            std::memcpy(gateup_input_tile.data() + tile_count * static_cast<size_t>(gate_exps->ne[0]),
                                        input_row, static_cast<size_t>(gate_exps->ne[0]) * sizeof(float));
                        }
                        ++tile_count;
                    }
                    if (tile_count == 0) {
                        continue;
                    }
                    bool ok = false;
                    if (q5_single_copy_8x8) {
                        ok = densecore::RunQ5KRepackedMoEFusedSwiGLURawProjection(
                            &backend, gate_row_start, up_row_start, gateup_input_tile.data(), gateup_qtile.data(),
                            qrow_bytes, gateup_tile_out.data(), static_cast<int64_t>(tile_count), row_count,
                            gate_exps->ne[0], /*numa_node=*/0, /*allow_parallel=*/false);
                    } else if (q4_repacked_gateup) {
                        RecordLFM2NativeMoEW1W3Kernel(lfm2_w1w3_work_ctx, "q4k_repacked");
                        ok = densecore::RunQ4KRepackedMoEFusedSwiGLUProjection(
                            &backend, gate_row_start, up_row_start, gateup_input_tile.data(), gateup_qtile.data(),
                            qrow_bytes, gateup_tile_out.data(), static_cast<int64_t>(tile_count), row_count,
                            gate_exps->ne[0], /*numa_node=*/0, /*allow_parallel=*/false);
                    } else {
                        ok = densecore::RunMoEKQuantRawBatchedFusedSwiGLU(
                            &backend, static_cast<int>(gate_exps->type), gate_row_start, up_row_start,
                            gateup_qtile.data(), qrow_bytes, gateup_tile_out.data(), static_cast<int64_t>(tile_count),
                            row_count, gate_exps->ne[0], /*numa_node=*/0, /*allow_parallel=*/false);
                    }
                    if (!ok) {
                        shared_q8->repacked_swiglu_failed.store(1, std::memory_order_relaxed);
                        return;
                    }
                    const size_t emitted_start = tile_start - tile_count;
                    for (size_t tm = 0; tm < tile_count; ++tm) {
                        const Qwen35MoEAssignment& assignment = assignment_data[emitted_start + tm];
                        float* out = reinterpret_cast<float*>(
                            static_cast<char*>(dst->data) +
                            static_cast<size_t>(row_start) * static_cast<size_t>(dst->nb[0]) +
                            static_cast<size_t>(assignment.topk_index) * static_cast<size_t>(dst->nb[1]) +
                            static_cast<size_t>(assignment.token) * static_cast<size_t>(dst->nb[2]));
                        const float* src = gateup_tile_out.data() + tm * static_cast<size_t>(row_count);
                        if (dst->nb[0] == static_cast<int64_t>(sizeof(float))) {
                            std::memcpy(out, src, static_cast<size_t>(row_count) * sizeof(float));
                        } else {
                            for (int64_t row = 0; row < row_count; ++row) {
                                *reinterpret_cast<float*>(reinterpret_cast<char*>(out) +
                                                          static_cast<size_t>(row) *
                                                              static_cast<size_t>(dst->nb[0])) = src[row];
                            }
                        }
                    }
                }
                group_start = group_end;
                continue;
            }
            for (size_t ai = group_start; ai < group_end; ++ai) {
                const Qwen35MoEAssignment& assignment = assignment_data[ai];
                const uint8_t* qrow = Qwen35SharedQ8RowPtr(shared_q8, assignment.token, 0);
                if (!qrow) {
                    if (shared_q8) {
                        shared_q8->repacked_swiglu_failed.store(1, std::memory_order_relaxed);
                    }
                    return;
                }
                float* out = reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                                      static_cast<size_t>(row_start) *
                                                          static_cast<size_t>(dst->nb[0]) +
                                                      static_cast<size_t>(assignment.topk_index) *
                                                          static_cast<size_t>(dst->nb[1]) +
                                                      static_cast<size_t>(assignment.token) *
                                                          static_cast<size_t>(dst->nb[2]));
                const InferenceExecutionPhase phase = GetCurrentExecutionPhase();
                const bool q4_repacked_gateup =
                    q4_gateup && shared_q8 && compact_gateup_rows && (row_count % 8) == 0 &&
                    Qwen35GateUpQ4KRepackKernelAvailable() &&
                    ((shared_q8->prefer_q4k_repacked_swiglu && phase == InferenceExecutionPhase::Decode) ||
                     lfm2_q4k_repacked_prefill);
                if (q5_single_copy_8x8 || q4_repacked_gateup) {
                    const float* input_row =
                        q5_single_copy_8x8 ? Qwen35NativeMoEGateUpInputRowPtr(input, assignment.token) : nullptr;
                    const bool q5_row_aligned = !q5_single_copy_8x8 || ((row_start % 8) == 0 && (row_count % 8) == 0);
                    const bool ok =
                        q5_single_copy_8x8
                            ? (input_row && q5_row_aligned &&
                               densecore::RunQ5KRepackedMoEFusedSwiGLURawProjection(
                                   &densecore::GetCpuBackend(), gate_row_start, up_row_start, input_row, qrow,
                                   shared_q8->row_bytes, out, /*rows=*/1, row_count, gate_exps->ne[0],
                                   /*numa_node=*/0, /*allow_parallel=*/false))
                            : densecore::RunQ4KRepackedMoEFusedSwiGLUProjection(
                                  &densecore::GetCpuBackend(), gate_row_start, up_row_start, nullptr, qrow,
                                  shared_q8->row_bytes, out, /*rows=*/1, row_count, gate_exps->ne[0],
                                  /*numa_node=*/0, /*allow_parallel=*/false);
                    if (ok && !q5_single_copy_8x8) {
                        RecordLFM2NativeMoEW1W3Kernel(lfm2_w1w3_work_ctx, "q4k_repacked");
                    }
                    if (!ok) {
                        if (shared_q8) {
                            shared_q8->repacked_swiglu_failed.store(1, std::memory_order_relaxed);
                        }
                        return;
                    }
                    continue;
                }
                const bool ok = Qwen35NativeMoEKQ8KFusedSwiGLURows(lfm2_w1w3_work_ctx, gate_exps->type, gate_row_start,
                                                                    up_row_start, qrow, gate_exps->ne[0], row_count,
                                                                    weight_row_bytes, out);
                if (!ok) {
                    if (shared_q8) {
                        shared_q8->repacked_swiglu_failed.store(1, std::memory_order_relaxed);
                    }
                    return;
                }
            }
            group_start = group_end;
        }
        if (!q5_single_copy_8x8) {
            ProbeQwen35NativeMoEGateUpReference(dst, gate_exps, up_exps, input, selected_experts, row_start, row_end,
                                                shared_q8);
        }
        return;
    }
    if (q5_single_copy_8x8) {
        if (shared_q8) {
            shared_q8->repacked_swiglu_failed.store(1, std::memory_order_relaxed);
        }
        return;
    }
    for (int64_t token = 0; token < n_tokens; ++token) {
        const uint8_t* qrow = use_shared_q8 ? Qwen35SharedQ8RowPtr(shared_q8, token, 0) : nullptr;
        if (!qrow) {
            if (!Qwen35NativeMoEGateUpQuantizeInput(input, token, qbuf)) continue;
            qrow = qbuf.data();
        }
        for (int64_t k = 0; k < top_k; ++k) {
            int32_t expert = -1;
            if (!Qwen35NativeMoEDownQ5KReadExpert(selected_experts, gate_exps, token, k, &expert)) continue;
            const char* gate_base = static_cast<const char*>(gate_exps->data) +
                                    static_cast<size_t>(expert) * static_cast<size_t>(gate_exps->nb[2]);
            const char* up_base = static_cast<const char*>(up_exps->data) +
                                  static_cast<size_t>(expert) * static_cast<size_t>(up_exps->nb[2]);
            if (row_count <= 0) {
                continue;
            }
            const char* gate_row_start = gate_base + static_cast<size_t>(row_start) * static_cast<size_t>(gate_exps->nb[1]);
            const char* up_row_start = up_base + static_cast<size_t>(row_start) * static_cast<size_t>(up_exps->nb[1]);
            float* out_start = reinterpret_cast<float*>(static_cast<char*>(dst->data) +
                                                        static_cast<size_t>(row_start) * static_cast<size_t>(dst->nb[0]) +
                                                        static_cast<size_t>(k) * static_cast<size_t>(dst->nb[1]) +
                                                        static_cast<size_t>(token) * static_cast<size_t>(dst->nb[2]));
            if (!Qwen35NativeMoEKQ8KFusedSwiGLURows(lfm2_w1w3_work_ctx, gate_exps->type, gate_row_start, up_row_start,
                                                    qrow, gate_exps->ne[0], row_count,
                                                    static_cast<size_t>(gate_exps->nb[1]), out_start)) {
                if (shared_q8) {
                    shared_q8->repacked_swiglu_failed.store(1, std::memory_order_relaxed);
                }
                return;
            }
        }
    }
    ProbeQwen35NativeMoEGateUpReference(dst, gate_exps, up_exps, input, selected_experts, row_start, row_end,
                                        shared_q8);
}

static void cb_qwen35_native_moe_gateup_raw_qxk_swiglu(struct ggml_tensor* dst, int ith, int nth,
                                                        void* userdata) {
    const auto start = std::chrono::steady_clock::now();
    auto* shared_q8 = static_cast<Qwen35SharedQ8RowsUserData*>(userdata);
    int effective_ith = ith;
    int effective_nth = nth;
    if (!RemapNativeMoECallbackTask(shared_q8, ith, nth, &effective_ith, &effective_nth)) {
        return;
    }
    RunQwen35NativeMoEGateUpRawQXKSwiGLU(dst, dst ? dst->src[0] : nullptr, dst ? dst->src[1] : nullptr,
                                         dst ? dst->src[2] : nullptr, dst ? dst->src[3] : nullptr, effective_ith,
                                         effective_nth, shared_q8);
    if (ith == 0) {
        const int selected_experts = dst ? static_cast<int>(std::max<int64_t>(0, dst->ne[1])) : 0;
        InferenceWorkContext* work_ctx =
            shared_q8 && shared_q8->work_ctx ? shared_q8->work_ctx : GetCurrentWorkContext();
        if (GetCurrentExecutionPhase() == InferenceExecutionPhase::Decode) {
            RecordNativeMoEGraphCallbackExecution(work_ctx, selected_experts, effective_nth);
        }
        const auto end = std::chrono::steady_clock::now();
        const auto wall_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
        const bool swiglu_failed =
            shared_q8 && shared_q8->repacked_swiglu_failed.load(std::memory_order_relaxed) != 0;
        const char* reject_reason = nullptr;
        if (swiglu_failed) {
            reject_reason = shared_q8->q5k_gateup_8x8_single_copy_required
                                ? "qwen35_gateup_q5k_single_copy_failed"
                            : shared_q8->prefer_q4k_repacked_swiglu
                                ? "lfm2_w1w3_repacked_swiglu_failed"
                                : "lfm2_w1w3_range_swiglu_failed";
        }
        RecordNativeMoEFastDecodeDecision(work_ctx, /*candidate=*/true, /*used=*/!swiglu_failed, reject_reason,
                                         /*w1w3_used=*/!swiglu_failed, /*w2_used=*/false, swiglu_failed ? 0 : wall_ns);
    }
}

static bool CanUseQwen35NativeMoEGateUpRawQXKSwiGLU(const TransformerModel* model, const ggml_tensor* gate_exps,
                                                    const ggml_tensor* up_exps, const ggml_tensor* input,
                                                    const ggml_tensor* selected_experts) {
    auto reject = [](const char* reason) {
        if (IsQwen35NativeMoEDownQ5KDiagEnabled()) {
            std::fprintf(stderr, "[W1W3_QXK_DIAG] gateup_raw rejected: %s\n", reason);
        }
        return false;
    };
    if (!model || !gate_exps || !up_exps || !input || !selected_experts) return reject("missing_arg");
    const bool qwen_native_moe = (model->variant == ModelVariant::QWEN35 || model->variant == ModelVariant::QWEN36) &&
                                 model->arch_flags.is_hybrid_ssm;
    const bool lfm2_native_moe = model->variant == ModelVariant::LFM2MOE && model->arch_flags.is_lfm2_shortconv;
    if (!qwen_native_moe && !lfm2_native_moe) {
        return reject("unsupported_model");
    }
    const InferenceExecutionPhase phase = GetCurrentExecutionPhase();
    if (phase != InferenceExecutionPhase::Decode && phase != InferenceExecutionPhase::Prefill) {
        return reject("unsupported_phase");
    }
    const BatchSpec* current_batch = GetCurrentBatch();
    const auto& fast_config = ResolveFastPathRuntimeConfig(current_batch);
    const bool fast_moe_enabled =
        ShouldEnableNativeMoEFastPathByDefault(model, phase, fast_config.native_moe_fast_decode);
    if (!fast_moe_enabled) {
        return reject("native_moe_fast_decode_disabled");
    }
    if (current_batch && !current_batch->lora_map.empty()) return reject("dynamic_lora");
    const bool qwen_supported_gateup =
        qwen_native_moe && ((gate_exps->type == GGML_TYPE_Q4_K && up_exps->type == GGML_TYPE_Q4_K) ||
                            (gate_exps->type == GGML_TYPE_Q5_K && up_exps->type == GGML_TYPE_Q5_K));
    const bool lfm2_supported_gateup =
        lfm2_native_moe && ((gate_exps->type == GGML_TYPE_Q4_K && up_exps->type == GGML_TYPE_Q4_K) ||
                            (gate_exps->type == GGML_TYPE_Q5_K && up_exps->type == GGML_TYPE_Q5_K));
    if ((!qwen_supported_gateup && !lfm2_supported_gateup) || input->type != GGML_TYPE_F32 ||
        selected_experts->type != GGML_TYPE_I32) {
        return reject("unsupported_type");
    }
    if (!gate_exps->data || !up_exps->data) return reject("missing_weight_data");
    if (gate_exps->view_src || up_exps->view_src) return reject("weight_view");
    if (input->ne[1] <= 0 || input->ne[1] > NativeMoEFastPathMaxDirectTokens(model)) {
        return reject("unsupported_token_count");
    }
    if (selected_experts->ne[1] != input->ne[1]) return reject("selected_expert_token_mismatch");
    if (gate_exps->ne[0] != input->ne[0] || up_exps->ne[0] != input->ne[0] ||
        gate_exps->ne[1] != up_exps->ne[1] || gate_exps->ne[2] != up_exps->ne[2]) {
        return reject("shape_mismatch");
    }
    const int64_t block_size = ggml_blck_size(gate_exps->type);
    if (block_size <= 0 || (gate_exps->ne[0] % block_size) != 0) return reject("bad_kquant_alignment");
    if (IsQwen35NativeMoEDownQ5KDiagEnabled()) {
        std::fprintf(stderr, "[W1W3_QXK_DIAG] gateup_raw accepted type=%d\n", static_cast<int>(gate_exps->type));
    }
    return true;
}

struct QwenLikeNativeMoEGraphPlan {
    bool qwen_native_moe = false;
    bool lfm2_native_moe = false;
    bool target_requires_native_moe = false;
    InferenceExecutionPhase graph_phase = InferenceExecutionPhase::Unknown;
    bool lfm2_debug_reference = false;
    bool lfm2_fast_only_decode = false;
};

struct QwenLikeNativeMoEWeights {
    ggml_tensor* gate_exps = nullptr;
    ggml_tensor* up_exps = nullptr;
    ggml_tensor* down_exps = nullptr;
    ggml_tensor* gate_up_exps = nullptr;
    ggml_tensor* raw_gate_exps = nullptr;
    ggml_tensor* raw_up_exps = nullptr;
    ggml_tensor* raw_down_exps = nullptr;
    bool use_fused_gate_up = false;
    ggml_type w1w3_type = GGML_TYPE_COUNT;
    bool native_q5_gateup_single_copy = false;
};

static QwenLikeNativeMoEGraphPlan ResolveQwenLikeNativeMoEGraphPlan(const TransformerModel* model) {
    QwenLikeNativeMoEGraphPlan plan;
    plan.qwen_native_moe =
        model && (model->variant == ModelVariant::QWEN35 || model->variant == ModelVariant::QWEN36) &&
        model->arch_flags.is_hybrid_ssm;
    plan.lfm2_native_moe = model && model->variant == ModelVariant::LFM2MOE && model->arch_flags.is_lfm2_shortconv;
    plan.target_requires_native_moe = ModelRequiresNativeMoEFastPath(model);
    plan.graph_phase = GetCurrentExecutionPhase();
    plan.lfm2_debug_reference = plan.lfm2_native_moe && IsDebugLFM2NativeMoEReferenceEnabled();
    plan.lfm2_fast_only_decode =
        plan.lfm2_native_moe && plan.graph_phase == InferenceExecutionPhase::Decode && !plan.lfm2_debug_reference;
    return plan;
}

static bool ShouldUseQwenLikeGateUpQ4KRepackedSwiGLU(const QwenLikeNativeMoEGraphPlan& graph_plan,
                                                     ggml_type w1w3_type, InferenceExecutionPhase phase,
                                                     bool kernel_available) {
    if (!graph_plan.qwen_native_moe) {
        return false;
    }
    // C4 x86 validation showed the single-token LFM2 Q4_K repacked SwiGLU
    // path regresses decode throughput versus the raw k-quant row path.
    return w1w3_type == GGML_TYPE_Q4_K && phase == InferenceExecutionPhase::Decode && kernel_available;
}

static const char* ValidateQwenLikeNativeMoERouter(
    const QwenLikeNativeMoEGraphPlan& plan, const densecore::models::DecoderLayerSpec* layer_spec) {
    if (layer_spec && plan.qwen_native_moe &&
        layer_spec->ffn.router != densecore::models::DecoderMoERouter::SoftmaxTopK) {
        return "router_not_softmax_topk";
    }
    if (layer_spec && plan.lfm2_native_moe &&
        layer_spec->ffn.router != densecore::models::DecoderMoERouter::GroupedSigmoidTopK) {
        return "router_not_grouped_sigmoid_topk";
    }
    return nullptr;
}

static bool ResolveQwenLikeNativeMoEWeights(TransformerModel* model, TransformerLayer* layer,
                                            const QwenLikeNativeMoEGraphPlan& plan, int64_t n_tokens,
                                            QwenLikeNativeMoEWeights* weights, const char** reject_reason) {
    if (!model || !layer || !weights) {
        if (reject_reason) *reject_reason = "missing_graph_inputs";
        return false;
    }
    weights->gate_exps = GetLayerTensorAny(layer, {"ffn_gate_exps.weight", "ffn_gate_exps"});
    weights->up_exps = GetLayerTensorAny(layer, {"ffn_up_exps.weight", "ffn_up_exps"});
    weights->down_exps = GetLayerTensorAny(layer, {"ffn_down_exps.weight", "ffn_down_exps"});
    weights->gate_up_exps = GetLayerTensorAny(layer, {"ffn_gate_up_exps.cpu_repack_fused"});
    if (!weights->gate_exps || !weights->up_exps || !weights->down_exps) {
        if (reject_reason) *reject_reason = "missing_moe_weight_tensor";
        return false;
    }

    weights->raw_gate_exps = weights->gate_exps;
    weights->raw_up_exps = weights->up_exps;
    weights->raw_down_exps = weights->down_exps;
    if (!plan.lfm2_native_moe) {
        weights->gate_exps = UseCpuRepackAliasForTokenCount(model, weights->gate_exps, n_tokens);
        weights->up_exps = UseCpuRepackAliasForTokenCount(model, weights->up_exps, n_tokens);
        weights->down_exps = UseCpuRepackAliasForTokenCount(model, weights->down_exps, n_tokens);
        if (weights->gate_up_exps) {
            weights->gate_up_exps = UseCpuRepackAliasForTokenCount(model, weights->gate_up_exps, n_tokens);
        }
    }
    weights->use_fused_gate_up =
        weights->gate_up_exps && weights->gate_up_exps->type == weights->gate_exps->type &&
        weights->gate_up_exps->ne[0] == weights->gate_exps->ne[0] &&
        weights->gate_up_exps->ne[1] == weights->gate_exps->ne[1] + weights->up_exps->ne[1] &&
        weights->gate_up_exps->ne[2] == weights->gate_exps->ne[2];
    weights->w1w3_type = weights->use_fused_gate_up ? weights->gate_up_exps->type : weights->gate_exps->type;
    weights->native_q5_gateup_single_copy =
        !weights->use_fused_gate_up && weights->raw_gate_exps && weights->raw_up_exps &&
        weights->raw_gate_exps->type == GGML_TYPE_Q5_K && weights->raw_up_exps->type == GGML_TYPE_Q5_K &&
        model->q5k_8x8_repacked_tensors.find(weights->raw_gate_exps) != model->q5k_8x8_repacked_tensors.end() &&
        model->q5k_8x8_repacked_tensors.find(weights->raw_up_exps) != model->q5k_8x8_repacked_tensors.end();
    if ((plan.qwen_native_moe || plan.lfm2_native_moe) && weights->w1w3_type == GGML_TYPE_Q5_K &&
        !weights->native_q5_gateup_single_copy) {
        if (reject_reason) {
            *reject_reason =
                plan.qwen_native_moe ? "qwen35_gateup_q5k_single_copy_missing" : "lfm2_gateup_q5k_single_copy_missing";
        }
        return false;
    }
    return true;
}

static bool ValidateQwenLikeNativeMoEShapes(const QwenLikeNativeMoEWeights& weights, ggml_tensor* routed_input,
                                            ggml_tensor* gate_logits, int64_t n_tokens, int64_t n_embd,
                                            int64_t n_experts) {
    return n_tokens > 0 && n_embd > 0 && n_experts > 0 && routed_input && gate_logits &&
           gate_logits->ne[1] == n_tokens && weights.gate_exps && weights.up_exps && weights.down_exps &&
           weights.gate_exps->ne[0] == n_embd && weights.up_exps->ne[0] == n_embd &&
           weights.gate_exps->ne[2] == n_experts && weights.up_exps->ne[2] == n_experts &&
           weights.down_exps->ne[2] == n_experts && weights.gate_exps->ne[1] == weights.up_exps->ne[1] &&
           weights.down_exps->ne[0] == weights.gate_exps->ne[1] && weights.down_exps->ne[1] == n_embd;
}

static const char* RecordQwenLikeNativeMoEFastDecodeAdmission(const TransformerModel* model,
                                                              const QwenLikeNativeMoEGraphPlan& plan,
                                                              const QwenLikeNativeMoEWeights& weights,
                                                              int64_t n_tokens, int64_t n_experts,
                                                              int64_t n_expert_used) {
    InferenceWorkContext* work_ctx = GetCurrentWorkContext();
    if (!work_ctx) {
        return nullptr;
    }
    const BatchSpec* current_batch = GetCurrentBatch();
    const auto& fast_config = ResolveFastPathRuntimeConfig(current_batch);
    const bool dynamic_lora_active = current_batch && !current_batch->lora_map.empty();
    const bool supported_lfm2_w2_quant =
        weights.down_exps->type == GGML_TYPE_Q4_K || weights.down_exps->type == GGML_TYPE_Q5_K ||
        weights.down_exps->type == GGML_TYPE_Q6_K || weights.down_exps->type == GGML_TYPE_Q8_0;
    const bool supported_qwen_w2_quant =
        weights.down_exps->type == GGML_TYPE_Q4_K || weights.down_exps->type == GGML_TYPE_Q5_K ||
        weights.down_exps->type == GGML_TYPE_Q6_K || weights.down_exps->type == GGML_TYPE_Q8_0;
    const bool supported_w1w3_quant =
        plan.lfm2_native_moe ? (weights.w1w3_type == GGML_TYPE_Q4_K || weights.w1w3_type == GGML_TYPE_Q5_K)
                             : (weights.w1w3_type == GGML_TYPE_Q4_K || weights.w1w3_type == GGML_TYPE_Q5_K);
    const bool supported_w2_quant = plan.lfm2_native_moe ? supported_lfm2_w2_quant : supported_qwen_w2_quant;
    const bool supported_quant = supported_w1w3_quant && supported_w2_quant;
    const bool supported_shape =
        (plan.graph_phase == InferenceExecutionPhase::Decode || plan.graph_phase == InferenceExecutionPhase::Prefill) &&
        n_tokens > 0 && n_tokens <= NativeMoEFastPathMaxDirectTokens(model);
    const bool selected_experts_available = n_expert_used > 0 && n_expert_used <= n_experts;
    const bool fast_kernel_available = supported_quant;
    const bool fast_decode_candidate =
        plan.graph_phase == InferenceExecutionPhase::Decode && n_experts > 0 &&
        (plan.qwen_native_moe || plan.lfm2_native_moe);
    const bool fast_decode_enabled =
        ShouldEnableNativeMoEFastPathByDefault(model, plan.graph_phase, fast_config.native_moe_fast_decode);
    const bool fast_decode_supported =
        fast_decode_candidate && fast_decode_enabled && supported_shape && selected_experts_available &&
        supported_quant && !dynamic_lora_active && fast_kernel_available;
    if (!fast_decode_candidate || fast_decode_supported) {
        return nullptr;
    }

    const char* reason = "none";
    if (!fast_decode_enabled) {
        reason = "native_moe_fast_decode_disabled";
    } else if (!supported_shape) {
        reason = plan.lfm2_native_moe ? "callback_shape_mismatch" : "unsupported_shape";
    } else if (!selected_experts_available) {
        reason = "missing_selected_experts";
    } else if (dynamic_lora_active) {
        reason = "dynamic_lora";
    } else if (!supported_w1w3_quant) {
        reason = "unsupported_w1w3_quant";
    } else if (!supported_w2_quant) {
        reason = "unsupported_w2_quant";
    } else if (!fast_kernel_available) {
        reason = "no_fast_kernel";
    }
    RecordNativeMoEFastDecodeDecision(work_ctx, /*candidate=*/true, /*used=*/false, reason,
                                     /*w1w3_used=*/false, /*w2_used=*/false);
    if (supported_w2_quant) {
        RecordNativeMoEFastW2Q5KDecision(work_ctx, /*candidate=*/true, /*used=*/false, reason);
    }
    return plan.lfm2_fast_only_decode ? reason : nullptr;
}

static void RecordQwenLikeNativeMoEGraphCensus(const QwenLikeNativeMoEGraphPlan& plan,
                                               const QwenLikeNativeMoEWeights& weights, ggml_tensor* routed_input,
                                               int64_t n_tokens, int64_t n_embd, int64_t n_expert_used,
                                               int native_moe_callback_tasks) {
    InferenceWorkContext* work_ctx = GetCurrentWorkContext();
    if (!work_ctx) {
        return;
    }
    const InferenceExecutionPhase phase = GetCurrentExecutionPhase();
    RecordGraphBuildMatmulCensus(work_ctx, phase, "moe_native", weights.w1w3_type, n_tokens,
                                 weights.use_fused_gate_up ? weights.gate_up_exps->ne[1] : weights.gate_exps->ne[1],
                                 n_embd, weights.use_fused_gate_up ? weights.gate_up_exps->name : weights.gate_exps->name,
                                 routed_input->name, n_tokens <= 1 || phase == InferenceExecutionPhase::Decode);
    if (!weights.use_fused_gate_up) {
        RecordGraphBuildMatmulCensus(work_ctx, phase, "moe_native", weights.up_exps->type, n_tokens,
                                     weights.up_exps->ne[1], n_embd, weights.up_exps->name, routed_input->name,
                                     n_tokens <= 1 || phase == InferenceExecutionPhase::Decode);
    }
    RecordGraphBuildMatmulCensus(work_ctx, phase, "moe_native", weights.down_exps->type, n_tokens, n_embd,
                                 weights.down_exps->ne[0], weights.down_exps->name, "qwen35_native_moe_swiglu",
                                 n_tokens <= 1 || phase == InferenceExecutionPhase::Decode);
    if (plan.qwen_native_moe || plan.lfm2_native_moe) {
        RecordQwen35MoEGraphPath(work_ctx, "native_graph", static_cast<int>(n_expert_used),
                                 static_cast<int>(n_expert_used), native_moe_callback_tasks, weights.w1w3_type,
                                 weights.down_exps->type);
    }
}

ggml_tensor* TryBuildQwen35NativeMoEGraph(ggml_context* ctx, ggml_cgraph* gf, TransformerModel* model,
                                          TransformerLayer* layer, int layer_idx, ggml_tensor* routed_input,
                                          ggml_tensor* gate_logits, int top_k,
                                          const densecore::models::DecoderLayerSpec* layer_spec) {
    const QwenLikeNativeMoEGraphPlan graph_plan = ResolveQwenLikeNativeMoEGraphPlan(model);
    if (!ctx || !gf || !model || !layer || !routed_input || !gate_logits ||
        (!graph_plan.qwen_native_moe && !graph_plan.lfm2_native_moe)) {
        return nullptr;
    }
    auto reject_native_moe = [&](const char* reason) -> ggml_tensor* {
        const char* safe_reason = reason && reason[0] != '\0' ? reason : "unknown";
        if (InferenceWorkContext* work_ctx = GetCurrentWorkContext()) {
            RecordNativeMoEFastDecodeDecision(work_ctx, /*candidate=*/true, /*used=*/false, safe_reason,
                                             /*w1w3_used=*/false, /*w2_used=*/false);
            RecordNativeMoEFastW2Q5KDecision(work_ctx, /*candidate=*/true, /*used=*/false, safe_reason);
        }
        if (graph_plan.target_requires_native_moe || IsQwen35NativeMoEDownQ5KDiagEnabled() ||
            IsMoEWiringDebugEnabled()) {
            const int64_t input0 = routed_input ? routed_input->ne[0] : -1;
            const int64_t input1 = routed_input ? routed_input->ne[1] : -1;
            const int64_t logits0 = gate_logits ? gate_logits->ne[0] : -1;
            const int64_t logits1 = gate_logits ? gate_logits->ne[1] : -1;
            std::fprintf(stderr,
                         "[NativeMoEAdmission] rejected reason=%s layer=%d variant=%d phase=%d input=[%lld,%lld] "
                         "logits=[%lld,%lld] top_k=%d qwen=%d lfm2=%d\n",
                         safe_reason, layer_idx, static_cast<int>(model->variant),
                         static_cast<int>(graph_plan.graph_phase), static_cast<long long>(input0),
                         static_cast<long long>(input1), static_cast<long long>(logits0),
                         static_cast<long long>(logits1), top_k, graph_plan.qwen_native_moe ? 1 : 0,
                         graph_plan.lfm2_native_moe ? 1 : 0);
        }
        return nullptr;
    };
    if (const char* router_reject = ValidateQwenLikeNativeMoERouter(graph_plan, layer_spec)) {
        return reject_native_moe(router_reject);
    }

    const int64_t n_tokens = routed_input->ne[1];
    QwenLikeNativeMoEWeights moe_weights;
    const char* weight_reject = nullptr;
    if (!ResolveQwenLikeNativeMoEWeights(model, layer, graph_plan, n_tokens, &moe_weights, &weight_reject)) {
        const bool q5_single_copy_missing =
            weight_reject &&
            (std::strcmp(weight_reject, "qwen35_gateup_q5k_single_copy_missing") == 0 ||
             std::strcmp(weight_reject, "lfm2_gateup_q5k_single_copy_missing") == 0);
        if (q5_single_copy_missing) {
            if (InferenceWorkContext* work_ctx = GetCurrentWorkContext()) {
                RecordMoEQ5KRepackedDecision(work_ctx, /*candidate=*/true, /*used=*/false, weight_reject);
                RecordNativeMoEFastDecodeDecision(work_ctx, /*candidate=*/true, /*used=*/false, weight_reject,
                                                 /*w1w3_used=*/false, /*w2_used=*/false);
            }
        }
        return reject_native_moe(weight_reject);
    }
    ggml_tensor* gate_exps = moe_weights.gate_exps;
    ggml_tensor* up_exps = moe_weights.up_exps;
    ggml_tensor* down_exps = moe_weights.down_exps;
    ggml_tensor* raw_gate_exps = moe_weights.raw_gate_exps;
    ggml_tensor* raw_up_exps = moe_weights.raw_up_exps;
    ggml_tensor* raw_down_exps = moe_weights.raw_down_exps;

    const int64_t n_embd = routed_input->ne[0];
    const int64_t n_experts = gate_logits->ne[0];
    const int64_t n_expert_used = std::max<int64_t>(1, std::min<int64_t>(top_k, n_experts));
    const int native_moe_callback_tasks =
        ResolveNativeMoEGraphCallbackTaskCount(model, GetCurrentBatch(), graph_plan.graph_phase, n_tokens,
                                               static_cast<int>(n_expert_used));
    if (!ValidateQwenLikeNativeMoEShapes(moe_weights, routed_input, gate_logits, n_tokens, n_embd, n_experts)) {
        if (IsMoEWiringDebugEnabled()) {
            std::fprintf(stderr,
                         "[Qwen35NativeMoE] rejected layer=%d input=[%lld,%lld] logits=[%lld,%lld] "
                         "gate=[%lld,%lld,%lld,%lld] up=[%lld,%lld,%lld,%lld] down=[%lld,%lld,%lld,%lld]\n",
                         layer_idx, static_cast<long long>(routed_input->ne[0]),
                         static_cast<long long>(routed_input->ne[1]), static_cast<long long>(gate_logits->ne[0]),
                         static_cast<long long>(gate_logits->ne[1]), static_cast<long long>(gate_exps->ne[0]),
                         static_cast<long long>(gate_exps->ne[1]), static_cast<long long>(gate_exps->ne[2]),
                         static_cast<long long>(gate_exps->ne[3]), static_cast<long long>(up_exps->ne[0]),
                         static_cast<long long>(up_exps->ne[1]), static_cast<long long>(up_exps->ne[2]),
                         static_cast<long long>(up_exps->ne[3]), static_cast<long long>(down_exps->ne[0]),
                         static_cast<long long>(down_exps->ne[1]), static_cast<long long>(down_exps->ne[2]),
                         static_cast<long long>(down_exps->ne[3]));
        }
        return reject_native_moe("shape_mismatch");
    }
    const ggml_type w1w3_type = moe_weights.w1w3_type;
    const bool native_q5_gateup_single_copy = moe_weights.native_q5_gateup_single_copy;
    if (const char* fast_decode_reject = RecordQwenLikeNativeMoEFastDecodeAdmission(
            model, graph_plan, moe_weights, n_tokens, n_experts, n_expert_used)) {
        return reject_native_moe(fast_decode_reject);
    }
    RecordQwenLikeNativeMoEGraphCensus(graph_plan, moe_weights, routed_input, n_tokens, n_embd, n_expert_used,
                                       native_moe_callback_tasks);

    ggml_tensor* routing_probs = gate_logits;
    ggml_tensor* selection_scores = gate_logits;
    if (graph_plan.lfm2_native_moe) {
        routing_probs = ggml_sigmoid(ctx, gate_logits);
        ggml_set_name(routing_probs, "lfm2_native_moe_sigmoid_probs");
        selection_scores = routing_probs;
        if (ggml_tensor* bias = layer->Get(model_keys::kMoeCorrectionBias);
            bias && bias->type == GGML_TYPE_F32 && bias->ne[0] == n_experts) {
            selection_scores = ggml_add(ctx, routing_probs, bias);
            ggml_set_name(selection_scores, "lfm2_native_moe_biased_selection_scores");
        }
    }

    ggml_tensor* selected_experts = ggml_argsort_top_k(ctx, selection_scores, static_cast<int>(n_expert_used));
    ggml_set_name(selected_experts, "qwen35_native_moe_topk");
    ggml_build_forward_expand(gf, selected_experts);

    ggml_tensor* gate = nullptr;
    ggml_tensor* up = nullptr;
    ggml_tensor* hidden = nullptr;
    if (!graph_plan.lfm2_debug_reference &&
        CanUseQwen35NativeMoEGateUpRawQXKSwiGLU(model, raw_gate_exps, raw_up_exps, routed_input, selected_experts)) {
        Qwen35SharedQ8RowsUserData* gateup_q8_ud =
            AllocateQwen35SharedQ8RowsUserData(ctx, routed_input, n_expert_used * n_tokens);
        if (gateup_q8_ud) {
            gateup_q8_ud->work_ctx = GetCurrentWorkContext();
            gateup_q8_ud->debug_layer_idx = layer_idx;
            gateup_q8_ud->requested_task_count = native_moe_callback_tasks;
            // Q4_K prefill is assignment-batched and stays on the raw-batched
            // k-quant path. Decode uses the validated repacked helper for
            // Qwen-like MoE variants when the kernel is available.
            gateup_q8_ud->prefer_q4k_repacked_swiglu = ShouldUseQwenLikeGateUpQ4KRepackedSwiGLU(
                graph_plan, w1w3_type, GetCurrentExecutionPhase(), Qwen35GateUpQ4KRepackKernelAvailable());
            gateup_q8_ud->q5k_gateup_8x8_single_copy = native_q5_gateup_single_copy;
            gateup_q8_ud->q5k_gateup_8x8_single_copy_required =
                (graph_plan.qwen_native_moe || graph_plan.lfm2_native_moe) && w1w3_type == GGML_TYPE_Q5_K;
            gateup_q8_ud->record_lfm2_w1w3_kernel = graph_plan.lfm2_native_moe;
        }
        ggml_tensor* args[] = {raw_gate_exps, raw_up_exps, routed_input, selected_experts};
        hidden = ggml_custom_4d(ctx, GGML_TYPE_F32, raw_gate_exps->ne[1], n_expert_used, n_tokens, 1, args, 4,
                                cb_qwen35_native_moe_gateup_raw_qxk_swiglu, native_moe_callback_tasks, gateup_q8_ud);
        ggml_set_name(hidden, "qwen35_native_moe_gateup_raw_qxk_swiglu");
    } else if (graph_plan.qwen_native_moe || graph_plan.lfm2_native_moe) {
        return reject_native_moe("w1w3_fast_callback_unavailable");
    } else {
        return reject_native_moe("unsupported_native_moe_graph");
    }
    if (!hidden) {
        hidden = ggml_swiglu_split(ctx, gate, up);
        ggml_set_name(hidden, "qwen35_native_moe_swiglu");
    }
    ggml_tensor* fused_out = nullptr;
    Qwen35SharedQ8RowsUserData* hidden_q8_ud = nullptr;
    ggml_tensor* fast_down_exps = graph_plan.qwen_native_moe ? raw_down_exps : down_exps;
    if (!graph_plan.lfm2_debug_reference &&
        CanFuseQwen35W2NormWeightsFromLogitsWithCustomCallback(model, fast_down_exps, hidden, selected_experts,
                                                                gate_logits)) {
        hidden_q8_ud = AllocateQwen35SharedQ8RowsUserData(ctx, hidden, selected_experts->ne[0] * selected_experts->ne[1]);
        if (hidden_q8_ud && graph_plan.lfm2_native_moe) {
            hidden_q8_ud->work_ctx = GetCurrentWorkContext();
            hidden_q8_ud->debug_layer_idx = layer_idx;
            hidden_q8_ud->requested_task_count = native_moe_callback_tasks;
            hidden_q8_ud->weighted_logits_lfm2_sigmoid = true;
            hidden_q8_ud->weighted_logits_norm_topk = model->moe_norm_topk_prob;
            hidden_q8_ud->weighted_logits_scale = model->moe_routed_scaling_factor;
        } else if (hidden_q8_ud) {
            hidden_q8_ud->work_ctx = GetCurrentWorkContext();
            hidden_q8_ud->debug_layer_idx = layer_idx;
            hidden_q8_ud->requested_task_count = native_moe_callback_tasks;
        }
        ggml_tensor* args[] = {fast_down_exps, hidden, selected_experts, gate_logits};
        fused_out = ggml_custom_4d(ctx, GGML_TYPE_F32, fast_down_exps->ne[1], selected_experts->ne[1], 1, 1, args, 4,
                                   cb_qwen35_native_moe_down_q5k_weighted_logits, native_moe_callback_tasks,
                                   hidden_q8_ud);
        ggml_set_name(fused_out, graph_plan.lfm2_native_moe ? "lfm2_native_moe_down_qxk_fast_weighted_logits"
                                                            : "qwen35_native_moe_down_q5k_fast_weighted_logits");
    }
    if (fused_out) {
        ggml_build_forward_expand(gf, fused_out);
        char name[80];
        std::snprintf(name, sizeof(name), "blk.%d.qwen35_native_moe_down_q5k_fast_out", layer_idx);
        ggml_set_name(fused_out, name);
        return fused_out;
    }

    ggml_tensor* weights = nullptr;
    if (graph_plan.lfm2_native_moe) {
        routing_probs = ggml_reshape_3d(ctx, routing_probs, 1, n_experts, n_tokens);
        weights = ggml_get_rows(ctx, routing_probs, selected_experts);
        ggml_set_name(weights, "lfm2_native_moe_weights");
        if (model->moe_norm_topk_prob) {
            weights = ggml_reshape_2d(ctx, weights, n_expert_used, n_tokens);
            ggml_tensor* weight_sum = ggml_sum_rows(ctx, weights);
            weight_sum = ggml_clamp(ctx, weight_sum, 6.103515625e-5f, INFINITY);
            weights = ggml_div(ctx, weights, weight_sum);
            weights = ggml_reshape_3d(ctx, weights, 1, n_expert_used, n_tokens);
            ggml_set_name(weights, "lfm2_native_moe_norm_weights");
        }
        if (model->moe_routed_scaling_factor != 0.0f && model->moe_routed_scaling_factor != 1.0f) {
            weights = ggml_scale(ctx, weights, model->moe_routed_scaling_factor);
            ggml_set_name(weights, "lfm2_native_moe_scaled_weights");
        }
    } else if (model->moe_norm_topk_prob) {
        weights = BuildMoETopKWeightsFromLogits(ctx, gate_logits, selected_experts, "qwen35_native_moe_norm_weights");
        if (!weights) {
            return reject_native_moe("topk_weights_build_failed");
        }
        if (model->moe_routed_scaling_factor != 0.0f && model->moe_routed_scaling_factor != 1.0f) {
            weights = ggml_scale(ctx, weights, model->moe_routed_scaling_factor);
            ggml_set_name(weights, "qwen35_native_moe_scaled_weights");
        }
    } else {
        ggml_tensor* probs = ggml_soft_max(ctx, gate_logits);
        ggml_set_name(probs, "qwen35_native_moe_probs");
        probs = ggml_reshape_3d(ctx, probs, 1, n_experts, n_tokens);
        weights = ggml_get_rows(ctx, probs, selected_experts);
        ggml_set_name(weights, "qwen35_native_moe_weights");
        if (model->moe_routed_scaling_factor != 0.0f && model->moe_routed_scaling_factor != 1.0f) {
            weights = ggml_scale(ctx, weights, model->moe_routed_scaling_factor);
        }
    }
    ggml_build_forward_expand(gf, weights);

    if (!graph_plan.lfm2_debug_reference &&
        CanFuseQwen35W2WeightedSumWithCustomCallback(model, fast_down_exps, hidden, selected_experts, weights)) {
        if (!hidden_q8_ud) {
            hidden_q8_ud = AllocateQwen35SharedQ8RowsUserData(ctx, hidden, selected_experts->ne[0] * selected_experts->ne[1]);
            if (hidden_q8_ud) {
                hidden_q8_ud->work_ctx = GetCurrentWorkContext();
                hidden_q8_ud->debug_layer_idx = layer_idx;
                hidden_q8_ud->requested_task_count = native_moe_callback_tasks;
            }
        }
        ggml_tensor* args[] = {fast_down_exps, hidden, selected_experts, weights};
        fused_out = ggml_custom_4d(ctx, GGML_TYPE_F32, fast_down_exps->ne[1], selected_experts->ne[1], 1, 1, args, 4,
                                   cb_qwen35_native_moe_down_q5k_weighted_sum, native_moe_callback_tasks,
                                   hidden_q8_ud);
        ggml_set_name(fused_out, "qwen35_native_moe_down_q5k_fast_weighted_sum");
    }
    if (fused_out) {
        ggml_build_forward_expand(gf, fused_out);
        char name[80];
        std::snprintf(name, sizeof(name), "blk.%d.qwen35_native_moe_down_q5k_fast_out", layer_idx);
        ggml_set_name(fused_out, name);
        return fused_out;
    }
    ggml_tensor* experts = nullptr;
    if (!graph_plan.lfm2_debug_reference &&
        CanReplaceQwen35W2WithCustomCallback(model, fast_down_exps, hidden, selected_experts)) {
        if (!hidden_q8_ud) {
            hidden_q8_ud = AllocateQwen35SharedQ8RowsUserData(ctx, hidden, selected_experts->ne[0] * selected_experts->ne[1]);
            if (hidden_q8_ud) {
                hidden_q8_ud->debug_layer_idx = layer_idx;
                hidden_q8_ud->requested_task_count = native_moe_callback_tasks;
            }
        }
        ggml_tensor* args[] = {fast_down_exps, hidden, selected_experts};
        experts = ggml_custom_4d(ctx, GGML_TYPE_F32, fast_down_exps->ne[1], selected_experts->ne[0],
                                 selected_experts->ne[1], 1, args, 3,
                                 cb_qwen35_native_moe_down_q5k, native_moe_callback_tasks, hidden_q8_ud);
        ggml_set_name(experts, "qwen35_native_moe_down_q5k_fast");
    }
    if (!experts) {
        if (graph_plan.qwen_native_moe || graph_plan.lfm2_native_moe) {
            return reject_native_moe("w2_fast_callback_unavailable");
        }
        if (IsQwen35NativeMoEDownQ5KDiagEnabled()) {
            std::fprintf(stderr, "[W2_Q5K_DIAG] rejected: layer=%d w2 custom callback unavailable\n", layer_idx);
        }
        return nullptr;
    }

    ggml_tensor* out =
        BuildMoeExpertWeightedSumWithWeights(ctx, experts, weights, n_embd, n_tokens, "qwen35_native_moe_expert_sum");
    if (!out) {
        return reject_native_moe("expert_weighted_sum_build_failed");
    }
    ggml_build_forward_expand(gf, out);
    char name[80];
    std::snprintf(name, sizeof(name), "blk.%d.qwen35_native_moe_out", layer_idx);
    ggml_set_name(out, name);
    return out;
}

using densecore::llm::runtime::IsMixedRoutingEnabled;
using densecore::llm::runtime::ResolveBackendRegistry;
using densecore::llm::runtime::ResolveFastPathRuntimeConfig;
using densecore::llm::runtime::ResolveHardwareTopology;
using densecore::llm::runtime::ResolveInferenceConfig;
using densecore::llm::runtime::ResolvePreferredAttentionDevice;
using densecore::llm::runtime::ResolvePreferredDevice;
using densecore::llm::runtime::ResolvePreferredMatmulDevice;
using densecore::llm::runtime::ResolvePreferredNormDevice;

static bool IsVerboseGraphBuildLoggingEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_VERBOSE_GRAPH_BUILD");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugInferenceStatsEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_INFERENCE_STATS");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugGemvSelectionEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_GEMV_SELECTION");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugMatmulPathLoggingEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_MATMUL_PATH");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugMatmulDispatchEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_MATMUL_DISPATCH");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

// Dispatch path labels for instrumentation
static const char* MatmulWeightTypeLabel(ggml_type wtype, bool is_packed_int4, bool is_packed_fp8) {
    if (is_packed_int4) return "PACKED_INT4";
    if (is_packed_fp8) return "PACKED_FP8";
    if (ggml_is_quantized(wtype)) return "GGML_QUANT";
    if (wtype == GGML_TYPE_F32) return "FLOAT_F32";
    if (wtype == GGML_TYPE_F16) return "FLOAT_F16";
    if (wtype == GGML_TYPE_BF16) return "FLOAT_BF16";
    return "UNKNOWN";
}

static const char* DetectedISATier() {
#if defined(__AVX512F__)
    return "AVX-512";
#elif defined(__AVX2__)
    return "AVX2";
#elif defined(__aarch64__)
    return "NEON";
#else
    return "SCALAR";
#endif
}

static void LogMatmulDispatch(const char* weight_name, const char* weight_type_label, int M, int N, int K,
                              const char* path_label, const char* fallback_reason = nullptr) {
    if (!IsDebugMatmulDispatchEnabled()) return;
    if (fallback_reason) {
        fprintf(stderr, "[DISPATCH] w=%s type=%s M=%d N=%d K=%d isa=%s path=%s fallback=%s\n",
                weight_name ? weight_name : "(unnamed)", weight_type_label, M, N, K, DetectedISATier(), path_label,
                fallback_reason);
    } else {
        fprintf(stderr, "[DISPATCH] w=%s type=%s M=%d N=%d K=%d isa=%s path=%s\n",
                weight_name ? weight_name : "(unnamed)", weight_type_label, M, N, K, DetectedISATier(), path_label);
    }
}

static bool IsDebugSSMQkvReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_SSM_QKV_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugSSMProjectionReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_SSM_PROJECTION_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugAttentionProjectionReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_ATTN_PROJECTION_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static void ValidateAttentionProjectionShape3D(const struct ggml_tensor* tensor, const char* tensor_name, int layer_idx,
                                               int ne0, int ne1, int ne2, int projected_dim, int n_heads) {
    if (!tensor) {
        throw densecore::InvalidArgumentException("Missing attention projection tensor for " +
                                                  std::string(tensor_name ? tensor_name : "unknown") + " in layer " +
                                                  std::to_string(layer_idx));
    }
    if (projected_dim <= 0 || n_heads <= 0 || ne2 <= 0) {
        throw densecore::InvalidArgumentException(
            "Invalid attention reshape parameters for " + std::string(tensor_name ? tensor_name : "unknown") +
            " in layer " + std::to_string(layer_idx) + ": projected_dim=" + std::to_string(projected_dim) +
            " n_heads=" + std::to_string(n_heads) + " n_tokens=" + std::to_string(ne2));
    }
    if ((projected_dim % n_heads) != 0) {
        throw densecore::InvalidArgumentException("Attention projection dimension is not divisible by head count for " +
                                                  std::string(tensor_name ? tensor_name : "unknown") + " in layer " +
                                                  std::to_string(layer_idx) + ": dim=" + std::to_string(projected_dim) +
                                                  " n_heads=" + std::to_string(n_heads));
    }
    const int64_t expected = static_cast<int64_t>(ne0) * static_cast<int64_t>(ne1) * static_cast<int64_t>(ne2);
    const int64_t actual = ggml_nelements(tensor);
    if (actual != expected) {
        throw densecore::InvalidArgumentException(
            "Attention reshape contract mismatch for " + std::string(tensor_name ? tensor_name : "unknown") +
            " in layer " + std::to_string(layer_idx) + ": tensor=[" + std::to_string(tensor->ne[0]) + "," +
            std::to_string(tensor->ne[1]) + "," + std::to_string(tensor->ne[2]) + "," + std::to_string(tensor->ne[3]) +
            "] projected_dim=" + std::to_string(projected_dim) + " n_heads=" + std::to_string(n_heads) + " target=[" +
            std::to_string(ne0) + "," + std::to_string(ne1) + "," + std::to_string(ne2) +
            "] actual_nelements=" + std::to_string(actual) + " expected_nelements=" + std::to_string(expected));
    }
}

static bool IsDebugFinalProjectionReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_FINAL_PROJECTION_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugHiddenSnapshotEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugGemma4SharedKVEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_GEMMA4_SHARED_KV");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugFfnProjectionReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_FFN_PROJECTION_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugSSMCoreReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_SSM_CORE_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

using densecore::llm::models::IsHybridSSMQkvWeightName;
using densecore::llm::models::ShouldForcePlainGgmlForHybridSSMQkv;

static std::array<std::atomic<uint64_t>, kHybridSSMDispatchWeightCount * kHybridSSMDispatchPathCount>
    g_hybrid_ssm_dispatch_counters{};

static std::size_t ResolveHybridSSMDispatchWeightIndex(const char* name) {
    if (!name) {
        return 3;
    }
    if (std::strstr(name, "qkv_mixed")) return 0;
    if (std::strcmp(name, "z") == 0 || std::strstr(name, ".z")) return 1;
    if (std::strstr(name, "ssm_out")) return 2;
    return 3;
}

static std::size_t ResolveHybridSSMDispatchPathIndex(const char* path) {
    if (!path) {
        return 4;
    }
    if (std::strstr(path, "PLAIN_GGML_CONSERVATIVE_FALLBACK")) return 0;
    if (std::strstr(path, "GEMV_QUANT")) return 1;
    if (std::strstr(path, "GGML_QUANT_NRC_M")) return 2;
    if (std::strstr(path, "GGML_NATIVE")) return 3;
    return 4;
}

static void LogHybridSSMQkvDispatch(const char* weight_name, ggml_type weight_type, int M, int N, int K,
                                    const char* chosen_path, bool used_batched_quant_nrc, bool used_native_q4k_vecdot,
                                    bool used_direct_int4_fastpath) {
    if (!densecore::llm::models::IsHybridSSMQkvWeightName(weight_name)) {
        return;
    }
    const std::size_t weight_index = ResolveHybridSSMDispatchWeightIndex(weight_name);
    const std::size_t path_index = ResolveHybridSSMDispatchPathIndex(chosen_path);
    g_hybrid_ssm_dispatch_counters[weight_index * kHybridSSMDispatchPathCount + path_index].fetch_add(
        1, std::memory_order_relaxed);
    if (!IsDebugMatmulDispatchEnabled()) {
        return;
    }
    fprintf(stderr,
            "[SSM_QKV_DISPATCH] w=%s ggml_type=%s M=%d N=%d K=%d path=%s batched_quant_nrc=%d native_q4k_vecdot=%d "
            "direct_int4_fastpath=%d\n",
            weight_name ? weight_name : "(unnamed)", ggml_type_name(weight_type), M, N, K,
            chosen_path ? chosen_path : "unknown", used_batched_quant_nrc ? 1 : 0, used_native_q4k_vecdot ? 1 : 0,
            used_direct_int4_fastpath ? 1 : 0);
}

// Env-tunable thresholds declared here, defined after ParsePositiveEnvInt.
static int GetBatchedMinM();
static int GetBatchedMinN();
static int GetBatchedMinK();

static void LogMatmulPathOnce(const char* path) {
    if (!path || !IsDebugMatmulPathLoggingEnabled()) {
        return;
    }

    static std::atomic<bool> logged_ggml_mul_mat{false};
    static std::atomic<bool> logged_gemv_batched_quant_nrc{false};
    std::atomic<bool>* once_flag = nullptr;

    if (std::strcmp(path, "ggml_mul_mat") == 0) {
        once_flag = &logged_ggml_mul_mat;
    } else if (std::strcmp(path, "gemv_batched_quant_nrc") == 0) {
        once_flag = &logged_gemv_batched_quant_nrc;
    } else {
        return;
    }

    bool expected = false;
    if (once_flag->compare_exchange_strong(expected, true, std::memory_order_relaxed)) {
        fprintf(stderr, "[DenseCore][MatmulPath] %s\n", path);
    }
}

[[maybe_unused]] static void LogMatmulValidationOnce(const char* path, bool ok, float max_abs_diff) {
    if (!path || !IsDebugMatmulPathLoggingEnabled()) {
        return;
    }

    static std::mutex mu;
    static std::unordered_map<std::string, bool> logged;
    std::lock_guard<std::mutex> lock(mu);
    if (logged[path]) {
        return;
    }
    logged[path] = true;
    fprintf(stderr, "[DenseCore][MatmulValidate] %s status=%s max_abs_diff=%.8f\n", path, ok ? "pass" : "fail",
            static_cast<double>(max_abs_diff));
}

[[maybe_unused]] static RuntimeToggleMode GetArmQ4KNativeVecDotMode();

static bool ShouldUseArmNativeQ4KVecDotValidated(ggml_type weight_type, const ggml_type_traits_cpu* type_traits_cpu,
                                                 const char* weight_name, const void* sample_row_ptr,
                                                 const void* sample_quant_input, const float* sample_input_f32, int N) {
#if defined(__aarch64__) || defined(_M_ARM64)
    if (weight_type != GGML_TYPE_Q4_K) {
        return type_traits_cpu && type_traits_cpu->vec_dot;
    }
    if (!type_traits_cpu || !type_traits_cpu->vec_dot || !sample_row_ptr || !sample_quant_input || !sample_input_f32 ||
        N <= 0) {
        return false;
    }

    const RuntimeToggleMode mode = GetArmQ4KNativeVecDotMode();
    if (mode == RuntimeToggleMode::Off) {
        return false;
    }
    if (mode == RuntimeToggleMode::On) {
        return true;
    }

    const bool is_hybrid_ssm_qkv = densecore::llm::models::IsHybridSSMQkvWeightName(weight_name);
    if (is_hybrid_ssm_qkv && densecore::llm::models::ShouldForcePlainGgmlForHybridSSMQkv()) {
        return false;  // Always fail-closed for hybrid SSM QKV on ARM unless specifically opted out
    }

    // Multi-sample validation: test multiple representative weight rows
    static std::atomic<int> state{0};
    int current = state.load(std::memory_order_acquire);
    if (current != 0) {
        return current == 1;
    }

    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    current = state.load(std::memory_order_relaxed);
    if (current == 0) {
        const auto* type_traits = ggml_get_type_traits(weight_type);
        bool ok = false;
        float global_max_abs_diff = 0.0f;
        if (type_traits && type_traits->to_float) {
            thread_local std::vector<float> dequant_buffer;
            dequant_buffer.resize(static_cast<size_t>(N));
            const size_t row_stride = ggml_row_size(weight_type, N);

            // Validate at least kMinValidationRows representative rows.
            // Rows are sampled at the base pointer (row 0) and at evenly
            // spaced offsets to detect position-dependent accuracy loss.
            static constexpr int kMinValidationRows = 4;
            ok = true;
            for (int sample = 0; sample < kMinValidationRows; ++sample) {
                const void* row_ptr =
                    reinterpret_cast<const char*>(sample_row_ptr) + static_cast<size_t>(sample) * row_stride;

                float native_sum = 0.0f;
                type_traits_cpu->vec_dot(N, &native_sum, 0, row_ptr, 0, sample_quant_input, 0, 1);

                type_traits->to_float(row_ptr, dequant_buffer.data(), N);
                float ref_sum = 0.0f;
                for (int i = 0; i < N; ++i) {
                    ref_sum += dequant_buffer[static_cast<size_t>(i)] * sample_input_f32[i];
                }

                const float diff = std::fabs(native_sum - ref_sum);
                global_max_abs_diff = std::max(global_max_abs_diff, diff);

                // Tighter tolerance than before (was 5e-4 relative, now 2e-4).
                // Fail closed: any single row exceeding tolerance disables the path.
                const float tol = std::max(1e-3f, 2e-4f * std::fabs(ref_sum));
                if (!std::isfinite(native_sum) || diff > tol) {
                    ok = false;
                    break;
                }
            }
        }

        state.store(ok ? 1 : 2, std::memory_order_release);
        LogMatmulValidationOnce("arm_q4k_native_vecdot", ok, global_max_abs_diff);
        current = ok ? 1 : 2;
    }

    return current == 1;
#else
    (void)weight_type;
    (void)weight_name;
    (void)sample_row_ptr;
    (void)sample_quant_input;
    (void)sample_input_f32;
    (void)N;
    return type_traits_cpu && type_traits_cpu->vec_dot;
#endif
}


static bool ResolveQ4KTrueBatchedKernelEnabledPolicy(densecore::simd::SimdLevel level, bool compiled_with_sve);

static bool IsQ4KTrueBatchedKernelEnabled() {
    return ResolveQ4KTrueBatchedKernelEnabledPolicy(
        densecore::simd::DetectSimdLevel(),
#if defined(__ARM_FEATURE_SVE)
        true
#else
        false
#endif
    );
}

static bool ResolveQ4KTrueBatchedKernelEnabledPolicy(densecore::simd::SimdLevel level, bool compiled_with_sve) {
    if (compiled_with_sve && (level == densecore::simd::SimdLevel::SVE || level == densecore::simd::SimdLevel::SVE2)) {
        return true;
    }
#if defined(__AVX2__) || defined(__AVX512F__) || defined(DENSECORE_X86)
    return level == densecore::simd::SimdLevel::AVX2 || level == densecore::simd::SimdLevel::AVX512 ||
           level == densecore::simd::SimdLevel::AMX;
#else
    return false;
#endif
}

static inline void SpinPause(int spin_count) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    if ((spin_count & 0x3F) != 0) {
        _mm_pause();
        return;
    }
#elif defined(__aarch64__)
    if ((spin_count & 0x3F) != 0) {
        asm volatile("yield");
        return;
    }
#endif
    std::this_thread::yield();
}

static uint64_t ComputeGemvBatchedQuantStamp(const BatchSpec* batch, int M, int slot_id, const void* src_data_ptr,
                                             const void* weight_data_ptr) {
    // FNV-1a over stable per-op metadata to avoid stale-buffer reuse when
    // slot_id repeats across different batched inputs.
    uint64_t hash = 1469598103934665603ull;
    const auto mix = [&](uint64_t v) {
        hash ^= v;
        hash *= 1099511628211ull;
    };

    mix(static_cast<uint64_t>(static_cast<uint32_t>(slot_id)));
    mix(static_cast<uint64_t>(static_cast<uint32_t>(M)));
    mix(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(src_data_ptr)));
    mix(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(weight_data_ptr)));
    if (batch) {
        mix(static_cast<uint64_t>(batch->tokens.size()));
        const int n_pos = std::min(M, static_cast<int>(batch->pos.size()));
        for (int i = 0; i < n_pos; ++i) {
            mix(static_cast<uint64_t>(static_cast<uint32_t>(batch->pos[static_cast<size_t>(i)])));
        }
    }

    // Zero is the reset sentinel for stamps.
    return hash == 0 ? 1 : hash;
}

static const KVRetentionPolicy& GetKVRetentionPolicy(const BatchSpec* batch = nullptr) {
    return ResolveFastPathRuntimeConfig(batch).kv_retention;
}

// Env-tunable thresholds for batched GEMM/GEMV routing
[[maybe_unused]] static int GetBatchedMinM() {
    static const int val = ParsePositiveEnvInt("DENSECORE_MATMUL_BATCHED_MIN_M", 2);
    return val;
}

[[maybe_unused]] static int GetBatchedMinN() {
    static const int val = ParsePositiveEnvInt("DENSECORE_MATMUL_BATCHED_MIN_N", 32);
    return val;
}

[[maybe_unused]] static int GetBatchedMinK() {
    static const int val = ParsePositiveEnvInt("DENSECORE_MATMUL_BATCHED_MIN_K", 32);
    return val;
}

[[maybe_unused]] static RuntimeToggleMode GetArmQ4KNativeVecDotMode() {
    return densecore::llm::config::LoadArmQ4KNativeVecDotMode();
}

[[maybe_unused]] static RuntimeToggleMode GetArmInt4DirectFastPathMode() {
    return densecore::llm::config::LoadArmInt4DirectFastPathMode();
}

static int ResolveTaskCount(const BatchSpec* batch, int work_items) {
    int n_tasks = ResolveInferenceConfig(batch).num_threads;
    if (n_tasks <= 0) {
        n_tasks = std::thread::hardware_concurrency();
        if (n_tasks <= 0) n_tasks = 4;
    }
    int physical_cores = ResolveHardwareTopology(batch).GetPhysicalCoreCount();
    if (physical_cores > 0) {
        n_tasks = std::min(n_tasks, physical_cores);
    }
    if (work_items > 0) {
        n_tasks = std::min(n_tasks, work_items);
    }
    return std::max(1, n_tasks);
}

struct NativeMoEGraphCallbackTaskPlan {
    bool use_per_op_task_count = false;
    int work_items = 0;
};

static NativeMoEGraphCallbackTaskPlan ResolveNativeMoEGraphCallbackTaskPlan(const TransformerModel* model,
                                                                            InferenceExecutionPhase phase,
                                                                            int64_t n_tokens, int top_k) {
    NativeMoEGraphCallbackTaskPlan plan;
    if (model && model->variant == ModelVariant::LFM2MOE && model->arch_flags.is_lfm2_shortconv &&
        phase == InferenceExecutionPhase::Decode && n_tokens == 1 && top_k > 1) {
        plan.use_per_op_task_count = true;
        // LFM2 high-topK decode keeps all configured workers available; zero means no work-item cap.
        plan.work_items = 0;
    }
    return plan;
}

static int ResolveNativeMoEGraphCallbackTaskCount(const TransformerModel* model, const BatchSpec* batch,
                                                  InferenceExecutionPhase phase, int64_t n_tokens, int top_k) {
    const NativeMoEGraphCallbackTaskPlan plan =
        ResolveNativeMoEGraphCallbackTaskPlan(model, phase, n_tokens, top_k);
    if (plan.use_per_op_task_count) {
        return ResolveTaskCount(batch, plan.work_items);
    }
    return GGML_N_TASKS_MAX;
}

static densecore::simd::SimdLevel GetRuntimeSimdLevel() {
    static const densecore::simd::SimdLevel level = densecore::simd::DetectSimdLevel();
    return level;
}

static const DecodePagedAttentionPolicy& ResolveDecodePagedAttentionPolicy(const BatchSpec* batch = nullptr) {
    return ResolveFastPathRuntimeConfig(batch).decode_paged_attention;
}

struct DecodeContextSummary {
    int min_context = 0;
    int max_context = 0;
    int avg_context = 0;
    bool valid = false;
};

bool IsDecodeOnlyBatchLayoutImpl(const BatchSpec& batch, int n_tokens_in_batch) {
    if (n_tokens_in_batch <= 0) {
        return false;
    }
    if (batch.num_seqs != n_tokens_in_batch) {
        return false;
    }
    if (static_cast<int>(batch.tokens.size()) != n_tokens_in_batch ||
        static_cast<int>(batch.seq_id.size()) != n_tokens_in_batch ||
        static_cast<int>(batch.pos.size()) != n_tokens_in_batch) {
        return false;
    }
    if (static_cast<int>(batch.block_tables.size()) != n_tokens_in_batch ||
        static_cast<int>(batch.n_past.size()) != n_tokens_in_batch) {
        return false;
    }

    std::vector<uint8_t> seen(static_cast<size_t>(n_tokens_in_batch), 0);
    for (int i = 0; i < n_tokens_in_batch; ++i) {
        const int seq_idx = batch.seq_id[static_cast<size_t>(i)];
        if (seq_idx < 0 || seq_idx >= n_tokens_in_batch) {
            return false;
        }
        if (seen[static_cast<size_t>(seq_idx)] != 0) {
            return false;
        }
        seen[static_cast<size_t>(seq_idx)] = 1;
    }

    return true;
}

static DecodeContextSummary SummarizeDecodeContext(const BatchSpec& batch, int n_tokens_in_batch) {
    DecodeContextSummary summary;
    if (!IsDecodeOnlyBatchLayoutImpl(batch, n_tokens_in_batch)) {
        return summary;
    }

    summary.min_context = std::numeric_limits<int>::max();
    summary.max_context = 0;
    int64_t sum_context = 0;
    std::vector<uint8_t> seen(static_cast<size_t>(batch.num_seqs), 0);

    for (int i = 0; i < n_tokens_in_batch; ++i) {
        const int seq_idx = batch.seq_id[static_cast<size_t>(i)];
        if (seq_idx < 0 || seq_idx >= batch.num_seqs) {
            return DecodeContextSummary{};
        }
        if (seen[static_cast<size_t>(seq_idx)] != 0) {
            return DecodeContextSummary{};
        }
        seen[static_cast<size_t>(seq_idx)] = 1;

        const int pos_i = batch.pos[static_cast<size_t>(i)];
        if (pos_i < 0) {
            return DecodeContextSummary{};
        }

        const auto& block_table = batch.block_tables[static_cast<size_t>(seq_idx)];
        if (block_table.empty()) {
            return DecodeContextSummary{};
        }
        const int logical_block = pos_i / BLOCK_SIZE;
        if (logical_block < 0 || logical_block >= static_cast<int>(block_table.size())) {
            return DecodeContextSummary{};
        }

        const int n_past_i = batch.n_past[static_cast<size_t>(seq_idx)];
        if (n_past_i < 0) {
            return DecodeContextSummary{};
        }

        const KVRetentionSpan retained =
            densecore::llm::config::ComputeKVRetentionSpan(n_past_i, GetKVRetentionPolicy(&batch));
        const int max_context_i = static_cast<int>(block_table.size()) * BLOCK_SIZE;
        const int context_len_i = std::max(1, std::min(retained.history_kept + 1, max_context_i));
        summary.min_context = std::min(summary.min_context, context_len_i);
        summary.max_context = std::max(summary.max_context, context_len_i);
        sum_context += context_len_i;
    }

    summary.avg_context = static_cast<int>((sum_context + n_tokens_in_batch - 1) / n_tokens_in_batch);
    summary.valid = true;
    return summary;
}

using DecodePagedFallbackReason = densecore::llm::attention::DecodePagedFallbackReason;
using DecodePagedDecision = densecore::llm::attention::DecodePagedDecision;
using BasePagedDecodeExecutionDecision = densecore::llm::attention::BasePagedDecodeExecutionDecision;

#ifdef DENSECORE_TEST_BUILD
static std::atomic<int> g_test_force_flash_attention_disabled{0};
#endif

static int ResolveAttentionQueryBasePosition(const BatchSpec& batch) {
    if (batch.n_past.empty()) {
        return 0;
    }
    int max_n_past = 0;
    for (int n_past_i : batch.n_past) {
        max_n_past = std::max(max_n_past, n_past_i);
    }
    return max_n_past;
}

static bool IsFlashAttentionDisabled() {
#ifdef DENSECORE_TEST_BUILD
    if (g_test_force_flash_attention_disabled.load(std::memory_order_relaxed) != 0) {
        return true;
    }
#endif
    return false;
}

static bool IsDebugPagedAttentionReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_PAGED_ATTN_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugPagedAttentionEagerReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_PAGED_ATTN_EAGER_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugAttentionCoreReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_ATTN_CORE_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugAddRmsNormReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_ADD_RMSNORM_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool ShouldRunAddRmsNormReferenceProbe(int layer_idx) {
    if (!IsDebugAddRmsNormReferenceEnabled()) {
        return false;
    }
    if (IsCurrentGraphBuildNoAlloc()) {
        return false;
    }
    int configured_layer = -1;
    if (const char* env = std::getenv("DENSECORE_DEBUG_ADD_RMSNORM_REFERENCE_LAYER")) {
        if (env[0] != '\0') {
            configured_layer = std::atoi(env);
        }
    }
    return configured_layer < 0 || configured_layer == layer_idx;
}

static bool IsDebugAttentionPostReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_ATTN_POST_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDebugKvRoundTripEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_KV_ROUNDTRIP");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static float ResolveGemma4AttentionLogitSoftcapRuntime(const TransformerModel* model) {
    return densecore::llm::attention::ResolveGemma4AttentionLogitSoftcapRuntime(model);
}

static bool ShouldRunKvRoundTripProbe(int layer, bool is_k) {
    if (!IsDebugKvRoundTripEnabled()) {
        return false;
    }
    if (IsCurrentGraphBuildNoAlloc()) {
        return false;
    }

    static const int target_layer = ParseIntEnv("DENSECORE_DEBUG_KV_ROUNDTRIP_LAYER", -1);
    static const int target_kind = ParseIntEnv("DENSECORE_DEBUG_KV_ROUNDTRIP_KIND", -1);  // -1 both, 0 V, 1 K
    static std::atomic<int> remaining_budget{ParsePositiveEnvInt("DENSECORE_DEBUG_KV_ROUNDTRIP_MAX_CALLS", 1)};

    if (target_layer >= 0 && layer != target_layer) {
        return false;
    }
    if (target_kind >= 0 && target_kind != (is_k ? 1 : 0)) {
        return false;
    }

    int remaining = remaining_budget.load(std::memory_order_relaxed);
    while (remaining > 0) {
        if (remaining_budget.compare_exchange_weak(remaining, remaining - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

static bool ShouldRunPagedAttentionReferenceProbe(int layer, int token_idx) {
    if (!IsDebugPagedAttentionReferenceEnabled()) {
        return false;
    }
    if (IsCurrentGraphBuildNoAlloc()) {
        return false;
    }

    static const int target_layer = ParseIntEnv("DENSECORE_DEBUG_PAGED_ATTN_REFERENCE_LAYER", -1);
    static const int target_token = ParseIntEnv("DENSECORE_DEBUG_PAGED_ATTN_REFERENCE_TOKEN", 0);
    static std::atomic<int> remaining_budget{ParsePositiveEnvInt("DENSECORE_DEBUG_PAGED_ATTN_REFERENCE_MAX_CALLS", 1)};

    if (target_layer >= 0 && layer != target_layer) {
        return false;
    }
    if (target_token >= 0 && token_idx != target_token) {
        return false;
    }

    int remaining = remaining_budget.load(std::memory_order_relaxed);
    while (remaining > 0) {
        if (remaining_budget.compare_exchange_weak(remaining, remaining - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

static bool ShouldRunPagedAttentionEagerReferenceProbe(int layer, int token_idx) {
    if (!IsDebugPagedAttentionEagerReferenceEnabled()) {
        return false;
    }
    if (IsCurrentGraphBuildNoAlloc()) {
        return false;
    }

    static const int target_layer = ParseIntEnv("DENSECORE_DEBUG_PAGED_ATTN_EAGER_REFERENCE_LAYER", -1);
    static const int target_token = ParseIntEnv("DENSECORE_DEBUG_PAGED_ATTN_EAGER_REFERENCE_TOKEN", 0);
    static std::atomic<int> remaining_budget{
        ParsePositiveEnvInt("DENSECORE_DEBUG_PAGED_ATTN_EAGER_REFERENCE_MAX_CALLS", 1)};

    if (target_layer >= 0 && layer != target_layer) {
        return false;
    }
    if (target_token >= 0 && token_idx != target_token) {
        return false;
    }

    int remaining = remaining_budget.load(std::memory_order_relaxed);
    while (remaining > 0) {
        if (remaining_budget.compare_exchange_weak(remaining, remaining - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

static bool ShouldRunAttentionCoreReferenceProbe(int layer, int token_idx) {
    if (!IsDebugAttentionCoreReferenceEnabled()) {
        return false;
    }
    if (IsCurrentGraphBuildNoAlloc()) {
        return false;
    }

    static const int target_layer = ParseIntEnv("DENSECORE_DEBUG_ATTN_CORE_REFERENCE_LAYER", -1);
    static const int target_token = ParseIntEnv("DENSECORE_DEBUG_ATTN_CORE_REFERENCE_TOKEN", 0);
    static std::atomic<int> remaining_budget{ParsePositiveEnvInt("DENSECORE_DEBUG_ATTN_CORE_REFERENCE_MAX_CALLS", 1)};

    if (target_layer >= 0 && layer != target_layer) {
        return false;
    }
    if (target_token >= 0 && token_idx != target_token) {
        return false;
    }

    int remaining = remaining_budget.load(std::memory_order_relaxed);
    while (remaining > 0) {
        if (remaining_budget.compare_exchange_weak(remaining, remaining - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

static bool ShouldRunAttentionPostReferenceProbe(int layer, int token_idx) {
    if (!IsDebugAttentionPostReferenceEnabled()) {
        return false;
    }
    if (IsCurrentGraphBuildNoAlloc()) {
        return false;
    }

    static const int target_layer = ParseIntEnv("DENSECORE_DEBUG_ATTN_POST_REFERENCE_LAYER", -1);
    static const int target_token = ParseIntEnv("DENSECORE_DEBUG_ATTN_POST_REFERENCE_TOKEN", 0);
    static std::atomic<int> remaining_budget{ParsePositiveEnvInt("DENSECORE_DEBUG_ATTN_POST_REFERENCE_MAX_CALLS", 1)};

    if (target_layer >= 0 && layer != target_layer) {
        return false;
    }
    if (target_token >= 0 && token_idx != target_token) {
        return false;
    }

    int remaining = remaining_budget.load(std::memory_order_relaxed);
    while (remaining > 0) {
        if (remaining_budget.compare_exchange_weak(remaining, remaining - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

static bool ShouldRunAttentionProjectionReferenceProbe(int layer) {
    if (!IsDebugAttentionProjectionReferenceEnabled()) {
        return false;
    }
    if (IsCurrentGraphBuildNoAlloc()) {
        return false;
    }

    static const int target_layer = ParseIntEnv("DENSECORE_DEBUG_ATTN_PROJECTION_REFERENCE_LAYER", -1);
    static std::atomic<int> remaining_budget{
        ParsePositiveEnvInt("DENSECORE_DEBUG_ATTN_PROJECTION_REFERENCE_MAX_CALLS", 8)};

    if (target_layer >= 0 && layer != target_layer) {
        return false;
    }

    int remaining = remaining_budget.load(std::memory_order_relaxed);
    while (remaining > 0) {
        if (remaining_budget.compare_exchange_weak(remaining, remaining - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

static bool ShouldRunFinalProjectionReferenceProbe() {
    if (!IsDebugFinalProjectionReferenceEnabled()) {
        return false;
    }
    if (IsCurrentGraphBuildNoAlloc()) {
        return false;
    }

    static std::atomic<int> remaining_budget{
        ParsePositiveEnvInt("DENSECORE_DEBUG_FINAL_PROJECTION_REFERENCE_MAX_CALLS", 1)};
    int remaining = remaining_budget.load(std::memory_order_relaxed);
    while (remaining > 0) {
        if (remaining_budget.compare_exchange_weak(remaining, remaining - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

static bool ShouldRunHiddenSnapshotProbe(int layer, const char* stage) {
    if (!IsDebugHiddenSnapshotEnabled()) {
        return false;
    }
    if (IsCurrentGraphBuildNoAlloc()) {
        return false;
    }

    static const int target_layer = ParseIntEnv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_LAYER", -2);
    if (target_layer >= -1 && layer != target_layer) {
        return false;
    }

    static const std::string target_stage = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_STAGE");
        return (env && env[0] != '\0') ? std::string(env) : std::string();
    }();
    if (!target_stage.empty() && stage && target_stage != stage) {
        return false;
    }

    static std::atomic<int> remaining_budget{ParsePositiveEnvInt("DENSECORE_DEBUG_HIDDEN_SNAPSHOT_MAX_CALLS", 4)};
    int remaining = remaining_budget.load(std::memory_order_relaxed);
    while (remaining > 0) {
        if (remaining_budget.compare_exchange_weak(remaining, remaining - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

static bool ShouldRunFfnProjectionReferenceProbe(int layer) {
    if (!IsDebugFfnProjectionReferenceEnabled()) {
        return false;
    }
    if (IsCurrentGraphBuildNoAlloc()) {
        return false;
    }

    static const int target_layer = ParseIntEnv("DENSECORE_DEBUG_FFN_PROJECTION_REFERENCE_LAYER", -1);
    static std::atomic<int> remaining_budget{
        ParsePositiveEnvInt("DENSECORE_DEBUG_FFN_PROJECTION_REFERENCE_MAX_CALLS", 8)};
    if (target_layer >= 0 && layer != target_layer) {
        return false;
    }
    int remaining = remaining_budget.load(std::memory_order_relaxed);
    while (remaining > 0) {
        if (remaining_budget.compare_exchange_weak(remaining, remaining - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

static bool IsDebugRmsNormReferenceEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_RMSNORM_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool ShouldRunRmsNormReferenceProbe(int layer) {
    if (!IsDebugRmsNormReferenceEnabled()) {
        return false;
    }
    if (IsCurrentGraphBuildNoAlloc()) {
        return false;
    }

    static const int target_layer = ParseIntEnv("DENSECORE_DEBUG_RMSNORM_REFERENCE_LAYER", -1);
    static std::atomic<int> remaining_budget{ParsePositiveEnvInt("DENSECORE_DEBUG_RMSNORM_REFERENCE_MAX_CALLS", 8)};
    if (target_layer >= 0 && layer != target_layer) {
        return false;
    }

    int remaining = remaining_budget.load(std::memory_order_relaxed);
    while (remaining > 0) {
        if (remaining_budget.compare_exchange_weak(remaining, remaining - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

static void DebugLogTensorFiniteStats(const char* tag, const struct ggml_tensor* tensor) {
    if (!tag || !tensor || !tensor->data || tensor->type != GGML_TYPE_F32) {
        return;
    }
    const float* data = reinterpret_cast<const float*>(tensor->data);
    const int n = ggml_nelements(tensor);
    if (n <= 0) {
        return;
    }

    int nan_ct = 0;
    int inf_ct = 0;
    int zero_ct = 0;
    int finite_ct = 0;
    float mn = std::numeric_limits<float>::infinity();
    float mx = -std::numeric_limits<float>::infinity();
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        const float v = data[i];
        if (std::isnan(v)) {
            ++nan_ct;
            continue;
        }
        if (!std::isfinite(v)) {
            ++inf_ct;
            continue;
        }
        if (v == 0.0f) {
            ++zero_ct;
        }
        mn = std::min(mn, v);
        mx = std::max(mx, v);
        sum += v;
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
        ++finite_ct;
    }

    if (!std::isfinite(mn)) mn = 0.0f;
    if (!std::isfinite(mx)) mx = 0.0f;
    const double mean = finite_ct > 0 ? (sum / finite_ct) : 0.0;
    const double rms = finite_ct > 0 ? std::sqrt(sum_sq / finite_ct) : 0.0;
    std::fprintf(
        stderr,
        "[%s] type=%d shape=[%ld,%ld,%ld] total=%d zero=%d nan=%d inf=%d min=%.6f max=%.6f mean=%.6f rms=%.6f\n", tag,
        static_cast<int>(tensor->type), static_cast<long>(tensor->ne[0]), static_cast<long>(tensor->ne[1]),
        static_cast<long>(tensor->ne[2]), n, zero_ct, nan_ct, inf_ct, mn, mx, mean, rms);
}

static void ComputeMatmulReferenceF32(const struct ggml_tensor* weight, const struct ggml_tensor* input,
                                      std::vector<float>* out) {
    constexpr int kMatmulReferenceMaxDequant = 16384;
    constexpr size_t kMatmulReferenceMaxQuantRowBytes = 1u << 16;
    if (!weight || !input || !out || !weight->data || !input->data || input->type != GGML_TYPE_F32) {
        out->clear();
        return;
    }
    const int M = static_cast<int>(input->ne[1]);
    if (M <= 0) {
        out->clear();
        return;
    }

    const int input_dim = static_cast<int>(input->ne[0]);
    const int ne0 = static_cast<int>(weight->ne[0]);
    const int ne1 = static_cast<int>(weight->ne[1]);
    const char* weight_base = reinterpret_cast<const char*>(weight->data);
    const char* input_base = reinterpret_cast<const char*>(input->data);

    if (weight->type == GGML_TYPE_F32 && input_dim == ne1 && ne0 > 0 && ne1 > 0) {
        out->assign(static_cast<size_t>(ne0) * static_cast<size_t>(M), 0.0f);
        for (int m = 0; m < M; ++m) {
            const char* src_col = input_base + static_cast<size_t>(m) * static_cast<size_t>(input->nb[1]);
            for (int out_idx = 0; out_idx < ne0; ++out_idx) {
                float sum = 0.0f;
                for (int in_idx = 0; in_idx < ne1; ++in_idx) {
                    const char* weight_row =
                        weight_base + static_cast<size_t>(in_idx) * static_cast<size_t>(weight->nb[1]);
                    const float w =
                        *reinterpret_cast<const float*>(weight_row + static_cast<size_t>(out_idx) * weight->nb[0]);
                    const float x =
                        *reinterpret_cast<const float*>(src_col + static_cast<size_t>(in_idx) * input->nb[0]);
                    sum += w * x;
                }
                (*out)[static_cast<size_t>(m) * static_cast<size_t>(ne0) + static_cast<size_t>(out_idx)] = sum;
            }
        }
        return;
    }

    const int N = ne0;
    const int K = ne1;
    if (N <= 0 || K <= 0 || input_dim != N) {
        out->clear();
        return;
    }

    out->assign(static_cast<size_t>(K) * static_cast<size_t>(M), 0.0f);
    if (weight->type == GGML_TYPE_F32) {
        for (int k = 0; k < K; ++k) {
            const char* row_ptr = weight_base + static_cast<size_t>(k) * static_cast<size_t>(weight->nb[1]);
            for (int m = 0; m < M; ++m) {
                const char* src_col = input_base + static_cast<size_t>(m) * static_cast<size_t>(input->nb[1]);
                float sum = 0.0f;
                for (int i = 0; i < N; ++i) {
                    const float w = *reinterpret_cast<const float*>(row_ptr + static_cast<size_t>(i) * weight->nb[0]);
                    const float x = *reinterpret_cast<const float*>(src_col + static_cast<size_t>(i) * input->nb[0]);
                    sum += w * x;
                }
                (*out)[static_cast<size_t>(m) * static_cast<size_t>(K) + static_cast<size_t>(k)] = sum;
            }
        }
        return;
    }

    const auto* type_traits_cpu = ggml_get_type_traits_cpu(weight->type);
    if (type_traits_cpu && type_traits_cpu->vec_dot) {
        const ggml_type vec_dot_type = type_traits_cpu->vec_dot_type;
        const auto* input_type_traits = ggml_get_type_traits_cpu(vec_dot_type);
        const size_t quant_row_size = ggml_row_size(vec_dot_type, static_cast<int64_t>(N));
        if (input_type_traits && input_type_traits->from_float && quant_row_size > 0 &&
            quant_row_size <= kMatmulReferenceMaxQuantRowBytes) {
            std::vector<uint8_t> quant_rows(quant_row_size * static_cast<size_t>(M));
            for (int m = 0; m < M; ++m) {
                const char* src_col = input_base + static_cast<size_t>(m) * static_cast<size_t>(input->nb[1]);
                std::vector<float> gathered_input(static_cast<size_t>(N), 0.0f);
                for (int i = 0; i < N; ++i) {
                    gathered_input[static_cast<size_t>(i)] =
                        *reinterpret_cast<const float*>(src_col + static_cast<size_t>(i) * input->nb[0]);
                }
                input_type_traits->from_float(gathered_input.data(),
                                              quant_rows.data() + quant_row_size * static_cast<size_t>(m),
                                              static_cast<int64_t>(N));
            }

            out->assign(static_cast<size_t>(K) * static_cast<size_t>(M), 0.0f);
            for (int k = 0; k < K; ++k) {
                const void* row_ptr = weight_base + static_cast<size_t>(k) * static_cast<size_t>(weight->nb[1]);
                for (int m = 0; m < M; ++m) {
                    float sum = 0.0f;
                    const void* q_ptr = quant_rows.data() + quant_row_size * static_cast<size_t>(m);
                    type_traits_cpu->vec_dot(N, &sum, 0, row_ptr, 0, q_ptr, 0, 1);
                    (*out)[static_cast<size_t>(m) * static_cast<size_t>(K) + static_cast<size_t>(k)] = sum;
                }
            }
            return;
        }
    }

    const auto* type_traits = ggml_get_type_traits(weight->type);
    if (!type_traits || !type_traits->to_float || N > kMatmulReferenceMaxDequant) {
        out->clear();
        return;
    }

    std::vector<float> dequant_row(static_cast<size_t>(N), 0.0f);
    for (int k = 0; k < K; ++k) {
        const void* row_ptr = weight_base + static_cast<size_t>(k) * static_cast<size_t>(weight->nb[1]);
        type_traits->to_float(row_ptr, dequant_row.data(), N);
        for (int m = 0; m < M; ++m) {
            const char* src_col = input_base + static_cast<size_t>(m) * static_cast<size_t>(input->nb[1]);
            float sum = 0.0f;
            for (int i = 0; i < N; ++i) {
                const float x = *reinterpret_cast<const float*>(src_col + static_cast<size_t>(i) * input->nb[0]);
                sum += dequant_row[static_cast<size_t>(i)] * x;
            }
            (*out)[static_cast<size_t>(m) * static_cast<size_t>(K) + static_cast<size_t>(k)] = sum;
        }
    }
}

static bool IsDecodeProfileEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_PROFILE_DECODE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsDecodeAttentionPathLoggingEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_LOG_DECODE_ATTENTION_PATH");
        if (env && env[0] != '\0' && std::strcmp(env, "0") != 0) {
            return true;
        }
        return IsDecodeProfileEnabled() || IsDebugInferenceStatsEnabled();
    }();
    return enabled;
}

using DecodeAttentionPathKind = densecore::llm::attention::DecodeAttentionPathKind;

static void RecordDecodePagedFallbackReason(DecodePagedFallbackReason reason, int layer, int N) {
    (void)layer;
    densecore::llm::attention::RecordDecodePagedFallbackReason(reason, N);
}

static void RecordSharedQuantReuse(bool reused_shared_buffer) {
    densecore::llm::attention::RecordSharedQuantReuse(reused_shared_buffer);
}

static void RecordDecodeAttentionPath(DecodeAttentionPathKind kind, int layer, int N, int n_past_val, int n_head,
                                      int n_head_kv, densecore::DeviceType preferred_device, bool native_layout,
                                      bool paged_candidate, bool paged_selected, bool portable_supported,
                                      bool offset_safe) {
    (void)layer;
    densecore::llm::attention::RecordDecodeAttentionPath(
        kind, N, n_past_val, n_head, n_head_kv, preferred_device, native_layout, paged_candidate, paged_selected,
        portable_supported, offset_safe, IsDebugInferenceStatsEnabled());
}

static bool IsForceSafeGqaDecodeEnabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_FORCE_SAFE_GQA_DECODE");
        if (!env || env[0] == '\0') return true;
        return std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool IsFlashAttentionIsaSupported() {
    static const bool supported = []() { return densecore::simd::HasX86Avx512OrBetter(GetRuntimeSimdLevel()); }();
    return supported;
}

static bool IsFusedQKVEnabled() {
    if (DENSECORE_DEFAULT_FUSED_QKV == 0) {
        return false;
    }
    const densecore::simd::SimdLevel simd = GetRuntimeSimdLevel();
    return densecore::simd::HasX86Avx2OrBetter(simd) || densecore::simd::IsArmFamily(simd);
}
}  // namespace

bool IsDecodeOnlyBatchLayout(const BatchSpec& batch, int n_tokens_in_batch) {
    return densecore::llm::attention::IsDecodeOnlyBatchLayoutImpl(batch, n_tokens_in_batch);
}

bool IsPagedDecodeCandidate(const PagedKVCache* cache, const BatchSpec& batch, int n_tokens_in_batch, int n_head,
                            int n_head_kv, int head_dim_q, int head_dim_kv) {
    return densecore::llm::attention::IsPagedDecodeCandidateImpl(cache, batch, n_tokens_in_batch, n_head, n_head_kv,
                                                                 head_dim_q, head_dim_kv);
}

bool IsPagedDecodeModeAlwaysOn() {
    return true;
}

DecodeRuntimeStatsSnapshot GetDecodeRuntimeStatsSnapshot() {
    DecodeRuntimeStatsSnapshot snapshot = densecore::llm::attention::GetDecodeRuntimeStatsSnapshotImpl();
    for (std::size_t i = 0; i < snapshot.hybrid_ssm_dispatch_counts.size(); ++i) {
        snapshot.hybrid_ssm_dispatch_counts[i] = g_hybrid_ssm_dispatch_counters[i].load(std::memory_order_relaxed);
    }
    return snapshot;
}

const char* GetDecodePagedFallbackReasonName(std::size_t index) {
    if (index >= kDecodePagedFallbackReasonCount) {
        return "unknown";
    }
    return densecore::llm::attention::DecodePagedFallbackReasonName(
        static_cast<densecore::llm::attention::DecodePagedFallbackReason>(index));
}

const char* GetHybridSSMDispatchWeightName(std::size_t index) {
    switch (index) {
    case 0: return "qkv_mixed";
    case 1: return "z";
    case 2: return "ssm_out";
    case 3:
    default: return "other";
    }
}

const char* GetHybridSSMDispatchPathName(std::size_t index) {
    switch (index) {
    case 0: return "PLAIN_GGML_CONSERVATIVE_FALLBACK";
    case 1: return "GEMV_QUANT";
    case 2: return "GGML_QUANT_NRC_M";
    case 3: return "GGML_NATIVE";
    case 4:
    default: return "OTHER";
    }
}

// Forward declaration for explicit work context
struct InferenceWorkContext;
static thread_local InferenceWorkContext* tls_work_ctx = nullptr;
static std::atomic<const BatchSpec*> g_shared_batch{nullptr};


// ============================================================================
// NEW: Robust KV Cache Update and Gather UserData
// ============================================================================
// This replaces the fragile cb_kv_manage approach that relied on ggml_pad
// assumptions. The new approach explicitly:
//   1. Writes current K/V to the PagedKVCache
//   2. Reads history from the cache into destination tensor
//   3. Appends current K/V to destination tensor
// ============================================================================

struct KVUpdateGatherUserData {
    PagedKVCache* cache;               // KV cache instance
    const BatchSpec* batch;            // Batch specification with block tables
    int layer;                         // Current transformer layer
    int head_dim;                      // Dimension per head
    int n_head_kv;                     // Number of KV heads
    int N;                             // Current batch size (new tokens)
    int n_past;                        // Number of past/history tokens
    bool is_k;                         // True for K tensor, false for V tensor
    bool read_only_shared_kv = false;  // Shared Gemma4 layers reuse cache without writing
    struct ggml_tensor* src_tensor;    // Pointer to Kcur/Vcur tensor (data accessed at runtime)
};

// Pool size for KVUpdateGatherUserData
static constexpr int kMaxKVUpdateGatherSlots = 256;

KVUpdateGatherUserData* GetKVUpdateGatherUserData(int layer, bool is_k);

// ============================================================================
// Multi-LoRA Batching Callback
// ============================================================================
// Applies per-request LoRA adapters during inference graph execution.
// Uses thread-local batch context to access the adapter-to-token mapping.
// The tensor name (e.g., "blk.0.attn_q") identifies which layer weights to use.
// ============================================================================

/**
 * @brief GGML callback for Multi-LoRA application during inference.
 *
 * 각 LoRA 가능 레이어(QKV, Output Projection, FFN)에서 호출됩니다.
 * Gather-Compute-Scatter 패턴으로 배치 내 각 토큰에 해당하는 LoRA 어댑터를 적용합니다.
 *
 * Time Complexity: O(N * R * D) where N=tokens, R=rank, D=hidden_dim
 * Space Complexity: O(N * max(R, D)) for scratch buffers (thread-local)
 *
 * @param dst Output tensor (base projection result, LoRA delta added in-place)
 * @param src0 First input (projection output tensor, same as dst)
 * @param src1 Second input (pre-projection hidden states, used for Gather)
 * @param ith Thread index (only thread 0 executes to avoid races)
 * @param nth Total threads
 * @param userdata Unused (uses GetCurrentBatch() for batch context)
 */
void cb_apply_multi_lora(struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1,
                         int ith, int nth, void* userdata) {
    (void)nth;
    (void)userdata;

    // Single-threaded execution: LoRA application is already parallelized internally
    if (ith != 0) return;

    if (!dst || !src0 || !dst->data || !src0->data) return;

    // Get current batch from explicit work context
    const BatchSpec* batch = GetCurrentBatch();
    if (!batch) return;

    // NOTE: ggml_map_custom2() allocates dst with ggml_dup_tensor(a), which only
    // duplicates metadata (shape/type) and does not copy payload. We must copy
    // base projection output (src0) into dst first, then apply LoRA deltas.
    if (dst->data != src0->data) {
        if (src0->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) && src0->ne[0] == dst->ne[0] &&
            src0->ne[1] == dst->ne[1]) {
            const size_t row_bytes = static_cast<size_t>(src0->ne[0]) * sizeof(float);
            for (int64_t row = 0; row < src0->ne[1]; ++row) {
                const auto* src_row = reinterpret_cast<const uint8_t*>(src0->data) + row * src0->nb[1];
                auto* dst_row = reinterpret_cast<uint8_t*>(dst->data) + row * dst->nb[1];
                memcpy(dst_row, src_row, row_bytes);
            }
        } else {
            memcpy(dst->data, src0->data, std::min(ggml_nbytes(dst), ggml_nbytes(src0)));
        }
    }

    // Early exit if no LoRA adapters in this batch
    if (batch->lora_map.empty()) return;

    // Validate tensor data - src1 is the pre-projection hidden states
    if (!src1 || !src1->data) return;

    // Extract layer name from tensor name (e.g., "blk.0.attn_q" -> "blk.0.attn_q")
    // The tensor name is set by apply_lora lambda in BuildTransformerGraph
    const char* layer_name = dst->name;
    if (!layer_name || layer_name[0] == '\0') return;

    // Convert GGML tensors to DenseCore Tensor format
    // src1 = pre-projection hidden states (input to LoRA)
    densecore::Tensor t_input;
    t_input.data = const_cast<void*>(src1->data);
    t_input.dtype = densecore::GgmlTypeToDType(src1->type);
    t_input.ndim = 2;
    t_input.shape[0] = src1->ne[1];  // tokens (GGML: ne[1] = rows after mul_mat)
    t_input.shape[1] = src1->ne[0];  // hidden_dim
    t_input.stride[0] = src1->nb[1];
    t_input.stride[1] = src1->nb[0];

    // dst = projection output (LoRA delta added in-place)
    densecore::Tensor t_output;
    t_output.data = dst->data;
    t_output.dtype = densecore::GgmlTypeToDType(dst->type);
    t_output.ndim = 2;
    t_output.shape[0] = dst->ne[1];  // tokens
    t_output.shape[1] = dst->ne[0];  // output_dim
    t_output.stride[0] = dst->nb[1];
    t_output.stride[1] = dst->nb[0];

    // Dispatch to CpuBackend (NUMA-aware, parallelized internally)
    densecore::CpuBackend& backend = densecore::GetCpuBackend();
    backend.ApplyMultiLoRA(t_input, std::string(layer_name), batch->lora_map, &t_output);
}

// ============================================================================
// Fused Add + RMSNorm Callback (AVX-512 Optimized)
// ============================================================================
// Combines residual connection (x += residual) and RMSNorm in a single pass
// to reduce memory bandwidth by loading/storing data once instead of twice.
// ============================================================================

/**
 * User data for fused Add+RMSNorm operation
 */
struct AddRMSNormUserData {
    const float* residual;                ///< Residual tensor data [n_embd, N]
    const float* rms_weight;              ///< RMSNorm weight [n_embd]
    int n_embd;                           ///< Embedding dimension
    int n_tokens;                         ///< Number of tokens
    float eps;                            ///< RMSNorm epsilon
    ptrdiff_t residual_row_stride = 0;    ///< Residual row stride in float elements
    std::vector<float> owned_rms_weight;  ///< Optional canonicalized RMS weight storage
    int layer_idx = -1;
    const int* token_seq_ids = nullptr;
    const char* stage = nullptr;
    const char* var_name = nullptr;
};

// Thread-local pool for AddRMSNorm user data
static constexpr int kMaxAddRMSNormSlots = 256;

AddRMSNormUserData* GetAddRMSNormUserData();

// =============================================================================
// Parallel GEMV User Data + Buffers
// =============================================================================
static constexpr int kMaxGemvUserDataSlots = 2048;
static constexpr size_t kMaxQuantInputBufferSize = 65536;  // 64KB for large N
static constexpr int kMaxSmallBatchColsHard = 16;
static constexpr size_t kMaxDequantBufferSize = 16384;

static int ResolveQuantBatchedTileCols(int requested_cols, int vec_dot_nrows, bool allow_true_batched_q4k) {
    int tile_cols = std::max(1, std::min(kMaxSmallBatchColsHard, requested_cols));
    if (!allow_true_batched_q4k && vec_dot_nrows > 0) {
        tile_cols = std::min(tile_cols, vec_dot_nrows);
    }
    return std::max(1, tile_cols);
}
/**
 * User data for parallel GEMV operation
 */
struct GemvUserData {
    struct ggml_tensor* weight_tensor;  // Weight tensor (data accessed at runtime)
    int N;                              // Input dimension
    int K;                              // Output dimension
    ggml_type weight_type;              // Tensor type (F32, Q4_K, Q8_0, etc.)
    ggml_type input_quant_type;         // Quantization type for input (Q8_K, Q8_0, or F32)
    bool force_reference_scalar = false;
    bool disable_q8_repacked_gemv = false;
    bool force_q8_repacked_gemv = false;
    int slot_id = -1;
    uint8_t* quant_input_shared = nullptr;
    std::atomic<uint64_t>* quantized_stamp = nullptr;
    uintptr_t model_identity = 0;
    bool dynamic_lora_active = false;
    InferenceWorkContext* work_ctx = nullptr;
    InferenceExecutionPhase phase_snapshot = InferenceExecutionPhase::Unknown;
    densecore::runtime::DenseCoreSemanticOp semantic_op = densecore::runtime::DenseCoreSemanticOp::Unknown;
    densecore::runtime::DenseCoreTensorRole tensor_role = densecore::runtime::DenseCoreTensorRole::Unknown;
    bool gemma4_decode_native = false;
    bool gemma4_decode_lm_head = false;
    bool lfm2_decode_lm_head = false;
    std::mutex lfm2_argmax_mutex;
    uintptr_t lfm2_argmax_key = 0;
    int lfm2_argmax_done = 0;
    int lfm2_argmax_nth = 0;
    int lfm2_argmax_best_token = -1;
    float lfm2_argmax_best_value = -std::numeric_limits<float>::infinity();
    std::chrono::steady_clock::time_point lfm2_argmax_begin{};
    uintptr_t lfm2_argmax_q6k_cache_weight = 0;
    int lfm2_argmax_q6k_cache_nth = 0;
    int lfm2_argmax_q6k_cache_k = 0;
    int lfm2_argmax_q6k_cache_n = 0;
    std::vector<std::shared_ptr<void>> lfm2_argmax_q6k_slice_cache;
};

struct GemvBatchedUserData {
    struct ggml_tensor* weight_tensor = nullptr;  // Weight tensor (data accessed at runtime)
    int N = 0;                                    // Input dimension
    int K = 0;                                    // Output dimension
    int M = 0;                                    // Number of input columns (tokens)
    ggml_type weight_type = GGML_TYPE_F32;
    bool force_reference_scalar = false;
    int slot_id = -1;
    ggml_type input_quant_type = GGML_TYPE_F32;
    size_t quant_row_stride = 0;  // Pre-computed aligned row stride for quantized input
    uint8_t* quant_input_shared = nullptr;
    std::atomic<uint64_t>* quantized_stamp = nullptr;
    InferenceWorkContext* work_ctx = nullptr;
    uint64_t qwen36_prefill_q4k_admission_key = 0;
    bool qwen36_prefill_q4k_probe = false;
    bool qwen36_prefill_q4k_admitted = false;
    bool require_q4k_true_batched = false;
    bool disable_quant_nrc_fast = false;
    bool gemma4_dense_prefill_native = false;
    bool gemma4_prefill_safe_batched = false;
    bool lfm2_q8_repacked_batched = false;
    bool qwen36_ssm_q8_repacked_batched = false;
    bool qwen36_ssm_q8_direct_batched = false;
    std::atomic<int> qwen36_prefill_q4k_probe_done{0};
    std::atomic<int> qwen36_prefill_q4k_probe_failures{0};
    std::atomic<int> qwen36_prefill_q4k_probe_internal_errors{0};
    std::atomic<uint32_t> qwen36_prefill_q4k_probe_max_abs_error_bits{0};
};

inline int ResolvePagedAttentionDecodeHeadTile(int n_head, int n_tokens, int n_tasks) {
    bool env_set = false;
    const int configured_head_tile =
        std::max(1, densecore::llm::config::ReadPositiveIntEnv("DENSECORE_PAGED_ATTN_DECODE_HEAD_TILE", 8, &env_set));
    if (n_head <= 0 || n_tokens <= 0 || n_tasks <= 0) {
        return configured_head_tile;
    }
    if (env_set) {
        return configured_head_tile;
    }

    if (n_tokens == 1) {
        const int active_threads = std::max(1, n_tasks);
        const int target_tasks = std::min(n_head, std::max(active_threads, active_threads * 2));
        int adaptive_head_tile = std::max(1, (n_head + std::max(1, target_tasks) - 1) / std::max(1, target_tasks));
        if (n_head <= active_threads / 2) {
            adaptive_head_tile = std::max(adaptive_head_tile, 2);
        }
        return std::clamp(adaptive_head_tile, 1, 8);
    }
    if (n_tokens <= 4) {
        const int target_tiles_per_token = std::max(1, (n_tasks + n_tokens - 1) / n_tokens);
        const int adaptive_head_tile = std::max(1, (n_head + target_tiles_per_token - 1) / target_tiles_per_token);
        return std::clamp(adaptive_head_tile, 1, 8);
    }
    return configured_head_tile;
}

/**
 * Custom callback for fused Add + RMSNorm
 *
 * Input tensor (src): Current tensor to add residual to and normalize
 * Output tensor (dst): Result of (src + residual) normalized with RMSNorm
 *
 * Uses AVX-512 fused kernel for optimal memory bandwidth utilization.
 */
void cb_residual_rmsnorm_fused(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                               void* userdata) {
    auto* ud = (AddRMSNormUserData*)userdata;
    if (!ud || !ud->rms_weight) return;
    if (!src || !dst || !src->data || !dst->data) return;
    if (src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return;
    if (src->nb[0] != static_cast<int64_t>(sizeof(float)) || dst->nb[0] != static_cast<int64_t>(sizeof(float))) return;

    const int n_embd = ud->n_embd;
    const int n_tokens = ud->n_tokens;
    const float eps = ud->eps;
    const bool has_residual = ud->residual != nullptr;
    const ptrdiff_t src_row_stride = static_cast<ptrdiff_t>(src->nb[1] / sizeof(float));
    const ptrdiff_t dst_row_stride = static_cast<ptrdiff_t>(dst->nb[1] / sizeof(float));
    const ptrdiff_t residual_row_stride = ud->residual_row_stride > 0 ? ud->residual_row_stride : n_embd;
    const bool run_reference_probe = ShouldRunAddRmsNormReferenceProbe(ud->layer_idx);

    // Partition work across tokens
    const int tokens_per_thread = (n_tokens + nth - 1) / nth;
    const int t_start = ith * tokens_per_thread;
    const int t_end = std::min(t_start + tokens_per_thread, n_tokens);

    if (t_start >= n_tokens) return;

    // Process assigned tokens
    for (int t = t_start; t < t_end; t++) {
        const float* x_ptr = reinterpret_cast<const float*>(src->data) + static_cast<ptrdiff_t>(t) * src_row_stride;
        float* out_ptr = reinterpret_cast<float*>(dst->data) + static_cast<ptrdiff_t>(t) * dst_row_stride;
        const float* res_ptr =
            has_residual ? (ud->residual + static_cast<ptrdiff_t>(t) * residual_row_stride) : nullptr;

        if (has_residual) {
            // Use unified AddRMSNorm dispatcher (Runtime AVX512/AVX2/Scalar)
            densecore::simd::AddRMSNorm(out_ptr, x_ptr, res_ptr, ud->rms_weight, static_cast<size_t>(n_embd), eps);
        } else {
            densecore::simd::RMSNorm(x_ptr, ud->rms_weight, out_ptr, static_cast<size_t>(n_embd), eps);
        }

        if (run_reference_probe) {
            static std::atomic<int> emitted{0};
            const int max_calls = ParsePositiveEnvInt("DENSECORE_DEBUG_ADD_RMSNORM_REFERENCE_MAX_CALLS", 8);
            const int prior = emitted.load(std::memory_order_relaxed);
            if (prior < max_calls && emitted.fetch_add(1, std::memory_order_relaxed) < max_calls) {
                std::vector<float> summed(static_cast<size_t>(n_embd));
                double sum_sq = 0.0;
                for (int i = 0; i < n_embd; ++i) {
                    float val = x_ptr[i];
                    if (res_ptr) {
                        val += res_ptr[i];
                    }
                    summed[static_cast<size_t>(i)] = val;
                    sum_sq += static_cast<double>(val) * static_cast<double>(val);
                }

                const float inv_rms = 1.0f / std::sqrt(static_cast<float>(sum_sq / std::max(1, n_embd)) + eps);
                float max_abs_diff = 0.0f;
                int first_bad_idx = -1;
                float first_actual = 0.0f;
                float first_ref = 0.0f;
                bool actual_nonfinite = false;
                bool ref_nonfinite = false;
                for (int i = 0; i < n_embd; ++i) {
                    const float ref = summed[static_cast<size_t>(i)] * inv_rms * ud->rms_weight[i];
                    const float actual = out_ptr[i];
                    actual_nonfinite = actual_nonfinite || !std::isfinite(actual);
                    ref_nonfinite = ref_nonfinite || !std::isfinite(ref);
                    const float diff = std::fabs(actual - ref);
                    if (diff > max_abs_diff) {
                        max_abs_diff = diff;
                        first_bad_idx = i;
                        first_actual = actual;
                        first_ref = ref;
                    }
                }
                const int seq_id = ud->token_seq_ids ? ud->token_seq_ids[t] : -1;
                fprintf(stderr,
                        "[ADD_RMS_REF] layer=%d stage=%s var=%s token=%d seq=%d has_residual=%d "
                        "max_abs_diff=%.9g first_idx=%d actual=%.9g ref=%.9g actual_nonfinite=%d "
                        "ref_nonfinite=%d\n",
                        ud->layer_idx, ud->stage ? ud->stage : "unknown", ud->var_name ? ud->var_name : "unknown", t,
                        seq_id, has_residual ? 1 : 0, max_abs_diff, first_bad_idx, first_actual, first_ref,
                        actual_nonfinite ? 1 : 0, ref_nonfinite ? 1 : 0);
            }
        }
    }
}

void cb_residual_rmsnorm_fused2(struct ggml_tensor* dst, const struct ggml_tensor* src,
                                const struct ggml_tensor* residual, int ith, int nth, void* userdata) {
    auto* ud = (AddRMSNormUserData*)userdata;
    if (!ud || !ud->rms_weight) return;
    if (!src || !residual || !dst || !src->data || !residual->data || !dst->data) return;
    if (src->type != GGML_TYPE_F32 || residual->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return;
    if (src->nb[0] != static_cast<int64_t>(sizeof(float)) ||
        residual->nb[0] != static_cast<int64_t>(sizeof(float)) ||
        dst->nb[0] != static_cast<int64_t>(sizeof(float))) {
        return;
    }

    const int n_embd = ud->n_embd;
    const int n_tokens = ud->n_tokens;
    if (n_embd <= 0 || n_tokens <= 0 || src->ne[0] != n_embd || residual->ne[0] != n_embd ||
        dst->ne[0] != n_embd) {
        return;
    }

    const float eps = ud->eps;
    const ptrdiff_t src_row_stride = static_cast<ptrdiff_t>(src->nb[1] / sizeof(float));
    const ptrdiff_t residual_row_stride = static_cast<ptrdiff_t>(residual->nb[1] / sizeof(float));
    const ptrdiff_t dst_row_stride = static_cast<ptrdiff_t>(dst->nb[1] / sizeof(float));
    const bool run_reference_probe = ShouldRunAddRmsNormReferenceProbe(ud->layer_idx);
    const int tokens_per_thread = (n_tokens + nth - 1) / nth;
    const int t_start = ith * tokens_per_thread;
    const int t_end = std::min(t_start + tokens_per_thread, n_tokens);
    if (t_start >= n_tokens) return;

    for (int t = t_start; t < t_end; ++t) {
        const float* x_ptr = reinterpret_cast<const float*>(src->data) + static_cast<ptrdiff_t>(t) * src_row_stride;
        const float* res_ptr =
            reinterpret_cast<const float*>(residual->data) + static_cast<ptrdiff_t>(t) * residual_row_stride;
        float* out_ptr = reinterpret_cast<float*>(dst->data) + static_cast<ptrdiff_t>(t) * dst_row_stride;
        densecore::simd::AddRMSNorm(out_ptr, x_ptr, res_ptr, ud->rms_weight, static_cast<size_t>(n_embd), eps);

        if (run_reference_probe) {
            static std::atomic<int> emitted{0};
            const int max_calls = ParsePositiveEnvInt("DENSECORE_DEBUG_ADD_RMSNORM_REFERENCE_MAX_CALLS", 8);
            const int prior = emitted.load(std::memory_order_relaxed);
            if (prior < max_calls && emitted.fetch_add(1, std::memory_order_relaxed) < max_calls) {
                double sum_sq = 0.0;
                for (int i = 0; i < n_embd; ++i) {
                    const float val = x_ptr[i] + res_ptr[i];
                    sum_sq += static_cast<double>(val) * static_cast<double>(val);
                }

                const float inv_rms = 1.0f / std::sqrt(static_cast<float>(sum_sq / std::max(1, n_embd)) + eps);
                float max_abs_diff = 0.0f;
                int first_bad_idx = -1;
                float first_actual = 0.0f;
                float first_ref = 0.0f;
                bool actual_nonfinite = false;
                bool ref_nonfinite = false;
                for (int i = 0; i < n_embd; ++i) {
                    const float ref = (x_ptr[i] + res_ptr[i]) * inv_rms * ud->rms_weight[i];
                    const float actual = out_ptr[i];
                    actual_nonfinite = actual_nonfinite || !std::isfinite(actual);
                    ref_nonfinite = ref_nonfinite || !std::isfinite(ref);
                    const float diff = std::fabs(actual - ref);
                    if (diff > max_abs_diff) {
                        max_abs_diff = diff;
                        first_bad_idx = i;
                        first_actual = actual;
                        first_ref = ref;
                    }
                }
                const int seq_id = ud->token_seq_ids ? ud->token_seq_ids[t] : -1;
                fprintf(stderr,
                        "[ADD_RMS_REF] layer=%d stage=%s var=%s token=%d seq=%d has_residual=1 "
                        "max_abs_diff=%.9g first_idx=%d actual=%.9g ref=%.9g actual_nonfinite=%d "
                        "ref_nonfinite=%d\n",
                        ud->layer_idx, ud->stage ? ud->stage : "unknown", ud->var_name ? ud->var_name : "unknown", t,
                        seq_id, max_abs_diff, first_bad_idx, first_actual, first_ref, actual_nonfinite ? 1 : 0,
                        ref_nonfinite ? 1 : 0);
            }
        }
    }
}

// ============================================================================
// Fused SiLU×Mul Callback (SwiGLU FFN Optimization)
// ============================================================================
// Computes: out = silu(gate) * up in a single pass
// Saves memory by avoiding intermediate silu(gate) tensor allocation.
// ============================================================================

/**
 * Custom callback for fused SiLU×Mul (SwiGLU FFN)
 *
 * Signature compatible with ggml_map_custom2:
 *   void (*)(struct ggml_tensor *dst, const struct ggml_tensor *a,
 *            const struct ggml_tensor *b, int ith, int nth, void *userdata)
 */
void cb_silu_mul_fused(struct ggml_tensor* dst, const struct ggml_tensor* a, const struct ggml_tensor* b, int ith,
                       int nth, void* userdata) {
    (void)userdata;  // Not needed since we use a/b directly

    // a = gate tensor (w1 output, SiLU input)
    // b = up tensor (w3 output)
    // dst = output tensor

    if (!a || !b || !dst || !a->data || !b->data || !dst->data) return;

    const float* gate = reinterpret_cast<const float*>(a->data);
    const float* up = reinterpret_cast<const float*>(b->data);
    float* out = reinterpret_cast<float*>(dst->data);

    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return;
    }
    if (a->nb[0] != static_cast<int64_t>(sizeof(float)) || b->nb[0] != static_cast<int64_t>(sizeof(float)) ||
        dst->nb[0] != static_cast<int64_t>(sizeof(float))) {
        return;
    }

    // Total elements (flattened)
    const size_t size = ggml_nelements(a);

    if (ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(dst)) {
        densecore::simd::SiLUMulParallel(out, gate, up, size, ith, nth);
        return;
    }

    const size_t begin = (size * static_cast<size_t>(ith)) / static_cast<size_t>(nth);
    const size_t end = (size * static_cast<size_t>(ith + 1)) / static_cast<size_t>(nth);
    const auto offset_bytes = [](const ggml_tensor* tensor, size_t flat_idx) -> size_t {
        size_t rem = flat_idx;
        size_t offset = 0;
        for (int dim = 0; dim < 4; ++dim) {
            const int64_t extent = tensor->ne[dim] > 0 ? tensor->ne[dim] : 1;
            const size_t coord = rem % static_cast<size_t>(extent);
            rem /= static_cast<size_t>(extent);
            offset += coord * static_cast<size_t>(tensor->nb[dim]);
        }
        return offset;
    };
    for (size_t flat = begin; flat < end; ++flat) {
        const float g = *reinterpret_cast<const float*>(reinterpret_cast<const char*>(a->data) + offset_bytes(a, flat));
        const float u = *reinterpret_cast<const float*>(reinterpret_cast<const char*>(b->data) + offset_bytes(b, flat));
        float* dst_ptr = reinterpret_cast<float*>(reinterpret_cast<char*>(dst->data) + offset_bytes(dst, flat));
        *dst_ptr = (g / (1.0f + std::exp(-g))) * u;
    }
}

void cb_gelu_mul_fused(struct ggml_tensor* dst, const struct ggml_tensor* a, const struct ggml_tensor* b, int ith,
                       int nth, void* userdata) {
    (void)userdata;
    if (!a || !b || !dst || !a->data || !b->data || !dst->data) return;
    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return;
    if (a->ne[0] != b->ne[0] || a->ne[1] != b->ne[1] || a->ne[2] != b->ne[2] || a->ne[3] != b->ne[3] ||
        dst->ne[0] != a->ne[0] || dst->ne[1] != a->ne[1] || dst->ne[2] != a->ne[2] || dst->ne[3] != a->ne[3]) {
        return;
    }
    if (a->nb[0] != static_cast<int64_t>(sizeof(float)) || b->nb[0] != static_cast<int64_t>(sizeof(float)) ||
        dst->nb[0] != static_cast<int64_t>(sizeof(float))) {
        return;
    }

    const size_t size = ggml_nelements(a);
    const size_t begin = (size * static_cast<size_t>(ith)) / static_cast<size_t>(nth);
    const size_t end = (size * static_cast<size_t>(ith + 1)) / static_cast<size_t>(nth);
    if (ggml_is_contiguous(a) && ggml_is_contiguous(b) && ggml_is_contiguous(dst)) {
        const float* gate = reinterpret_cast<const float*>(a->data);
        const float* up = reinterpret_cast<const float*>(b->data);
        float* out = reinterpret_cast<float*>(dst->data);
        for (size_t i = begin; i < end; ++i) {
            const float x = gate[i];
            const float x3 = x * x * x;
            const float gelu = 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x3)));
            out[i] = gelu * up[i];
        }
        return;
    }

    const auto offset_bytes = [](const ggml_tensor* tensor, size_t flat_idx) -> size_t {
        size_t rem = flat_idx;
        size_t offset = 0;
        for (int dim = 0; dim < 4; ++dim) {
            const int64_t extent = tensor->ne[dim] > 0 ? tensor->ne[dim] : 1;
            const size_t coord = rem % static_cast<size_t>(extent);
            rem /= static_cast<size_t>(extent);
            offset += coord * static_cast<size_t>(tensor->nb[dim]);
        }
        return offset;
    };
    for (size_t flat = begin; flat < end; ++flat) {
        const float x =
            *reinterpret_cast<const float*>(reinterpret_cast<const char*>(a->data) + offset_bytes(a, flat));
        const float up =
            *reinterpret_cast<const float*>(reinterpret_cast<const char*>(b->data) + offset_bytes(b, flat));
        const float x3 = x * x * x;
        const float gelu = 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x3)));
        float* out = reinterpret_cast<float*>(reinterpret_cast<char*>(dst->data) + offset_bytes(dst, flat));
        *out = gelu * up;
    }
}

void cb_gelu_tanh_unary(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata) {
    (void)userdata;
    if (!src || !dst || !src->data || !dst->data) return;

    const float* in = reinterpret_cast<const float*>(src->data);
    float* out = reinterpret_cast<float*>(dst->data);
    const size_t size = ggml_nelements(src);

    const size_t begin = (size * static_cast<size_t>(ith)) / static_cast<size_t>(nth);
    const size_t end = (size * static_cast<size_t>(ith + 1)) / static_cast<size_t>(nth);
    for (size_t i = begin; i < end; ++i) {
        const float x = in[i];
        const float x3 = x * x * x;
        out[i] = 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x3)));
    }
}

static void cb_apply_shared_scalar_gate(struct ggml_tensor* dst, const struct ggml_tensor* src,
                                        const struct ggml_tensor* gate_logits_scalar, int ith, int nth,
                                        void* userdata) {
    (void)userdata;
    if (!dst || !src || !gate_logits_scalar || !dst->data || !src->data || !gate_logits_scalar->data) {
        return;
    }

    const int hidden = static_cast<int>(src->ne[0]);
    const int tokens = static_cast<int>(std::max<int64_t>(1, src->ne[1]));
    const int scalar_elems = static_cast<int>(ggml_nelements(gate_logits_scalar));
    if (hidden <= 0 || tokens <= 0 || scalar_elems < tokens) {
        return;
    }

    const auto stable_sigmoid = [](float x) -> float {
        if (x >= 0.0f) {
            const float z = std::exp(-x);
            return 1.0f / (1.0f + z);
        }
        const float z = std::exp(x);
        return z / (1.0f + z);
    };

    const float* src_base = reinterpret_cast<const float*>(src->data);
    float* dst_base = reinterpret_cast<float*>(dst->data);
    const float* gate_base = reinterpret_cast<const float*>(gate_logits_scalar->data);
    const ptrdiff_t src_row_stride = static_cast<ptrdiff_t>(src->nb[1] / sizeof(float));
    const ptrdiff_t dst_row_stride = static_cast<ptrdiff_t>(dst->nb[1] / sizeof(float));
    const ptrdiff_t gate_row_stride = static_cast<ptrdiff_t>(gate_logits_scalar->nb[1] / sizeof(float));

    const int token_begin = (tokens * ith) / nth;
    const int token_end = (tokens * (ith + 1)) / nth;
    for (int t = token_begin; t < token_end; ++t) {
        const float gate = stable_sigmoid(gate_base[static_cast<ptrdiff_t>(t) * gate_row_stride]);
        const float* src_row = src_base + static_cast<ptrdiff_t>(t) * src_row_stride;
        float* dst_row = dst_base + static_cast<ptrdiff_t>(t) * dst_row_stride;
        densecore::simd::ScaleF32(dst_row, src_row, gate, static_cast<size_t>(hidden));
    }
}

// ============================================================================
// Fused QKV Projection Callback (Tensor-Level Parallelism)
// ============================================================================

// Computes Q, K, V projections in a single pass with intra-operator parallelism
// across the output dimension (dim_q + dim_k + dim_v). This enables
// multi-thread utilization during decode when batch_size=1.
// ============================================================================

/**
 * User data for fused Q/K/V projection operation
 */
struct QKVUserData {
    const float* w_q;  ///< Q weight [dim_q, n_embd] (row-major)
    const float* w_k;  ///< K weight [dim_k, n_embd] (row-major)
    const float* w_v;  ///< V weight [dim_v, n_embd] (row-major)
    int n_embd;        ///< Input embedding dimension
    int dim_q;         ///< Q output dimension (n_head * head_dim)
    int dim_k;         ///< K output dimension (n_head_kv * head_dim)
    int dim_v;         ///< V output dimension (n_head_kv * head_dim)
};

// ============================================================================
// SSM (Mamba2) Callback UserData Structs
// ============================================================================
struct SSMConv1DUserData {
    float* conv_state;
    const float* weight;
    int channels;
    int kernel_size;
    bool apply_silu = false;
    int layer_idx = -1;
    int ssm_ordinal = -1;
    const int* token_seq_ids = nullptr;
    const std::vector<std::vector<TransformerModel::SSMSequenceRuntimeState>*>* runtime_states = nullptr;
    Qwen36ProfileCounters* profile = nullptr;
};

// LFM2 / LFM2.5 double-gated short-conv mixer. The custom op reads the packed
// BCx projection [3*channels, N] and writes the gated conv output [channels, N],
// carrying per-sequence conv state across decode steps (reusing the same
// SSMSequenceRuntimeState.conv_state buffer; ssm_state is unused for LFM2).
struct LFM2ShortConvUserData {
    const float* conv_weight = nullptr;  // [channels * kernel], layout [channel * kernel + tap]
    int channels = 0;
    int kernel = 0;
    int conv_ordinal = -1;  // index among short-conv layers (state slot)
    int layer_idx = -1;
    const int* token_seq_ids = nullptr;
    const int* token_positions = nullptr;
    const std::vector<std::vector<TransformerModel::SSMSequenceRuntimeState>*>* runtime_states = nullptr;
    Qwen36ProfileCounters* profile = nullptr;
    InferenceWorkContext* work_ctx = nullptr;
    std::shared_ptr<densecore::kernels::Q4KRepackedGemvWeight> decode_in_proj_q4k_packed;
    const void* decode_in_proj_weight_data = nullptr;
    int decode_in_proj_rows = 0;
    int decode_in_proj_cols = 0;
    std::shared_ptr<densecore::kernels::Q4KRepackedGemvWeight> decode_out_proj_q4k_packed;
    const void* decode_out_proj_weight_data = nullptr;
    int decode_out_proj_rows = 0;
    int decode_out_proj_cols = 0;
    std::vector<float> decode_bcx;
    std::vector<uint8_t> decode_input_q8;
    std::vector<float> decode_y;
    std::vector<uint8_t> decode_y_q8;
    std::atomic<uint64_t> decode_in_proj_ready_stamp{0};
    std::atomic<int> decode_in_proj_done{0};
    std::atomic<uint64_t> decode_y_stamp{0};
    std::atomic<uint64_t> decode_out_proj_stamp{0};
};

struct ProjectionReferenceUserData {
    const struct ggml_tensor* weight_tensor = nullptr;
    const struct ggml_tensor* input_tensor = nullptr;
    const uint8_t* int4_packed = nullptr;
    const float* int4_scales = nullptr;
    const float* int4_zeros = nullptr;
    int int4_group_size = 0;
    int int4_k = 0;
    int int4_n = 0;
    const uint8_t* fp8_packed = nullptr;
    TransformerModel::FP8Format fp8_format = TransformerModel::FP8Format::E4M3FN;
    int fp8_k = 0;
    int fp8_n = 0;
    int layer_idx = -1;
    const int* token_seq_ids = nullptr;
    const char* stage = nullptr;
    const char* var_name = nullptr;
};

struct SharedScalarGateReferenceUserData {
    const struct ggml_tensor* shared_ffn_pre_gate = nullptr;
    const struct ggml_tensor* shared_gate_logits_scalar = nullptr;
    int layer_idx = -1;
    const int* token_seq_ids = nullptr;
    const char* stage = nullptr;
    const char* var_name = nullptr;
};

struct RmsNormReferenceUserData {
    const struct ggml_tensor* input_tensor = nullptr;
    const struct ggml_tensor* norm_weight = nullptr;
    int layer_idx = -1;
    const int* token_seq_ids = nullptr;
    const char* stage = nullptr;
    const char* var_name = nullptr;
};

struct HiddenSnapshotUserData {
    int layer_idx = -1;
    int token_idx = -1;
    const int* token_ids = nullptr;
    const int* token_seq_ids = nullptr;
    const char* stage = nullptr;
    const char* var_name = nullptr;
};

struct Gemma4KVSummaryUserData {
    int layer_idx = -1;
    int source_layer = -1;
    const char* action = nullptr;
    const char* kind = nullptr;
};

static Gemma4KVSummaryUserData* AllocateGemma4KVSummaryUserData(struct ggml_context* ctx_c);
static void cb_gemma4_kv_summary_probe(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                       void* userdata);

static struct ggml_tensor* MaybeAttachGemma4SharedKVProbe(struct ggml_context* ctx_c, struct ggml_tensor* tensor,
                                                          const char* action, const char* kind, int layer_idx,
                                                          int source_layer) {
    const char* env = std::getenv("DENSECORE_DEBUG_GEMMA4_SHARED_KV");
    const bool enabled = env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    if (!enabled || !ctx_c || !tensor) {
        return tensor;
    }
    auto* ud = AllocateGemma4KVSummaryUserData(ctx_c);
    if (!ud) {
        return tensor;
    }
    ud->layer_idx = layer_idx;
    ud->source_layer = source_layer;
    ud->action = action;
    ud->kind = kind;
    return ggml_map_custom1(ctx_c, tensor, cb_gemma4_kv_summary_probe, 1, ud);
}

static void ComputeMatmulReferenceInt4BindingF32(const ProjectionReferenceUserData* ud, std::vector<float>* out) {
    if (!ud || !out || !ud->input_tensor || !ud->input_tensor->data || ud->input_tensor->type != GGML_TYPE_F32 ||
        !ud->int4_packed || !ud->int4_scales || !ud->int4_zeros || ud->int4_group_size <= 0 || ud->int4_k <= 0 ||
        ud->int4_n <= 0) {
        out->clear();
        return;
    }
    const int K = ud->int4_k;
    const int N = ud->int4_n;
    const int M = static_cast<int>(ud->input_tensor->ne[1]);
    if (M <= 0 || static_cast<int>(ud->input_tensor->ne[0]) != K || (K % ud->int4_group_size) != 0 ||
        (ud->int4_group_size & 1) != 0) {
        out->clear();
        return;
    }

    const int num_groups = K / ud->int4_group_size;
    const int packed_k = (K + 1) / 2;
    out->assign(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);

    const char* input_base = reinterpret_cast<const char*>(ud->input_tensor->data);
    for (int m = 0; m < M; ++m) {
        const char* src_col = input_base + static_cast<size_t>(m) * static_cast<size_t>(ud->input_tensor->nb[1]);
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int g = 0; g < num_groups; ++g) {
                const float scale = ud->int4_scales[n * num_groups + g];
                const float zero = ud->int4_zeros[n * num_groups + g];
                const int k_start = g * ud->int4_group_size;
                const uint8_t* w_packed = ud->int4_packed + static_cast<size_t>(n) * packed_k +
                                          static_cast<size_t>(g) * (ud->int4_group_size / 2);
                for (int k = 0; k < ud->int4_group_size; ++k) {
                    const int byte_idx = k / 2;
                    const int nibble_idx = k % 2;
                    const uint8_t packed_byte = w_packed[byte_idx];
                    int8_t q = (nibble_idx == 0) ? static_cast<int8_t>(packed_byte & 0x0F)
                                                 : static_cast<int8_t>((packed_byte >> 4) & 0x0F);
                    if (q & 0x08) {
                        q = static_cast<int8_t>(q | static_cast<int8_t>(0xF0));
                    }
                    const float x = *reinterpret_cast<const float*>(src_col + static_cast<size_t>(k_start + k) *
                                                                                  ud->input_tensor->nb[0]);
                    sum += x * (scale * (static_cast<float>(q) - zero));
                }
            }
            (*out)[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)] = sum;
        }
    }
}

static void ComputeMatmulReferenceFp8BindingF32(const ProjectionReferenceUserData* ud, std::vector<float>* out) {
    if (!ud || !out || !ud->input_tensor || !ud->input_tensor->data || ud->input_tensor->type != GGML_TYPE_F32 ||
        !ud->fp8_packed || ud->fp8_k <= 0 || ud->fp8_n <= 0) {
        out->clear();
        return;
    }
    const int K = ud->fp8_k;
    const int N = ud->fp8_n;
    const int M = static_cast<int>(ud->input_tensor->ne[1]);
    if (M <= 0 || static_cast<int>(ud->input_tensor->ne[0]) != K) {
        out->clear();
        return;
    }

    out->assign(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
    std::vector<float> dequant_row(static_cast<size_t>(K), 0.0f);
    const char* input_base = reinterpret_cast<const char*>(ud->input_tensor->data);
    for (int n = 0; n < N; ++n) {
        const uint8_t* packed_row = ud->fp8_packed + static_cast<size_t>(n) * static_cast<size_t>(K);
        if (ud->fp8_format == TransformerModel::FP8Format::E5M2) {
            densecore::hwy_kernels::ConvertFP8E5M2ToFP32_Hwy(packed_row, dequant_row.data(), static_cast<int64_t>(K));
        } else {
            densecore::hwy_kernels::ConvertFP8E4M3FNToFP32_Hwy(packed_row, dequant_row.data(), static_cast<int64_t>(K));
        }
        for (int m = 0; m < M; ++m) {
            const char* src_col = input_base + static_cast<size_t>(m) * static_cast<size_t>(ud->input_tensor->nb[1]);
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                const float x =
                    *reinterpret_cast<const float*>(src_col + static_cast<size_t>(k) * ud->input_tensor->nb[0]);
                sum += dequant_row[static_cast<size_t>(k)] * x;
            }
            (*out)[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)] = sum;
        }
    }
}

static bool ShouldRunSharedScalarGateReferenceProbe(int layer_idx) {
    static const bool enabled = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_SHARED_SCALAR_GATE_REFERENCE");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    if (!enabled) return false;
    if (IsCurrentGraphBuildNoAlloc()) return false;
    static const int target_layer = []() {
        const char* env = std::getenv("DENSECORE_DEBUG_SHARED_SCALAR_GATE_REFERENCE_LAYER");
        if (!env || env[0] == '\0') return -1;
        char* end = nullptr;
        long parsed = std::strtol(env, &end, 10);
        return (end == env) ? -1 : static_cast<int>(parsed);
    }();
    static std::atomic<int> remaining_budget{[]() {
        const char* env = std::getenv("DENSECORE_DEBUG_SHARED_SCALAR_GATE_REFERENCE_MAX_CALLS");
        if (!env || env[0] == '\0') return 4;
        char* end = nullptr;
        long parsed = std::strtol(env, &end, 10);
        return (end == env || parsed <= 0) ? 4 : static_cast<int>(parsed);
    }()};
    if (target_layer >= 0 && layer_idx != target_layer) return false;
    int remaining = remaining_budget.load(std::memory_order_relaxed);
    while (remaining > 0) {
        if (remaining_budget.compare_exchange_weak(remaining, remaining - 1, std::memory_order_relaxed)) {
            return true;
        }
    }
    return false;
}

struct AttentionCoreReferenceUserData {
    const struct ggml_tensor* value_tensor = nullptr;
    const struct ggml_tensor* gate_tensor = nullptr;
    int layer_idx = -1;
    int n_head = 0;
    int n_head_kv = 0;
    int head_dim_q = 0;
    int head_dim_k = 0;
    int head_dim_v = 0;
    int n_past = 0;
    int sliding_window = -1;
    float attention_scale = 0.0f;
    float logit_softcap = 0.0f;
    const int* token_seq_ids = nullptr;
    const char* stage = nullptr;
    const char* var_name = nullptr;
};

#ifdef DENSECORE_TEST_BUILD
static std::atomic<int> g_test_capture_attention_layer{-1};
static std::vector<float>* g_test_capture_attention_out = nullptr;

static void cb_test_capture_attention_tensor(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                             void* userdata) {
    (void)nth;
    (void)userdata;
    if (!dst || !src || !dst->data || !src->data) return;
    std::memcpy(dst->data, src->data, ggml_nbytes(src));
    if (ith != 0 || !g_test_capture_attention_out || src->type != GGML_TYPE_F32) {
        return;
    }
    const int elems = static_cast<int>(ggml_nelements(src));
    g_test_capture_attention_out->resize(static_cast<size_t>(elems));
    std::memcpy(g_test_capture_attention_out->data(), src->data, static_cast<size_t>(elems) * sizeof(float));
}
#endif


// ============================================================================
// SSM Delta Callback Helper logic
// ============================================================================

struct GLMDSAPackUserData {
    int n_heads;
    int qk_nope_head_dim;
    int qk_rope_head_dim;
    int v_head_dim;
};
