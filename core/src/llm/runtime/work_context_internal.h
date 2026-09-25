#pragma once
// Complete ownership layout is private to llm/runtime implementation files.
#include "llm/attention/exec.h"
#include "llm/graph/common_types.h"
#include "llm/graph/reference_types.h"
#include "llm/graph/ssm_types.h"
#include "llm/matmul/work_state.h"
#include "llm/runtime/cpu_execution.h"
#include "llm/runtime/profile_types.h"
#include "runtime/inference_types_internal.h"
#include <mutex>
#include <string>
#include <vector>
struct InferenceWorkContext {
    InferenceWorkContext() : cpu_telemetry(densecore::llm::runtime::MakeCpuExecutionTelemetry(this)) {}
    densecore::CpuExecutionTelemetry cpu_telemetry;
    const BatchSpec* batch = nullptr;
    ModelVariant model_variant = ModelVariant::UNKNOWN;
    InferenceExecutionPhase phase = InferenceExecutionPhase::Unknown;
    bool graph_build_no_alloc = false;
    Qwen36ProfileCounters qwen36_profile;
    KVCacheUserData kv_pool[256];
    std::vector<Gemma4SharedKVState> gemma4_shared_kv_states;
    QKVUserData qkv_pool[256];
    KVUpdateGatherUserData kv_update_gather_pool[kMaxKVUpdateGatherSlots];
    int qkv_index = 0;
    AddRMSNormUserData add_rmsnorm_pool[kMaxAddRMSNormSlots];
    int add_rmsnorm_index = 0;
    uint64_t execution_generation = 0;
    MatmulWorkState matmul;
    PagedAttentionUserData paged_attention_userdata_pool[kMaxPagedAttentionUserDataSlots];
    int paged_attention_userdata_index = 0;
    std::vector<const void*> paged_attention_shared_k_block_ptrs;
    std::vector<const void*> paged_attention_shared_v_block_ptrs;
    mutable std::mutex profile_string_mutex;
    std::string moe_small_decode_parallel_last_reject_reason;
    std::string moe_q4k_repacked_last_reject_reason;
    std::string moe_q5k_repacked_last_reject_reason;
    std::string gemma4_moe_prefill_quant_batch_last_reject_reason;
    std::string gemma4_native_moe_prefill_last_reject_reason;
    std::string gemma4_dense_prefill_native_last_reject_reason;
    std::string gemma4_decode_native_last_reject_reason;
    std::string qwen_target_ggml_compute_last_reason;
    std::string qwen_target_ggml_compute_last_op;
    std::string qwen_target_ggml_compute_target;
    std::string native_moe_fast_decode_last_reject_reason;
    std::string native_moe_fast_w2_q5k_last_reject_reason;
    std::string q6k_gemv_effective_phase;
    std::string q6k_gemv_graph_phase;
    std::string q6k_gemv_callback_phase;
    mutable std::mutex matmul_dispatch_census_mutex;
    std::vector<MatmulDispatchCensusEntry> matmul_dispatch_census_entries;
    std::vector<MatmulShapeCensusEntry> decode_matmul_shape_entries;
    std::vector<MatmulShapeCensusEntry> prefill_matmul_shape_entries;
    std::vector<MatmulShapeCensusEntry> prefill_matmul_ggml_shape_entries;
    std::vector<MatmulShapeCensusEntry> q6k_gemv_shape_entries;
    std::vector<MatmulShapeCensusEntry> qwen36_prefill_slow_entries;
    SSMConv1DUserData ssm_conv1d_pool[128];
    int ssm_conv1d_index = 0;
    LFM2ShortConvUserData lfm2_shortconv_pool[128];
    int lfm2_shortconv_index = 0;
    ProjectionReferenceUserData projection_reference_pool[384];
    int projection_reference_index = 0;
    RmsNormReferenceUserData rmsnorm_reference_pool[256];
    int rmsnorm_reference_index = 0;
    AttentionCoreReferenceUserData attention_core_reference_pool[128];
    int attention_core_reference_index = 0;
    SSMQwen35DeltaUserData ssm_qwen35_delta_pool[128];
    int ssm_qwen35_delta_index = 0;
    std::vector<ggml_bf16_t> bf16_buffer;
};
