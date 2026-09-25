#pragma once
#include "llm/moe/native.h"
struct PagedAttentionUserData;
#include "densecore/backend/cpu_backend.h"
#include "densecore/models/decoder_model_spec.h"
#include "densecore/runtime/inference.h"
#include "llm/attention/internal.h"
#include "llm/config/runtime_config.h"
#include "llm/graph/common_types.h"
#include "llm/graph/policy_types.h"
#include "llm/graph/reference_types.h"
#include "llm/graph/ssm_types.h"
#include "llm/runtime/profile_types.h"
#include "runtime/inference_types_internal.h"
#include <atomic>
#include <cstdint>
#include <vector>
using densecore::llm::attention::BasePagedDecodeExecutionDecision;
using densecore::llm::attention::DecodeAttentionPathKind;
using densecore::llm::attention::DecodePagedDecision;
using densecore::llm::attention::DecodePagedFallbackReason;
using densecore::llm::config::DecodePagedAttentionPolicy;

// Private graph assembly/execution interface. These declarations share graph-owned
// callback arguments; mutable pools and implementations have a single owner.

// Recurrent and model-family graph operations
GLMDSAPackUserData* AllocateGLMDSAPackUserData(struct ggml_context* ctx_c);
LFM2ShortConvUserData* GetLFM2ShortConvUserData();
SSMConv1DUserData* GetSSMConv1DUserData();
SSMQwen35DeltaUserData* GetSSMQwen35DeltaUserData();
int ModelMRoPEMode(const TransformerModel* model);
bool ModelUsesMRoPE(const TransformerModel* model);
int PositionIdsPerToken(const TransformerModel* model);

// Reference checks and diagnostics
HiddenSnapshotUserData* AllocateHiddenSnapshotUserData(struct ggml_context* ctx_c);
void CheckSSMFiniteTensor(int layer_idx, const int* token_seq_ids, const struct ggml_tensor* tensor, const char* stage,
                          const char* var_name);
AttentionCoreReferenceUserData* GetAttentionCoreReferenceUserData();
ProjectionReferenceUserData* GetProjectionReferenceUserData();
RmsNormReferenceUserData* GetRmsNormReferenceUserData();
SharedScalarGateReferenceUserData* GetSharedScalarGateReferenceUserData();
bool IsSSMNonFiniteDebugEnabled();
struct ggml_tensor* MaybeAttachGemma4SharedKVProbe(struct ggml_context* ctx_c, struct ggml_tensor* tensor,
                                                   const char* action, const char* kind, int layer_idx,
                                                   int source_layer);
bool ShouldCaptureAttentionForTest(int layer);
bool ShouldRunSharedScalarGateReferenceProbe(int layer_idx);

// MoE graph construction and expert metadata
std::vector<densecore::CpuBackend::ExpertWeights> BuildExpertWeights(const TransformerLayer* layer,
                                                                     const TransformerModel* model);
void EnsureMoERebalanceThread(densecore::CpuBackend* backend);
void RecordMoEGraphWiring();

// Common graph operations and policy
AddRMSNormUserData* GetAddRMSNormUserData();
void MarkQwen36ProfileFlag(std::atomic<int>& counter);
void SetQwen36ProfileMax(std::atomic<int>& counter, int value);
void cb_apply_shared_scalar_gate(struct ggml_tensor* dst, const struct ggml_tensor* src,
                                 const struct ggml_tensor* gate_logits_scalar, int ith, int nth, void* userdata);
void cb_attention_core_reference_probe(struct ggml_tensor* dst, const struct ggml_tensor* src,
                                       const struct ggml_tensor* q_tensor, const struct ggml_tensor* k_tensor, int ith,
                                       int nth, void* userdata);
void cb_attention_post_reference_probe(struct ggml_tensor* dst, const struct ggml_tensor* src,
                                       const struct ggml_tensor* kqv_tensor, int ith, int nth, void* userdata);
void cb_compute_qkv_map2(struct ggml_tensor* dst, const struct ggml_tensor* a, const struct ggml_tensor* b, int ith,
                         int nth, void* userdata);
void cb_hidden_snapshot_probe(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata);
void cb_kv_write_only(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata);
void cb_lfm2_shortconv(struct ggml_tensor* dst, const struct ggml_tensor* a, const struct ggml_tensor* b, int ith,
                       int nth, void* userdata);
void cb_lfm2_shortconv_inout_q4k_decode(struct ggml_tensor* dst, int ith, int nth, void* userdata);
void cb_lfm2_shortconv_out_q4k_decode(struct ggml_tensor* dst, int ith, int nth, void* userdata);
void cb_pack_glm_dsa_k(struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1,
                       const struct ggml_tensor* src2, int ith, int nth, void* userdata);
void cb_pack_glm_dsa_q(struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1, int ith,
                       int nth, void* userdata);
void cb_pack_glm_dsa_v(struct ggml_tensor* dst, const struct ggml_tensor* src0, const struct ggml_tensor* src1, int ith,
                       int nth, void* userdata);
void cb_projection_reference_probe(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                   void* userdata);
void cb_residual_rmsnorm_fused(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                               void* userdata);
void cb_residual_rmsnorm_fused2(struct ggml_tensor* dst, const struct ggml_tensor* src,
                                const struct ggml_tensor* residual, int ith, int nth, void* userdata);
void cb_rmsnorm_reference_probe(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                void* userdata);
void cb_shared_scalar_gate_reference_probe(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                           void* userdata);
void cb_ssm_alpha_beta_qk_project_map3(struct ggml_tensor* dst, const struct ggml_tensor* placeholder,
                                       const struct ggml_tensor* input_src, const struct ggml_tensor* qkv, int ith,
                                       int nth, void* userdata);
void cb_ssm_conv1d(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth, void* userdata);
void cb_test_capture_attention_tensor(struct ggml_tensor* dst, const struct ggml_tensor* src, int ith, int nth,
                                      void* userdata);
struct ggml_tensor* ggml_glm_dsa_attention(struct ggml_context* ctx, struct ggml_tensor* q_cur,
                                           struct ggml_tensor* k_cur, struct ggml_tensor* v_cur,
                                           struct ggml_tensor* index_q, struct ggml_tensor* index_weights,
                                           struct ggml_tensor* index_k, PagedAttentionUserData* userdata);
struct ggml_tensor* ggml_kv_update_and_gather(struct ggml_context* ctx, struct ggml_tensor* src, int n_total,
                                              int n_tasks, KVCacheUserData* userdata);

// Attention and KV graph operations
const Gemma4SharedKVState* GetGemma4SharedKVState(int source_layer);
QKVUserData* GetQKVUserData();
void SetGemma4SharedKVState(int source_layer, ggml_tensor* k, ggml_tensor* v);
struct ggml_tensor* ggml_rope_precomputed_table(struct ggml_context* ctx, struct ggml_tensor* input,
                                                struct ggml_tensor* pos, const TransformerModel* model, int rope_dim,
                                                const BatchSpec* batch);

namespace densecore::llm::graph::detail {

// Common graph operations and policy
ggml_tensor* BuildFusedGateUpSiluMul(struct ggml_context* ctx, ggml_tensor* gate_up, int64_t n_ff, int64_t n_tokens,
                                     const char* name);
void RecordDecodePagedFallbackReason(DecodePagedFallbackReason reason, int layer, int N);
bool ShouldUsePrefillLastLogitsOnly(const TransformerModel* model, const BatchSpec& batch, int n_tokens);
bool ShouldUsePrefillPerSequenceLastLogits(const TransformerModel* model, const BatchSpec& batch, int n_tokens);

// Reference checks and diagnostics
void DebugLogSharedExpertTensor(const char* stage, int layer_idx, const struct ggml_tensor* tensor);
bool IsDebugAttentionCoreReferenceEnabled();
bool IsDebugAttentionPostReferenceEnabled();
bool IsDebugInferenceStatsEnabled();
bool IsDebugSSMProjectionReferenceEnabled();
bool IsDebugSSMQkvReferenceEnabled();
bool IsMoEWiringDebugEnabled();
bool ShouldRunAttentionProjectionReferenceProbe(int layer);
bool ShouldRunFfnProjectionReferenceProbe(int layer);
bool ShouldRunFinalProjectionReferenceProbe();
bool ShouldRunHiddenSnapshotProbe(int layer, const char* stage);
bool ShouldRunRmsNormReferenceProbe(int layer);

// MoE graph construction and expert metadata
ggml_tensor* GetLayerTensorAny(TransformerLayer* layer, std::initializer_list<const char*> keys);
bool ShouldRunMoESharedDenseBranch(const TransformerModel* model, const densecore::models::DecoderLayerSpec* layer_spec,
                                   bool is_gemma4_moe, const struct ggml_tensor* ffn_gate,
                                   const struct ggml_tensor* ffn_up, const struct ggml_tensor* ffn_down);



// Attention and KV graph operations
bool IsFlashAttentionDisabled();
bool IsFlashAttentionIsaSupported();
bool IsFusedQKVEnabled();
void RecordDecodeAttentionPath(DecodeAttentionPathKind kind, int layer, int N, int n_past_val, int n_head,
                               int n_head_kv, densecore::DeviceType preferred_device, bool native_layout,
                               bool paged_candidate, bool paged_selected, bool portable_supported, bool offset_safe);
int ResolveAttentionQueryBasePosition(const BatchSpec& batch);
const DecodePagedAttentionPolicy& ResolveDecodePagedAttentionPolicy(const BatchSpec* batch = nullptr);
float ResolveGemma4AttentionLogitSoftcapRuntime(const TransformerModel* model);
void ValidateAttentionProjectionShape3D(const struct ggml_tensor* tensor, const char* tensor_name, int layer_idx,
                                        int ne0, int ne1, int ne2, int projected_dim, int n_heads);

// Recurrent and model-family graph operations
bool IsSerialMultiSequenceSsmForced();
int ResolveSsmConvTaskCount(ModelVariant variant, int num_seqs, int n_tokens, int conv_channels, int num_threads,
                            bool force_serial_multi_seq);
int ResolveSsmDeltaTaskCount(ModelVariant variant, int num_seqs, int num_v_heads, int num_threads,
                             bool force_serial_multi_seq);

}  // namespace densecore::llm::graph::detail
