#include "densecore/runtime/inference.h"

#include "densecore/backend/flash_attention.h"
#include "densecore/backend/hardware_topology.h"  // For compute thread affinity
#include "densecore/backend/matmul_backend.h"
#include "densecore/models/model_graph_capabilities.h"
#include "densecore/models/transformer_graph_builder.h"  // Strategy Pattern for graph building
#include "densecore/runtime/optimization_bridge.h"       // Runtime SIMD dispatch
#include "ggml-cpu.h"                                    // For ggml_get_type_traits_cpu (vec_dot)
#include "ggml.h"                                        // Required for ggml_tensor definition
#include "models/model_inference_policy.h"
#include "runtime/batched_activation_pack_cache.h"
#include "runtime/qwen36_gateup_rowpair.h"
#include "runtime/inference_types_internal.h"  // Shared internal types

#ifndef GGML_KQ_MASK_PAD
#define GGML_KQ_MASK_PAD 32
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
extern "C" {
void ggml_gemv_q4_K_8x4_q8_K(int n, float* s, size_t bs, const void* vx, const void* vy, int nr, int nc);
}
#endif

#include "densecore/backend/cpu_backend.h"
#include "densecore/exceptions.h"
#include "densecore/hal/backend_registry.h"
#include "densecore/kernels/paged_attention.h"
#include "densecore/memory/kv_cache.h"  // Added for KV cache
#include "densecore/memory/memory_pool.h"
#include "densecore/models/decoder_model_spec.h"
#include "densecore/models/model_execution_contract.h"
#include "densecore/models/qwen35_ssm_math.h"
#include "densecore/moe/moe_routing.h"
#include "densecore/quantization/int4_types.h"  // For TensorInt4
#include "densecore/runtime/dtype_utils.h"      // For GgmlTypeToDType
#include "densecore/runtime/ggml_compute_policy.h"
#include "densecore/runtime/scheduler.h"
#include "densecore/simd/simd_ops.h"
#include "kernels/hwy/hwy_kernels.h"
#include "kernels/kernel_caps.h"
#include "kernels/q4k_repacked_gemv.h"
#include "llm/attention/exec.h"
#include "llm/attention/callback_ops.h"
#include "llm/attention/diagnostics.h"
#include "llm/attention/internal.h"
#include "llm/config/runtime_config.h"
#include "llm/decoder/spec_runtime.h"
#include "llm/graph/planning.h"
#include "llm/models/common/family_internal.h"
#include "llm/runtime/deps.h"
#include "llm/runtime/cpu_execution.h"
#include "backend/cpu_moe_execution.h"
#include "llm/runtime/work_context.h"
#include "llm/runtime/spin_wait.h"
#include "llm/matmul/graph_ops.h"
#include "llm/matmul/diagnostics.h"
#include "llm/matmul/execution_policy.h"
#include "llm/matmul/kquant_batched_kernels.h"
#include "llm/matmul/q8_small_batch.h"
#include "llm/matmul/q6_small_batch.h"
#include "llm/matmul/arm_m4.h"
#include "llm/runtime/profile_helpers.h"
#include "llm/runtime/tensor_view.h"
#include "llm/runtime/profile_types.h"
#include "models/gemma4_packed_expert_layout.h"
#include "runtime/kernel_admission.h"
#include "runtime/runtime_env.h"

#ifndef DENSECORE_DEFAULT_INT4_SINGLE_THREADING_LAYER
#define DENSECORE_DEFAULT_INT4_SINGLE_THREADING_LAYER 1
#endif

#ifndef DENSECORE_DEFAULT_PRECOMPUTED_ROPE
#define DENSECORE_DEFAULT_PRECOMPUTED_ROPE 1
#endif

#ifndef DENSECORE_DEFAULT_FUSED_RESIDUAL_RMSNORM
#define DENSECORE_DEFAULT_FUSED_RESIDUAL_RMSNORM 0
#endif

#ifndef DENSECORE_DEFAULT_FUSED_QKV
#define DENSECORE_DEFAULT_FUSED_QKV 1
#endif

namespace densecore {
bool RunQ5KRepackedMoEFusedSwiGLURawProjection(CpuBackend* backend, const void* gate_weight_ptr,
                                               const void* up_weight_ptr, const float* input_data,
                                               const uint8_t* qinput_data, size_t qinput_row_bytes, float* output_data,
                                               int64_t rows, int64_t cols, int64_t input_cols, int numa_node,
                                               bool allow_parallel);
bool RunQ4KRepackedMoEProjection(CpuBackend* backend, const void* weight_ptr, const uint8_t* qinput_data,
                                 size_t qinput_row_bytes, float* output_data, int64_t rows, int64_t cols,
                                 int64_t input_cols, int numa_node, bool allow_parallel);
bool RunQ6KRepackedMoEProjection(CpuBackend* backend, const void* weight_ptr, const uint8_t* qinput_data,
                                 size_t qinput_row_bytes, float* output_data, int64_t rows, int64_t cols,
                                 int64_t input_cols, int numa_node, bool allow_parallel);
bool RunQ6KRepackedMoEProjectionCached(CpuBackend* backend, const void* weight_ptr, const uint8_t* qinput_data,
                                       size_t qinput_row_bytes, float* output_data, int64_t rows, int64_t cols,
                                       int64_t input_cols, int numa_node, bool allow_parallel,
                                       std::shared_ptr<void>* packed_cache);
bool RunQ4KRepackedMoEFusedSwiGLUProjection(CpuBackend* backend, const void* gate_weight_ptr, const void* up_weight_ptr,
                                            const float* input_data, const uint8_t* qinput_data,
                                            size_t qinput_row_bytes, float* output_data, int64_t rows, int64_t cols,
                                            int64_t input_cols, int numa_node, bool allow_parallel);
bool RunQ4KPrepackedMoEFusedSwiGLUTileRange(const void* fused_weight_ptr, const uint8_t* qinput_data,
                                            float* output_data, int64_t cols, int64_t input_cols, int tile_start,
                                            int tile_end);
bool RunMoEQ4KRawBatchedProjection(CpuBackend* backend, const void* weight_ptr, const uint8_t* qinput_data,
                                   size_t qinput_row_bytes, float* out_data, int64_t M, int64_t N, int64_t K,
                                   int numa_node, bool allow_parallel);
bool RunMoEKQuantRawBatchedProjection(CpuBackend* backend, int ggml_type_id, const void* weight_ptr,
                                      const uint8_t* qinput_data, size_t qinput_row_bytes, float* out_data, int64_t M,
                                      int64_t N, int64_t K, int numa_node, bool allow_parallel);
bool RunMoEKQuantRawBatchedFusedSwiGLU(CpuBackend* backend, int ggml_type_id, const void* gate_weight_ptr,
                                       const void* up_weight_ptr, const uint8_t* qinput_data, size_t qinput_row_bytes,
                                       float* out_data, int64_t M, int64_t N, int64_t K, int numa_node,
                                       bool allow_parallel);
}  // namespace densecore


#include "llm/graph/support_internal.h"
using namespace densecore::llm::graph::detail;

// Runtime KV gather callbacks retain their current owner until its separate extraction.
#include "runtime/inference_runtime.inl"

#include "llm/attention/graph_ops.h"


#include "llm/moe/exec.h"
#include "llm/ssm/exec.h"
#include "llm/graph/construction_ops.h"
#include "llm/ssm/internal.h"

#include "llm/graph/build_internal.h"



#include "runtime/inference_hybrid_ssm_rebind.inl"


#include "runtime/inference_test_hooks.inl"
