#pragma once

#include "densecore/backend/cpu_backend.h"

namespace densecore::internal {

inline bool SamePackedExpertSource(const CpuBackend::ExpertPackedInt4Weight& a,
                                  const CpuBackend::ExpertPackedInt4Weight& b) {
    return a.packed_weights == b.packed_weights && a.scales == b.scales && a.zeros == b.zeros &&
           a.group_size == b.group_size && a.K == b.K && a.N == b.N;
}

// Weights are immutable during one registration lifetime. Shape, quantization,
// sidecars and source identity are all part of the dequantized representation.
inline bool SameExpertSource(const CpuBackend::ExpertWeights& a, const CpuBackend::ExpertWeights& b) {
    return a.w1.ptr == b.w1.ptr && a.w2.ptr == b.w2.ptr && a.w3.ptr == b.w3.ptr &&
           a.w1.size == b.w1.size && a.w2.size == b.w2.size && a.w3.size == b.w3.size &&
           a.hidden_dim == b.hidden_dim && a.intermediate_dim == b.intermediate_dim &&
           a.w1_type == b.w1_type && a.w2_type == b.w2_type && a.w3_type == b.w3_type &&
           a.w1_tensor == b.w1_tensor && a.w2_tensor == b.w2_tensor && a.w3_tensor == b.w3_tensor &&
           a.gate_up_tensor == b.gate_up_tensor && a.w2_scale_tensor == b.w2_scale_tensor &&
           a.use_gelu_activation == b.use_gelu_activation && a.force_safe_reference == b.force_safe_reference &&
           a.uses_canonical_gemma4_packed_layout == b.uses_canonical_gemma4_packed_layout &&
           SamePackedExpertSource(a.w1_int4, b.w1_int4) && SamePackedExpertSource(a.w2_int4, b.w2_int4) &&
           SamePackedExpertSource(a.w3_int4, b.w3_int4);
}

}  // namespace densecore::internal
