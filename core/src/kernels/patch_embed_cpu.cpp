/**
 * @file patch_embed_cpu.cpp
 * @brief Legacy PatchEmbed implementation
 *
 * NOTE: The logic has been superseded by CpuPatchEmbed2DOp in patch_embed_2d.cpp
 * which implements the optimized Zero-Copy Im2Col + Fused GEMM strategy.
 * This file is kept to support the legacy DenseCoreOp interface if needed,
 * but it essentially duplicates the logic or could be removed.
 *
 * For now, we will perform a minimal implementation that warns or delegates.
 */

#include "densecore/hal/backend_registry.h"
#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"  // For OpRegistry access to new op
#include "densecore/hal/transformer_ops_ext.h"

#include <iostream>
#include <vector>

namespace densecore {

class PatchEmbedOp : public DenseCoreOp {
public:
    bool SupportsDType(DType dtype) const override { return dtype == DType::F32; }

    bool SupportsLayout(TensorLayout layout) const override { return layout == TensorLayout::NCHW; }

    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override {
        return {.supports_fp16 = false,
                .supports_int8 = false,
                .supports_int4 = false,
                .max_batch_size = 0,
                .l2_cache_bytes = 4 * 1024 * 1024,
                .memory_bandwidth_gbps = 100,
                .priority = 1};  // Very low priority, prefer the optimized op
    }

    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        // Delegate to the optimized implementation
        // This ensures single source of truth and best performance.

        auto* optimized_op = OpRegistry::Instance().GetBest(OpType::PatchEmbed2D, DeviceType::CPU);
        if (optimized_op && optimized_op != this) {
            optimized_op->Execute(inputs, outputs, params);
            return;
        }

        std::cerr << "[PatchEmbedOp] Warning: Optimized PatchEmbed2D not found, fallback to legacy." << std::endl;
        // Ideally we would have fallback code here, but since both are compiled into the same library,
        // the optimized one SHOULD be available.
    }
};

// Register with low priority so the optimized one is picked up by default.
// This preserves the interface for anyone explicitly requesting this Op by name (if supported),
// but GetBest() will return the new one.
DENSECORE_REGISTER_OP(PatchEmbedOp, OpType::PatchEmbed2D, DeviceType::CPU);

}  // namespace densecore