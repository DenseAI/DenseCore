/**
 * @file graph_optimizer.cpp
 * @brief Graph-level optimization implementation
 *
 * Pattern Matcher 기반 그래프 최적화 구현.
 */

#include "graph_optimizer.h"

#include <algorithm>
#include <iostream>

namespace densecore {

// =============================================================================
// Global Singleton
// =============================================================================

GraphOptimizer& GetGlobalGraphOptimizer() {
    static GraphOptimizer instance;
    return instance;
}

// =============================================================================
// Pattern Registration
// =============================================================================

void GraphOptimizer::RegisterDefaultPatterns() {
    // Pattern 1: SwiGLU - silu(x) * y
    // 이미 inference.cpp에서 cb_silu_mul_fused로 직접 처리되므로
    // 여기서는 통계만 수집
    RegisterPattern({.name = "SwiGLU",
                     .match = MatchSwiGLU,
                     .replace = nullptr,  // No-op (already handled at construction time)
                     .priority = 100});

    // Pattern 2: Add + RMSNorm
    RegisterPattern({.name = "AddRMSNorm",
                     .match = MatchAddRMSNorm,
                     .replace = nullptr,  // Handled by cb_residual_rmsnorm_fused
                     .priority = 90});

    // Pattern 3: Gated MLP (future)
    RegisterPattern({.name = "GatedMLP",
                     .match = MatchGatedMLP,
                     .replace = nullptr,  // Reserved for future
                     .priority = 80});

    // Pattern 4: QKV sibling projections from same normalized hidden state
    RegisterPattern({.name = "QKV",
                     .match = MatchQKVProjection,
                     .replace = nullptr,  // Stats-only for now
                     .priority = 70});
}

void GraphOptimizer::RegisterPattern(OptimizationPattern pattern) {
    patterns_.push_back(std::move(pattern));
    // Sort by priority (descending)
    std::sort(patterns_.begin(), patterns_.end(),
              [](const OptimizationPattern& a, const OptimizationPattern& b) { return a.priority > b.priority; });
}

// =============================================================================
// Optimization Pass
// =============================================================================

int GraphOptimizer::Optimize(ggml_cgraph* graph) {
    if (!enabled_ || !graph) {
        return 0;
    }

    stats_.Reset();

    // Traverse graph in reverse topological order
    // This ensures that consumers are processed before producers
    int n_nodes = ggml_graph_n_nodes(graph);
    for (int i = n_nodes - 1; i >= 0; --i) {
        ggml_tensor* node = ggml_graph_node(graph, i);
        if (!node) continue;

        // Try each pattern in priority order
        for (const auto& pattern : patterns_) {
            if (pattern.match && pattern.match(node)) {
                stats_.total_patterns++;

                // Apply replacement if defined
                if (pattern.replace) {
                    pattern.replace(node);
                    stats_.nodes_eliminated++;
                }

                // Update specific counters
                if (pattern.name == "SwiGLU") {
                    stats_.swiglu_fusions++;
                } else if (pattern.name == "AddRMSNorm") {
                    stats_.rmsnorm_fusions++;
                } else if (pattern.name == "QKV") {
                    stats_.qkv_fusions++;
                }

                // Only one pattern per node
                break;
            }
        }
    }

    return stats_.total_patterns;
}

// =============================================================================
// Pattern Matchers
// =============================================================================

bool GraphOptimizer::MatchSwiGLU(ggml_tensor* node) {
    // Pattern: ggml_mul(ggml_silu(a), b)
    // GGML_OP_MUL with src[0] being SILU (GGML_OP_UNARY with GGML_UNARY_OP_SILU)
    if (node->op != GGML_OP_MUL) {
        return false;
    }

    // Helper lambda to check if a tensor is SiLU activation
    auto is_silu = [](ggml_tensor* t) -> bool {
        if (!t) return false;
        if (t->op != GGML_OP_UNARY) return false;
        return ggml_get_unary_op(t) == GGML_UNARY_OP_SILU;
    };

    // Check if first operand is SiLU
    if (is_silu(node->src[0])) {
        return true;
    }

    // Check if second operand is SiLU (commutative)
    if (is_silu(node->src[1])) {
        return true;
    }

    return false;
}

bool GraphOptimizer::MatchAddRMSNorm(ggml_tensor* node) {
    // Pattern: ggml_rms_norm(ggml_add(x, residual))
    // Check for this common transformer pattern
    if (node->op != GGML_OP_RMS_NORM) {
        return false;
    }

    // Check if input is an ADD operation
    if (node->src[0] && node->src[0]->op == GGML_OP_ADD) {
        return true;
    }

    return false;
}

bool GraphOptimizer::MatchGatedMLP(ggml_tensor* node) {
    // Pattern: MUL_MAT → SiLU×MUL → MUL_MAT
    // This is a more complex pattern for the full FFN block
    // For now, return false - reserved for future implementation

    // The pattern would be:
    //   node is MUL_MAT (down_proj)
    //   node->src[1] is the result of SiLU×MUL
    //   That result comes from two MUL_MATs (gate_proj and up_proj)

    if (node->op != GGML_OP_MUL_MAT) {
        return false;
    }

    if (!node->src[1]) return false;
    ggml_tensor* mid = node->src[1];

    // Case A: SwiGLU style - map_custom2 or mul of two matmuls
    if (mid->op == GGML_OP_MAP_CUSTOM2 || mid->op == GGML_OP_MUL) {
        return mid->src[0] && mid->src[1] && mid->src[0]->op == GGML_OP_MUL_MAT && mid->src[1]->op == GGML_OP_MUL_MAT;
    }

    // Case B: GELU MLP - down(matmul(gelu(matmul(x))))
    if (mid->op == GGML_OP_UNARY) {
        const auto uop = ggml_get_unary_op(mid);
        if (uop == GGML_UNARY_OP_GELU || uop == GGML_UNARY_OP_GELU_ERF || uop == GGML_UNARY_OP_GELU_QUICK) {
            return mid->src[0] && mid->src[0]->op == GGML_OP_MUL_MAT;
        }
    }

    return false;
}

bool GraphOptimizer::MatchQKVProjection(ggml_tensor* node) {
    // Heuristic: one of Q/K/V matmuls where input is shared normalized activation.
    if (node->op != GGML_OP_MUL_MAT) {
        return false;
    }
    if (!node->src[1]) {
        return false;
    }
    ggml_tensor* shared_input = node->src[1];
    // Typical attention prelude source op: RMSNorm / MUL / ADD / UNARY
    if (shared_input->op == GGML_OP_RMS_NORM || shared_input->op == GGML_OP_MUL || shared_input->op == GGML_OP_ADD ||
        shared_input->op == GGML_OP_UNARY) {
        return true;
    }
    return false;
}

}  // namespace densecore
