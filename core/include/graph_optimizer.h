/**
 * @file graph_optimizer.h
 * @brief Graph-level optimization for LLM inference
 *
 * Pattern Matcher-based graph optimization system.
 * Analyzes GGML graphs and replaces known patterns with fused versions.
 *
 * Supported optimizations:
 *   1. SwiGLU Pattern: silu(x) * y -> FusedSiLUMul
 *   2. LayerNorm Pattern: rms_norm(x) * w -> FusedRMSNormMul
 *   3. GatedMLP Pattern: matmul -> silu*mul -> matmul
 */

#ifndef DENSECORE_GRAPH_OPTIMIZER_H
#define DENSECORE_GRAPH_OPTIMIZER_H

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" {
#include "ggml.h"
}

namespace densecore {

/**
 * @brief Graph optimization statistics
 */
struct OptimizationStats {
    int swiglu_fusions = 0;    ///< Number of SiLU*Mul -> FusedSiLUMul fusions applied
    int rmsnorm_fusions = 0;   ///< Number of Add+RMSNorm -> FusedAddRMSNorm fusions applied
    int qkv_fusions = 0;       ///< Number of separate Q/K/V -> FusedQKV fusions applied
    int total_patterns = 0;    ///< Total number of matched patterns
    int nodes_eliminated = 0;  ///< Number of nodes eliminated

    void Reset() {
        swiglu_fusions = 0;
        rmsnorm_fusions = 0;
        qkv_fusions = 0;
        total_patterns = 0;
        nodes_eliminated = 0;
    }
};

/**
 * @brief Graph optimization pattern definition
 */
struct OptimizationPattern {
    std::string name;                           ///< Pattern name
    std::function<bool(ggml_tensor*)> match;    ///< Matching function
    std::function<void(ggml_tensor*)> replace;  ///< Replacement function
    int priority;                               ///< Priority (higher = applied first)
};

/**
 * @brief Graph optimizer for LLM inference
 *
 * Analyzes GGML graphs and replaces with optimized fused operations.
 *
 * Usage example:
 * @code
 *   GraphOptimizer optimizer;
 *   optimizer.RegisterDefaultPatterns();
 *   optimizer.Optimize(graph);
 *   std::cout << optimizer.GetStats().swiglu_fusions << " SwiGLU fusions applied\n";
 * @endcode
 */
class GraphOptimizer {
public:
    GraphOptimizer() = default;
    ~GraphOptimizer() = default;

    // Non-copyable
    GraphOptimizer(const GraphOptimizer&) = delete;
    GraphOptimizer& operator=(const GraphOptimizer&) = delete;

    /**
     * @brief Register default optimization patterns
     *
     * LLM inference-specific patterns:
     *   - SwiGLU: silu(x) * y
     *   - Add+RMSNorm: x + residual -> rms_norm
     */
    void RegisterDefaultPatterns();

    /**
     * @brief Register a custom optimization pattern
     *
     * @param pattern Optimization pattern to register
     */
    void RegisterPattern(OptimizationPattern pattern);

    /**
     * @brief Optimize GGML graph (in-place)
     *
     * Applies registered patterns in reverse topological order.
     * Does not modify graph structure directly; replaces node op types and callbacks.
     *
     * @param graph GGML computation graph to optimize
     * @return Number of optimizations applied
     */
    int Optimize(ggml_cgraph* graph);

    /**
     * @brief Get statistics from the last optimization pass
     */
    const OptimizationStats& GetStats() const { return stats_; }

    /**
     * @brief Enable or disable optimization
     */
    void SetEnabled(bool enabled) { enabled_ = enabled; }
    bool IsEnabled() const { return enabled_; }

private:
    // =========================================================================
    // Pattern Matchers
    // =========================================================================

    /**
     * @brief Detect SwiGLU pattern: ggml_mul(ggml_silu(a), b)
     *
     * @param node Node to examine
     * @return true if pattern matches
     */
    static bool MatchSwiGLU(ggml_tensor* node);

    /**
     * @brief Detect Add+RMSNorm pattern: ggml_rms_norm(ggml_add(x, residual))
     */
    static bool MatchAddRMSNorm(ggml_tensor* node);

    /**
     * @brief Detect complete GatedMLP pattern
     */
    static bool MatchGatedMLP(ggml_tensor* node);

    /**
     * @brief Detect Q/K/V sibling projection groups sharing the same input
     */
    static bool MatchQKVProjection(ggml_tensor* node);

private:
    std::vector<OptimizationPattern> patterns_;
    OptimizationStats stats_;
    bool enabled_ = true;
};

/**
 * @brief Global GraphOptimizer singleton
 *
 * Shared instance across the inference pipeline.
 *
 * @note TODO(TechnicalDebt): Consider refactoring to dependency injection.
 *       A global singleton introduces tight coupling and makes unit testing
 *       more difficult. Pass GraphOptimizer& explicitly to components that
 *       need it (e.g., InferenceEngine ctor) for better testability and
 *       explicit dependency management.
 */
GraphOptimizer& GetGlobalGraphOptimizer();

}  // namespace densecore

#endif  // DENSECORE_GRAPH_OPTIMIZER_H
