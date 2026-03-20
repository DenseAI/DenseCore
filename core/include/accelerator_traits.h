/**
 * @file accelerator_traits.h
 * @brief Hardware capability traits for Unified Memory Architecture (UMA)
 * backends
 *
 * This header provides a compile-time and runtime mechanism to query hardware
 * capabilities. Essential for adapting inference strategies to different
 * accelerator architectures:
 *
 * **Apple Silicon (M1-M4):**
 * - Unified memory eliminates CPU-GPU copies
 * - ANE (Neural Engine) requires graph-based execution
 * - Optimal quantization: INT8 for ANE, FP16 for GPU
 *
 * **Qualcomm Hexagon/Adreno:**
 * - Unified memory with explicit cache management
 * - DSP benefits from INT8/INT4 quantization
 * - Graph compilation reduces per-op overhead
 *
 * **Generic CPU (x86/ARM64):**
 * - All memory is "unified" by default
 * - Immediate execution mode (no graph capture needed)
 * - Strong memory model (no explicit sync)
 */

#ifndef DENSECORE_ACCELERATOR_TRAITS_H
#define DENSECORE_ACCELERATOR_TRAITS_H

#include <cstddef>
#include <cstdint>

namespace densecore {

/**
 * @brief Quantization type preferences for different accelerators
 *
 * Ordered by typical accelerator preference for edge inference.
 */
enum class QuantType : uint8_t {
    FP32 = 0,  ///< Full precision (baseline, no quantization)
    FP16 = 1,  ///< Half precision (Metal GPU, ARM NEON)
    BF16 = 2,  ///< Brain float (Intel AMX, future Apple chips)
    INT8 = 3,  ///< 8-bit integer (Apple ANE, Qualcomm DSP)
    Q4_0 = 4,  ///< 4-bit basic (GGML style, per-block scale)
    Q4_K = 5   ///< 4-bit k-quants (GGML, higher quality)
};

/**
 * @brief Memory synchronization direction
 *
 * Even with unified memory, some ARM SoCs require explicit cache coherency
 * operations. This enum specifies the direction of synchronization.
 *
 * **Apple Silicon:** No explicit sync needed (hardware coherent)
 * **Qualcomm:** May need DSP cache flush/invalidate
 * **Generic CPU:** No-op (strong memory model on x86, barriers on ARM)
 */
enum class MemorySyncDirection : uint8_t {
    HostToDevice,  ///< Flush host caches, invalidate device caches
    DeviceToHost,  ///< Flush device caches, invalidate host caches
    Bidirectional  ///< Full barrier (safest, slowest)
};

/**
 * @brief Hardware capability descriptor for UMA accelerators
 *
 * Use this struct to query accelerator capabilities at runtime and adapt
 * inference strategies accordingly. Factory methods provide common profiles.
 *
 * Example usage:
 * @code
 * auto traits = AcceleratorTraits::AppleSilicon();
 * if (traits.supports_graph_execution) {
 *     backend.BeginCapture();
 *     // Record operations...
 *     auto graph = backend.EndCapture();
 *     graph.Compile();  // NPU-specific optimization
 * }
 * @endcode
 */
struct AcceleratorTraits {
    // =========================================================================
    // Memory Model
    // =========================================================================

    /**
     * @brief True if CPU and accelerator share the same physical memory
     *
     * When true, use `AllocateUnified()` instead of separate allocations.
     * Eliminates all PCIe/DMA transfer overhead.
     */
    bool supports_unified_memory = false;

    /**
     * @brief Alignment requirement for unified memory allocations (bytes)
     *
     * - Apple Metal: 16KB for optimal MTLStorageModeShared
     * - Qualcomm: 4KB typical
     * - CPU: 64 bytes (cache line)
     */
    size_t unified_memory_alignment = 64;

    /**
     * @brief True if explicit cache sync is required on this hardware
     *
     * Even with UMA, some architectures need cache flush/invalidate.
     * Set to false for hardware-coherent systems like Apple Silicon.
     */
    bool requires_explicit_sync = false;

    // =========================================================================
    // Execution Model
    // =========================================================================

    /**
     * @brief True if accelerator benefits from graph-based execution
     *
     * NPUs (Apple ANE, Qualcomm Hexagon) prefer compiling a sequence of
     * operations into a single executable. When true, use BeginCapture/
     * EndCapture to record operations instead of executing immediately.
     */
    bool supports_graph_execution = false;

    /**
     * @brief Maximum nodes in a single operation graph
     *
     * NPUs may have limits on graph complexity. 0 = no limit.
     */
    size_t max_graph_nodes = 0;

    /**
     * @brief True if graph compilation is required before execution
     *
     * Some NPUs require an explicit compile step that may take hundreds
     * of milliseconds. Plan for this in initialization.
     */
    bool requires_graph_compilation = false;

    // =========================================================================
    // Quantization Support
    // =========================================================================

    /**
     * @brief Preferred quantization format for this accelerator
     *
     * - Apple ANE: INT8 (limited INT4 support)
     * - Qualcomm Hexagon: INT8 preferred, Q4_K available
     * - Metal GPU: FP16 for maximum throughput
     * - CPU: Q4_K for memory-bound inference
     */
    QuantType preferred_quantization = QuantType::FP32;

    /**
     * @brief True if hardware has native quantized matmul units
     *
     * When true, keep weights quantized during computation.
     * When false, dequantize to FP16/FP32 before matmul.
     */
    bool has_native_quantized_matmul = false;

    // =========================================================================
    // Factory Methods for Common Hardware Profiles
    // =========================================================================

    /**
     * @brief Apple Silicon (M1/M2/M3/M4) profile
     *
     * Features:
     * - 16KB aligned unified memory, hardware coherent
     * - ANE supports graph execution with INT8
     * - Metal GPU prefers FP16
     */
    static AcceleratorTraits AppleSilicon() {
        AcceleratorTraits t;
        t.supports_unified_memory = true;
        t.unified_memory_alignment = 16384;  // 16KB for Metal
        t.requires_explicit_sync = false;    // Hardware coherent
        t.supports_graph_execution = true;   // For ANE
        t.max_graph_nodes = 4096;            // CoreML limit
        t.requires_graph_compilation = true;
        t.preferred_quantization = QuantType::INT8;
        t.has_native_quantized_matmul = true;  // ANE has INT8 units
        return t;
    }

    /**
     * @brief Qualcomm Hexagon/Adreno profile
     *
     * Features:
     * - Unified memory with explicit cache management
     * - DSP benefits from graph compilation
     * - INT8 preferred for Hexagon DSP
     */
    static AcceleratorTraits QualcommHexagon() {
        AcceleratorTraits t;
        t.supports_unified_memory = true;
        t.unified_memory_alignment = 4096;  // 4KB page
        t.requires_explicit_sync = true;    // Need cache flush
        t.supports_graph_execution = true;
        t.max_graph_nodes = 2048;
        t.requires_graph_compilation = true;
        t.preferred_quantization = QuantType::INT8;
        t.has_native_quantized_matmul = true;
        return t;
    }

    /**
     * @brief Generic CPU profile (x86-64 / ARM64)
     *
     * Features:
     * - All memory is inherently unified
     * - Immediate execution (no graph benefit)
     * - Q4_K preferred for memory-bandwidth savings
     */
    static AcceleratorTraits GenericCPU() {
        AcceleratorTraits t;
        t.supports_unified_memory = true;
        t.unified_memory_alignment = 64;     // Cache line
        t.requires_explicit_sync = false;    // Strong memory model
        t.supports_graph_execution = false;  // Immediate mode
        t.max_graph_nodes = 0;
        t.requires_graph_compilation = false;
        t.preferred_quantization = QuantType::Q4_K;
        t.has_native_quantized_matmul = false;  // Dequant + FP32 matmul
        return t;
    }

    /**
     * @brief MediaTek Dimensity/APU profile
     *
     * Similar to Qualcomm with graph-based NPU execution.
     */
    static AcceleratorTraits MediaTekAPU() {
        AcceleratorTraits t;
        t.supports_unified_memory = true;
        t.unified_memory_alignment = 4096;
        t.requires_explicit_sync = true;
        t.supports_graph_execution = true;
        t.max_graph_nodes = 1024;
        t.requires_graph_compilation = true;
        t.preferred_quantization = QuantType::INT8;
        t.has_native_quantized_matmul = true;
        return t;
    }

    /**
     * @brief AWS Graviton (3/4) profile
     *
     * Features:
     * - Graviton3: SVE with 256-bit vectors, BF16 support
     * - Graviton4: SVE2 with improved instructions
     * - Unified memory (single-socket, no NUMA complexity)
     * - Immediate execution mode (no NPU)
     */
    static AcceleratorTraits AWSGraviton() {
        AcceleratorTraits t;
        t.supports_unified_memory = true;
        t.unified_memory_alignment = 64;     // Cache line
        t.requires_explicit_sync = false;    // Coherent
        t.supports_graph_execution = false;  // No NPU
        t.max_graph_nodes = 0;
        t.requires_graph_compilation = false;
        t.preferred_quantization = QuantType::BF16;  // Graviton3+ has BF16 support
        t.has_native_quantized_matmul = false;       // Use dequant + FP32/BF16 matmul
        return t;
    }

    /**
     * @brief Intel Sapphire Rapids / Emerald Rapids profile
     *
     * Features:
     * - AMX (Advanced Matrix Extensions) for BF16/INT8 tiles
     * - AVX-512 with BF16 instructions
     * - VNNI for INT8 acceleration
     */
    static AcceleratorTraits IntelAMX() {
        AcceleratorTraits t;
        t.supports_unified_memory = true;
        t.unified_memory_alignment = 64;
        t.requires_explicit_sync = false;
        t.supports_graph_execution = false;
        t.max_graph_nodes = 0;
        t.requires_graph_compilation = false;
        t.preferred_quantization = QuantType::BF16;  // AMX excels at BF16
        t.has_native_quantized_matmul = true;        // AMX_INT8
        return t;
    }
};

/**
 * @brief Get name string for QuantType enum
 */
inline const char* QuantTypeName(QuantType type) {
    switch (type) {
    case QuantType::FP32: return "FP32";
    case QuantType::FP16: return "FP16";
    case QuantType::BF16: return "BF16";
    case QuantType::INT8: return "INT8";
    case QuantType::Q4_0: return "Q4_0";
    case QuantType::Q4_K: return "Q4_K";
    default: return "UNKNOWN";
    }
}

}  // namespace densecore

#endif  // DENSECORE_ACCELERATOR_TRAITS_H
