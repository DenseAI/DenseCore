/**
 * @file apple_silicon.h
 * @brief Apple Silicon (M1/M2/M3/M4) detection and optimization utilities
 *
 * This header provides utilities for:
 * 1. Chip generation detection (M1, M1 Pro, M2, etc.)
 * 2. Core topology (P-cores vs E-cores)
 * 3. AMX (Apple Matrix Extensions) detection
 * 4. Memory bandwidth information
 * 5. Neural Engine capabilities
 *
 * Usage:
 * @code
 *   using namespace densecore::apple;
 *
 *   ChipGeneration chip = DetectChipGeneration();
 *   std::cout << "Running on: " << ChipGenerationName(chip) << std::endl;
 *
 *   int compute_threads = GetOptimalThreadCount(); // P-cores only
 *   if (HasAMX()) {
 *     // Use AMX-accelerated matrix operations
 *   }
 * @endcode
 *
 * @see metal_backend.h for GPU acceleration
 * @see ane_backend.h for Neural Engine integration
 *
 * Copyright (c) 2025 DenseCore Authors
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DENSECORE_APPLE_SILICON_H
#define DENSECORE_APPLE_SILICON_H

#ifdef __APPLE__

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace densecore {
namespace apple {

// ============================================================================
// Chip Generation Detection
// ============================================================================

/**
 * @brief Apple Silicon chip generation enumeration
 *
 * Ordered by release date. Higher values = newer chips.
 * Each generation brings improved Neural Engine, GPU cores, and memory
 * bandwidth.
 */
enum class ChipGeneration {
    Unknown = 0,  ///< Unable to detect (pre-M1 or unsupported)

    // M1 Family (November 2020)
    M1 = 1,        ///< 8-core (4P+4E), 7-8 GPU, 11 TOPS ANE
    M1_Pro = 2,    ///< 10-core (8P+2E), 14-16 GPU
    M1_Max = 3,    ///< 10-core (8P+2E), 24-32 GPU
    M1_Ultra = 4,  ///< 20-core (16P+4E), 48-64 GPU

    // M2 Family (June 2022)
    M2 = 5,        ///< 8-core (4P+4E), 8-10 GPU, 15.8 TOPS ANE
    M2_Pro = 6,    ///< 10-12 core, 16-19 GPU
    M2_Max = 7,    ///< 12-core (8P+4E), 30-38 GPU
    M2_Ultra = 8,  ///< 24-core, 60-76 GPU

    // M3 Family (October 2023)
    M3 = 9,       ///< 8-core (4P+4E), 8-10 GPU, 18 TOPS ANE, 3nm
    M3_Pro = 10,  ///< 11-12 core, 14-18 GPU
    M3_Max = 11,  ///< 14-16 core, 30-40 GPU

    // M4 Family (November 2024)
    M4 = 12,      ///< 10-core (4P+6E), 10 GPU, 38 TOPS ANE, 3nm
    M4_Pro = 13,  ///< 12-14 core, 16-20 GPU
    M4_Max = 14,  ///< 16 core, 40 GPU (projected)
};

/**
 * @brief Detect the current Apple Silicon chip generation
 *
 * Uses sysctlbyname("hw.cpusubtype") and IOKit to identify the chip.
 * Falls back to Unknown if detection fails.
 *
 * @return ChipGeneration enum value
 */
ChipGeneration DetectChipGeneration();

/**
 * @brief Get human-readable name for a chip generation
 *
 * @param gen ChipGeneration enum value
 * @return String like "M1", "M2 Pro", "M4", etc.
 */
const char* ChipGenerationName(ChipGeneration gen);

/**
 * @brief Check if the chip is part of the M1 family
 */
inline bool IsM1Family(ChipGeneration gen) {
    return gen >= ChipGeneration::M1 && gen <= ChipGeneration::M1_Ultra;
}

/**
 * @brief Check if the chip is part of the M2 family
 */
inline bool IsM2Family(ChipGeneration gen) {
    return gen >= ChipGeneration::M2 && gen <= ChipGeneration::M2_Ultra;
}

/**
 * @brief Check if the chip is part of the M3 family
 */
inline bool IsM3Family(ChipGeneration gen) {
    return gen >= ChipGeneration::M3 && gen <= ChipGeneration::M3_Max;
}

/**
 * @brief Check if the chip is part of the M4 family
 */
inline bool IsM4Family(ChipGeneration gen) {
    return gen >= ChipGeneration::M4 && gen <= ChipGeneration::M4_Max;
}

// ============================================================================
// Core Topology
// ============================================================================

/**
 * @brief Get the number of Performance cores (P-cores)
 *
 * P-cores are optimized for single-threaded performance.
 * Use these for compute-intensive tasks like matrix operations.
 *
 * @return Number of P-cores (typically 4-12)
 */
int GetPerformanceCoreCount();

/**
 * @brief Get the number of Efficiency cores (E-cores)
 *
 * E-cores are optimized for power efficiency.
 * Good for background tasks and I/O-bound operations.
 *
 * @return Number of E-cores (typically 4-6)
 */
int GetEfficiencyCoreCount();

/**
 * @brief Get total CPU core count
 * @return P-cores + E-cores
 */
inline int GetTotalCoreCount() {
    return GetPerformanceCoreCount() + GetEfficiencyCoreCount();
}

/**
 * @brief Get optimal thread count for compute workloads
 *
 * For LLM inference, using only P-cores typically gives better performance
 * than using all cores, since E-cores have lower IPC and can cause
 * thread synchronization bottlenecks.
 *
 * @return Recommended thread count for compute (usually == P-core count)
 */
int GetOptimalComputeThreadCount();

/**
 * @brief Pin current thread to P-cores
 *
 * Uses thread_policy_set() with THREAD_AFFINITY to prefer P-cores.
 * Note: macOS doesn't guarantee strict affinity, this is a hint.
 *
 * @return true if policy was set successfully
 */
bool PinToPerformanceCores();

/**
 * @brief Pin current thread to E-cores
 *
 * Useful for I/O threads or background tasks.
 *
 * @return true if policy was set successfully
 */
bool PinToEfficiencyCores();

// ============================================================================
// Apple Matrix Extensions (AMX)
// ============================================================================

/**
 * @brief Check if AMX (Apple Matrix Extensions) is available
 *
 * AMX is a coprocessor on Apple Silicon that accelerates matrix operations
 * directly on the CPU. DenseCore currently relies on Accelerate.framework's
 * BLAS routines to reach AMX for FP32 GEMV/GEMM.
 *
 * This signal does not imply that DenseCore has custom AMX kernels for
 * quantized INT4 paths; those still use the custom Apple CPU kernel path.
 *
 * @return true if AMX is available
 */
bool HasAMX();

/**
 * @brief Get AMX matrix block size
 *
 * AMX operates on fixed-size tiles. Knowing this helps optimize
 * matrix dimensions for AMX acceleration.
 *
 * @return Block size in elements (typically 16 or 32)
 */
int GetAMXBlockSize();

/**
 * @brief AMX-accelerated GEMV using Accelerate.framework
 *
 * Falls back to NEON if AMX is not optimal for the given dimensions.
 * Uses cblas_sgemv internally.
 *
 * Computes: output = weight @ input (GEMV: [M,K] @ [K] = [M])
 *
 * @param output Output vector [M]
 * @param input Input vector [K]
 * @param weight Weight matrix [M, K] (row-major)
 * @param M Number of output elements
 * @param K Input dimension
 */
void GemvAccelerate(float* output, const float* input, const float* weight, int M, int K);

/**
 * @brief AMX-accelerated GEMM using Accelerate.framework
 *
 * Uses cblas_sgemm internally with optimal settings for Apple Silicon.
 *
 * Computes: C = A @ B ([M,K] @ [K,N] = [M,N])
 *
 * @param C Output matrix [M, N]
 * @param A Input matrix A [M, K]
 * @param B Input matrix B [K, N]
 * @param M Rows of A and C
 * @param N Columns of B and C
 * @param K Columns of A, rows of B
 */
void GemmAccelerate(float* C, const float* A, const float* B, int M, int N, int K);

/**
 * @brief Check whether DenseCore has Apple CPU INT4 kernels available
 *
 * These kernels are implemented in DenseCore and tuned for Apple Silicon's
 * current NEON core path. They complement Accelerate, which does not cover
 * DenseCore's custom quantized INT4 layouts.
 */
bool HasCustomInt4Kernels();

/**
 * @brief Apple-tuned INT4 GEMM over an output-column slice
 *
 * Computes C[:, n_start:n_end] = A @ W^T for a packed INT4 weight matrix.
 * The slice form allows the caller to parallelize across output columns
 * without temporary buffers.
 */
void GemmInt4CustomRange(float* C, const float* A, const uint8_t* W_int4, const float* scales, const float* zero_points,
                         int M, int N, int K, int group_size, int n_start = 0, int n_end = -1);

// ============================================================================
// Memory Information
// ============================================================================

/**
 * @brief Memory bandwidth and capacity information
 */
struct MemoryInfo {
    uint64_t total_bytes;      ///< Total unified memory in bytes
    uint64_t available_bytes;  ///< Currently available memory
    float bandwidth_gbps;      ///< Memory bandwidth in GB/s
};

/**
 * @brief Get unified memory information
 *
 * @return MemoryInfo struct with capacity and bandwidth
 */
MemoryInfo GetMemoryInfo();

/**
 * @brief Get memory bandwidth for a specific chip generation
 *
 * @param gen ChipGeneration enum value
 * @return Memory bandwidth in GB/s
 */
float GetMemoryBandwidth(ChipGeneration gen);

// ============================================================================
// Neural Engine (ANE) Information
// ============================================================================

/**
 * @brief Neural Engine capabilities
 */
struct NeuralEngineInfo {
    bool available;      ///< Whether ANE is available via CoreML
    int tops;            ///< Compute power in Trillion Operations Per Second
    int cores;           ///< Number of Neural Engine cores (typically 16)
    bool supports_int8;  ///< INT8 quantization support
    bool supports_fp16;  ///< FP16 support
};

/**
 * @brief Get Neural Engine information
 *
 * @return NeuralEngineInfo struct
 */
NeuralEngineInfo GetNeuralEngineInfo();

/**
 * @brief Check if Neural Engine is available for compute
 *
 * ANE is available on all M-series chips via CoreML.
 *
 * @return true if ANE can be used
 */
bool IsNeuralEngineAvailable();

/**
 * @brief Get Neural Engine TOPS for current chip
 *
 * @return Trillion Operations Per Second (11-38 depending on chip)
 */
int GetNeuralEngineTOPS();

// ============================================================================
// Thermal and Power
// ============================================================================

/**
 * @brief Current thermal state
 */
enum class ThermalState {
    Nominal = 0,   ///< Normal operation
    Fair = 1,      ///< Slightly elevated temperature
    Serious = 2,   ///< Approaching thermal limit
    Critical = 3,  ///< Performance throttling active
};

/**
 * @brief Get current thermal state
 *
 * Useful for adaptive scheduling - reduce parallelism when thermal throttling.
 *
 * @return Current ThermalState
 */
ThermalState GetThermalState();

/**
 * @brief Power mode preference
 */
enum class PowerMode {
    LowPower = 0,   ///< Prefer efficiency over performance
    Automatic = 1,  ///< System-managed (default)
    HighPower = 2,  ///< Prefer performance over efficiency
};

/**
 * @brief Get current power mode
 * @return Current PowerMode
 */
PowerMode GetPowerMode();

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Check if running on Apple Silicon (vs Intel Mac)
 *
 * @return true if current machine uses Apple Silicon
 */
bool IsAppleSilicon();

/**
 * @brief Check whether Apple hybrid scheduling is enabled
 *
 * Reads `DENSECORE_APPLE_HYBRID`. The Apple runtime defaults to enabled
 * unless the variable is explicitly set to `"0"`.
 *
 * @return true if hybrid scheduling should be used
 */
inline bool IsAppleHybridEnabled() {
    const char* env = std::getenv("DENSECORE_APPLE_HYBRID");
    if (!env || env[0] == '\0') {
        return true;
    }
    return std::strcmp(env, "0") != 0;
}

/**
 * @brief Check if running in Rosetta 2 translation
 *
 * When true, the binary is x86_64 running on ARM via translation.
 * Native ARM builds will return false.
 *
 * @return true if running under Rosetta 2
 */
bool IsRunningRosetta();

/**
 * @brief Get macOS version as a packed integer
 *
 * Format: major * 10000 + minor * 100 + patch
 * Example: macOS 14.2.1 -> 140201
 *
 * @return Packed version number
 */
int GetMacOSVersion();

/**
 * @brief Check if macOS version is at least the specified version
 *
 * @param major Major version (e.g., 14 for Sonoma)
 * @param minor Minor version (default 0)
 * @return true if current macOS >= specified version
 */
bool MacOSVersionAtLeast(int major, int minor = 0);

}  // namespace apple
}  // namespace densecore

#endif  // __APPLE__

#endif  // DENSECORE_APPLE_SILICON_H
