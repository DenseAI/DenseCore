/**
 * @file apple_silicon.mm
 * @brief Apple Silicon detection and optimization implementation
 *
 * Objective-C++ implementation using:
 * - sysctl for CPU information
 * - IOKit for chip identification
 * - Accelerate.framework for BLAS operations
 * - ProcessInfo for thermal/power state
 *
 * Copyright (c) 2024 DenseCore Authors
 * SPDX-License-Identifier: Apache-2.0
 */

#include "densecore/backend/apple/apple_silicon.h"

#include "densecore/simd/simd_ops.h"

#ifdef __APPLE__

#import <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#import <mach/mach.h>
#include <string>
#import <sys/sysctl.h>
#import <sys/types.h>
#include <vector>

#include "runtime/runtime_env.h"

namespace densecore {
namespace apple {

bool HasAMX();

// ============================================================================
// Internal Helpers
// ============================================================================

namespace {

/**
 * @brief Get sysctl string value
 */
std::string GetSysctlString(const char* name) {
    size_t size = 0;
    if (sysctlbyname(name, nullptr, &size, nullptr, 0) != 0) {
        return "";
    }
    std::string result(size, '\0');
    if (sysctlbyname(name, &result[0], &size, nullptr, 0) != 0) {
        return "";
    }
    // Remove trailing null
    while (!result.empty() && result.back() == '\0') {
        result.pop_back();
    }
    return result;
}

/**
 * @brief Get sysctl integer value
 */
int64_t GetSysctlInt(const char* name) {
    int64_t value = 0;
    size_t size = sizeof(value);
    if (sysctlbyname(name, &value, &size, nullptr, 0) != 0) {
        return -1;
    }
    return value;
}

/**
 * @brief Cached chip generation (computed once)
 */
ChipGeneration g_cached_chip = ChipGeneration::Unknown;
bool g_chip_detected = false;

/**
 * @brief Detect chip from machdep.cpu.brand_string
 */
ChipGeneration DetectFromBrandString(const std::string& brand) {
    // M4 family
    if (brand.find("M4 Max") != std::string::npos)
        return ChipGeneration::M4_Max;
    if (brand.find("M4 Pro") != std::string::npos)
        return ChipGeneration::M4_Pro;
    if (brand.find("M4") != std::string::npos)
        return ChipGeneration::M4;

    // M3 family
    if (brand.find("M3 Max") != std::string::npos)
        return ChipGeneration::M3_Max;
    if (brand.find("M3 Pro") != std::string::npos)
        return ChipGeneration::M3_Pro;
    if (brand.find("M3") != std::string::npos)
        return ChipGeneration::M3;

    // M2 family
    if (brand.find("M2 Ultra") != std::string::npos)
        return ChipGeneration::M2_Ultra;
    if (brand.find("M2 Max") != std::string::npos)
        return ChipGeneration::M2_Max;
    if (brand.find("M2 Pro") != std::string::npos)
        return ChipGeneration::M2_Pro;
    if (brand.find("M2") != std::string::npos)
        return ChipGeneration::M2;

    // M1 family
    if (brand.find("M1 Ultra") != std::string::npos)
        return ChipGeneration::M1_Ultra;
    if (brand.find("M1 Max") != std::string::npos)
        return ChipGeneration::M1_Max;
    if (brand.find("M1 Pro") != std::string::npos)
        return ChipGeneration::M1_Pro;
    if (brand.find("M1") != std::string::npos)
        return ChipGeneration::M1;

    return ChipGeneration::Unknown;
}

std::string ToLowerASCII(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

bool StartsWith(const std::string& s, const char* prefix) {
    return s.rfind(prefix, 0) == 0;
}

ChipGeneration DetectFromTargetType(const std::string& target_type_raw) {
    const std::string target = ToLowerASCII(target_type_raw);

    // M4 family (T813x)
    if (StartsWith(target, "t813")) {
        return ChipGeneration::M4;
    }

    // M3 family (known desktop/mobile codenames)
    if (StartsWith(target, "j514") || StartsWith(target, "j515") || StartsWith(target, "j516")) {
        return ChipGeneration::M3;
    }

    // M2 family
    if (StartsWith(target, "j413") || StartsWith(target, "j473") || StartsWith(target, "j474") ||
        StartsWith(target, "j475")) {
        return ChipGeneration::M2;
    }

    // M1 family
    if (StartsWith(target, "j274") || StartsWith(target, "j293") || StartsWith(target, "j313") ||
        StartsWith(target, "j273") || StartsWith(target, "j375")) {
        return ChipGeneration::M1;
    }

    return ChipGeneration::Unknown;
}

bool ParseEnvBool(const char* name, bool default_value) {
    return densecore::env::ParseBoolEnv(name, default_value);
}

int ParseEnvInt(const char* name, int default_value) {
    return densecore::env::ParsePositiveEnvInt(name, default_value);
}

bool EnableAMXInt4Path() {
    static const bool enabled = ParseEnvBool("DENSECORE_APPLE_AMX_INT4", true);
    return enabled;
}

int GetAMXInt4TileN() {
    static const int tile_n = std::max(16, ParseEnvInt("DENSECORE_APPLE_AMX_INT4_TILE_N", 128));
    return tile_n;
}

int GetAMXInt4MinSliceN() {
    static const int min_slice_n =
        std::max(16, ParseEnvInt("DENSECORE_APPLE_AMX_INT4_MIN_SLICE_N", 64));
    return min_slice_n;
}

int GetAMXInt4MinK() {
    static const int min_k = std::max(32, ParseEnvInt("DENSECORE_APPLE_AMX_INT4_MIN_K", 128));
    return min_k;
}

void DequantizeInt4TileRows(float* dst, const uint8_t* weights, const float* scales,
                            const float* zero_points, int rows, int K, int group_size) {
    const int packed_k = K / 2;
    const int groups_per_row = K / group_size;
    for (int row = 0; row < rows; ++row) {
        const uint8_t* w_row = weights + static_cast<size_t>(row) * packed_k;
        const float* s_row = scales + static_cast<size_t>(row) * groups_per_row;
        const float* z_row =
            zero_points ? zero_points + static_cast<size_t>(row) * groups_per_row : nullptr;
        float* out_row = dst + static_cast<size_t>(row) * K;
        for (int k = 0; k < K; ++k) {
            const uint8_t packed = w_row[k / 2];
            int q = (k & 1) ? int((packed >> 4) & 0x0F) : int(packed & 0x0F);
            if (q > 7) {
                q -= 16;
            }
            const int group_idx = k / group_size;
            const float zero = z_row ? z_row[group_idx] : 0.0f;
            out_row[k] = s_row[group_idx] * (static_cast<float>(q) - zero);
        }
    }
}

bool TryGemmInt4AMXRange(float* C, const float* A, const uint8_t* W_int4, const float* scales,
                         const float* zero_points, int M, int N, int K, int group_size, int n_start,
                         int n_end) {
    if (!C || !A || !W_int4 || !scales || M <= 0 || N <= 0 || K <= 0 || group_size <= 0) {
        return false;
    }
    if (!HasAMX() || !EnableAMXInt4Path()) {
        return false;
    }
    if ((K % group_size) != 0 || K < GetAMXInt4MinK()) {
        return false;
    }

    const int col_start = std::max(0, n_start);
    const int col_end = (n_end < 0) ? N : std::min(N, n_end);
    const int slice_n = col_end - col_start;
    if (slice_n < GetAMXInt4MinSliceN()) {
        return false;
    }

    const int tile_n = GetAMXInt4TileN();
    thread_local std::vector<float> weight_tile;

    static bool warned_once = false;
    if (!warned_once) {
        std::cout << "[AppleSilicon] INT4 GEMM using AMX-oriented tiled dequant + Accelerate path"
                  << std::endl;
        warned_once = true;
    }

    for (int col = col_start; col < col_end; col += tile_n) {
        const int cols_this_tile = std::min(tile_n, col_end - col);
        weight_tile.resize(static_cast<size_t>(cols_this_tile) * static_cast<size_t>(K));

        const int packed_k = K / 2;
        const int groups_per_row = K / group_size;
        const uint8_t* w_tile = W_int4 + static_cast<size_t>(col) * packed_k;
        const float* s_tile = scales + static_cast<size_t>(col) * groups_per_row;
        const float* z_tile =
            zero_points ? zero_points + static_cast<size_t>(col) * groups_per_row : nullptr;
        DequantizeInt4TileRows(weight_tile.data(), w_tile, s_tile, z_tile, cols_this_tile, K,
                               group_size);

        if (M == 1) {
            cblas_sgemv(CblasRowMajor, CblasNoTrans, cols_this_tile, K, 1.0f, weight_tile.data(), K,
                        A, 1, 0.0f, C + col, 1);
        } else {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, cols_this_tile, K, 1.0f, A, K,
                        weight_tile.data(), K, 0.0f, C + col, N);
        }
    }

    return true;
}

}  // anonymous namespace

// ============================================================================
// Chip Generation Detection
// ============================================================================

ChipGeneration DetectChipGeneration() {
    if (g_chip_detected) {
        return g_cached_chip;
    }

    g_chip_detected = true;

    // First, check if we're on Apple Silicon
    if (!IsAppleSilicon()) {
        g_cached_chip = ChipGeneration::Unknown;
        return g_cached_chip;
    }

    // Try to get brand string
    std::string brand = GetSysctlString("machdep.cpu.brand_string");
    if (!brand.empty()) {
        g_cached_chip = DetectFromBrandString(brand);
        if (g_cached_chip != ChipGeneration::Unknown) {
            return g_cached_chip;
        }
    }

    // Fallback: Use IOKit target-type when brand string is unavailable.
    @autoreleasepool {
        io_registry_entry_t entry =
            IORegistryEntryFromPath(kIOMasterPortDefault, "IOService:/AppleARMPE");

        if (entry != MACH_PORT_NULL) {
            CFTypeRef property = IORegistryEntryCreateCFProperty(entry, CFSTR("target-type"),
                                                                 kCFAllocatorDefault, 0);

            if (property != nullptr) {
                if (CFGetTypeID(property) == CFDataGetTypeID()) {
                    CFDataRef data = (CFDataRef)property;
                    const char* str = (const char*)CFDataGetBytePtr(data);
                    std::string target(str, CFDataGetLength(data));
                    g_cached_chip = DetectFromTargetType(target);
                } else if (CFGetTypeID(property) == CFStringGetTypeID()) {
                    const CFStringRef cfstr = static_cast<CFStringRef>(property);
                    char buffer[128] = {0};
                    if (CFStringGetCString(cfstr, buffer, sizeof(buffer), kCFStringEncodingUTF8)) {
                        g_cached_chip = DetectFromTargetType(buffer);
                    }
                }
                CFRelease(property);
            }
            IOObjectRelease(entry);
        }
    }

    return g_cached_chip;
}

const char* ChipGenerationName(ChipGeneration gen) {
    switch (gen) {
    case ChipGeneration::Unknown:
        return "Unknown";
    case ChipGeneration::M1:
        return "M1";
    case ChipGeneration::M1_Pro:
        return "M1 Pro";
    case ChipGeneration::M1_Max:
        return "M1 Max";
    case ChipGeneration::M1_Ultra:
        return "M1 Ultra";
    case ChipGeneration::M2:
        return "M2";
    case ChipGeneration::M2_Pro:
        return "M2 Pro";
    case ChipGeneration::M2_Max:
        return "M2 Max";
    case ChipGeneration::M2_Ultra:
        return "M2 Ultra";
    case ChipGeneration::M3:
        return "M3";
    case ChipGeneration::M3_Pro:
        return "M3 Pro";
    case ChipGeneration::M3_Max:
        return "M3 Max";
    case ChipGeneration::M4:
        return "M4";
    case ChipGeneration::M4_Pro:
        return "M4 Pro";
    case ChipGeneration::M4_Max:
        return "M4 Max";
    default:
        return "Unknown";
    }
}

// ============================================================================
// Core Topology
// ============================================================================

int GetPerformanceCoreCount() {
    int64_t count = GetSysctlInt("hw.perflevel0.physicalcpu");
    if (count <= 0) {
        count = GetSysctlInt("hw.perflevel0.logicalcpu");
    }
    if (count > 0) {
        return static_cast<int>(count);
    }

    // Fallback: estimate from total physical/logical cores.
    int64_t total = GetSysctlInt("hw.physicalcpu");
    if (total <= 0) {
        total = GetSysctlInt("hw.ncpu");
    }
    if (total > 0) {
        return static_cast<int>(std::max<int64_t>(1, total / 2));
    }

    return 4;  // Conservative default
}

int GetEfficiencyCoreCount() {
    int64_t count = GetSysctlInt("hw.perflevel1.physicalcpu");
    if (count <= 0) {
        count = GetSysctlInt("hw.perflevel1.logicalcpu");
    }
    if (count > 0) {
        return static_cast<int>(count);
    }

    // Fallback: derive from total minus performance cores.
    int64_t total = GetSysctlInt("hw.physicalcpu");
    if (total <= 0) {
        total = GetSysctlInt("hw.ncpu");
    }
    int p_cores = GetPerformanceCoreCount();
    if (total > p_cores) {
        return static_cast<int>(total - p_cores);
    }

    return 4;  // Conservative default
}

int GetOptimalComputeThreadCount() {
    // For LLM inference, P-cores only gives best latency
    return GetPerformanceCoreCount();
}

bool PinToPerformanceCores() {
    // macOS doesn't have strict CPU affinity, but we can set QoS hints
    @autoreleasepool {
        // Set high priority QoS class for current thread
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
        return true;
    }
}

bool PinToEfficiencyCores() {
    @autoreleasepool {
        // Set background QoS class to prefer E-cores
        pthread_set_qos_class_self_np(QOS_CLASS_BACKGROUND, 0);
        return true;
    }
}

// ============================================================================
// AMX (Apple Matrix Extensions)
// ============================================================================

bool HasAMX() {
    // All Apple Silicon chips expose AMX to system BLAS libraries such as
    // Accelerate. DenseCore routes larger INT4 tiles through an AMX-oriented
    // dequantize-and-GEMM path, with NEON retained as the small-range fallback.
    return IsAppleSilicon();
}

int GetAMXBlockSize() {
    // AMX operates on 32x32 or 16x16 blocks depending on data type
    // For FP32, it's typically 16x16
    return 16;
}

void GemvAccelerate(float* output, const float* input, const float* weight, int M, int K) {
    // Use BLAS sgemv: y = alpha * A * x + beta * y
    // A is M x K (row-major), x is K, y is M
    cblas_sgemv(CblasRowMajor, CblasNoTrans, M, K,  // Matrix dimensions
                1.0f,                               // alpha
                weight, K,                          // A and leading dimension
                input, 1,                           // x and incX
                0.0f,                               // beta
                output, 1);                         // y and incY
}

void GemmAccelerate(float* C, const float* A, const float* B, int M, int N, int K) {
    // Use BLAS sgemm: C = alpha * A * B + beta * C
    // A is M x K, B is K x N, C is M x N (all row-major)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N,
                K,      // Matrix dimensions
                1.0f,   // alpha
                A, K,   // A and leading dimension
                B, N,   // B and leading dimension
                0.0f,   // beta
                C, N);  // C and leading dimension
}

bool HasCustomInt4Kernels() {
    // DenseCore now exposes two Apple CPU INT4 paths:
    // 1. AMX-oriented tiled dequantization + Accelerate GEMM/GEMV
    // 2. NEON fallback for small or unsupported ranges
    return IsAppleSilicon();
}

void GemmInt4CustomRange(float* C, const float* A, const uint8_t* W_int4, const float* scales,
                         const float* zero_points, int M, int N, int K, int group_size, int n_start,
                         int n_end) {
    if (!HasCustomInt4Kernels() || !C || !A || !W_int4 || !scales)
        return;
    if (group_size <= 0 || (K % group_size) != 0)
        return;

    if (TryGemmInt4AMXRange(C, A, W_int4, scales, zero_points, M, N, K, group_size, n_start,
                            n_end)) {
        return;
    }

    const int col_start = std::max(0, n_start);
    const int col_end = (n_end < 0) ? N : std::min(N, n_end);
    if (col_start >= col_end)
        return;

    const int slice_n = col_end - col_start;
    const int packed_k = K / 2;
    const int groups_per_row = K / group_size;
    const uint8_t* w_slice = W_int4 + static_cast<size_t>(col_start) * packed_k;
    const float* scale_slice = scales + static_cast<size_t>(col_start) * groups_per_row;
    thread_local std::vector<float> zero_fallback;
    if (!zero_points) {
        zero_fallback.assign(static_cast<size_t>(slice_n) * static_cast<size_t>(groups_per_row),
                             0.0f);
    }
    const float* zero_slice = zero_points
                                  ? (zero_points + static_cast<size_t>(col_start) * groups_per_row)
                                  : zero_fallback.data();

    for (int m = 0; m < M; ++m) {
        simd::GemmInt4Fp32_NEON(C + static_cast<size_t>(m) * N + col_start,
                                A + static_cast<size_t>(m) * K, w_slice, scale_slice, zero_slice, 1,
                                slice_n, K, group_size);
    }
}

// ============================================================================
// Memory Information
// ============================================================================

MemoryInfo GetMemoryInfo() {
    MemoryInfo info = {0, 0, 0.0f};

    // Get total physical memory
    info.total_bytes = static_cast<uint64_t>(GetSysctlInt("hw.memsize"));

    // Get available memory via mach API
    vm_size_t page_size;
    mach_port_t mach_port = mach_host_self();
    vm_statistics64_data_t vm_stats;
    mach_msg_type_number_t count = sizeof(vm_stats) / sizeof(natural_t);

    if (host_page_size(mach_port, &page_size) == KERN_SUCCESS &&
        host_statistics64(mach_port, HOST_VM_INFO64, (host_info64_t)&vm_stats, &count) ==
            KERN_SUCCESS) {
        info.available_bytes = static_cast<uint64_t>(vm_stats.free_count) * page_size;
    }

    // Get bandwidth based on chip
    info.bandwidth_gbps = GetMemoryBandwidth(DetectChipGeneration());

    return info;
}

float GetMemoryBandwidth(ChipGeneration gen) {
    switch (gen) {
    case ChipGeneration::M1:
        return 68.25f;
    case ChipGeneration::M1_Pro:
        return 200.0f;
    case ChipGeneration::M1_Max:
        return 400.0f;
    case ChipGeneration::M1_Ultra:
        return 800.0f;

    case ChipGeneration::M2:
        return 100.0f;
    case ChipGeneration::M2_Pro:
        return 200.0f;
    case ChipGeneration::M2_Max:
        return 400.0f;
    case ChipGeneration::M2_Ultra:
        return 800.0f;

    case ChipGeneration::M3:
        return 100.0f;
    case ChipGeneration::M3_Pro:
        return 150.0f;
    case ChipGeneration::M3_Max:
        return 400.0f;

    case ChipGeneration::M4:
        return 120.0f;
    case ChipGeneration::M4_Pro:
        return 273.0f;  // Estimated
    case ChipGeneration::M4_Max:
        return 546.0f;  // Estimated

    default:
        return 100.0f;  // Conservative estimate
    }
}

// ============================================================================
// Neural Engine
// ============================================================================

NeuralEngineInfo GetNeuralEngineInfo() {
    NeuralEngineInfo info = {false, 0, 16, true, true};

    if (!IsAppleSilicon()) {
        return info;
    }

    info.available = true;
    info.tops = GetNeuralEngineTOPS();

    return info;
}

bool IsNeuralEngineAvailable() {
    return IsAppleSilicon();
}

int GetNeuralEngineTOPS() {
    ChipGeneration gen = DetectChipGeneration();

    switch (gen) {
    case ChipGeneration::M1:
    case ChipGeneration::M1_Pro:
    case ChipGeneration::M1_Max:
        return 11;
    case ChipGeneration::M1_Ultra:
        return 22;

    case ChipGeneration::M2:
    case ChipGeneration::M2_Pro:
    case ChipGeneration::M2_Max:
        return 15;
    case ChipGeneration::M2_Ultra:
        return 31;

    case ChipGeneration::M3:
    case ChipGeneration::M3_Pro:
    case ChipGeneration::M3_Max:
        return 18;

    case ChipGeneration::M4:
    case ChipGeneration::M4_Pro:
    case ChipGeneration::M4_Max:
        return 38;

    default:
        return 11;  // Conservative M1 baseline
    }
}

// ============================================================================
// Thermal and Power
// ============================================================================

ThermalState GetThermalState() {
    @autoreleasepool {
        NSProcessInfoThermalState state = [[NSProcessInfo processInfo] thermalState];
        switch (state) {
        case NSProcessInfoThermalStateNominal:
            return ThermalState::Nominal;
        case NSProcessInfoThermalStateFair:
            return ThermalState::Fair;
        case NSProcessInfoThermalStateSerious:
            return ThermalState::Serious;
        case NSProcessInfoThermalStateCritical:
            return ThermalState::Critical;
        default:
            return ThermalState::Nominal;
        }
    }
}

PowerMode GetPowerMode() {
    @autoreleasepool {
        if ([[NSProcessInfo processInfo] isLowPowerModeEnabled]) {
            return PowerMode::LowPower;
        }
        return PowerMode::Automatic;
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

bool IsAppleSilicon() {
    // Check if we're running native ARM64
#if defined(__arm64__) || defined(__aarch64__)
    // Check for Rosetta translation
    if (IsRunningRosetta()) {
        return true;  // Still Apple Silicon, just translated
    }
    return true;
#else
    // Running as x86_64 - check if under Rosetta
    return IsRunningRosetta();
#endif
}

bool IsRunningRosetta() {
    int ret = 0;
    size_t size = sizeof(ret);
    if (sysctlbyname("sysctl.proc_translated", &ret, &size, nullptr, 0) == 0) {
        return ret == 1;
    }
    return false;
}

int GetMacOSVersion() {
    @autoreleasepool {
        NSOperatingSystemVersion version = [[NSProcessInfo processInfo] operatingSystemVersion];
        return static_cast<int>(version.majorVersion * 10000 + version.minorVersion * 100 +
                                version.patchVersion);
    }
}

bool MacOSVersionAtLeast(int major, int minor) {
    @autoreleasepool {
        NSOperatingSystemVersion required = {major, minor, 0};
        return [[NSProcessInfo processInfo] isOperatingSystemAtLeastVersion:required];
    }
}

}  // namespace apple
}  // namespace densecore

#endif  // __APPLE__
