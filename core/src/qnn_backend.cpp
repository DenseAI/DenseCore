/**
 * @file qnn_backend.cpp
 * @brief Qualcomm QNN Backend Implementation
 *
 * Implements the QnnBackend using the Qualcomm AI Stack (QNN/HTP).
 * Note: This is an architectural reference implementation. Real QNN SDK
 * requires linking against libQnnHtp.so and including generic QNN headers.
 */

#include "../include/qnn_backend.h"
#include "../include/cpu_backend.h"

#include <array>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#if defined(_WIN32)
#include <malloc.h>
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace densecore {
namespace {

constexpr const char* kQnnForceDisableEnv = "DENSECORE_QNN_FORCE_DISABLE";
constexpr const char* kQnnForceEnableEnv = "DENSECORE_QNN_FORCE_ENABLE";
constexpr const char* kQnnLibPathEnv = "DENSECORE_QNN_LIB_PATH";
constexpr const char* kQnnExplicitLibEnv = "DENSECORE_QNN_LIB";
constexpr const char* kQnnAllowUnsupportedArchEnv = "DENSECORE_QNN_ALLOW_UNSUPPORTED_ARCH";

inline bool IsTruthyEnvValue(const char* value) {
    if (!value) {
        return false;
    }
    std::string normalized;
    for (const char* p = value; *p != '\0'; ++p) {
        if (!std::isspace(static_cast<unsigned char>(*p))) {
            normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(*p))));
        }
    }
    return normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on";
}

inline bool EnvFlagEnabled(const char* name) {
    return IsTruthyEnvValue(std::getenv(name));
}

inline std::string GetEnvString(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string();
}

inline bool IsArm64Target() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return true;
#else
    return false;
#endif
}

inline bool IsSupportedQnnPlatform() {
#if defined(__linux__) || defined(__ANDROID__) || defined(_WIN32)
    return true;
#else
    return false;
#endif
}

#if defined(_WIN32)
using DynamicLibHandle = HMODULE;
#else
using DynamicLibHandle = void*;
#endif

DynamicLibHandle OpenDynamicLibrary(const std::string& path) {
#if defined(_WIN32)
    return LoadLibraryA(path.c_str());
#else
    return dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

void CloseDynamicLibrary(DynamicLibHandle handle) {
    if (!handle) {
        return;
    }
#if defined(_WIN32)
    FreeLibrary(handle);
#else
    dlclose(handle);
#endif
}

inline char PathListSeparator() {
#if defined(_WIN32)
    return ';';
#else
    return ':';
#endif
}

std::vector<std::string> SplitPathList(const std::string& path_list) {
    std::vector<std::string> result;
    if (path_list.empty()) {
        return result;
    }

    const char sep = PathListSeparator();
    size_t start = 0;
    while (start < path_list.size()) {
        size_t end = path_list.find(sep, start);
        if (end == std::string::npos) {
            end = path_list.size();
        }
        if (end > start) {
            result.emplace_back(path_list.substr(start, end - start));
        }
        start = end + 1;
    }
    return result;
}

void PushUnique(std::vector<std::string>* out, std::unordered_set<std::string>* seen, const std::string& value) {
    if (!out || !seen || value.empty()) {
        return;
    }
    if (seen->insert(value).second) {
        out->push_back(value);
    }
}

std::string JoinPath(const std::string& dir, const std::string& file) {
    if (dir.empty()) {
        return file;
    }
    if (file.empty()) {
        return dir;
    }
#if defined(_WIN32)
    const char preferred_sep = '\\';
    const bool ends_with_sep = dir.back() == '\\' || dir.back() == '/';
#else
    const char preferred_sep = '/';
    const bool ends_with_sep = dir.back() == '/';
#endif
    if (ends_with_sep) {
        return dir + file;
    }
    return dir + preferred_sep + file;
}

const std::vector<std::string>& CandidateQnnLibraryNames() {
#if defined(_WIN32)
    static const std::vector<std::string> kNames = {
        "QnnHtp.dll", "QnnHtpPrepare.dll", "QnnHtpV69Stub.dll", "QnnHtpV73Stub.dll", "QnnHtpV75Stub.dll",
    };
#else
    static const std::vector<std::string> kNames = {
        "libQnnHtp.so",        "libQnnHtp.so.1",      "libQnnHtp.so.2",
        "libQnnHtpPrepare.so", "libQnnHtpV68Stub.so", "libQnnHtpV69Stub.so",
        "libQnnHtpV70Stub.so", "libQnnHtpV73Stub.so", "libQnnHtpV75Stub.so",
    };
#endif
    return kNames;
}

std::vector<std::string> CandidateQnnLibraryPaths() {
    std::vector<std::string> candidates;
    std::unordered_set<std::string> seen;

    const std::string explicit_library = GetEnvString(kQnnExplicitLibEnv);
    if (!explicit_library.empty()) {
        PushUnique(&candidates, &seen, explicit_library);
    }

    for (const std::string& name : CandidateQnnLibraryNames()) {
        PushUnique(&candidates, &seen, name);
    }

    std::vector<std::string> search_dirs;
    std::unordered_set<std::string> seen_dirs;
    auto append_dirs_from_env = [&](const char* env_name) {
        for (const std::string& entry : SplitPathList(GetEnvString(env_name))) {
            if (!entry.empty() && seen_dirs.insert(entry).second) {
                search_dirs.push_back(entry);
            }
        }
    };

    append_dirs_from_env(kQnnLibPathEnv);
#if defined(_WIN32)
    append_dirs_from_env("PATH");
#else
    append_dirs_from_env("LD_LIBRARY_PATH");
    static constexpr std::array<const char*, 8> kDefaultDirs = {
        "/usr/lib",    "/usr/lib64",    "/usr/local/lib", "/usr/local/lib64",
        "/vendor/lib", "/vendor/lib64", "/system/lib",    "/system/lib64",
    };
    for (const char* dir : kDefaultDirs) {
        if (seen_dirs.insert(dir).second) {
            search_dirs.emplace_back(dir);
        }
    }
#endif

    for (const std::string& dir : search_dirs) {
        for (const std::string& name : CandidateQnnLibraryNames()) {
            PushUnique(&candidates, &seen, JoinPath(dir, name));
        }
    }

    return candidates;
}

struct QnnRuntimeProbeResult {
    bool available = false;
    bool forced = false;
    bool runtime_library_loaded = false;
    std::string loaded_library_path;
    std::string reason;
};

QnnRuntimeProbeResult ProbeQnnRuntime() {
    QnnRuntimeProbeResult result;

    if (EnvFlagEnabled(kQnnForceDisableEnv)) {
        result.reason = std::string(kQnnForceDisableEnv) + "=1";
        return result;
    }

    const bool force_enable = EnvFlagEnabled(kQnnForceEnableEnv);
    if (!force_enable) {
        if (!IsSupportedQnnPlatform()) {
            result.reason = "unsupported platform";
            return result;
        }
        if (!IsArm64Target() && !EnvFlagEnabled(kQnnAllowUnsupportedArchEnv)) {
            result.reason = "unsupported architecture (set DENSECORE_QNN_ALLOW_UNSUPPORTED_ARCH=1 to override)";
            return result;
        }
    }

    const std::vector<std::string> candidates = CandidateQnnLibraryPaths();
    for (const std::string& path : candidates) {
        DynamicLibHandle handle = OpenDynamicLibrary(path);
        if (!handle) {
            continue;
        }
        CloseDynamicLibrary(handle);

        result.available = true;
        result.runtime_library_loaded = true;
        result.loaded_library_path = path;
        result.reason = "runtime library detected";
        return result;
    }

    if (force_enable) {
        result.available = true;
        result.forced = true;
        result.reason = std::string(kQnnForceEnableEnv) + "=1 (runtime library not resolved)";
        return result;
    }

    result.reason = "QNN HTP runtime library not found";
    return result;
}

inline bool IsF32Tensor(const Tensor& t) {
    return t.IsValid() && t.dtype == DType::F32;
}

}  // namespace

// ============================================================================
// Private Implementation (PImpl)
// ============================================================================

struct QnnBackend::Impl {
    bool initialized = false;
    bool runtime_library_loaded = false;
    bool forced_availability = false;
    std::string runtime_library_path;
    std::string availability_reason;
    // Qnn_BackendHandle_t backendHandle = nullptr;
    // Qnn_DeviceHandle_t deviceHandle = nullptr;
    // Qnn_ContextHandle_t contextHandle = nullptr;

    Impl() {
        const QnnRuntimeProbeResult probe = ProbeQnnRuntime();
        initialized = probe.available;
        forced_availability = probe.forced;
        runtime_library_loaded = probe.runtime_library_loaded;
        runtime_library_path = probe.loaded_library_path;
        availability_reason = probe.reason;

        if (!initialized) {
            std::cerr << "[QnnBackend] Runtime unavailable: " << availability_reason << std::endl;
            return;
        }

        if (runtime_library_loaded) {
            std::cout << "[QnnBackend] Initializing Qualcomm Hexagon NPU Backend (runtime=" << runtime_library_path
                      << ")" << std::endl;
        } else {
            std::cout << "[QnnBackend] Initializing in forced mode without resolved QNN runtime library" << std::endl;
        }
    }

    ~Impl() {
        if (initialized) {
            std::cout << "[QnnBackend] Shutting down NPU Backend..." << std::endl;
            // QnnContext_free(contextHandle);
            // QnnBackend_free(backendHandle);
        }
    }
};

// ============================================================================
// Public Interface
// ============================================================================

QnnBackend::QnnBackend() : impl_(std::make_unique<Impl>()) {}

QnnBackend::~QnnBackend() = default;

BackendCapabilityManifest QnnBackend::GetCapabilityManifest() const {
    BackendCapabilityManifest manifest;

    // Current implementation routes compute through optimized CPU fallback
    // while QNN runtime probing and backend admission are handled natively.
    manifest.fallback_ops = {
        OpType::Embedding,      OpType::MatMul,     OpType::MatMulTransB, OpType::GemmInt4,
        OpType::RMSNorm,        OpType::AddRMSNorm, OpType::LayerNorm,    OpType::Softmax,
        OpType::SiLU,           OpType::GELU,       OpType::RoPE,         OpType::FusedQKVProjection,
        OpType::FlashAttention,
    };

    manifest.allow_cpu_fallback = true;
    manifest.declared_complete = false;
    return manifest;
}

bool QnnBackend::CheckAvailability() {
    return ProbeQnnRuntime().available;
}

void* QnnBackend::AllocateDevice(size_t size_bytes, size_t alignment) {
    if (size_bytes == 0) {
        return nullptr;
    }
    if (alignment == 0) alignment = 4096;  // QNN usually likes page alignment
    // Use posix_memalign or NPU-aware allocator (ION/DMA-BUF)
    void* ptr = nullptr;
#if defined(_WIN32)
    ptr = _aligned_malloc(size_bytes, alignment);
#else
    if (posix_memalign(&ptr, alignment, size_bytes) != 0) {
        return nullptr;
    }
#endif
    // Ideally register this memory with QnnMem_register()
    return ptr;
}

void QnnBackend::FreeDevice(void* ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
    // QnnMem_deRegister()
}

void QnnBackend::CopyToDevice(void* dst, const void* src, size_t size_bytes) {
    // For UMA, this is just memcpy usually, unless dealing with non-shared device memory
    std::memcpy(dst, src, size_bytes);
}

void QnnBackend::CopyFromDevice(void* dst, const void* src, size_t size_bytes) {
    std::memcpy(dst, src, size_bytes);
}

void QnnBackend::BeginCapture() {
    ComputeBackend::BeginCapture();
    // QnnGraph_create(...)
}

std::unique_ptr<OperationGraph> QnnBackend::EndCapture() {
    // QnnGraph_finalize(...)
    return ComputeBackend::EndCapture();
}

void QnnBackend::ExecuteGraph(const OperationGraph& graph) {
    // QnnGraph_execute(...)
    // Mapping inputs/outputs using tensor IDs locally
    if (graph.Nodes().empty()) return;
}

// ============================================================================
// Operation Fallbacks
// ============================================================================

void QnnBackend::MatMul(const Tensor& A, const Tensor& B, Tensor* C) {
    if (capturing_) {
        // QnnGraph_addNode(..., QNN_OP_MAT_MUL, ...)
        return;
    }
    if (!IsF32Tensor(A) || !IsF32Tensor(B) || !C || !IsF32Tensor(*C) || A.ndim != 2 || B.ndim != 2) {
        return;
    }

    const int64_t M = A.shape[0];
    const int64_t K = A.shape[1];
    const int64_t Kb = B.shape[0];
    const int64_t N = B.shape[1];
    if (M <= 0 || K <= 0 || N <= 0 || K != Kb) {
        return;
    }

    // Delegate fallback execution to optimized CPU backend kernels (BLAS/SIMD).
    GetCpuBackend().MatMul(A, B, C);
}

void QnnBackend::MatMulTransB(const Tensor& A, const Tensor& B, Tensor* C) {
    if (capturing_) {
        // QnnGraph_addNode(..., QNN_OP_MAT_MUL, transpose_b=true)
        return;
    }
    if (!IsF32Tensor(A) || !IsF32Tensor(B) || !C || !IsF32Tensor(*C) || A.ndim != 2 || B.ndim != 2) {
        return;
    }

    const int64_t M = A.shape[0];
    const int64_t K = A.shape[1];
    const int64_t N = B.shape[0];
    if (M <= 0 || K <= 0 || N <= 0 || B.shape[1] != K) {
        return;
    }

    // Delegate fallback execution to optimized CPU backend kernels (BLAS/SIMD).
    GetCpuBackend().MatMulTransB(A, B, C);
}

void QnnBackend::GemmInt4(const Tensor& A, const Tensor& W, const Tensor& scales, const Tensor& zero_points, Tensor* C,
                          int group_size) {
    if (capturing_) {
        return;
    }
    if (!IsF32Tensor(A) || !W.IsValid() || !C || !IsF32Tensor(*C) || A.ndim != 2 || W.ndim != 2 ||
        scales.dtype != DType::F32) {
        return;
    }

    const int64_t M = A.shape[0];
    const int64_t K = A.shape[1];
    const int64_t N = W.shape[0];
    const int64_t K_packed = W.shape[1];
    if (M <= 0 || N <= 0 || K <= 0 || K != K_packed * 2) {
        return;
    }

    if (group_size <= 0) {
        group_size = 32;
    }
    if (K % group_size != 0) {
        return;
    }

    const int64_t groups_per_row = K / group_size;
    if (scales.ndim < 2 || scales.shape[0] != N || scales.shape[1] < groups_per_row) {
        return;
    }
    const bool has_zero_points = zero_points.IsValid() && zero_points.dtype == DType::F32 && zero_points.ndim >= 2 &&
                                 zero_points.shape[0] == N && zero_points.shape[1] >= groups_per_row;

    Tensor effective_zero_points = zero_points;
    std::vector<float> zero_points_buffer;
    if (!has_zero_points) {
        zero_points_buffer.assign(static_cast<size_t>(N * groups_per_row), 0.0f);
        effective_zero_points = Tensor::Make2D(zero_points_buffer.data(), N, groups_per_row, DType::F32);
    }

    // Delegate fallback execution to optimized CPU backend kernels (SIMD/HWY).
    GetCpuBackend().GemmInt4(A, W, scales, effective_zero_points, C, group_size);
}

void QnnBackend::Softmax(const Tensor& input, Tensor* output) {
    if (capturing_) {
        return;
    }
    GetCpuBackend().Softmax(input, output);
}

void QnnBackend::SoftmaxInplace(Tensor* data) {
    if (capturing_) {
        return;
    }
    GetCpuBackend().SoftmaxInplace(data);
}

void QnnBackend::SiLU(const Tensor& input, Tensor* output) {
    if (capturing_) {
        return;
    }
    GetCpuBackend().SiLU(input, output);
}

void QnnBackend::GELU(const Tensor& input, Tensor* output) {
    if (capturing_) {
        return;
    }
    GetCpuBackend().GELU(input, output);
}

void QnnBackend::RMSNorm(const Tensor& input, const Tensor& weight, Tensor* output, float eps) {
    if (capturing_) {
        return;
    }
    GetCpuBackend().RMSNorm(input, weight, output, eps);
}

void QnnBackend::AddRMSNorm(const Tensor& input, const Tensor& residual, const Tensor& weight, Tensor* output,
                            float eps) {
    if (capturing_) {
        return;
    }
    GetCpuBackend().AddRMSNorm(input, residual, weight, output, eps);
}

void QnnBackend::LayerNorm(const Tensor& input, const Tensor& gamma, const Tensor& beta, Tensor* output, float eps) {
    if (capturing_) {
        return;
    }
    GetCpuBackend().LayerNorm(input, gamma, beta, output, eps);
}

void QnnBackend::RoPE(const Tensor& input, const Tensor& cos_sin, const int* positions, Tensor* output, int rope_dim) {
    if (capturing_) {
        return;
    }
    GetCpuBackend().RoPE(input, cos_sin, positions, output, rope_dim);
}

void QnnBackend::FusedQKVProjection(const Tensor& input, const Tensor& wq, const Tensor& wk, const Tensor& wv,
                                    Tensor* q_out, Tensor* k_out, Tensor* v_out) {
    if (capturing_) {
        return;
    }
    MatMulTransB(input, wq, q_out);
    MatMulTransB(input, wk, k_out);
    MatMulTransB(input, wv, v_out);
}

void QnnBackend::FlashAttention(const Tensor& Q, const Tensor& K, const Tensor& V, Tensor* output, float scale,
                                bool causal, int n_head_kv) {
    if (capturing_) {
        return;
    }
    GetCpuBackend().FlashAttention(Q, K, V, output, scale, causal, n_head_kv);
}

void QnnBackend::Synchronize() {
    // QnnGraph_execute is synchronous usually, or wait on event
}

}  // namespace densecore
