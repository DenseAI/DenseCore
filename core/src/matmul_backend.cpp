/**
 * @file matmul_backend.cpp
 * @brief Matmul backend selection and configuration
 */

#include "../include/matmul_backend.h"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <string>

#include "../include/inference.h"
#include "../include/simd_ops.h"

namespace densecore {

namespace {

bool ParseEnvBool(const char* value, bool default_value) {
    if (!value) return default_value;
    std::string v(value);
    if (v == "1" || v == "true" || v == "TRUE" || v == "on" || v == "ON") return true;
    if (v == "0" || v == "false" || v == "FALSE" || v == "off" || v == "OFF") return false;
    return default_value;
}

int ParseEnvInt(const char* value, int default_value) {
    if (!value) return default_value;
    char* end = nullptr;
    long v = std::strtol(value, &end, 10);
    if (!end || end == value) return default_value;
    if (v < 0) return default_value;
    return static_cast<int>(v);
}

int64_t ParseEnvInt64(const char* value, int64_t default_value) {
    if (!value) return default_value;
    char* end = nullptr;
    long long v = std::strtoll(value, &end, 10);
    if (!end || end == value) return default_value;
    if (v < 0) return default_value;
    return static_cast<int64_t>(v);
}

}  // namespace

MatmulConfig GetMatmulConfig() {
    MatmulConfig cfg;
    const auto& icfg = InferenceConfig::Instance();

    cfg.enable_onednn = icfg.enable_onednn;
    cfg.onednn_mode = 0;
    cfg.pack_m_hint = icfg.onednn_pack_m;
    cfg.heuristics.min_m = icfg.onednn_min_m;
    cfg.heuristics.min_n = icfg.onednn_min_n;
    cfg.heuristics.min_k = icfg.onednn_min_k;
    cfg.heuristics.min_mnk = icfg.onednn_min_mnk;

    const char* env_onednn = std::getenv("DENSECORE_ONEDNN");
    if (env_onednn) {
        cfg.enable_onednn = ParseEnvBool(env_onednn, cfg.enable_onednn);
    }
    const char* env_force = std::getenv("FORCE_ONEDNN");
    if (env_force) {
        if (ParseEnvBool(env_force, false)) {
            cfg.onednn_mode = 1;
            cfg.enable_onednn = true;
        } else {
            cfg.onednn_mode = -1;
            cfg.enable_onednn = false;
        }
    }

    cfg.pack_m_hint = ParseEnvInt(std::getenv("DENSECORE_ONEDNN_PACK_M"), cfg.pack_m_hint);
    cfg.heuristics.min_m = ParseEnvInt(std::getenv("DENSECORE_ONEDNN_MIN_M"), cfg.heuristics.min_m);
    cfg.heuristics.min_n = ParseEnvInt(std::getenv("DENSECORE_ONEDNN_MIN_N"), cfg.heuristics.min_n);
    cfg.heuristics.min_k = ParseEnvInt(std::getenv("DENSECORE_ONEDNN_MIN_K"), cfg.heuristics.min_k);
    cfg.heuristics.min_mnk = ParseEnvInt64(std::getenv("DENSECORE_ONEDNN_MIN_MNK"), cfg.heuristics.min_mnk);

    return cfg;
}

MatmulBackendKind SelectMatmulBackend(const MatmulParams& params, bool is_prefill) {
    const MatmulConfig cfg = GetMatmulConfig();
    static std::atomic<bool> logged_force_fallback{false};
    if (cfg.onednn_mode < 0) {
        return MatmulBackendKind::DenseCore;
    }

    if (!cfg.enable_onednn || !is_prefill || params.M <= 1) {
        return MatmulBackendKind::DenseCore;
    }

    if (cfg.onednn_mode == 0 &&
        (params.M < cfg.heuristics.min_m || params.N < cfg.heuristics.min_n || params.K < cfg.heuristics.min_k)) {
        return MatmulBackendKind::DenseCore;
    }

    const int64_t mnk = params.M * params.N * params.K;
    if (cfg.onednn_mode == 0 && mnk < cfg.heuristics.min_mnk) {
        return MatmulBackendKind::DenseCore;
    }

    MatmulBackend& one = GetOneDnnMatmulBackend();
    if (cfg.onednn_mode == 0) {
        if (!one.IsAvailable() || !one.Supports(params)) {
            return MatmulBackendKind::DenseCore;
        }
    } else {
        if (!one.Supports(params)) {
            if (!logged_force_fallback.exchange(true)) {
                std::cout << "[oneDNN] FORCE_ONEDNN enabled but params unsupported; falling back to DenseCore\n";
            }
            return MatmulBackendKind::DenseCore;
        }
    }

    return MatmulBackendKind::OneDNN;
}

void PrepareMatmulWeights(const void* weight, int64_t K, int64_t N, DType dtype, const char* name) {
    const MatmulConfig cfg = GetMatmulConfig();
    if (!cfg.enable_onednn) return;

    MatmulParams params;
    if (dtype == DType::BF16) {
        params.a_type = DType::BF16;
    } else if (dtype == DType::INT8) {
        params.a_type = DType::INT8;
    } else {
        params.a_type = dtype;
    }
    params.b_type = dtype;
    params.c_type = DType::F32;
    params.M = cfg.pack_m_hint;
    params.K = K;
    params.N = N;
    params.ldb = K;
    params.trans_b = true;
    params.b = weight;

    MatmulBackend& one = GetOneDnnMatmulBackend();
    if (!one.IsAvailable() || !one.Supports(params)) return;
    one.PrepareWeights(params, name);
}

}  // namespace densecore
