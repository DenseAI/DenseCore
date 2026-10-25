#pragma once
#include "llm/speculative/model_identity.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace densecore::llm::speculative::qualification {
inline bool IsDigest(const std::string& text) {
    return text.size() == 64 && text != std::string(64,'0') &&
        std::all_of(text.begin(), text.end(), [](char c) { return (c>='0' && c<='9') || (c>='a' && c<='f'); });
}
inline bool CanonicalIds(const std::vector<int>& values) {
    return !values.empty() && values.front() >= 0 &&
        std::adjacent_find(values.begin(), values.end(), std::greater_equal<int>()) == values.end();
}
struct CpuExecutionProfile {
    std::string cpu_identity; // Canonical vendor/family/model/stepping or ARM identity, never filename.
    // Compiler/build/ISA/kernel-contract fingerprint EXCLUDING registry DATA.
    // A full .so SHA here would become self-referential when adding profiles.
    std::string build_sha256;
    std::string simd;
    std::string runtime_policy_sha256; // All effective kernel/scheduler controls except MTP on/off.
    uint32_t pool_threads=0, prefill_threads=0, decode_threads=0;
    std::vector<int> affinity_cpus, numa_nodes;
    std::string numa_policy, weight_placement;
    bool Complete() const {
        return !cpu_identity.empty() && cpu_identity != "unknown" && IsDigest(build_sha256) &&
            !simd.empty() && simd != "unknown" && IsDigest(runtime_policy_sha256) &&
            pool_threads && prefill_threads && decode_threads && CanonicalIds(affinity_cpus) &&
            CanonicalIds(numa_nodes) && !numa_policy.empty() && !weight_placement.empty();
    }
    bool operator==(const CpuExecutionProfile& p) const {
        return cpu_identity==p.cpu_identity && build_sha256==p.build_sha256 && simd==p.simd &&
            runtime_policy_sha256==p.runtime_policy_sha256 && pool_threads==p.pool_threads &&
            prefill_threads==p.prefill_threads && decode_threads==p.decode_threads &&
            affinity_cpus==p.affinity_cpus && numa_nodes==p.numa_nodes &&
            numa_policy==p.numa_policy && weight_placement==p.weight_placement;
    }
};
enum class PrefixMode { Unspecified, Disabled, KVOnly, RecurrentSnapshot };
struct RuntimeConditions {
    CpuExecutionProfile cpu;
    uint32_t context=0, concurrency=0;
    PrefixMode prefix = PrefixMode::Unspecified;
    std::string kv_dtype;
    bool Complete() const {
        return cpu.Complete() && context && concurrency >= 1 && concurrency <= 8 &&
            (prefix==PrefixMode::Disabled || prefix==PrefixMode::KVOnly || prefix==PrefixMode::RecurrentSnapshot) &&
            !kv_dtype.empty() && kv_dtype!="unknown";
    }
};
struct ValidationEvidence {
    // These SHA references identify curated immutable raw proof artifacts.
    // Runtime never promotes arbitrary files/environment variables into entries.
    std::string quality_sha256, resource_sha256, latency_sha256, throughput_sha256, decode_sha256;
    std::string measured_binary_sha256; // Full binary used by those measurements.
    bool final_answers_complete=false, greedy_parity=false, distribution_parity=false;
    bool rollback_isolation=false, grammar_stream_cancel=false, resource_pass=false;
    bool latency_pass=false, throughput_pass=false, all_applicable_cells_passed=false, aa_calibrated=false;
    uint32_t abba_pairs=0;
    double aa_noise_fraction=0;
    double decode_speedup=0, decode_speedup_lower_bound=0;
    // Worst applicable cell. Do not offset a failed cell with another cell.
    double worst_latency_ratio=0, worst_throughput_ratio=0;
    bool CompleteAndPassed() const {
        if (!IsDigest(quality_sha256) || !IsDigest(resource_sha256) || !IsDigest(latency_sha256) ||
            !IsDigest(throughput_sha256) || !IsDigest(decode_sha256) || !IsDigest(measured_binary_sha256) ||
            !final_answers_complete || !greedy_parity || !distribution_parity || !rollback_isolation ||
            !grammar_stream_cancel || !resource_pass || !latency_pass || !throughput_pass ||
            !all_applicable_cells_passed || !aa_calibrated || abba_pairs < 5 ||
            !std::isfinite(aa_noise_fraction) || aa_noise_fraction < 0 ||
            !std::isfinite(decode_speedup) || !std::isfinite(decode_speedup_lower_bound) ||
            decode_speedup_lower_bound < 1.15 || decode_speedup < decode_speedup_lower_bound ||
            !std::isfinite(worst_latency_ratio) || worst_latency_ratio <= 0 ||
            !std::isfinite(worst_throughput_ratio) || worst_throughput_ratio <= 0) return false;
        const double tolerance = std::max(0.03,aa_noise_fraction);
        return worst_latency_ratio <= 1+tolerance && worst_throughput_ratio >= 1-tolerance;
    }
};
struct QualificationProfile {
    std::string id, model_sha256;
    CpuExecutionProfile cpu;
    uint32_t context=0, concurrency_min=0, concurrency_max=0;
    PrefixMode prefix = PrefixMode::Unspecified;
    std::string kv_dtype;
    int draft_n_max=0;
    ValidationEvidence evidence;
    bool Eligible(const RuntimeConditions& runtime, int requested_upper) const {
        return !id.empty() && IsDigest(model_sha256) && cpu.Complete() && cpu==runtime.cpu &&
            context==runtime.context && concurrency_min>=1 && concurrency_max<=8 &&
            concurrency_min<=concurrency_max && runtime.concurrency>=concurrency_min &&
            runtime.concurrency<=concurrency_max && prefix==runtime.prefix && kv_dtype==runtime.kv_dtype &&
            draft_n_max>=1 && draft_n_max<=3 && draft_n_max<=requested_upper && evidence.CompleteAndPassed();
    }
};
// Curated profiles can be added after complete qualification. None are shipped
// yet. Adding an entry changes DATA, not the build/kernel-contract fingerprint.
inline const std::vector<QualificationProfile>& ProductionProfiles() {
    static const std::vector<QualificationProfile> profiles{
#include "qualification_registry_data.inl"
    };
    return profiles;
}
enum class SelectionMode { None, MTP };
struct Selection {
    SelectionMode mode = SelectionMode::None;
    int draft_n_max = 0;
    std::string profile_id;
    std::string reason;
};
struct CostFit { bool fits=false; std::string reason; };
using CostEvaluator = std::function<CostFit(int)>;

inline Selection SelectAutomatic(const std::vector<QualificationProfile>& profiles,
                                  const RuntimeConditions& runtime, int requested_upper,
                                  LazyModelIdentity& identity, const CostEvaluator& cost_fit) {
    const auto none = [](std::string reason) { return Selection{SelectionMode::None,0,{},std::move(reason)}; };
    if (profiles.empty()) return none("no qualified automatic MTP profiles");
    if (!runtime.Complete() || requested_upper<1 || requested_upper>3) return none("incomplete automatic MTP execution profile");
    std::vector<const QualificationProfile*> candidates;
    for (const auto& profile : profiles) if (profile.Eligible(runtime,requested_upper)) candidates.push_back(&profile);
    if (candidates.empty()) return none("no qualified MTP profile matches CPU/context/concurrency/prefix conditions");
    if (!cost_fit) return none("automatic MTP resource evaluator unavailable");
    const auto& model = identity.Get();
    if (!model.stable_single_file || !model.matches_loaded_file || !model.stamp.valid || !model.stamp.bytes || !IsDigest(model.sha256)) {
        return none(model.reason.empty() ? "automatic MTP model identity is not bound to stable loaded bytes" : model.reason);
    }
    candidates.erase(std::remove_if(candidates.begin(), candidates.end(), [&](const auto* profile) {
        return profile->model_sha256 != model.sha256;
    }), candidates.end());
    if (candidates.empty()) return none("no qualified MTP profile matches exact model SHA256");
    // Compare verified speed, respecting the larger of 3% and A/A noise.
    // A tied K2 is the default; do not choose K3 from insignificant noise.
    const QualificationProfile* fastest = *std::max_element(candidates.begin(), candidates.end(), [](auto* a, auto* b) {
        return a->evidence.decode_speedup < b->evidence.decode_speedup;
    });
    const QualificationProfile* selected = fastest;
    for (const auto* profile : candidates) {
        const double noise = std::max({0.03,fastest->evidence.aa_noise_fraction,profile->evidence.aa_noise_fraction});
        if (fastest->evidence.decode_speedup <= profile->evidence.decode_speedup*(1+noise)) {
            if (profile->draft_n_max==2 || (selected->draft_n_max!=2 && profile->draft_n_max<selected->draft_n_max)) selected=profile;
        }
    }
    try {
        const CostFit fit = cost_fit(selected->draft_n_max);
        if (!fit.fits) return none("qualified MTP does not fit at current concurrency: " + fit.reason);
    } catch (const std::exception& error) { return none(std::string("automatic MTP resource evaluation failed: ") + error.what()); }
    return {SelectionMode::MTP,selected->draft_n_max,selected->id,"matched qualified automatic MTP profile " + selected->id};
}
inline Selection SelectAutomatic(const RuntimeConditions& runtime, int requested_upper,
                                  LazyModelIdentity& identity, const CostEvaluator& cost_fit) {
    return SelectAutomatic(ProductionProfiles(),runtime,requested_upper,identity,cost_fit);
}
} // namespace densecore::llm::speculative::qualification
