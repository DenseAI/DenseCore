#pragma once
#include "runtime/engine_internal.h"
#include "runtime/worker_internal.h"
#include "backend/thread_pool_impl.h"
#include "llm/speculative/qualification.h"
#include <map>
#include <set>
#include <optional>
#include <iomanip>
#include <locale>
#include <type_traits>
#if __has_include("densecore_native_fingerprint.h")
#include "densecore_native_fingerprint.h"
#else
namespace densecore::runtime::automatic_profile_detail {
inline constexpr const char* kNativeBuildFingerprint="";
inline constexpr std::array<const char*,0> kKnownNativeEnvironment{};
}
#endif
#if defined(__linux__)
#include <sched.h>
extern char** environ;
#endif

namespace densecore::runtime {
namespace automatic_profile_detail {
namespace qualification=densecore::llm::speculative::qualification;
using Environment=std::map<std::string,std::string>;
struct Inputs {
    ResourceReader read_file;
    ResourceBudget resources;
    std::vector<int> affinity;
    Environment environment;
    int pool_threads=0,physical_cores=0;
    densecore::simd::SimdLevel simd=densecore::simd::SimdLevel::NONE;
    std::string build_sha256;
};
inline bool CpuIds(const std::string& text,std::vector<int>& ids) {
    ids.clear();
    std::istringstream input(ResourceWord(text));std::string part;
    while(std::getline(input,part,',')) {
        const auto dash=part.find('-');uint64_t first=0,last=0;
        if(!ResourceUnsigned(part.substr(0,dash),first)) return false;
        if(dash==std::string::npos) last=first;
        else if(!ResourceUnsigned(part.substr(dash+1),last)) return false;
        if(last<first || last>65535) return false;
        for(auto id=first;id<=last;++id) ids.push_back(static_cast<int>(id));
    }
    return qualification::CanonicalIds(ids);
}
inline Environment ReadEnvironment() {
    Environment result;
#if defined(__linux__)
    for(char** item=::environ;item && *item;++item) {
        const std::string entry(*item);const auto split=entry.find('=');
        if(split!=std::string::npos) result.emplace(entry.substr(0,split),entry.substr(split+1));
    }
#endif
    return result;
}
inline bool Relevant(const std::string& key) {
    return key.rfind("DENSECORE_",0)==0 || key.rfind("GGML_",0)==0 ||
           key.rfind("OMP_",0)==0 || key.rfind("KMP_",0)==0;
}
inline bool ModeOnly(const std::string& key) {
    return key=="DENSECORE_SPEC_TYPE" || key=="DENSECORE_SPEC_DRAFT_N_MAX";
}
inline void Field(std::ostringstream& out,const std::string& key,const std::string& value) {
    out<<key.size()<<':'<<key<<'='<<value.size()<<':'<<value<<'\n';
}
template<class T> inline void Value(std::ostringstream& out,const char* name,T value) {
    out<<name<<'=';
    if constexpr(std::is_enum_v<T>) out<<static_cast<int>(value);
    else out<<std::hexfloat<<value;
    out<<'\n';
}
inline std::string CompiledIsa() {
    std::string result;
#if defined(__AVX512F__)
    result+="avx512f;";
#endif
#if defined(__AVX512VNNI__)
    result+="avx512vnni;";
#endif
#if defined(__AVX512BF16__)
    result+="avx512bf16;";
#endif
#if defined(__AVX2__)
    result+="avx2;";
#endif
#if defined(__FMA__)
    result+="fma;";
#endif
#if defined(__aarch64__)
    result+="aarch64;";
#endif
#if defined(__ARM_FEATURE_SVE)
    result+="sve;";
#endif
#if defined(__ARM_FEATURE_SVE2)
    result+="sve2;";
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
    result+="i8mm;";
#endif
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    result+="bf16;";
#endif
#ifdef DENSECORE_TEST_BUILD
    result+="test-build;";
#else
    result+="production-build;";
#endif
    return result;
}
inline std::string Trim(const std::string& value) {
    const auto first=value.find_first_not_of(" \t\r\n");
    if(first==std::string::npos)return {};
    return value.substr(first,value.find_last_not_of(" \t\r\n")-first+1);
}
inline std::string CpuIdentity(const Inputs& in) {
    std::string text;if(!in.read_file("/proc/cpuinfo",text)) return {};
    std::map<int,Environment> records;Environment record;int id=-1;
    auto finish=[&](){if(id>=0) records[id]=record;record.clear();id=-1;};
    std::istringstream lines(text);std::string line;
    while(std::getline(lines,line)) {
        if(ResourceWord(line).empty()){finish();continue;}
        const auto colon=line.find(':');if(colon==std::string::npos) return {};
        const auto key=Trim(line.substr(0,colon)),value=Trim(line.substr(colon+1));
        if(key=="processor") {uint64_t number=0;if(!ResourceUnsigned(value,number)||number>65535)return {};id=static_cast<int>(number);}
        record[key]=value;
    }
    finish();std::ostringstream result;
    for(int cpu:in.affinity) {
        const auto found=records.find(cpu);if(found==records.end()) return {};
        const auto& values=found->second;
        const bool x86=values.count("vendor_id")!=0;
        const std::vector<std::string> keys=x86
            ? std::vector<std::string>{"vendor_id","cpu family","model","stepping","flags"}
            : std::vector<std::string>{"CPU implementer","CPU architecture","CPU variant","CPU part","CPU revision","Features"};
        result<<"cpu="<<cpu<<';';
        for(const auto& key:keys) {
            auto it=values.find(key);if(it==values.end()||it->second.empty())return {};
            Field(result,key,it->second);
        }
        if(auto it=values.find("microcode");it!=values.end())Field(result,"microcode",it->second);
    }
    return result.str();
}
inline void EffectivePolicy(std::ostringstream& out,const EngineState& state) {
#define DC_PROFILE(x) Value(out,#x,(x))
    const auto& f=state.fast_path_config;const auto& w=f.worker;const auto& i=state.inference_config;
    DC_PROFILE(i.use_flash_attention);DC_PROFILE(i.flash_attention_block_size);DC_PROFILE(i.enable_prefetch);
    DC_PROFILE(i.prefetch_lookahead);DC_PROFILE(i.num_threads);DC_PROFILE(i.decode_threads);
    DC_PROFILE(i.prefill_threads);DC_PROFILE(i.enable_split_thread_policy);
    DC_PROFILE(w.validate_mul);DC_PROFILE(w.runtime_path_logging);DC_PROFILE(w.scheduler_stall_debug);
    DC_PROFILE(w.moe_trace_dump);DC_PROFILE(w.sampler_trace_dump);DC_PROFILE(w.determinism_boundary_debug);
    DC_PROFILE(w.prefix_cache_reuse_disabled);DC_PROFILE(w.hybrid_ssm_snapshot_restore_disabled);
    DC_PROFILE(w.qwen36_prefix_cache_reuse_enabled);DC_PROFILE(w.qwen36_hybrid_ssm_snapshot_restore_enabled);
    DC_PROFILE(w.graph_cache_reuse_disabled);DC_PROFILE(w.moe_trace_plumbing_disabled);DC_PROFILE(w.moe_graph_summary);
    DC_PROFILE(w.zero_fill_prefill_input_buffer);DC_PROFILE(w.zero_fill_prefill_graph_buffer);DC_PROFILE(w.zero_fill_prefill_kv_blocks);
    DC_PROFILE(w.debug_sample);DC_PROFILE(w.debug_sample_top);DC_PROFILE(w.debug_sampler_trace);DC_PROFILE(w.debug_first_token_raw);
    DC_PROFILE(w.prefill_thread_override);DC_PROFILE(w.callback_mode);DC_PROFILE(w.qwen35_prefix_cache_reuse_enabled);
    DC_PROFILE(f.engine_debug.verbose_token_trace);DC_PROFILE(f.engine_debug.runtime_path_logging);
    DC_PROFILE(f.engine_debug.runtime_path_token_logging);DC_PROFILE(f.engine_debug.parity_debug);
    DC_PROFILE(f.decode_paged_attention.mode);DC_PROFILE(f.decode_paged_attention.min_context_tokens);
    DC_PROFILE(f.decode_paged_attention.min_batched_context_tokens);DC_PROFILE(f.decode_paged_attention.min_head_dim);
    DC_PROFILE(f.decode_paged_attention.min_heads);DC_PROFILE(f.decode_paged_attention.allow_quantized_auto);
    DC_PROFILE(f.decode_paged_attention.debug_log);DC_PROFILE(f.kv_retention.sliding_window);DC_PROFILE(f.kv_retention.sink_tokens);
    DC_PROFILE(f.kv_retention.enabled);DC_PROFILE(f.prefill_graph_cache.enabled);DC_PROFILE(f.prefill_graph_cache.lru_size);
    DC_PROFILE(f.prefill_graph_cache.max_bytes);DC_PROFILE(f.qwen36_prefill_q4k_batched);DC_PROFILE(f.qwen36_ssm_q8_prefill_amx);
    DC_PROFILE(f.qwen36_ssm_q8_prefill_amx_min_tokens);DC_PROFILE(f.native_moe_fast_decode);DC_PROFILE(f.q4k_repacked_gemv);
    DC_PROFILE(f.q4k_repacked_gemv_allow_prefill);DC_PROFILE(f.q4k_repacked_gemv_probe);DC_PROFILE(f.q4k_repacked_gemv_disable_on_thrash);
    DC_PROFILE(f.q4k_repacked_gemv_thrash_repack_mb);DC_PROFILE(f.q4k_repacked_gemv_thrash_eviction_ratio);
    DC_PROFILE(f.q4k_repacked_gemv_thrash_repack_cache_fraction);DC_PROFILE(f.qact_cache);
    DC_PROFILE(f.matmul_dispatch_census);DC_PROFILE(f.bench_respect_threads);DC_PROFILE(state.numa_node_id);DC_PROFILE(state.pinning_policy);
#undef DC_PROFILE
}
inline qualification::RuntimeConditions Capture(const EngineState& state,const TransformerModel& model,
    uint32_t context,uint32_t concurrency,ggml_type kv,const Inputs& in) {
    qualification::RuntimeConditions result;
    if(!context||!concurrency||concurrency>8||kv<0||kv>=GGML_TYPE_COUNT||state.n_threads<=0||in.pool_threads<=0||in.physical_cores<=0||
       !qualification::CanonicalIds(in.affinity)||!qualification::IsDigest(in.build_sha256)||
       in.resources.cpu_quota_state==ResourceState::Unknown) return result;
    // Initial qualified policy deliberately excludes partial affinity/cpuset masks:
    // lazy pool affinity may otherwise differ from the calling thread.
    std::string nodes,online,node_cpus;
    std::vector<int> node_ids,online_ids,node_cpu_ids;
    if(!in.read_file("/sys/devices/system/node/online",nodes)||!CpuIds(nodes,node_ids)||node_ids!=std::vector<int>{0}||
       !in.read_file("/sys/devices/system/cpu/online",online)||!CpuIds(online,online_ids)||
       !in.read_file("/sys/devices/system/node/node0/cpulist",node_cpus)||!CpuIds(node_cpus,node_cpu_ids)||
       in.affinity!=online_ids||in.affinity!=node_cpu_ids||state.numa_node_id>0||state.numa_node_id < -1) return result;
    std::ostringstream policy;policy.imbue(std::locale::classic());
    for(const auto& item:in.environment) if(Relevant(item.first)) {
        if(std::find_if(kKnownNativeEnvironment.begin(),kKnownNativeEnvironment.end(),
              [&](const char* key){return item.first==key;})==kKnownNativeEnvironment.end())return result;
        if(!ModeOnly(item.first))Field(policy,item.first,item.second);
    }
    for(int cpu:in.affinity) for(const char* key:{"core_id","physical_package_id","thread_siblings_list"}) {
        std::string value;
        if(!in.read_file("/sys/devices/system/cpu/cpu"+std::to_string(cpu)+"/topology/"+key,value)||ResourceWord(value).empty())return result;
        Field(policy,std::to_string(cpu)+":"+key,ResourceWord(value));
    }
    if(in.resources.cpu_quota_state==ResourceState::Finite &&
       (!(in.resources.cpu_quota>0)||!std::isfinite(in.resources.cpu_quota)))return result;
    Value(policy,"quota_state",in.resources.cpu_quota_state);Value(policy,"quota_fraction",in.resources.cpu_quota);
    Value(policy,"resource_effective_threads",in.resources.effective_threads);
    Value(policy,"physical_core_policy_count",in.physical_cores);
    Value(policy,"head_graph_threads",state.n_threads);
    EffectivePolicy(policy,state);
    const bool allow_over=AllowDecodeThreadsOverBase();Value(policy,"allow_decode_over_base",allow_over);
    const auto simd=in.simd;const auto base=state.n_threads;
    const int cap=ResolveFinalWorkerThreadCap(&model,in.physical_cores,base,state.fast_path_config.bench_respect_threads);
    // Current ResolvePrefillThreadPolicySelection explicitly ignores prompt length.
    // The code fingerprint invalidates profiles if that source policy changes.
    for(int batch=1;batch<=8;++batch) {
        int prefill=base,decode=base;
        if(!state.fast_path_config.bench_respect_threads && state.inference_config.enable_split_thread_policy) {
            prefill=state.inference_config.prefill_threads>0 ? state.inference_config.prefill_threads
                : ResolvePrefillThreadPolicySelection(&model,batch,static_cast<int>(std::min<uint32_t>(context,INT_MAX)),in.physical_cores,base,simd).threads;
            decode=state.inference_config.decode_threads>0 ? state.inference_config.decode_threads
                : ResolveDecodeThreadPolicySelection(&model,batch,in.physical_cores,base,simd).threads;
        }
        if(state.fast_path_config.worker.prefill_thread_override>0) prefill=state.fast_path_config.worker.prefill_thread_override;
        if(!allow_over) decode=std::min(decode,base);
        prefill=std::max(1,std::min(prefill,cap));decode=std::max(1,std::min(decode,cap));
        Value(policy,("prefill_batch"+std::to_string(batch)).c_str(),prefill);
        Value(policy,("decode_batch"+std::to_string(batch)).c_str(),decode);
        if(batch==1){result.cpu.prefill_threads=prefill;result.cpu.decode_threads=decode;}
    }
    result.cpu.cpu_identity=CpuIdentity(in);if(result.cpu.cpu_identity.empty())return {};
    result.cpu.build_sha256=qualification::Sha256Hex(in.build_sha256+CompiledIsa());
    result.cpu.simd=densecore::simd::SimdLevelName(simd);
    result.cpu.runtime_policy_sha256=qualification::Sha256Hex(policy.str());
    result.cpu.pool_threads=in.pool_threads;result.cpu.affinity_cpus=in.affinity;result.cpu.numa_nodes={0};
    // Single physical node is proven by sysfs, not guessed from requested IDs.
    result.cpu.numa_policy="single-physical-node:0";result.cpu.weight_placement="single-physical-node:0";
    // This describes the proposed MTP execution, not the current NONE baseline:
    // worker admission disables prefix reuse whenever speculation_selected == 2.
    result.context=context;result.concurrency=concurrency;result.prefix=qualification::PrefixMode::Disabled;
    result.kv_dtype=ggml_type_name(kv);
    return result;
}
} // namespace automatic_profile_detail

inline densecore::llm::speculative::qualification::RuntimeConditions CaptureAutomaticRuntimeConditions(
    const EngineState& state,const TransformerModel& model,uint32_t context,uint32_t concurrency,ggml_type kv) {
    using namespace automatic_profile_detail;
    Inputs in;in.read_file=ReadResourceFile;in.resources=DetectResourceBudget();in.environment=ReadEnvironment();
    in.build_sha256=kNativeBuildFingerprint;in.simd=densecore::simd::DetectSimdLevel();
#if defined(__linux__)
    cpu_set_t cpus;CPU_ZERO(&cpus);
    if(sched_getaffinity(0,sizeof(cpus),&cpus)!=0)return {};
    for(int cpu=0;cpu<CPU_SETSIZE;++cpu)if(CPU_ISSET(cpu,&cpus))in.affinity.push_back(cpu);
#else
    return {};
#endif
    const auto arm=ResolveArmComputeAffinityPolicy();
    if(!arm.core_ids.empty() && arm.core_ids!=in.affinity)return {};
    in.physical_cores=arm.core_ids.empty() ? densecore::HardwareTopology::GetInstance().GetPhysicalCoreCount() : arm.core_ids.size();
    if(!state.backend_registry)return {};
    auto* cpu=dynamic_cast<densecore::CpuBackend*>(state.backend_registry->Get(densecore::DeviceType::CPU));
    if(!cpu)return {};
    in.pool_threads=cpu->GetThreadPool(0).GetNumThreads();
    return Capture(state,model,context,concurrency,kv,in);
}
} // namespace densecore::runtime
