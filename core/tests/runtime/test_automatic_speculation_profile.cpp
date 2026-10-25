#include <gtest/gtest.h>
#include "runtime/automatic_speculation_profile.h"

namespace {
using namespace densecore::runtime;
namespace detail = densecore::runtime::automatic_profile_detail;
namespace qualification = densecore::llm::speculative::qualification;
struct AutomaticCaptureFixture : testing::Test {
    EngineState state;
    TransformerModel model;
    detail::Inputs inputs;
    std::map<std::string,std::string> files;
    void SetUp() override {
        state.n_threads=4;
        state.inference_config.enable_split_thread_policy=false;
        inputs.pool_threads=4;inputs.physical_cores=4;
        inputs.simd=densecore::simd::SimdLevel::AVX2;
        inputs.affinity={0,1,2,3};inputs.build_sha256=std::string(64,'a');
        inputs.resources.cpu_quota_state=ResourceState::Unlimited;
        inputs.resources.effective_threads=4;
        files["/sys/devices/system/node/online"]="0\n";
        files["/sys/devices/system/cpu/online"]="0-3\n";
        files["/sys/devices/system/node/node0/cpulist"]="0-3\n";
        std::string cpuinfo;
        for(int cpu=0;cpu<4;++cpu) {
            cpuinfo+="processor : "+std::to_string(cpu)+"\nvendor_id : UnitVendor\ncpu family : 6\nmodel : 85\nstepping : 7\nflags : fpu avx2\nmicrocode : 0x1\n\n";
            const auto base="/sys/devices/system/cpu/cpu"+std::to_string(cpu)+"/topology/";
            files[base+"core_id"]=std::to_string(cpu);
            files[base+"physical_package_id"]="0";
            files[base+"thread_siblings_list"]=std::to_string(cpu);
        }
        files["/proc/cpuinfo"]=cpuinfo;
        inputs.read_file=[this](const std::string& path,std::string& result) {
            const auto found=files.find(path);if(found==files.end())return false;
            result=found->second;return true;
        };
    }
    qualification::RuntimeConditions Capture(uint32_t c=1) {
        return detail::Capture(state,model,2048,c,GGML_TYPE_F16,inputs);
    }
};
TEST_F(AutomaticCaptureFixture, CompleteProposedMtpProfileUsesActualPoolAndDisabledPrefix) {
    const auto first=Capture();ASSERT_TRUE(first.Complete());
    EXPECT_EQ(first.cpu.pool_threads,4u);EXPECT_EQ(first.prefix,qualification::PrefixMode::Disabled);
    EXPECT_EQ(first.cpu.affinity_cpus,inputs.affinity);
    EXPECT_EQ(first.cpu, Capture(8).cpu); // Concurrency is matched separately by registry range.
}
TEST_F(AutomaticCaptureFixture, RawFractionalQuotaCannotAliasSameRoundedThreadCount) {
    inputs.resources.cpu_quota_state=ResourceState::Finite;
    inputs.resources.effective_threads=1;inputs.resources.cpu_quota=.1;
    const auto fractional=Capture();ASSERT_TRUE(fractional.Complete());
    inputs.resources.cpu_quota=1;
    const auto whole=Capture();ASSERT_TRUE(whole.Complete());
    EXPECT_NE(fractional.cpu.runtime_policy_sha256,whole.cpu.runtime_policy_sha256);
}
TEST_F(AutomaticCaptureFixture, UnknownEnvironmentAndQuotaFailClosed) {
    inputs.environment["DENSECORE_UNRECOGNIZED_FUTURE_KERNEL_CONTROL"]="1";
    EXPECT_FALSE(Capture().Complete());inputs.environment.clear();
    inputs.resources.cpu_quota_state=ResourceState::Unknown;EXPECT_FALSE(Capture().Complete());
}
TEST_F(AutomaticCaptureFixture, MissingTopologyAndPartialAffinityRemainUnqualified) {
    inputs.affinity={0,1};EXPECT_FALSE(Capture().Complete());inputs.affinity={0,1,2,3};
    files.erase("/sys/devices/system/node/online");EXPECT_FALSE(Capture().Complete());
    files["/sys/devices/system/node/online"]="0-1";EXPECT_FALSE(Capture().Complete());
}
TEST_F(AutomaticCaptureFixture, EffectivePolicyCpuFeaturesAndBuildChangeIdentity) {
    const auto original=Capture();ASSERT_TRUE(original.Complete());
    state.fast_path_config.worker.graph_cache_reuse_disabled=!state.fast_path_config.worker.graph_cache_reuse_disabled;
    EXPECT_NE(original.cpu.runtime_policy_sha256,Capture().cpu.runtime_policy_sha256);
    inputs.build_sha256=std::string(64,'b');EXPECT_NE(original.cpu.build_sha256,Capture().cpu.build_sha256);
    files["/proc/cpuinfo"].append("processor : 0\nvendor_id : DifferentVendor\ncpu family : 6\nmodel : 85\nstepping : 7\nflags : fpu avx2\n\n");
    EXPECT_NE(original.cpu.cpu_identity,Capture().cpu.cpu_identity);
}
TEST_F(AutomaticCaptureFixture, AllCpuFeatureWordsAndMultiwordKeysAreRetained) {
    const auto original=Capture();ASSERT_TRUE(original.Complete());
    auto& text=files["/proc/cpuinfo"];
    const auto at=text.find("fpu avx2");ASSERT_NE(at,std::string::npos);
    text.replace(at,8,"fpu avx2 avx512f");
    const auto changed=Capture();ASSERT_TRUE(changed.Complete());
    EXPECT_NE(original.cpu.cpu_identity,changed.cpu.cpu_identity);
}
TEST_F(AutomaticCaptureFixture, InvalidDtypeAndZeroDimensionsFailClosed) {
    EXPECT_FALSE(detail::Capture(state,model,2048,1,static_cast<ggml_type>(GGML_TYPE_COUNT),inputs).Complete());
    EXPECT_FALSE(detail::Capture(state,model,0,1,GGML_TYPE_F16,inputs).Complete());
    EXPECT_FALSE(detail::Capture(state,model,2048,9,GGML_TYPE_F16,inputs).Complete());
}
} // namespace
