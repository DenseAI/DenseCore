#include <gtest/gtest.h>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <functional>
#include <limits>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>
#include "llm/speculative/qualification.h"
#include "runtime/engine_internal.h"
#include "runtime/speculation_config.h"
#include <cstring>
#include <cstdlib>

namespace {
using namespace densecore::llm::speculative::qualification;
#include "fixtures/mtp_sha256_nist.inl"
std::string DecodeHex(const std::string& text) {
    const auto digit=[](char c) { return c<='9' ? c-'0' : c-'a'+10; };
    std::string bytes(text.size()/2,'\0');
    for (size_t i=0;i<bytes.size();++i) bytes[i]=static_cast<char>((digit(text[2*i])<<4)|digit(text[2*i+1]));
    return bytes;
}
std::string Payload() {
    std::string bytes(131119,'\0');
    for (size_t i=0;i<bytes.size();++i) bytes[i]=static_cast<char>((i*131+i/7)%256);
    return bytes;
}
struct TemporaryFile {
    std::string path;
    explicit TemporaryFile(const std::string& bytes) {
        char name[]="/tmp/densecore_mtp_identity_XXXXXX";
        const int descriptor=mkstemp(name);
        if (descriptor<0) throw std::runtime_error("temporary model fixture unavailable");
        close(descriptor); path=name; Write(bytes);
    }
    void Write(const std::string& bytes) const {
        std::ofstream out(path,std::ios::binary|std::ios::trunc); out.write(bytes.data(),bytes.size());
        if (!out) throw std::runtime_error("temporary fixture write failed");
    }
    ~TemporaryFile() { std::remove(path.c_str()); }
};
RuntimeConditions Runtime() {
    RuntimeConditions r;
    r.cpu.cpu_identity="unit CPU vendor/family/model/stepping";
    r.cpu.build_sha256=Sha256Hex("compiler+ISA+kernel-contract; excluding registry data");
    r.cpu.simd="avx2"; r.cpu.runtime_policy_sha256=Sha256Hex("effective scheduler/kernel controls");
    r.cpu.pool_threads=2;r.cpu.prefill_threads=2;r.cpu.decode_threads=2;
    r.cpu.affinity_cpus={0,2};r.cpu.numa_nodes={0};r.cpu.numa_policy="disabled";r.cpu.weight_placement="default";
    r.context=2048;r.concurrency=4;r.prefix=PrefixMode::RecurrentSnapshot;r.kv_dtype="f16";
    return r;
}
ValidationEvidence Evidence() {
    ValidationEvidence e;
    e.quality_sha256=Sha256Hex("complete final-answer/token/distribution/raw proof");
    e.resource_sha256=Sha256Hex("resource proof");e.latency_sha256=Sha256Hex("latency proof");
    e.throughput_sha256=Sha256Hex("throughput proof");e.decode_sha256=Sha256Hex("decode raw ABBA proof");
    e.measured_binary_sha256=Sha256Hex("full measured binary including its old registry data");
    e.final_answers_complete=e.greedy_parity=e.distribution_parity=e.rollback_isolation=true;
    e.grammar_stream_cancel=e.resource_pass=e.latency_pass=e.throughput_pass=true;
    e.all_applicable_cells_passed=e.aa_calibrated=true;e.abba_pairs=5;e.aa_noise_fraction=0.01;
    e.decode_speedup=1.25;e.decode_speedup_lower_bound=1.20;
    e.worst_latency_ratio=e.worst_throughput_ratio=1;
    return e;
}
QualificationProfile Profile(int k=2) {
    const auto r=Runtime();QualificationProfile p;
    p.id="unit-only K"+std::to_string(k);p.model_sha256=Sha256Hex("model bytes");p.cpu=r.cpu;
    p.context=r.context;p.concurrency_min=1;p.concurrency_max=4;p.prefix=r.prefix;p.kv_dtype=r.kv_dtype;
    p.draft_n_max=k;p.evidence=Evidence();return p;
}
ModelIdentity Identity() {
    ModelIdentity m;m.stable_single_file=m.matches_loaded_file=true;m.sha256=Sha256Hex("model bytes");
    m.stamp.valid=true;m.stamp.bytes=11;m.stamp.device=7;m.stamp.inode=42;return m;
}
CostFit Fits(int) { return {true,{}}; }
Selection Select(const std::vector<QualificationProfile>& p,const RuntimeConditions& r=Runtime(),int upper=2) {
    LazyModelIdentity identity([] {return Identity();});return SelectAutomatic(p,r,upper,identity,Fits);
}
}

TEST(MtpSha256, AllNistShortAndSelectedLongByteVectors) {
    ASSERT_EQ(std::size(kNistSha256Vectors),68u);
    for (const auto& vector:kNistSha256Vectors) {
        SCOPED_TRACE(vector.bits);
        auto bytes=DecodeHex(vector.message_hex);bytes.resize(vector.bits/8);
        EXPECT_EQ(Sha256Hex(bytes),vector.digest);
        Sha256 streamed;
        for (size_t offset=0;offset<bytes.size();offset+=7) streamed.Update(bytes.data()+offset,std::min<size_t>(7,bytes.size()-offset));
        EXPECT_EQ(streamed.HexDigest(),vector.digest);
    }
}
TEST(MtpSha256, PythonHashlibBinaryPayloadAndStreamingBoundaries) {
    const auto bytes=Payload();
    constexpr auto expected="29098d0a31def7329f43e0535f986ff08989c484fc6bb0cfc7d92f6a4d6de7d8";
    for (size_t chunk:{1,7,55,56,63,64,65,111,112,127,128,4096,65536}) {
        Sha256 hash;
        for (size_t offset=0;offset<bytes.size();offset+=chunk) hash.Update(bytes.data()+offset,std::min(chunk,bytes.size()-offset));
        EXPECT_EQ(hash.HexDigest(),expected)<<chunk;
    }
    std::istringstream input(bytes);EXPECT_EQ(Sha256Stream(input),expected);
    EXPECT_EQ(Sha256Hex(std::string(1000000,'a')),"cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}
TEST(MtpSha256, DigestDoesNotFinalizeAndInvalidInputFails) {
    Sha256 hash;hash.Update("a");const auto prefix=hash.HexDigest();
    hash.Update("bc");EXPECT_EQ(hash.HexDigest(),"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_NE(prefix,hash.HexDigest());EXPECT_THROW(hash.Update(nullptr,1),std::invalid_argument);
    std::istringstream broken("unused");broken.setstate(std::ios::badbit);
    EXPECT_THROW(Sha256Stream(broken),std::ios_base::failure);
}
TEST(MtpModelIdentity, HashIsBoundToUnchangedLoadedSnapshot) {
    TemporaryFile file(Payload());const auto before=CaptureFileStamp(file.path);ASSERT_TRUE(before);
    std::ifstream loaded(file.path,std::ios::binary);std::string bytes((std::istreambuf_iterator<char>(loaded)),{});
    const auto after=CaptureFileStamp(file.path);ASSERT_TRUE(after);ASSERT_EQ(*before,*after);
    const auto result=HashLoadedModel(file.path,*after,false);
    EXPECT_TRUE(result.stable_single_file);EXPECT_TRUE(result.matches_loaded_file);
    EXPECT_EQ(result.stamp,*after);EXPECT_EQ(result.sha256,Sha256Hex(bytes));
}
TEST(MtpModelIdentity, DifferentLoadedFileAndSplitIdentityFailBeforeReading) {
    TemporaryFile file("model bytes"),replacement("other bytes");
    const auto loaded=CaptureFileStamp(file.path);ASSERT_TRUE(loaded);
    ASSERT_EQ(std::rename(replacement.path.c_str(),file.path.c_str()),0);
    int reads=0;
    EXPECT_FALSE(HashLoadedModel(file.path,*loaded,false,[&](uint64_t){++reads;}).stable_single_file);
    EXPECT_EQ(reads,0);
    EXPECT_FALSE(HashLoadedModel("/missing/split-file",*loaded,true,[&](uint64_t){++reads;}).stable_single_file);
    EXPECT_EQ(reads,0);EXPECT_FALSE(CaptureFileStamp("/missing/densecore-model"));
    EXPECT_FALSE(CaptureFileStamp("/tmp"));
}
TEST(MtpModelIdentity, ReplacementDuringHashAndProviderFailureAreCachedClosed) {
    TemporaryFile file(Payload()),replacement(Payload());
    const auto loaded=CaptureFileStamp(file.path);ASSERT_TRUE(loaded);
    bool replaced=false;
    const auto result=HashLoadedModel(file.path,*loaded,false,[&](uint64_t) {
        if (!replaced) {ASSERT_EQ(std::rename(replacement.path.c_str(),file.path.c_str()),0);replaced=true;}
    });
    EXPECT_FALSE(result.stable_single_file);EXPECT_TRUE(replaced);
    int calls=0;LazyModelIdentity lazy([&]() -> ModelIdentity {++calls;throw std::runtime_error("identity fault");});
    EXPECT_FALSE(lazy.Get().stable_single_file);EXPECT_FALSE(lazy.Get().stable_single_file);EXPECT_EQ(calls,1);
}
TEST(MtpQualification, ProductionEmptyDoesNotHashOrEvaluateCost) {
    ASSERT_TRUE(ProductionProfiles().empty());int hashes=0,costs=0;
    LazyModelIdentity identity([&]{++hashes;return Identity();});
    const auto result=SelectAutomatic(Runtime(),2,identity,[&](int){++costs;return CostFit{true,{}};});
    EXPECT_EQ(result.mode,SelectionMode::None);EXPECT_EQ(hashes,0);EXPECT_EQ(costs,0);
}
TEST(MtpQualification, ExactModelCpuContextConcurrencyPrefixAndKvMatch) {
    EXPECT_EQ(Select({Profile()}).mode,SelectionMode::MTP);
    using Change=std::function<void(RuntimeConditions&)>;
    const std::vector<Change> changes={
        [](auto& r){r.cpu.cpu_identity+="other";},[](auto& r){r.cpu.build_sha256=Sha256Hex("other build");},
        [](auto& r){r.cpu.simd="scalar";},[](auto& r){r.cpu.runtime_policy_sha256=Sha256Hex("other flags");},
        [](auto& r){++r.cpu.pool_threads;},[](auto& r){++r.cpu.prefill_threads;},[](auto& r){++r.cpu.decode_threads;},
        [](auto& r){r.cpu.affinity_cpus={0,1};},[](auto& r){r.cpu.numa_nodes={0,1};},
        [](auto& r){r.cpu.numa_policy="interleave";},[](auto& r){r.cpu.weight_placement="node1";},
        [](auto& r){++r.context;},[](auto& r){r.concurrency=5;},
        [](auto& r){r.prefix=PrefixMode::Disabled;},[](auto& r){r.kv_dtype="q8_0";}};
    for (size_t i=0;i<changes.size();++i) {
        auto runtime=Runtime();changes[i](runtime);int hashes=0;
        LazyModelIdentity identity([&]{++hashes;return Identity();});
        EXPECT_EQ(SelectAutomatic({Profile()},runtime,2,identity,Fits).mode,SelectionMode::None)<<i;
        EXPECT_EQ(hashes,0)<<i;
    }
    auto p=Profile();p.model_sha256=Sha256Hex("different model bytes");
    EXPECT_EQ(Select({p}).mode,SelectionMode::None);
}
TEST(MtpQualification, EveryRequiredEvidenceGateAndArtifactIsEnforced) {
    const std::vector<bool ValidationEvidence::*> gates={
        &ValidationEvidence::final_answers_complete,&ValidationEvidence::greedy_parity,&ValidationEvidence::distribution_parity,
        &ValidationEvidence::rollback_isolation,&ValidationEvidence::grammar_stream_cancel,&ValidationEvidence::resource_pass,
        &ValidationEvidence::latency_pass,&ValidationEvidence::throughput_pass,&ValidationEvidence::all_applicable_cells_passed,
        &ValidationEvidence::aa_calibrated};
    for (auto gate:gates) {auto p=Profile();p.evidence.*gate=false;EXPECT_EQ(Select({p}).mode,SelectionMode::None);}
    const std::vector<std::string ValidationEvidence::*> artifacts={
        &ValidationEvidence::quality_sha256,&ValidationEvidence::resource_sha256,&ValidationEvidence::latency_sha256,
        &ValidationEvidence::throughput_sha256,&ValidationEvidence::decode_sha256,&ValidationEvidence::measured_binary_sha256};
    for (auto artifact:artifacts) {auto p=Profile();(p.evidence.*artifact).clear();EXPECT_EQ(Select({p}).mode,SelectionMode::None);}
    using Change=std::function<void(ValidationEvidence&)>;
    const std::vector<Change> invalid={[](auto& e){e.abba_pairs=4;},[](auto& e){e.decode_speedup_lower_bound=1.149;},
        [](auto& e){e.decode_speedup=1.19;},[](auto& e){e.worst_latency_ratio=1.031;},
        [](auto& e){e.worst_throughput_ratio=0.969;},[](auto& e){e.aa_noise_fraction=-1;},
        [](auto& e){e.decode_speedup=std::numeric_limits<double>::infinity();},
        [](auto& e){e.worst_latency_ratio=std::numeric_limits<double>::quiet_NaN();}};
    for (auto change:invalid) {auto p=Profile();change(p.evidence);EXPECT_EQ(Select({p}).mode,SelectionMode::None);}
}
TEST(MtpQualification, UpperKAndMeasuredSpeedNoiseChooseOnlyQualifiedDraftCounts) {
    auto one=Profile(1),two=Profile(2),three=Profile(3);
    one.evidence.decode_speedup=1.21;three.evidence.decode_speedup=1.26;
    EXPECT_EQ(Select({one,two,three},Runtime(),1).draft_n_max,1);
    EXPECT_EQ(Select({one,two,three},Runtime(),2).draft_n_max,2);
    EXPECT_EQ(Select({one,two,three},Runtime(),3).draft_n_max,2);
    EXPECT_EQ(Select({three},Runtime(),2).mode,SelectionMode::None);
    three.evidence.decode_speedup=1.35;
    EXPECT_EQ(Select({one,two,three},Runtime(),3).draft_n_max,3);
    three.evidence.aa_noise_fraction=0.10;
    EXPECT_EQ(Select({one,two,three},Runtime(),3).draft_n_max,2);
}
TEST(MtpQualification, CurrentConcurrencyCostFailureSelectsNoneAndNeverLowersConcurrency) {
    auto runtime=Runtime();int hashes=0,costs=0;
    LazyModelIdentity identity([&]{++hashes;return Identity();});
    const auto result=SelectAutomatic({Profile()},runtime,2,identity,[&](int k) {
        ++costs;EXPECT_EQ(k,2);EXPECT_EQ(runtime.concurrency,4u);return CostFit{false,"required=4096 available=1024"};});
    EXPECT_EQ(result.mode,SelectionMode::None);EXPECT_EQ(result.draft_n_max,0);
    EXPECT_NE(result.reason.find("current concurrency"),std::string::npos);
    EXPECT_EQ(runtime.concurrency,4u);EXPECT_EQ(costs,1);EXPECT_EQ(hashes,1);
    const auto error=SelectAutomatic({Profile()},runtime,2,identity,[](int)->CostFit {throw std::runtime_error("cost fault");});
    EXPECT_EQ(error.mode,SelectionMode::None);EXPECT_EQ(hashes,1);
}
TEST(MtpQualification, IdentityIsLazyOnceAcrossConcurrencyPlanningAndRejectsUnboundBytes) {
    int calls=0;LazyModelIdentity identity([&]{++calls;return Identity();});auto runtime=Runtime();
    runtime.concurrency=8;EXPECT_EQ(SelectAutomatic({Profile()},runtime,2,identity,Fits).mode,SelectionMode::None);EXPECT_EQ(calls,0);
    for (int c:{4,2,1}) {runtime.concurrency=c;EXPECT_EQ(SelectAutomatic({Profile()},runtime,2,identity,Fits).mode,SelectionMode::MTP);}
    EXPECT_EQ(calls,1);
    using Change=std::function<void(ModelIdentity&)>;
    for (Change change:std::vector<Change>{[](auto& i){i.stable_single_file=false;},[](auto& i){i.matches_loaded_file=false;},
            [](auto& i){i.stamp.valid=false;},[](auto& i){i.stamp.bytes=0;},[](auto& i){i.sha256="not SHA256";}}) {
        LazyModelIdentity invalid([&]{auto i=Identity();change(i);return i;});
        EXPECT_EQ(SelectAutomatic({Profile()},Runtime(),2,invalid,Fits).mode,SelectionMode::None);
    }
    LazyModelIdentity unused([]{return Identity();});
    EXPECT_EQ(SelectAutomatic({Profile()},Runtime(),2,unused,{}).mode,SelectionMode::None);
}
TEST(MtpQualification, RealLoadedFileHashMatchesRegardlessOfPathName) {
    TemporaryFile file("model bytes");const auto stamp=CaptureFileStamp(file.path);ASSERT_TRUE(stamp);
    LazyModelIdentity identity([&]{return HashLoadedModel(file.path,*stamp,false);});
    EXPECT_EQ(SelectAutomatic({Profile()},Runtime(),2,identity,Fits).mode,SelectionMode::MTP);
}

namespace {
std::unique_ptr<TransformerModel> AutomaticTinyModel() {
    auto owner = std::make_unique<TransformerModel>();
    auto& source = *owner;
    source.arch = ModelArch::QWEN35;
    source.variant = ModelVariant::QWEN35;
    source.arch_flags.is_hybrid_ssm = true;
    source.arch_flags.requires_q_norm = true;
    source.arch_flags.requires_k_norm = true;
    source.hparams.n_embd = 2;
    source.hparams.n_vocab = 2;
    source.hparams.n_head = 1;
    source.hparams.n_head_kv = 1;
    source.hparams.n_embd_head_k = 2;
    source.hparams.n_embd_head_v = 2;
    source.hparams.n_rot = 2;
    source.hparams.n_ctx = 16;
    source.hparams.n_layer = 4;
    source.layers.resize(4);
    source.hybrid_layer_is_ssm = {0,0,0,0};
    source.hparams.n_ff = 2;
    source.ctx_w = ggml_init({1024 * 1024, nullptr, false});
    if (!source.ctx_w) throw std::bad_alloc();
    const auto weight = [&](int columns, int rows, float value) {
        auto* t = ggml_new_tensor_2d(source.ctx_w, GGML_TYPE_F32, columns, rows);
        std::fill_n(static_cast<float*>(t->data), columns * rows, value);
        return t;
    };
    source.tok_embeddings = weight(2, 2, 1);
    source.output_norm = weight(2, 1, 1);
    source.output = weight(2, 2, 0);
    const float identity[] = {1, 0, 0, 1};
    std::memcpy(source.output->data, identity, sizeof(identity));
    source.mtp_heads.resize(1);
    auto& head = source.mtp_heads[0];
    head.gguf_layer_index = 4;
    head.tensors["nextn.enorm.weight"] = weight(2, 1, 1);
    head.tensors["nextn.hnorm.weight"] = weight(2, 1, 1);
    auto* eh = weight(4, 2, 0);
    const float hidden_selector[] = {0, 0, 1, 0, 0, 0, 0, 1};
    std::memcpy(eh->data, hidden_selector, sizeof(hidden_selector));
    head.tensors["nextn.eh_proj.weight"] = eh;
    for (const char* name : {"attn_norm.weight", "post_attention_norm.weight", "attn_q_norm.weight",
                             "attn_k_norm.weight"}) head.tensors[name] = weight(2, 1, 1);
    for (const char* name : {"attn_k.weight", "attn_v.weight", "attn_output.weight",
                             "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"}) {
        head.tensors[name] = weight(2, 2, 0);
    }
    head.tensors["attn_q.weight"] = weight(2, 4, 0); // [q | gate] per head
    return owner;
}

struct AutoEnv {
    std::string key, previous; bool present;
    AutoEnv(const char* name,const char* value):key(name),present(std::getenv(name)!=nullptr) {
        if(present) previous=std::getenv(name);
        if(value) setenv(name,value,1);else unsetenv(name);
    }
    ~AutoEnv(){if(present)setenv(key.c_str(),previous.c_str(),1);else unsetenv(key.c_str());}
};
RuntimeConditions AutoRuntime(int context,int concurrency,ggml_type type) {
    auto r=Runtime();r.context=context;r.concurrency=concurrency;r.prefix=PrefixMode::Disabled;
    r.kv_dtype=ggml_type_name(type);return r;
}
QualificationProfile AutoProfile(const KVCacheConfig& config,int k=1) {
    auto p=Profile(k);
    auto r=BindAutomaticPrefillChunk(AutoRuntime(config.max_seq_len,config.max_num_seqs,config.effective_cache_type),config.max_prefill_chunk_tokens);
    p.cpu=r.cpu;p.context=r.context;p.prefix=r.prefix;p.kv_dtype=r.kv_dtype;p.concurrency_max=8;return p;
}
}
TEST(MtpAutomaticIntegration, SelectedKIsReservedBeforeKvAndConfigurePreservesRequestedAuto) {
    AutoEnv context("DENSECORE_MAX_SEQ_LEN","16"),concurrency("DENSECORE_MAX_NUM_SEQS","4"),memory("DENSECORE_KV_AVAILABLE_MB_HINT","2048");
    auto model=AutomaticTinyModel();
    const auto plain=ComputeSafeKVCacheConfig(model.get(),GGML_TYPE_F16);
    std::vector<QualificationProfile> profiles{AutoProfile(plain)};
    int hashes=0;LazyModelIdentity identity([&]{++hashes;return Identity();});
    AutomaticSpeculationContext automatic{&profiles,&identity,AutoRuntime};
    auto selected=ComputeSafeKVCacheConfig(model.get(),GGML_TYPE_F16,DENSECORE_SPEC_AUTO,2,&automatic);
    ASSERT_EQ(selected.speculation_selected,DENSECORE_SPEC_MTP);
    EXPECT_EQ(selected.speculation_requested,DENSECORE_SPEC_AUTO);EXPECT_EQ(selected.speculation_selected_k,1);
    EXPECT_EQ(selected.max_num_seqs,plain.max_num_seqs);EXPECT_EQ(hashes,1);
    EXPECT_NE(selected.speculation_reason.find(profiles[0].id),std::string::npos);
    EXPECT_EQ(selected.memory_plan.mtp_checkpoint_bytes,CalculateSpeculationMemory(*model,16,4,1).Total());
    EngineState state;auto entry=std::make_unique<ModelEntry>();entry->model=std::move(model);entry->kv_config=selected;
    entry->resource_lease=std::make_unique<densecore::runtime::ResourcePlanLease>(selected.memory_plan);
    entry->kv_cache.reset(InitPagedKVCache(entry->model.get(),4,16,selected.effective_cache_type,-1));
    ASSERT_TRUE(entry->kv_cache);state.default_model_id="default";state.models.emplace("default",std::move(entry));
    std::string error;ASSERT_TRUE(ConfigureSpeculation(&state,DENSECORE_SPEC_AUTO,2,&error))<<error;
    EXPECT_EQ(state.speculation_requested.load(),DENSECORE_SPEC_AUTO);EXPECT_EQ(state.speculation_selected.load(),DENSECORE_SPEC_MTP);
    EXPECT_EQ(state.spec_draft_n_max.load(),1);EXPECT_EQ(state.speculation_reserved_bytes,selected.memory_plan.mtp_checkpoint_bytes);
    EXPECT_TRUE(state.models["default"]->mtp_target_checkpoints);
    EXPECT_NE(state.speculation_reason.find(profiles[0].id),std::string::npos);
}
TEST(MtpAutomaticIntegration, EmptyRegistryAndWrongPrefillPolicySkipIdentity) {
    AutoEnv context("DENSECORE_MAX_SEQ_LEN","16"),concurrency("DENSECORE_MAX_NUM_SEQS","4"),memory("DENSECORE_KV_AVAILABLE_MB_HINT","2048");
    auto model=AutomaticTinyModel();int captures=0,hashes=0;
    LazyModelIdentity identity([&]{++hashes;return Identity();});std::vector<QualificationProfile> profiles;
    AutomaticSpeculationContext automatic{&profiles,&identity,[&](int c,int n,ggml_type t){++captures;return AutoRuntime(c,n,t);}};
    auto selected=ComputeSafeKVCacheConfig(model.get(),GGML_TYPE_F16,DENSECORE_SPEC_AUTO,2,&automatic);
    EXPECT_EQ(selected.speculation_selected,DENSECORE_SPEC_NONE);EXPECT_EQ(captures,0);EXPECT_EQ(hashes,0);
    profiles={AutoProfile(selected)};profiles[0].cpu.runtime_policy_sha256=Sha256Hex("different actual prefill chunk");
    selected=ComputeSafeKVCacheConfig(model.get(),GGML_TYPE_F16,DENSECORE_SPEC_AUTO,2,&automatic);
    EXPECT_EQ(selected.speculation_selected,DENSECORE_SPEC_NONE);EXPECT_GT(captures,0);EXPECT_EQ(hashes,0);
}
TEST(MtpAutomaticIntegration, MtpCostFailureKeepsSameConcurrencyAndExplicitNeverFallsBack) {
    AutoEnv context("DENSECORE_MAX_SEQ_LEN","16"),concurrency("DENSECORE_MAX_NUM_SEQS","8"),memory("DENSECORE_KV_AVAILABLE_MB_HINT","2048");
    auto model=AutomaticTinyModel();model->hparams.n_vocab=5000000; // Cost-only output rows; never execute this graph.
    const auto plain=ComputeSafeKVCacheConfig(model.get(),GGML_TYPE_F16);
    std::vector<QualificationProfile> profiles{AutoProfile(plain,2)};LazyModelIdentity identity([]{return Identity();});
    AutomaticSpeculationContext automatic{&profiles,&identity,AutoRuntime};
    const auto selected=ComputeSafeKVCacheConfig(model.get(),GGML_TYPE_F16,DENSECORE_SPEC_AUTO,2,&automatic);
    EXPECT_EQ(selected.max_num_seqs,plain.max_num_seqs);EXPECT_EQ(selected.speculation_selected,DENSECORE_SPEC_NONE);
    EXPECT_EQ(selected.memory_plan.mtp_checkpoint_bytes,0u);
    EXPECT_THROW(ComputeSafeKVCacheConfig(model.get(),GGML_TYPE_F16,DENSECORE_SPEC_MTP,2),densecore::OutOfMemoryException);
}
TEST(MtpAutomaticIntegration, WeightMutationRequiresNoneAndRejectsBeforeTensorLookup) {
    EngineState state;auto entry=std::make_unique<ModelEntry>();entry->model=AutomaticTinyModel();
    auto* weights=static_cast<float*>(entry->model->tok_embeddings->data);
    state.default_model_id="default";state.models.emplace("default",std::move(entry));
    float values[4]={3,4,5,6};DenseCoreTensor tensor{values,{2,2,0,0},2,0};
    state.speculation_selected.store(DENSECORE_SPEC_NONE);
    ASSERT_EQ(DenseCoreSetWeight(&state,"model.embed_tokens.weight",&tensor),DENSECORE_STATUS_OK);EXPECT_EQ(weights[0],3);
    for(int requested:{DENSECORE_SPEC_AUTO,DENSECORE_SPEC_MTP}) {
        state.speculation_requested.store(requested);state.speculation_selected.store(DENSECORE_SPEC_MTP);values[0]=9;
        EXPECT_EQ(DenseCoreSetWeight(&state,"model.embed_tokens.weight",&tensor),DENSECORE_STATUS_UNSUPPORTED_OPERATION);
        EXPECT_EQ(DenseCoreSetWeight(&state,"nonexistent.weight",&tensor),DENSECORE_STATUS_UNSUPPORTED_OPERATION);EXPECT_EQ(weights[0],3);
    }
}

TEST(MtpAutomaticIntegration, MissingPreKvReservationAndInvalidPreselectionFailClosed) {
    EngineState state;auto entry=std::make_unique<ModelEntry>();entry->model=AutomaticTinyModel();entry->kv_cache=std::make_unique<PagedKVCache>();
    auto& config=entry->kv_config;config.max_seq_len=16;config.max_num_seqs=1;config.speculation_preselected=true;
    config.speculation_requested=DENSECORE_SPEC_AUTO;config.speculation_requested_upper=2;
    config.speculation_selected=DENSECORE_SPEC_MTP;config.speculation_selected_k=1;
    config.speculation_profile="injected qualified profile";config.memory_plan.available_bytes=1024ULL*1024*1024;
    config.memory_plan.resource_capacity_bytes=config.memory_plan.available_bytes;config.memory_plan.kv_bytes=1024;
    entry->resource_lease=std::make_unique<densecore::runtime::ResourcePlanLease>(config.memory_plan);
    state.default_model_id="default";state.models.emplace("default",std::move(entry));std::string error;
    EXPECT_FALSE(ConfigureSpeculation(&state,DENSECORE_SPEC_AUTO,2,&error));EXPECT_NE(error.find("before primary KV"),std::string::npos);
    config.speculation_profile.clear();EXPECT_FALSE(ConfigureSpeculation(&state,DENSECORE_SPEC_AUTO,2,&error));
    EXPECT_NE(error.find("preselected qualification"),std::string::npos);
    config.speculation_profile="injected qualified profile";config.speculation_selected_k=3;
    EXPECT_FALSE(ConfigureSpeculation(&state,DENSECORE_SPEC_AUTO,2,&error));EXPECT_NE(error.find("preselected qualification"),std::string::npos);
    config.speculation_selected_k=1;
    EXPECT_FALSE(ConfigureSpeculation(&state,DENSECORE_SPEC_AUTO,3,&error));EXPECT_NE(error.find("does not match pre-KV"),std::string::npos);
    EXPECT_EQ(state.speculation_selected.load(),DENSECORE_SPEC_NONE);
}
