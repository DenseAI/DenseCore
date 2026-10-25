#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <limits>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>
#include "densecore/exceptions.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "densecore/models/model_types.h"
#include "densecore/runtime/inference.h"
#include "kernels/q4k_repacked_gemv.h"
#include "llm/config/runtime_config.h"
#include "llm/matmul/userdata.h"
#include "llm/runtime/work_context.h"
#include "llm/runtime/work_context_internal.h"

namespace densecore::testing {
ggml_tensor* SmartMulMatWithPhaseTest(ggml_context*,ggml_tensor*,ggml_tensor*,TransformerModel*,int);
bool IsQwen35SsmQkvQ5KProjectionForTest(const TransformerModel*,const ggml_tensor*,const ggml_tensor*);
bool IsQwen35OrdinaryDenseQ4KProjectionForTest(const TransformerModel*,const ggml_tensor*,const ggml_tensor*);
}
namespace {
std::unique_ptr<TransformerModel> MakePoint8B() {
    auto model=std::make_unique<TransformerModel>();
    model->variant=ModelVariant::QWEN35;
    model->arch_flags.is_hybrid_ssm=true;
    model->hparams.n_embd=1024;model->hparams.n_layer=24;model->hparams.n_experts=0;
    model->layers.resize(24);model->hybrid_layer_is_ssm.assign(24,1);
    model->hybrid_layer_is_ssm[3]=0;
    return model;
}
struct CacheScope {
    size_t floor=densecore::kernels::Q4KRepackedGemvRuntimeCacheBudgetFloorBytes();
    CacheScope(){densecore::kernels::RetireRepackedWeightCaches();densecore::kernels::Q4KRepackedGemvRaiseRuntimeCacheBudgetFloor(64*1024*1024);}
    ~CacheScope(){densecore::kernels::RetireRepackedWeightCaches();if(floor)densecore::kernels::Q4KRepackedGemvSetRuntimeCacheBudgetFloor(floor);else densecore::kernels::Q4KRepackedGemvResetRuntimeCacheBudgetFloorForTest();}
};
struct RunOptions {
    InferenceExecutionPhase phase=InferenceExecutionPhase::Decode;
    densecore::env::RuntimeToggleMode mode=densecore::env::RuntimeToggleMode::On;
    bool lora=false,reference=false,legacy=false,required=false,fused_ssm=false;
};
std::vector<float> RunProjection(TransformerModel* model,ggml_tensor* weight,const std::vector<float>& values,
                       int columns,int first,int threads,bool padded,RunOptions options={}) {
    const int reduction=static_cast<int>(weight->ne[0]),outputs=static_cast<int>(weight->ne[1]);
    std::unique_ptr<InferenceWorkContext,decltype(&DestroyInferenceWorkContext)> work(CreateInferenceWorkContext(),DestroyInferenceWorkContext);
    struct Restore{InferenceWorkContext* previous;~Restore(){SetCurrentWorkContext(previous);}} restore{GetCurrentWorkContext()};
    SetCurrentWorkContext(work.get());ResetInferenceWorkContext(work.get());work->model_variant=ModelVariant::QWEN35;
    BatchSpec batch{};batch.num_seqs=columns;batch.tokens.assign(columns,0);
    for(int c=0;c<columns;++c){batch.seq_id.push_back(c);batch.pos.push_back(51+c);batch.n_past.push_back(51+c);}
    if(options.lora)batch.lora_map[nullptr]={0};
    densecore::llm::config::FastPathRuntimeConfig config;config.q4k_repacked_gemv=options.mode;
    config.q4k_repacked_gemv_probe=false;config.q4k_repacked_gemv_disable_on_thrash=false;
    InferenceDependencies deps;deps.fast_path_config=&config;batch.deps=&deps;
    SetCurrentBatch(&batch);SetCurrentExecutionPhase(options.phase);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({2*1024*1024,nullptr,false}),ggml_free);
    auto* storage=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,reduction+(padded?16:0),columns);
    auto* input=padded?ggml_view_2d(ctx.get(),storage,reduction,columns,storage->nb[1],0):storage;
    for(int c=0;c<columns;++c)std::memcpy(static_cast<char*>(input->data)+c*input->nb[1],values.data()+(first+c)*reduction,reduction*sizeof(float));
    auto* output=densecore::testing::SmartMulMatWithPhaseTest(ctx.get(),weight,input,model,static_cast<int>(options.phase));
    struct Params{ggml_custom_op_t fun;int tasks;void* userdata;} params{};
    if(output->op!=GGML_OP_CUSTOM)throw std::runtime_error("Qwen ordinary fixture lost maintained callback");
    std::memcpy(&params,output->op_params,sizeof(params));
    if(columns==1)EXPECT_EQ(params.fun,cb_gemv_custom); // Protect the actual old M1 primitive.
    if(params.fun==cb_gemv_batched_custom){
        auto* ud=static_cast<GemvBatchedUserData*>(params.userdata);
        EXPECT_TRUE(ud->qwen35_dense_canonical_q4k_columns);
        if(options.legacy)ud->qwen35_dense_canonical_q4k_columns=false;
        ud->force_reference_scalar=options.reference;
        ud->require_q4k_true_batched=options.required;
        ud->qwen36_ssm_q8_repacked_batched=options.fused_ssm;
    }
    params.tasks=threads;std::memcpy(output->op_params,&params,sizeof(params));
    auto* graph=ggml_new_graph_custom(ctx.get(),128,false);ggml_build_forward_expand(graph,output);
    EXPECT_EQ(ggml_graph_compute_with_ctx(ctx.get(),graph,threads),GGML_STATUS_SUCCESS);
    if(columns>1&&(options.phase!=InferenceExecutionPhase::Decode||options.lora||options.reference||options.legacy||options.required||options.fused_ssm))
        EXPECT_EQ(GetInferenceWorkContextProfile(work.get())->gemv_custom_total_ops.load(),0);
    const auto packed_ops=GetInferenceWorkContextProfile(work.get())->q4k_repacked_gemv_used_ops.load();
    if(options.mode==densecore::env::RuntimeToggleMode::Off)EXPECT_EQ(packed_ops,0);
    const bool ordinary_decode=options.phase==InferenceExecutionPhase::Decode && !options.lora &&
        !options.reference && !options.legacy && !options.required && !options.fused_ssm;
    if(ordinary_decode && options.mode!=densecore::env::RuntimeToggleMode::Off &&
       weight->nb[1]==ggml_row_size(GGML_TYPE_Q4_K,reduction) &&
       densecore::kernels::Q4KRealPackedGemvKernelAvailable() &&
       densecore::kernels::Q4KRepackedGemvCacheLimitBytes()>=ggml_nbytes(weight))
        EXPECT_GT(packed_ops,0); // Real execution evidence, not mode spelling alone.
    std::vector<float> result(outputs*columns);
    for(int c=0;c<columns;++c)std::memcpy(result.data()+c*outputs,static_cast<char*>(output->data)+c*output->nb[1],outputs*sizeof(float));
    return result;
}
}

TEST(Qwen35CanonicalMatmulTest, ActualOrdinaryShapesPreserveOldM1BitsAndExclusions) {
    CacheScope cache;
    for(bool down:{false,true})for(bool padded_weight:{false,true}) {
        SCOPED_TRACE(down);
        SCOPED_TRACE(padded_weight);
        const int reduction=down?3584:1024,outputs=down?1024:3584;
        std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({8*1024*1024,nullptr,false}),ggml_free);
        ASSERT_NE(ctx,nullptr);
        auto* storage=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_Q4_K,reduction,outputs*(padded_weight?2:1));
        std::memset(storage->data,0,ggml_nbytes(storage));
        auto* weight=padded_weight?ggml_view_2d(ctx.get(),storage,reduction,outputs,2*storage->nb[1],0):storage;
        ggml_set_name(weight,down?"blk.3.ffn_down.weight":"blk.0.ffn_gate.weight");
        std::vector<float> row(reduction),values(8*reduction);
        for(int out=0;out<outputs;++out){
            for(int i=0;i<reduction;++i)row[i]=.31f*std::sin((i+7)*(out+3)*.0017f)+.125f*std::cos((i%37-out%19)*.11f);
            ggml_quantize_chunk(GGML_TYPE_Q4_K,row.data(),static_cast<char*>(weight->data)+out*weight->nb[1],0,1,reduction,nullptr);
        }
        for(int c=0;c<8;++c)for(int i=0;i<reduction;++i)values[c*reduction+i]=std::sin((i+1)*.017f+c*.31f)+.37f*std::cos((i+7)*.031f-c*.13f);
        auto model=MakePoint8B();model->layers[down?3:0].Set(down?model_keys::kFfnDown:model_keys::kFfnGate,weight);
        for(auto mode:{densecore::env::RuntimeToggleMode::On,densecore::env::RuntimeToggleMode::Off,densecore::env::RuntimeToggleMode::Auto}){
            SCOPED_TRACE(static_cast<int>(mode));
            RunOptions options;options.mode=mode;
            std::vector<std::vector<float>> baseline;
            for(int c=0;c<8;++c)baseline.push_back(RunProjection(model.get(),weight,values,1,c,1,false,options));
            for(int threads:{1,4,5,8})for(bool padded:{false,true})for(int columns:{1,2,3,4,8}){
                SCOPED_TRACE(threads);
                SCOPED_TRACE(padded);
                SCOPED_TRACE(columns);
                const auto result=RunProjection(model.get(),weight,values,columns,0,threads,padded,options);
                for(int c=0;c<columns;++c)EXPECT_EQ(std::memcmp(result.data()+c*outputs,baseline[c].data(),outputs*sizeof(float)),0)<<"column="<<c;
            }
        }
        for(int exclusion=0;exclusion<6;++exclusion){
            RunOptions options;
            if(exclusion==0)options.phase=InferenceExecutionPhase::Prefill;
            if(exclusion==1)options.phase=InferenceExecutionPhase::Unknown;
            options.lora=exclusion==2;options.reference=exclusion==3;options.required=exclusion==4;options.fused_ssm=exclusion==5;
            const auto current=RunProjection(model.get(),weight,values,4,0,5,true,options);
            options.legacy=true;
            const auto old=RunProjection(model.get(),weight,values,4,0,5,true,options);
            EXPECT_EQ(std::memcmp(current.data(),old.data(),old.size()*sizeof(float)),0)<<"exclusion="<<exclusion;
        }
        densecore::kernels::RetireRepackedWeightCaches();densecore::kernels::Q4KRepackedGemvSetRuntimeCacheBudgetFloor(0);
        ASSERT_EQ(densecore::kernels::Q4KRepackedGemvCacheLimitBytes(),0u); // Reject a manual cache override in this control.
        const auto refused=RunProjection(model.get(),weight,values,4,0,5,true);
        RunOptions raw;raw.mode=densecore::env::RuntimeToggleMode::Off;
        for(int c=0;c<4;++c){const auto old=RunProjection(model.get(),weight,values,1,c,5,true,raw);EXPECT_EQ(std::memcmp(refused.data()+c*outputs,old.data(),outputs*sizeof(float)),0);}
        densecore::kernels::Q4KRepackedGemvRaiseRuntimeCacheBudgetFloor(64*1024*1024);
    }
}

TEST(Qwen35CanonicalMatmulTest, AdmissionRequiresActualOrdinaryOwnershipAndExcludesRecurrentNames) {
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({512*1024,nullptr,true}),ggml_free);
    ASSERT_NE(ctx,nullptr);
    auto model=MakePoint8B();auto* input=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,1024,4);
    auto* weight=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_Q4_K,1024,3584);
    auto* foreign=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_Q4_K,1024,3584);
    ggml_set_name(weight,"blk.0.ffn_gate.weight");ggml_set_name(foreign,weight->name);
    model->layers[0].Set(model_keys::kFfnGate,weight);
    const auto admitted=[&](ggml_tensor* tensor){return densecore::testing::IsQwen35OrdinaryDenseQ4KProjectionForTest(model.get(),tensor,input);};
    EXPECT_TRUE(admitted(weight));EXPECT_FALSE(admitted(foreign));
    model->layers[0].Set(model_keys::kAttnQWeight,foreign);ggml_set_name(foreign,"blk.0.attn_q.weight");
    EXPECT_FALSE(admitted(foreign)); // Even an attention-key alias on an SSM layer is excluded.
    model->layers[3].Set(model_keys::kAttnQWeight,foreign);ggml_set_name(foreign,"blk.3.attn_q.weight");
    EXPECT_TRUE(admitted(foreign));model->layers[3].Set(model_keys::kAttnQWeight,nullptr);
    for(const char* name:{"blk.0.attn_qkv.weight","qkv_mixed.weight","linear_attn.in_proj_qkv.weight","in_proj_z.weight","blk.0.z.weight","ssm_out.weight","linear_attn.out_proj.weight","blk.0.attn_gate.weight","blk.0.ffn_gate_exps.weight"}){
        SCOPED_TRACE(name);ggml_set_name(weight,name);EXPECT_FALSE(admitted(weight));
    }
    ggml_set_name(weight,"blk.0.ffn_gate.weight");model->output=weight;EXPECT_FALSE(admitted(weight));model->output=nullptr;
    model->hparams.n_experts=2;EXPECT_FALSE(admitted(weight));model->hparams.n_experts=0;
    model->variant=ModelVariant::QWEN36;EXPECT_FALSE(admitted(weight));model->variant=ModelVariant::QWEN35;
    model->hparams.n_embd=2048;EXPECT_FALSE(admitted(weight));model->hparams.n_embd=1024;
    auto* q6=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_Q6_K,1024,3584);model->layers[0].Set(model_keys::kFfnGate,q6);ggml_set_name(q6,"blk.0.ffn_gate.weight");EXPECT_FALSE(admitted(q6));
    model->layers[0].Set(model_keys::kFfnGate,weight);
    auto* half_input=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F16,1024,4);
    EXPECT_FALSE(densecore::testing::IsQwen35OrdinaryDenseQ4KProjectionForTest(model.get(),weight,half_input));
}

namespace {
constexpr int kSsmQ5Input = 1024, kSsmQ5Output = 6144;
struct SsmQ5Options {
    InferenceExecutionPhase phase=InferenceExecutionPhase::Decode;
    bool lora=false, reference=false, legacy=false, required=false;
    bool fused_repacked=false, fused_direct=false, amx=false;
    bool no_context=false, no_batch=false, mixed_tokens=false, wrong_context_model=false;
};
std::vector<float> RunSsmQ5(TransformerModel* model, ggml_tensor* weight,
                            const std::vector<float>& values, int columns, int first, int threads, bool padded, SsmQ5Options options={}) {
    std::unique_ptr<InferenceWorkContext,decltype(&DestroyInferenceWorkContext)> work(
        CreateInferenceWorkContext(), DestroyInferenceWorkContext);
    struct Restore { InferenceWorkContext* previous; ~Restore(){SetCurrentWorkContext(previous);} } restore{GetCurrentWorkContext()};
    SetCurrentWorkContext(work.get()); ResetInferenceWorkContext(work.get()); work->model_variant=ModelVariant::QWEN35;
    BatchSpec batch{}; batch.num_seqs=columns; batch.tokens.assign(columns,0);
    for(int c=0;c<columns;++c) {batch.seq_id.push_back(c);batch.pos.push_back(37+c);batch.n_past.push_back(37+c);}
    densecore::llm::config::FastPathRuntimeConfig config;
    InferenceDependencies deps;deps.fast_path_config=&config;batch.deps=&deps;
    SetCurrentBatch(&batch);SetCurrentExecutionPhase(InferenceExecutionPhase::Decode);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({2*1024*1024,nullptr,false}),ggml_free);
    auto* storage=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,kSsmQ5Input+(padded?16:0),columns);
    std::fill_n(static_cast<float*>(storage->data),ggml_nelements(storage),std::numeric_limits<float>::quiet_NaN());
    auto* input=padded?ggml_view_2d(ctx.get(),storage,kSsmQ5Input,columns,storage->nb[1],0):storage;
    for(int c=0;c<columns;++c)
        std::memcpy(static_cast<char*>(input->data)+c*input->nb[1],values.data()+(first+c)*kSsmQ5Input,kSsmQ5Input*sizeof(float));
    auto* output=densecore::testing::SmartMulMatWithPhaseTest(ctx.get(),weight,input,model,static_cast<int>(InferenceExecutionPhase::Decode));
    struct Params {ggml_custom_op_t fun;int tasks;void* userdata;} params{};
    if(output->op!=GGML_OP_CUSTOM)throw std::runtime_error("SSM Q5 fixture lost native custom callback");
    std::memcpy(&params,output->op_params,sizeof(params));
    if(columns==1)EXPECT_EQ(params.fun,cb_gemv_custom); // Protect the untouched old singleton selector.
    if(params.fun==cb_gemv_batched_custom) {
        auto* ud=static_cast<GemvBatchedUserData*>(params.userdata);
        EXPECT_TRUE(ud->qwen35_ssm_qkv_canonical_q5k_columns);
        if(options.legacy)ud->qwen35_ssm_qkv_canonical_q5k_columns=false;
        ud->force_reference_scalar=options.reference;
        ud->require_q4k_true_batched=options.required;
        ud->qwen36_ssm_q8_repacked_batched=options.fused_repacked;
        ud->qwen36_ssm_q8_direct_batched=options.fused_direct;
        if(options.amx)ud->qwen38_q4k_8x8_weight=weight;
        if(options.no_context)ud->work_ctx=nullptr;
    }
    if(options.lora)batch.lora_map[nullptr]={0};
    if(options.mixed_tokens)batch.tokens.push_back(0);
    if(options.no_batch)BindInferenceWorkContextBatch(work.get(),nullptr);
    if(options.wrong_context_model)work->model_variant=ModelVariant::QWEN36;
    SetCurrentExecutionPhase(options.phase);
    const bool excluded=options.legacy || options.reference || options.required || options.lora ||
        options.fused_repacked || options.fused_direct || options.amx || options.no_context || options.no_batch ||
        options.mixed_tokens || options.wrong_context_model || options.phase!=InferenceExecutionPhase::Decode;
    if(options.required) {
        // The original Q5 path explicitly rejects a Q4-only required kernel.
        // Direct invocation keeps that exception on this test's C++ stack.
        EXPECT_THROW(params.fun(output,0,1,params.userdata),densecore::InvalidArgumentException);
        EXPECT_EQ(GetInferenceWorkContextProfile(work.get())->gemv_custom_total_ops.load(),0);
        return {};
    }
    params.tasks=threads;std::memcpy(output->op_params,&params,sizeof(params));
    auto* graph=ggml_new_graph_custom(ctx.get(),128,false);ggml_build_forward_expand(graph,output);
    EXPECT_EQ(ggml_graph_compute_with_ctx(ctx.get(),graph,threads),GGML_STATUS_SUCCESS);
    if(columns>1) {
        const auto calls=GetInferenceWorkContextProfile(work.get())->gemv_custom_total_ops.load();
        if(excluded)EXPECT_EQ(calls,0);
        else EXPECT_EQ(calls,static_cast<uint64_t>(columns));
    }
    std::vector<float> result(kSsmQ5Output*columns);
    for(int c=0;c<columns;++c)
        std::memcpy(result.data()+c*kSsmQ5Output,static_cast<char*>(output->data)+c*output->nb[1],kSsmQ5Output*sizeof(float));
    return result;
}
}

TEST(Qwen35CanonicalMatmulTest, SsmQ5ProjectionMatchesInitializedCpuPrimitiveAndOwnedOldM1Bits) {
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({16*1024*1024,nullptr,false}),ggml_free);
    ASSERT_NE(ctx,nullptr); // Base GGML initialization must precede any reference conversion.
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(),ggml_backend_free);
    ASSERT_NE(cpu,nullptr); // The CPU-traits oracle requires CPU initialization, not just base GGML.
    const auto* q5=ggml_get_type_traits_cpu(GGML_TYPE_Q5_K);
    const auto* q8=ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    ASSERT_NE(q5,nullptr);ASSERT_NE(q8,nullptr);ASSERT_NE(q5->vec_dot,nullptr);ASSERT_NE(q8->from_float,nullptr);
    ASSERT_EQ(q5->vec_dot_type,GGML_TYPE_Q8_K);
    ASSERT_EQ(ggml_row_size(GGML_TYPE_Q8_K,kSsmQ5Input),1168u);
    std::vector<float> values(8*kSsmQ5Input),row(kSsmQ5Input);
    for(int c=0;c<8;++c)for(int i=0;i<kSsmQ5Input;++i)
        values[c*kSsmQ5Input+i]=std::sin((i+1)*.017f+c*.31f)+.37f*std::cos((i+7)*.031f-c*.13f);
    for(bool padded_weight:{false,true}) {
        SCOPED_TRACE(padded_weight);
        auto* storage=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_Q5_K,kSsmQ5Input,kSsmQ5Output*(padded_weight?2:1));
        std::memset(storage->data,0xA5,ggml_nbytes(storage)); // Poison gaps that are not semantic Q5 rows.
        auto* weight=padded_weight?ggml_view_2d(ctx.get(),storage,kSsmQ5Input,kSsmQ5Output,2*storage->nb[1],0):storage;
        ggml_set_name(weight,"blk.0.attn_qkv.weight");
        for(int out=0;out<kSsmQ5Output;++out) {
            for(int i=0;i<kSsmQ5Input;++i)
                row[i]=.31f*std::sin((i+7)*(out+3)*.0017f)+.125f*std::cos((i%37-out%19)*.11f);
            ggml_quantize_chunk(GGML_TYPE_Q5_K,row.data(),static_cast<char*>(weight->data)+out*weight->nb[1],0,1,kSsmQ5Input,nullptr);
        }
        auto model=MakePoint8B();model->layers[0].Set(model_keys::kAttnQkvWeight,weight);
        std::vector<std::vector<float>> expected;
        for(int c=0;c<8;++c) {
            alignas(64) std::array<unsigned char,1168> quant{};
            q8->from_float(values.data()+c*kSsmQ5Input,quant.data(),kSsmQ5Input);
            std::vector<float> reference(kSsmQ5Output);
            for(int out=0;out<kSsmQ5Output;++out)
                q5->vec_dot(kSsmQ5Input,&reference[out],0,static_cast<char*>(weight->data)+out*weight->nb[1],0,quant.data(),0,1);
            ASSERT_TRUE(std::all_of(reference.begin(),reference.end(),[](float x){return std::isfinite(x);}));
            ASSERT_GT(std::abs(*std::max_element(reference.begin(),reference.end(),[](float a,float b){return std::abs(a)<std::abs(b);})),1e-3f);
            const auto singleton=RunSsmQ5(model.get(),weight,values,1,c,1,false);
            ASSERT_EQ(std::memcmp(singleton.data(),reference.data(),kSsmQ5Output*sizeof(float)),0)
                <<"old M1 must first equal initialized independent CPU nrc1; column="<<c;
            expected.push_back(std::move(reference));
        }
        for(int threads:{1,4,5,8})for(bool padded_input:{false,true})for(int columns:{1,2,3,4,8}) {
            SCOPED_TRACE(threads);
            SCOPED_TRACE(padded_input);
            SCOPED_TRACE(columns);
            const auto actual=RunSsmQ5(model.get(),weight,values,columns,0,threads,padded_input);
            for(int c=0;c<columns;++c)
                EXPECT_EQ(std::memcmp(actual.data()+c*kSsmQ5Output,expected[c].data(),kSsmQ5Output*sizeof(float)),0)<<"column="<<c;
        }
        for(int exclusion=0;exclusion<13;++exclusion) {
            SCOPED_TRACE(exclusion);
            SsmQ5Options options;
            if(exclusion==0)options.phase=InferenceExecutionPhase::Prefill;
            if(exclusion==1)options.phase=InferenceExecutionPhase::Unknown;
            options.lora=exclusion==2;options.reference=exclusion==3;options.required=exclusion==4;
            options.fused_repacked=exclusion==5;options.fused_direct=exclusion==6;options.amx=exclusion==7;
            options.no_context=exclusion==8;options.no_batch=exclusion==9;options.mixed_tokens=exclusion==10;
            options.wrong_context_model=exclusion==11;options.legacy=exclusion==12;
            const auto current=RunSsmQ5(model.get(),weight,values,4,0,5,true,options);
            options.legacy=true;
            const auto previous=RunSsmQ5(model.get(),weight,values,4,0,5,true,options);
            ASSERT_EQ(current.size(),previous.size());
            if(!current.empty())EXPECT_EQ(std::memcmp(current.data(),previous.data(),current.size()*sizeof(float)),0);
        }
    }
}

TEST(Qwen35CanonicalMatmulTest, SsmQ5AdmissionRequiresExactRegisteredPoint8NonMoeQkv) {
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({1024*1024,nullptr,true}),ggml_free);
    ASSERT_NE(ctx,nullptr);
    auto model=MakePoint8B();
    auto* input=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,1024,4);
    auto* weight=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_Q5_K,1024,6144);
    auto* foreign=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_Q5_K,1024,6144);
    ggml_set_name(weight,"blk.0.attn_qkv.weight");ggml_set_name(foreign,weight->name);
    model->layers[0].Set(model_keys::kAttnQkvWeight,weight);
    const auto admitted=[&](ggml_tensor* w,ggml_tensor* x=nullptr){
        return densecore::testing::IsQwen35SsmQkvQ5KProjectionForTest(model.get(),w,x?x:input);
    };
    EXPECT_TRUE(admitted(weight));EXPECT_FALSE(admitted(foreign));
    model->layers[0].Set(model_keys::kAttnQkvWeight,nullptr);
    model->layers[3].Set(model_keys::kAttnQkvWeight,weight);EXPECT_FALSE(admitted(weight)); // Attention layer.
    model->layers[3].Set(model_keys::kAttnQkvWeight,nullptr);
    model->layers[0].Set(model_keys::kAttnQWeight,weight);EXPECT_FALSE(admitted(weight));
    model->layers[0].Set(model_keys::kAttnQWeight,nullptr);
    model->layers[0].Set(model_keys::kAttnGate,weight);EXPECT_FALSE(admitted(weight));
    model->layers[0].Set(model_keys::kAttnGate,nullptr);
    model->layers[0].Set(model_keys::kSSMOut,weight);EXPECT_FALSE(admitted(weight));
    model->layers[0].Set(model_keys::kSSMOut,nullptr);
    model->layers[0].Set(model_keys::kAttnQkvWeight,weight);
    model->output=weight;EXPECT_FALSE(admitted(weight));model->output=nullptr;
    for(auto variant:{ModelVariant::QWEN36,ModelVariant::QWEN38,ModelVariant::GEMMA4}) {
        model->variant=variant;EXPECT_FALSE(admitted(weight));
    }
    model->variant=ModelVariant::QWEN35;
    model->hparams.n_experts=2;EXPECT_FALSE(admitted(weight));model->hparams.n_experts=0;
    model->layers[0].is_moe=true;EXPECT_FALSE(admitted(weight));model->layers[0].is_moe=false;
    model->layers[1].experts.resize(1);EXPECT_FALSE(admitted(weight));model->layers[1].experts.clear();
    model->arch_flags.is_glm_moe=true;EXPECT_FALSE(admitted(weight));model->arch_flags.is_glm_moe=false;
    model->arch_flags.is_hybrid_ssm=false;EXPECT_FALSE(admitted(weight));model->arch_flags.is_hybrid_ssm=true;
    model->hparams.n_embd=2048;EXPECT_FALSE(admitted(weight));model->hparams.n_embd=1024;
    model->hparams.n_layer=32;EXPECT_FALSE(admitted(weight));model->hparams.n_layer=24;
    for(auto type:{GGML_TYPE_Q4_K,GGML_TYPE_Q6_K,GGML_TYPE_F32}) {
        auto* wrong=ggml_new_tensor_2d(ctx.get(),type,1024,6144);ggml_set_name(wrong,weight->name);
        model->layers[0].Set(model_keys::kAttnQkvWeight,wrong);EXPECT_FALSE(admitted(wrong));
    }
    auto* wrong_shape=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_Q5_K,1024,6136);ggml_set_name(wrong_shape,weight->name);
    model->layers[0].Set(model_keys::kAttnQkvWeight,wrong_shape);EXPECT_FALSE(admitted(wrong_shape));
    auto* fused=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_Q5_K,1024,8192);ggml_set_name(fused,"qwen35_ssm_qkv_gate_fused_proj");
    model->layers[0].Set(model_keys::kAttnQkvWeight,fused);EXPECT_FALSE(admitted(fused));
    model->layers[0].Set(model_keys::kAttnQkvWeight,weight);
    EXPECT_FALSE(admitted(weight,ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F16,1024,4)));
    EXPECT_FALSE(admitted(weight,ggml_transpose(ctx.get(),ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,4,1024))));
    EXPECT_FALSE(admitted(weight,ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,1024,4,2)));
    EXPECT_FALSE(admitted(weight,ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,1024,16)));
    EXPECT_TRUE(admitted(weight));
}

namespace {
struct SsmGateOutQ4Options {
    densecore::env::RuntimeToggleMode mode=densecore::env::RuntimeToggleMode::On;
    InferenceExecutionPhase phase=InferenceExecutionPhase::Decode;
    bool foreign=false,lora=false,reference=false,no_context=false,no_batch=false;
    bool mixed_tokens=false,mixed_seqs=false,required=false,fused=false,amx=false,metadata_excluded=false;
};
std::vector<float> RunSsmGateOutQ4(TransformerModel* model,ggml_tensor* owned,
                                  const std::vector<float>& values,int columns,int first,int threads,bool padded,
                                  SsmGateOutQ4Options options={}) {
    const int n=static_cast<int>(owned->ne[0]),k=static_cast<int>(owned->ne[1]);
    ggml_tensor lookalike=*owned;auto* weight=options.foreign?&lookalike:owned;
    std::unique_ptr<InferenceWorkContext,decltype(&DestroyInferenceWorkContext)> work(CreateInferenceWorkContext(),DestroyInferenceWorkContext);
    struct Restore{InferenceWorkContext* previous;~Restore(){SetCurrentWorkContext(previous);}}restore{GetCurrentWorkContext()};
    SetCurrentWorkContext(work.get());ResetInferenceWorkContext(work.get());work->model_variant=model->variant;
    BatchSpec batch{};batch.num_seqs=columns;batch.tokens.assign(columns,0);
    for(int c=0;c<columns;++c){batch.seq_id.push_back(c);batch.pos.push_back(37+c);batch.n_past.push_back(37+c);}
    densecore::llm::config::FastPathRuntimeConfig config;config.q4k_repacked_gemv=options.mode;
    config.q4k_repacked_gemv_probe=false;config.q4k_repacked_gemv_disable_on_thrash=false;
    InferenceDependencies deps;deps.fast_path_config=&config;batch.deps=&deps;
    SetCurrentBatch(&batch);SetCurrentExecutionPhase(InferenceExecutionPhase::Decode);
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({2*1024*1024,nullptr,false}),ggml_free);
    auto* backing=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,n+(padded?16:0),columns);
    std::fill_n(static_cast<float*>(backing->data),ggml_nelements(backing),std::numeric_limits<float>::quiet_NaN());
    auto* input=padded?ggml_view_2d(ctx.get(),backing,n,columns,backing->nb[1],0):backing;
    for(int c=0;c<columns;++c)std::memcpy(static_cast<char*>(input->data)+c*input->nb[1],values.data()+(first+c)*n,n*sizeof(float));
    auto* output=densecore::testing::SmartMulMatWithPhaseTest(ctx.get(),weight,input,model,static_cast<int>(InferenceExecutionPhase::Decode));
    struct Params{ggml_custom_op_t fun;int tasks;void* userdata;}params{};
    if(output->op!=GGML_OP_CUSTOM)throw std::runtime_error("SSM gate/out Q4 fixture lost maintained custom callback");
    std::memcpy(&params,output->op_params,sizeof(params));
    if(columns==1)EXPECT_EQ(params.fun,cb_gemv_custom);
    if(params.fun==cb_gemv_batched_custom){
        auto* ud=static_cast<GemvBatchedUserData*>(params.userdata);
        EXPECT_FALSE(ud->qwen35_dense_canonical_q4k_columns); // These are SSM roles, never ordinary DenseProjection.
        EXPECT_FALSE(ud->qwen35_ssm_qkv_canonical_q5k_columns);
        ud->force_reference_scalar=options.reference;ud->require_q4k_true_batched=options.required;
        ud->qwen36_ssm_q8_repacked_batched=options.fused;
        if(options.amx)ud->qwen38_q4k_8x8_weight=weight;
        if(options.no_context)ud->work_ctx=nullptr;
    }
    if(options.lora)batch.lora_map[nullptr]={0};
    if(options.mixed_tokens)batch.tokens.push_back(0);
    if(options.mixed_seqs)batch.num_seqs=columns-1;
    if(options.no_batch)BindInferenceWorkContextBatch(work.get(),nullptr);
    SetCurrentExecutionPhase(options.phase);
    params.tasks=threads;std::memcpy(output->op_params,&params,sizeof(params));
    auto* graph=ggml_new_graph_custom(ctx.get(),128,false);ggml_build_forward_expand(graph,output);
    EXPECT_EQ(ggml_graph_compute_with_ctx(ctx.get(),graph,threads),GGML_STATUS_SUCCESS);
    const auto* profile=GetInferenceWorkContextProfile(work.get());
    if(columns==1){
        const bool contiguous=weight->nb[1]==ggml_row_size(GGML_TYPE_Q4_K,n);
        if(options.mode==densecore::env::RuntimeToggleMode::Off||!contiguous||
           densecore::kernels::Q4KRepackedGemvCacheLimitBytes()==0)
            EXPECT_EQ(profile->q4k_repacked_gemv_used_ops.load(),0);
        else if(densecore::kernels::Q4KRealPackedGemvKernelAvailable())
            EXPECT_GT(profile->q4k_repacked_gemv_used_ops.load(),0); // Funded On/Auto must actually pack old M1.
    }
    const bool excluded=options.foreign||options.lora||options.reference||options.required||options.fused||options.amx||
        options.no_context||options.no_batch||options.mixed_tokens||options.mixed_seqs||options.metadata_excluded||
        options.phase!=InferenceExecutionPhase::Decode;
    if(columns>1&&excluded)EXPECT_EQ(profile->gemv_custom_total_ops.load(),0);
    if(columns>1&&!excluded){
        EXPECT_EQ(profile->gemv_custom_total_ops.load(),columns);
        const bool contiguous=weight->nb[1]==ggml_row_size(GGML_TYPE_Q4_K,n);
        if(options.mode==densecore::env::RuntimeToggleMode::Off||!contiguous||
           densecore::kernels::Q4KRepackedGemvCacheLimitBytes()==0)
            EXPECT_EQ(profile->q4k_repacked_gemv_used_ops.load(),0);
        else if(densecore::kernels::Q4KRealPackedGemvKernelAvailable())
            EXPECT_GT(profile->q4k_repacked_gemv_used_ops.load(),0);
    }
    std::vector<float> result(k*columns);
    for(int c=0;c<columns;++c)std::memcpy(result.data()+c*k,static_cast<char*>(output->data)+c*output->nb[1],k*sizeof(float));
    return result;
}
}

TEST(Qwen35CanonicalMatmulTest, SsmGateOutQ4RealShapesPreserveInitializedOldM1Bits) {
    CacheScope cache;
    std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({16*1024*1024,nullptr,false}),ggml_free);
    ASSERT_NE(ctx,nullptr);
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(),ggml_backend_free);
    ASSERT_NE(cpu,nullptr);
    const auto* q4=ggml_get_type_traits_cpu(GGML_TYPE_Q4_K);const auto* q8=ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
    ASSERT_NE(q4,nullptr);ASSERT_NE(q8,nullptr);ASSERT_NE(q4->vec_dot,nullptr);ASSERT_NE(q8->from_float,nullptr);
    ASSERT_EQ(q4->vec_dot_type,GGML_TYPE_Q8_K);
    for(bool gate:{true,false})for(bool padded_weight:{false,true}) {
        SCOPED_TRACE(gate);
        SCOPED_TRACE(padded_weight);
        const int n=gate?1024:2048,k=gate?2048:1024;
        const char* key=gate?model_keys::kAttnGate:model_keys::kSSMOut;
        auto model=MakePoint8B();
        auto* storage=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_Q4_K,n,k*(padded_weight?2:1));
        std::memset(storage->data,0xA5,ggml_nbytes(storage));
        auto* weight=padded_weight?ggml_view_2d(ctx.get(),storage,n,k,2*storage->nb[1],0):storage;
        ggml_set_name(weight,gate?"blk.0.attn_gate.weight":"blk.0.ssm_out.weight");
        model->layers[0].Set(key,weight);
        std::vector<float> row(n),values(8*n);
        for(int out=0;out<k;++out){
            for(int i=0;i<n;++i)row[i]=.31f*std::sin((i+7)*(out+3)*.0017f)+.125f*std::cos((i%37-out%19)*.11f);
            ggml_quantize_chunk(GGML_TYPE_Q4_K,row.data(),static_cast<char*>(weight->data)+out*weight->nb[1],0,1,n,nullptr);
        }
        for(int c=0;c<8;++c)for(int i=0;i<n;++i)values[c*n+i]=std::sin((i+1)*.017f+c*.31f)+.37f*std::cos((i+7)*.031f-c*.13f);
        for(auto mode:{densecore::env::RuntimeToggleMode::Off,densecore::env::RuntimeToggleMode::On,densecore::env::RuntimeToggleMode::Auto}){
            SCOPED_TRACE(static_cast<int>(mode));
            ASSERT_GE(densecore::kernels::Q4KRepackedGemvCacheLimitBytes(),static_cast<size_t>(k)*ggml_row_size(GGML_TYPE_Q4_K,n));
            SsmGateOutQ4Options options;options.mode=mode;
            std::vector<std::vector<float>> baseline;
            for(int c=0;c<8;++c){
                auto singleton=RunSsmGateOutQ4(model.get(),weight,values,1,c,1,false,options);
                ASSERT_TRUE(std::all_of(singleton.begin(),singleton.end(),[](float x){return std::isfinite(x);}));
                ASSERT_GT(std::abs(*std::max_element(singleton.begin(),singleton.end(),[](float a,float b){return std::abs(a)<std::abs(b);})),1e-3f);
                if(mode==densecore::env::RuntimeToggleMode::Off){
                    alignas(64)std::array<unsigned char,2336> quant{};
                    ASSERT_LE(ggml_row_size(GGML_TYPE_Q8_K,n),quant.size());q8->from_float(values.data()+c*n,quant.data(),n);
                    std::vector<float> raw(k);
                    for(int out=0;out<k;++out)q4->vec_dot(n,&raw[out],0,static_cast<char*>(weight->data)+out*weight->nb[1],0,quant.data(),0,1);
                    ASSERT_EQ(std::memcmp(raw.data(),singleton.data(),k*sizeof(float)),0)<<"Off oldM1 must equal initialized CPU nrc1";
                }
                baseline.push_back(std::move(singleton));
            }
            for(int threads:{1,4,5,8})for(bool padded_input:{false,true})for(int columns:{1,2,3,4,8}){
                SCOPED_TRACE(threads);
                SCOPED_TRACE(padded_input);
                SCOPED_TRACE(columns);
                const auto actual=RunSsmGateOutQ4(model.get(),weight,values,columns,0,threads,padded_input,options);
                for(int c=0;c<columns;++c)if(std::memcmp(actual.data()+c*k,baseline[c].data(),k*sizeof(float))!=0){
                    const auto mismatch=std::mismatch(baseline[c].begin(),baseline[c].end(),actual.data()+c*k,
                        [](float a,float b){return std::memcmp(&a,&b,sizeof(float))==0;});
                    const int index=static_cast<int>(mismatch.first-baseline[c].begin());
                    ADD_FAILURE()<<std::setprecision(10)<<"SSM-Q4 gate="<<gate<<" padded_weight="<<padded_weight
                        <<" mode="<<(mode==densecore::env::RuntimeToggleMode::Off?"off":mode==densecore::env::RuntimeToggleMode::On?"on":"auto")
                        <<" threads="<<threads<<" padded_input="<<padded_input<<" width="<<columns<<" column="<<c
                        <<" first_difference="<<index<<" old_m1="<<baseline[c][index]<<" matrix="<<actual[c*k+index];
                }
            }
        }
        densecore::kernels::RetireRepackedWeightCaches();densecore::kernels::Q4KRepackedGemvSetRuntimeCacheBudgetFloor(0);
        ASSERT_EQ(densecore::kernels::Q4KRepackedGemvCacheLimitBytes(),0u);
        for(auto mode:{densecore::env::RuntimeToggleMode::On,densecore::env::RuntimeToggleMode::Auto}){
            SsmGateOutQ4Options options;options.mode=mode;
            const auto matrix=RunSsmGateOutQ4(model.get(),weight,values,4,0,5,true,options);
            for(int c=0;c<4;++c){
                const auto singleton=RunSsmGateOutQ4(model.get(),weight,values,1,c,5,true,options);
                EXPECT_EQ(std::memcmp(matrix.data()+c*k,singleton.data(),k*sizeof(float)),0)<<"explicit cache refusal column="<<c;
            }
        }
        densecore::kernels::Q4KRepackedGemvRaiseRuntimeCacheBudgetFloor(64*1024*1024);
        // Exclusions compare against the same tensor bytes/role name but an
        // unregistered descriptor. This remains a legacy matrix baseline even
        // after a pointer-owned canonical extension is implemented.
        for(int exclusion=0;exclusion<20;++exclusion){
            SCOPED_TRACE(exclusion);SsmGateOutQ4Options options;
            if(exclusion==0)options.phase=InferenceExecutionPhase::Prefill;
            if(exclusion==1)options.phase=InferenceExecutionPhase::Unknown;
            options.lora=exclusion==2;options.reference=exclusion==3;options.no_context=exclusion==4;
            options.no_batch=exclusion==5;options.mixed_tokens=exclusion==6;options.required=exclusion==7;
            options.fused=exclusion==8;options.amx=exclusion==9;options.foreign=exclusion==10;
            options.metadata_excluded=exclusion>=11;
            options.mixed_seqs=exclusion==12;
            ggml_tensor clipped=*weight;
            std::unique_ptr<ggml_context,decltype(&ggml_free)> extra(nullptr,ggml_free);
            ggml_tensor* tested=weight;
            if(exclusion==11)model->hybrid_layer_is_ssm[0]=0;
            if(exclusion==13){model->layers[0].Set(key,nullptr);model->layers[0].Set(model_keys::kAttnQkvWeight,weight);}
            if(exclusion==14)model->hparams.n_layer=32;
            if(exclusion==15){clipped.ne[1]-=8;tested=&clipped;model->layers[0].Set(key,tested);}
            if(exclusion==16){
                extra.reset(ggml_init({4*1024*1024,nullptr,false}));ASSERT_NE(extra,nullptr);
                tested=ggml_new_tensor_2d(extra.get(),GGML_TYPE_Q8_0,n,k);ggml_set_name(tested,weight->name);
                for(int out=0;out<k;++out){
                    for(int i=0;i<n;++i)row[i]=.31f*std::sin((i+7)*(out+3)*.0017f)+.125f*std::cos((i%37-out%19)*.11f);
                    ggml_quantize_chunk(GGML_TYPE_Q8_0,row.data(),static_cast<char*>(tested->data)+out*tested->nb[1],0,1,n,nullptr);
                }
                model->layers[0].Set(key,tested);
            }
            if(exclusion==17)model->variant=ModelVariant::QWEN36;
            if(exclusion==18)model->hparams.n_experts=2;
            if(exclusion==19)model->hparams.n_embd=2048;
            const auto current=RunSsmGateOutQ4(model.get(),tested,values,4,0,5,true,options);
            options.foreign=true;
            const auto legacy=RunSsmGateOutQ4(model.get(),tested,values,4,0,5,true,options);
            ASSERT_EQ(current.size(),legacy.size());
            EXPECT_EQ(std::memcmp(current.data(),legacy.data(),current.size()*sizeof(float)),0);
            model->hybrid_layer_is_ssm[0]=1;model->variant=ModelVariant::QWEN35;
            model->hparams.n_embd=1024;model->hparams.n_layer=24;model->hparams.n_experts=0;
            model->layers[0].Set(model_keys::kAttnQkvWeight,nullptr);model->layers[0].Set(key,weight);
        }
    }
}
