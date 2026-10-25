#pragma once

#include "densecore/models/model_types.h"
#include "densecore/memory/kv_cache.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace densecore::llm::speculative {
inline constexpr uint64_t kHeadBootstrapTokens = 16;
namespace head_memory_detail {
inline uint64_t Add(uint64_t a,uint64_t b) {
    if (b>UINT64_MAX-a) throw std::overflow_error("MTP head memory size overflow");
    return a+b;
}
inline uint64_t Mul(uint64_t a,uint64_t b) {
    if (b && a>UINT64_MAX/b) throw std::overflow_error("MTP head memory size overflow");
    return a*b;
}
inline uint64_t Align64(uint64_t bytes) {return Mul(Add(bytes,63)/64,64);}
inline uint64_t Dimension(uint64_t value) {
    if (!value || value>static_cast<uint64_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("invalid MTP head memory dimension");
    return value;
}
struct Dimensions {uint64_t embedding,heads,kv_heads,key,value,vocab;};
inline Dimensions Shape(const TransformerModel& model) {
    const auto& hp=model.hparams;
    Dimensions d{Dimension(hp.n_embd),Dimension(hp.n_head),Dimension(hp.n_head_kv),0,0,Dimension(hp.n_vocab)};
    d.key=Dimension(hp.n_embd_head_k ? hp.n_embd_head_k : d.embedding/d.heads);
    d.value=Dimension(hp.n_embd_head_v ? hp.n_embd_head_v : d.key);
    return d;
}
inline uint64_t FfnWidth(const TransformerModel& model,const char* key,uint64_t fallback) {
    if (model.mtp_heads.size()==1) {
        if (const auto* tensor=model.mtp_heads[0].Find(key)) {
            if (tensor->ne[1]<=0) throw std::invalid_argument("invalid MTP head FFN dimension");
            return Dimension(static_cast<uint64_t>(tensor->ne[1]));
        }
    }
    // Shape-only budget callers may not have loaded the NextN tensors yet.
    return Dimension(fallback);
}
}

// One independent F16 head cache per active session. KVBlockAllocator aligns
// each K/V block stride separately to 64 bytes, not the combined token row.
inline uint64_t HeadKVCacheBytes(const TransformerModel& model,uint64_t context) {
    using namespace head_memory_detail;
    if (!context) throw std::invalid_argument("MTP head context must be positive");
    const auto d=Shape(model);
    const auto blocks=Add(context,BLOCK_SIZE-1)/BLOCK_SIZE;
    const auto k_stride=Align64(Mul(Mul(Mul(BLOCK_SIZE,sizeof(ggml_fp16_t)),d.key),d.kv_heads));
    const auto v_stride=Align64(Mul(Mul(Mul(BLOCK_SIZE,sizeof(ggml_fp16_t)),d.value),d.kv_heads));
    return Mul(blocks,Add(k_stride,v_stride));
}

// Linear GGML arena: payloads of intermediate nodes remain live until the whole
// head invocation finishes. Reserve a conservative sum, including full-vocab
// logits, GQA history gathers and selected-expert tensors; packed weights and
// persistent expert descriptors are deliberately outside this arena.
struct HeadGraphMemory {
    uint64_t metadata=64ULL*1024*1024;
    uint64_t activations=0,attention=0,feed_forward=0,logits=0;
    uint64_t Total() const {
        using namespace head_memory_detail;
        return Add(Add(metadata,activations),Add(Add(attention,feed_forward),logits));
    }
};
inline HeadGraphMemory CalculateHeadGraphMemory(const TransformerModel& model,uint64_t queries,uint64_t keys) {
    using namespace head_memory_detail;
    if (!queries || queries>kHeadBootstrapTokens || keys<queries)
        throw std::invalid_argument("MTP head invocation exceeds its bounded query/key shape");
    const auto d=Shape(model);
    HeadGraphMemory memory;
    // Fusion, norms, residuals, gated attention/output projection and Q/K/V.
    auto activations=Mul(Mul(queries,d.embedding),32);
    activations=Add(activations,Mul(Mul(Mul(queries,d.heads),Add(d.key,d.value)),4));
    memory.activations=Mul(activations,sizeof(float));
    // Logical GQA broadcasts; no all-query-head expansion of cached K/V.
    const auto history=Mul(Mul(Mul(keys,d.kv_heads),Add(d.key,d.value)),3);
    const auto scores=Mul(Mul(Mul(queries,keys),d.heads),4);
    memory.attention=Mul(Add(history,scores),sizeof(float));
    const uint64_t fallback=model.hparams.n_ff ? model.hparams.n_ff : Mul(d.embedding,4);
    uint64_t ffn=0;
    if (model.hparams.n_experts) {
        const auto experts=Dimension(model.hparams.n_experts);
        const auto topk=Dimension(model.hparams.n_experts_used);
        if (topk>experts) throw std::invalid_argument("MTP head top-k exceeds expert count");
        const auto expert_ffn=FfnWidth(model,"ffn_gate_exps.weight",fallback);
        const auto shared_ffn=FfnWidth(model,"ffn_gate_shexp.weight",fallback);
        // Gate/up/SwiGLU, Q8 row sidecars, assignment/selected outputs and reduction.
        ffn=Mul(Mul(queries,topk),Add(Mul(expert_ffn,8),Mul(d.embedding,4)));
        ffn=Add(ffn,Mul(Mul(queries,shared_ffn),6));
        ffn=Add(ffn,Mul(Mul(queries,experts),8)); // router, sort and probability rows
        ffn=Add(ffn,Mul(Mul(queries,topk),16)); // IDs, weights and assignment metadata
    } else {
        ffn=Mul(Mul(queries,FfnWidth(model,"ffn_gate.weight",fallback)),6);
    }
    memory.feed_forward=Mul(ffn,sizeof(float));
    memory.logits=Mul(Mul(Mul(queries,d.vocab),2),sizeof(float));
    (void)memory.Total();
    return memory;
}
} // namespace densecore::llm::speculative
