#pragma once
#include "runtime/callback_scratch.h"
#include <vector>

namespace densecore::llm::attention {
// Slice declared first: vector storage is destroyed before its numeric charge.
// No callback/model/work-context/weight pointers survive in this TLS workspace.
struct PagedScratch {
    densecore::runtime::CallbackScratchSlice slice;
    std::vector<const uint8_t*> scalar_k_blocks,scalar_v_blocks;
    std::vector<const void*> k_blocks,v_blocks;
    std::vector<float> key,value,scores;
    void ClearStorage() noexcept {
        std::vector<const uint8_t*>().swap(scalar_k_blocks);std::vector<const uint8_t*>().swap(scalar_v_blocks);
        std::vector<const void*>().swap(k_blocks);std::vector<const void*>().swap(v_blocks);
        std::vector<float>().swap(key);std::vector<float>().swap(value);std::vector<float>().swap(scores);
    }
    template<class T> static void Ensure(std::vector<T>& data,size_t count) {
        if(data.capacity()<count){std::vector<T>().swap(data);data.resize(count);}
    }
#ifdef DENSECORE_TEST_BUILD
    static std::atomic<bool>& FailNextAllocationForTest(){static std::atomic<bool> flag{false};return flag;}
#endif
    bool Prepare(const std::shared_ptr<densecore::runtime::CallbackScratchBudget>& owner,
                 size_t blocks,size_t k,size_t v,size_t context) noexcept {
        if(!owner) {
            // Legacy/unknown paths keep their old policy. Retire any previously
            // charged storage before releasing its owner, never the reverse.
            if(slice.Bytes()){ClearStorage();slice.Reset();}return true;
        }
        if(!slice.IsOwner(owner)) {
            // Thread reuse across engines does not require the new smaller
            // engine to budget an old larger workspace. Nothing borrows these
            // callback-only buffers after a previous callback has returned.
            ClearStorage();slice.Reset();
        }
        const size_t sk=std::max(blocks,scalar_k_blocks.capacity());
        const size_t sv=std::max(blocks,scalar_v_blocks.capacity());
        const size_t hk=std::max(blocks,k_blocks.capacity()),hv=std::max(blocks,v_blocks.capacity());
        const size_t nk=std::max(k,key.capacity()),nv=std::max(v,value.capacity()),ns=std::max(context,scores.capacity());
        const auto add=[](uint64_t a,uint64_t b){return densecore::runtime::resource_detail::Add(a,b);};
        uint64_t bytes=2048;
        for(size_t count:{sk,sv,hk,hv}) {
            if(count>UINT64_MAX/sizeof(void*))return false;bytes=add(bytes,count*sizeof(void*));
        }
        for(size_t count:{nk,nv,ns}) {
            if(count>UINT64_MAX/sizeof(float))return false;bytes=add(bytes,count*sizeof(float));
        }
        if(!slice.Grow(owner,bytes))return false;
        try {
#ifdef DENSECORE_TEST_BUILD
            if(FailNextAllocationForTest().exchange(false))throw std::bad_alloc();
#endif
            // Each old vector is freed before replacement; there is no growing
            // temporary retaining both old and new allocation capacities.
            Ensure(scalar_k_blocks,sk);Ensure(scalar_v_blocks,sv);Ensure(k_blocks,hk);Ensure(v_blocks,hv);
            Ensure(key,nk);Ensure(value,nv);Ensure(scores,ns);
            return true;
        } catch(...) {
            ClearStorage();slice.Reset();return false;
        }
    }
};
inline PagedScratch& ThreadPagedScratch(){static thread_local PagedScratch scratch;return scratch;}
}
