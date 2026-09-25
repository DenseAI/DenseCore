#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

#include "densecore/models/model_types.h"
#include "llm/models/common/family_internal.h"

namespace {

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value) : name_(name ? name : "") {
        const char* prev = std::getenv(name_.c_str());
        if (prev) {
            had_prev_ = true;
            prev_value_ = prev;
        }
        Set(value);
    }

    ~ScopedEnvVar() {
        if (had_prev_) {
            Set(prev_value_.c_str());
        } else {
            Set(nullptr);
        }
    }

private:
    void Set(const char* value) {
#if defined(_WIN32)
        _putenv_s(name_.c_str(), value ? value : "");
#else
        if (value) {
            setenv(name_.c_str(), value, 1);
        } else {
            unsetenv(name_.c_str());
        }
#endif
    }

    std::string name_;
    bool had_prev_ = false;
    std::string prev_value_;
};

}  // namespace

TEST(ModelFamilyInternalTest, DetectsHybridSsmQkvWeightNames) {
    EXPECT_TRUE(densecore::llm::models::IsHybridSSMQkvWeightName("blk.0.attn_qkv.weight"));
    EXPECT_TRUE(densecore::llm::models::IsHybridSSMQkvWeightName("linear_attn.in_proj_qkv.weight"));
    EXPECT_FALSE(densecore::llm::models::IsHybridSSMQkvWeightName("blk.0.attn_q.weight"));
}

TEST(ModelFamilyInternalTest, HybridSsmForcePlainGgmlRespectsEnv) {
    ScopedEnvVar force_ggml("DENSECORE_HYBRID_SSM_QKV_FORCE_GGML", "1");
    densecore::llm::models::ResetHybridSSMQkvForceGgmlCache();
    EXPECT_TRUE(densecore::llm::models::ShouldForcePlainGgmlForHybridSSMQkv());

    ScopedEnvVar force_ggml_off("DENSECORE_HYBRID_SSM_QKV_FORCE_GGML", "0");
    densecore::llm::models::ResetHybridSSMQkvForceGgmlCache();
    EXPECT_FALSE(densecore::llm::models::ShouldForcePlainGgmlForHybridSSMQkv());
}

TEST(ModelFamilyInternalTest, DetectsGemmaSharedKvSourceLayer) {
    TransformerModel model{};
    model.arch_flags.is_gemma4 = true;
    model.gemma4_layer_kv_source = {0, 0, 2, 2};

    EXPECT_TRUE(densecore::llm::models::IsGemma4SharedKVSourceLayer(&model, 0));
    EXPECT_FALSE(densecore::llm::models::IsGemma4SharedKVSourceLayer(&model, 1));
    EXPECT_TRUE(densecore::llm::models::IsGemma4SharedKVSourceLayer(&model, 2));
    EXPECT_FALSE(densecore::llm::models::IsGemma4SharedKVSourceLayer(&model, 3));
}
