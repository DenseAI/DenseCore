#ifndef DENSECORE_MODELS_GRAPH_REGISTRY_H
#define DENSECORE_MODELS_GRAPH_REGISTRY_H

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "densecore/hal/operation_graph.h"
#include "densecore/hal/tensor.h"

namespace densecore {

// Interface for model-specific graph builders
class GraphBuilder {
public:
    virtual ~GraphBuilder() = default;

    // Builds the computation graph given input shapes
    // This allows dynamic reshaping based on input resolution/length
    virtual std::unique_ptr<OperationGraph> Build(const std::vector<Tensor>& inputs,
                                                  const std::string& variant_name  // e.g., "vit_base_patch16_224"
                                                  ) = 0;
};

class GraphRegistry {
public:
    static GraphRegistry& Instance() {
        static GraphRegistry instance;
        return instance;
    }

    using BuilderFactory = std::function<std::unique_ptr<GraphBuilder>()>;

    void Register(const std::string& architecture, BuilderFactory factory) {
        std::lock_guard<std::mutex> lock(mu_);
        builders_[architecture] = factory;
    }

    std::unique_ptr<GraphBuilder> GetBuilder(const std::string& architecture) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = builders_.find(architecture);
        if (it != builders_.end()) {
            return it->second();
        }
        return nullptr;
    }

private:
    std::mutex mu_;
    std::unordered_map<std::string, BuilderFactory> builders_;
};

#define DENSECORE_REGISTER_GRAPH(Architecture, BuilderClass)                                                    \
    namespace {                                                                                                 \
    struct Architecture##Registrar {                                                                            \
        Architecture##Registrar() {                                                                             \
            ::densecore::GraphRegistry::Instance().Register(#Architecture,                                      \
                                                            []() { return std::make_unique<BuilderClass>(); }); \
        }                                                                                                       \
    };                                                                                                          \
    static Architecture##Registrar global_##Architecture##_registrar;                                           \
    }

}  // namespace densecore

#endif  // DENSECORE_MODELS_GRAPH_REGISTRY_H
