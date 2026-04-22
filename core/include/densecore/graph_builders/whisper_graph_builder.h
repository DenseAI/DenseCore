/**
 * @file whisper_graph_builder.h
 * @brief Whisper graph builder header
 */

#ifndef DENSECORE_WHISPER_GRAPH_BUILDER_H
#define DENSECORE_WHISPER_GRAPH_BUILDER_H

#include "densecore/models/graph_registry.h"
#include "densecore/models/model_types.h"
#include <memory>

namespace densecore {

class WhisperGraphBuilder : public GraphBuilder {
public:
    explicit WhisperGraphBuilder(const WhisperModel* model, const WhisperHParams& hparams);
    ~WhisperGraphBuilder() override = default;

    std::unique_ptr<OperationGraph> Build(const std::vector<Tensor>& inputs, const std::string& variant_name) override;

private:
    const WhisperModel* model_;
    WhisperHParams hparams_;
};

}  // namespace densecore

#endif  // DENSECORE_WHISPER_GRAPH_BUILDER_H
