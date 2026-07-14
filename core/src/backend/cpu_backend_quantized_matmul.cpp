#include "densecore/backend/cpu_backend.h"

#include "ggml-cpu.h"
#include "ggml.h"
#include "thread_pool_impl.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace densecore {
namespace {

bool IsSupportedWeightType(ggml_type type) {
    return type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_K;
}

struct GgmlContextDeleter {
    void operator()(ggml_context* ctx) const {
        if (ctx) ggml_free(ctx);
    }
};

class GgmlMatMulWorkspace {
public:
    ~GgmlMatMulWorkspace() {
        entries_.clear();
        if (threadpool_) ggml_threadpool_free(threadpool_);
    }

    bool Run(const Tensor& input, const CpuBackend::GgmlQuantizedMatrixView& weight, Tensor* output, int threads) {
        threads = std::max(1, threads);
        if (!EnsureThreadPool(threads)) return false;
        if (cached_rows_ != input.shape[0]) {
            entries_.clear();
            cached_rows_ = input.shape[0];
        }

        const GraphKey key{weight.data, weight.type_id, input.shape[0], weight.rows, weight.cols};
        auto it = entries_.find(key);
        if (it == entries_.end()) {
            auto entry = BuildEntry(key);
            if (!entry) return false;
            it = entries_.emplace(key, std::move(entry)).first;
        }

        GraphEntry& entry = *it->second;
        entry.weight->data = const_cast<void*>(weight.data);
        entry.input->data = input.data;
        entry.output->data = output->data;
        return ggml_graph_compute(entry.graph, &entry.plan) == GGML_STATUS_SUCCESS;
    }

private:
    struct GraphKey {
        const void* weight = nullptr;
        int32_t type_id = -1;
        int64_t m = 0;
        int64_t n = 0;
        int64_t k = 0;

        bool operator==(const GraphKey& other) const {
            return weight == other.weight && type_id == other.type_id && m == other.m && n == other.n && k == other.k;
        }
    };

    struct GraphKeyHash {
        size_t operator()(const GraphKey& key) const {
            size_t hash = std::hash<const void*>{}(key.weight);
            const auto mix = [&](uint64_t value) {
                hash ^= std::hash<uint64_t>{}(value) + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
            };
            mix(static_cast<uint64_t>(key.type_id));
            mix(static_cast<uint64_t>(key.m));
            mix(static_cast<uint64_t>(key.n));
            mix(static_cast<uint64_t>(key.k));
            return hash;
        }
    };

    struct GraphEntry {
        std::vector<uint8_t> context_buffer;
        std::unique_ptr<ggml_context, GgmlContextDeleter> context;
        ggml_tensor* weight = nullptr;
        ggml_tensor* input = nullptr;
        ggml_tensor* output = nullptr;
        ggml_cgraph* graph = nullptr;
        ggml_cplan plan{};
        std::vector<uint8_t> work_buffer;
    };

    bool EnsureThreadPool(int threads) {
        if (threadpool_ && threads_ == threads) return true;
        entries_.clear();
        cached_rows_ = -1;
        if (threadpool_) ggml_threadpool_free(threadpool_);
        auto params = ggml_threadpool_params_default(threads);
        threadpool_ = ggml_threadpool_new(&params);
        threads_ = threadpool_ ? threads : 0;
        return threadpool_ != nullptr;
    }

    std::unique_ptr<GraphEntry> BuildEntry(const GraphKey& key) {
        constexpr size_t kGraphNodes = 4;
        auto entry = std::make_unique<GraphEntry>();
        const size_t context_bytes = 8 * ggml_tensor_overhead() + ggml_graph_overhead_custom(kGraphNodes, false) + 4096;
        entry->context_buffer.resize(context_bytes);

        ggml_init_params params{};
        params.mem_size = entry->context_buffer.size();
        params.mem_buffer = entry->context_buffer.data();
        params.no_alloc = true;
        entry->context.reset(ggml_init(params));
        if (!entry->context) return nullptr;

        entry->weight = ggml_new_tensor_2d(entry->context.get(), static_cast<ggml_type>(key.type_id), key.k, key.n);
        entry->input = ggml_new_tensor_2d(entry->context.get(), GGML_TYPE_F32, key.k, key.m);
        if (!entry->weight || !entry->input) return nullptr;
        entry->output = ggml_mul_mat(entry->context.get(), entry->weight, entry->input);
        if (!entry->output || entry->output->ne[0] != key.n || entry->output->ne[1] != key.m) return nullptr;

        entry->graph = ggml_new_graph_custom(entry->context.get(), kGraphNodes, false);
        if (!entry->graph) return nullptr;
        ggml_build_forward_expand(entry->graph, entry->output);
        entry->plan = ggml_graph_plan(entry->graph, threads_, threadpool_);
        entry->work_buffer.resize(entry->plan.work_size);
        entry->plan.work_data = entry->work_buffer.empty() ? nullptr : entry->work_buffer.data();
        return entry;
    }

    int threads_ = 0;
    int64_t cached_rows_ = -1;
    ggml_threadpool* threadpool_ = nullptr;
    std::unordered_map<GraphKey, std::unique_ptr<GraphEntry>, GraphKeyHash> entries_;
};

bool RunGgmlQ4MatMul(const Tensor& input, const CpuBackend::GgmlQuantizedMatrixView& weight, Tensor* output,
                     int threads) {
    static thread_local GgmlMatMulWorkspace workspace;
    return workspace.Run(input, weight, output, threads);
}

}  // namespace

bool CpuBackend::MatMulGgmlQuantizedTransB(const Tensor& A, const GgmlQuantizedMatrixView& W, Tensor* C,
                                           int numa_node_id) {
    if (!A.IsValid() || !W.IsValid() || !C || !C->IsValid() || A.dtype != DType::F32 || C->dtype != DType::F32 ||
        A.ndim != 2 || C->ndim != 2 || A.shape[1] != W.cols || C->shape[0] != A.shape[0] || C->shape[1] != W.rows) {
        return false;
    }

    const ggml_type weight_type = static_cast<ggml_type>(W.type_id);
    if (!IsSupportedWeightType(weight_type) || W.cols % ggml_blck_size(weight_type) != 0 ||
        W.row_bytes != ggml_row_size(weight_type, W.cols)) {
        return false;
    }

    auto& pool = GetThreadPool(numa_node_id);
    // ggml owns the maintained Q4_0/Q4_K kernels. The cached graph keeps the
    // weights in raw GGUF blocks and avoids rebuilding plans on resident calls.
    return RunGgmlQ4MatMul(A, W, C, pool.GetNumThreads());
}

}  // namespace densecore
