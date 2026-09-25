#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace densecore::runtime {

// One entry per execution context, not one buffer per graph node. Consumers keep
// immutable snapshots so replacing an entry never invalidates an active kernel.
class BatchedActivationPackCache {
public:
    struct Key {
        uint64_t generation;
        const void* operation;
        const void* source;
        int rows;
        int cols;

        bool operator==(const Key& other) const {
            return generation == other.generation && operation == other.operation && source == other.source &&
                   rows == other.rows && cols == other.cols;
        }
    };
    using Snapshot = std::shared_ptr<const std::vector<uint8_t>>;

    template <typename Fill> Snapshot GetOrFill(const Key& key, Fill&& fill) {
        if (!key.generation || !key.operation || !key.source || key.rows <= 0 || key.cols <= 0) return {};
        std::lock_guard<std::mutex> lock(mutex_);
        if (buffer_ && key_ == key) return buffer_;
        // Reuse capacity only after all earlier consumers have released it.
        if (!buffer_ || buffer_.use_count() != 1) buffer_ = std::make_shared<std::vector<uint8_t>>();
        key_ = {};  // A failed fill must not publish an old key with modified bytes.
        fill(*buffer_);
#ifdef DENSECORE_TEST_BUILD
        ++successful_fills_;
#endif
        key_ = key;
        return buffer_;
    }

#ifdef DENSECORE_TEST_BUILD
    uint64_t SuccessfulFillCountForTest() {
        std::lock_guard<std::mutex> lock(mutex_);
        return successful_fills_;
    }
#endif

private:
#ifdef DENSECORE_TEST_BUILD
    uint64_t successful_fills_ = 0;
#endif
    std::mutex mutex_;
    Key key_{};
    std::shared_ptr<std::vector<uint8_t>> buffer_;
};

}  // namespace densecore::runtime
