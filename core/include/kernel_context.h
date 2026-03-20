/**
 * @file kernel_context.h
 * @brief Thread-local context for hardware acceleration state
 *
 * Holds persistent resources like scratchpad memory and hardware configurations
 * (e.g., AMX tile config) to avoid re-initialization overhead per kernel call.
 */

#ifndef DENSECORE_KERNEL_CONTEXT_H
#define DENSECORE_KERNEL_CONTEXT_H

#include <cstddef>
#include <cstdint>

#include "aligned_allocator.h"

namespace densecore {

class KernelContext {
public:
    /**
     * @brief Create context with initial scratchpad size
     * @param scratchpad_bytes Initial size of scratchpad memory (default: 64KB)
     */
    explicit KernelContext(size_t scratchpad_bytes = 64 * 1024);
    ~KernelContext();

    // Prevent copying (hardware state is non-copyable)
    KernelContext(const KernelContext&) = delete;
    KernelContext& operator=(const KernelContext&) = delete;

    // Move semantics
    KernelContext(KernelContext&&) noexcept;
    KernelContext& operator=(KernelContext&&) noexcept;

    /**
     * @brief Configure AMX tiles for matrix multiplication
     *
     * Sets up 2D register tiles (palette 1) for TMUL operations.
     * This operation is expensive (requires syscall or intense setup), so
     * we only re-configure if parameters change.
     *
     * @param rows Number of rows (tile height)
     * @param cols Number of columns (tile width in bytes)
     */
    void ConfigureAMX(int rows, int cols);

    /**
     * @brief Get pointer to aligned scratchpad memory
     */
    void* ScratchpadData() { return scratchpad_.get(); }

    /**
     * @brief Get current scratchpad size
     */
    size_t ScratchpadSize() const { return scratchpad_size_; }

    /**
     * @brief Resize scratchpad (preserves content not guaranteed)
     */
    void ResizeScratchpad(size_t new_size);

private:
    AlignedPtr<uint8_t> scratchpad_;
    size_t scratchpad_size_ = 0;

    // AMX State caching
    bool amx_configured_ = false;
    int cached_rows_ = 0;
    int cached_cols_ = 0;
};

}  // namespace densecore

#endif  // DENSECORE_KERNEL_CONTEXT_H
