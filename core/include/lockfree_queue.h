/**
 * @file lockfree_queue.h
 * @brief Lock-Free MPSC Queue with Tagged Pointer ABA Protection
 *
 * Implements a Michael-Scott style lock-free queue using tagged pointers
 * to solve the ABA problem. On x86-64, virtual addresses only use 48 bits,
 * so we pack a 16-bit version counter into the upper bits.
 *
 * THREADING CONTRACT:
 * -------------------
 * This queue is designed for Multi-Producer Single-Consumer (MPSC) pattern:
 * - Multiple threads can safely call push() concurrently
 * - Only ONE thread should call pop() (the worker thread)
 *
 * If MPMC is needed, use the pop() with care or add proper MPMC support.
 */

#ifndef DENSECORE_LOCKFREE_QUEUE_H
#define DENSECORE_LOCKFREE_QUEUE_H

#include <atomic>
#include <cstdint>
#include <mutex>
#include <new>  // For std::hardware_destructive_interference_size
#include <queue>
#include <string>

namespace densecore {

// Cache line size for padding to prevent false sharing
// Use standard 64 bytes to avoid ABI instability warnings across compilers
constexpr size_t CACHE_LINE_SIZE = 64;

/**
 * @brief Tagged pointer for ABA protection on 64-bit systems
 *
 * On x86-64, canonical addresses only use 48 bits. We exploit the upper
 * 16 bits to store a version counter that increments on each CAS operation.
 * This prevents the ABA problem where a pointer is freed and reallocated
 * to the same address.
 */
struct TaggedPtr {
    // On x86-64, only bits [0:47] are used for virtual addresses
    // Bits [48:63] must be sign-extended copies of bit 47 for canonical addresses
    // We use the packed representation where tag occupies upper bits
    static constexpr uint64_t PTR_MASK = 0x0000FFFFFFFFFFFFULL;
    static constexpr int TAG_SHIFT = 48;

    uint64_t packed;

    TaggedPtr() noexcept : packed(0) {}

    TaggedPtr(void* ptr, uint16_t tag) noexcept { packed = Pack(ptr, tag); }

    void* Ptr() const noexcept {
        // Sign-extend for canonical address (handles kernel addresses too)
        uint64_t addr = packed & PTR_MASK;
        // Check if bit 47 is set (kernel address)
        if (addr & (1ULL << 47)) {
            addr |= 0xFFFF000000000000ULL;  // Sign extend
        }
        return reinterpret_cast<void*>(addr);
    }

    uint16_t Tag() const noexcept { return static_cast<uint16_t>(packed >> TAG_SHIFT); }

    static uint64_t Pack(void* ptr, uint16_t tag) noexcept {
        uint64_t addr = reinterpret_cast<uint64_t>(ptr) & PTR_MASK;
        return (static_cast<uint64_t>(tag) << TAG_SHIFT) | addr;
    }

    bool operator==(const TaggedPtr& other) const noexcept { return packed == other.packed; }

    bool operator!=(const TaggedPtr& other) const noexcept { return packed != other.packed; }
};

static_assert(sizeof(TaggedPtr) == sizeof(uint64_t), "TaggedPtr must be 64 bits for atomic operations");

/**
 * @brief Lock-free MPSC queue node
 */
template <typename T> struct alignas(CACHE_LINE_SIZE) LockFreeNode {
    T* data;
    std::atomic<TaggedPtr> next;

    LockFreeNode() : data(nullptr), next(TaggedPtr(nullptr, 0)) {}
    explicit LockFreeNode(T* d) : data(d), next(TaggedPtr(nullptr, 0)) {}
};

/**
 * @brief Lock-Free Queue with Tagged Pointer ABA Protection
 *
 * Based on Michael-Scott queue algorithm with modifications for
 * tagged pointers. Uses a dummy node to simplify edge cases.
 *
 * Memory ordering:
 * - push(): Uses release semantics to ensure data is visible before linking
 * - pop(): Uses acquire semantics to see all writes from push()
 *
 * @tparam T Element type (must be a pointer type in practice)
 */
template <typename T> class LockFreeQueue {
public:
    using Node = LockFreeNode<T>;

    LockFreeQueue() {
        // Create dummy node (sentinel)
        Node* dummy = new Node();
        TaggedPtr initial(dummy, 0);
        head_.store(initial, std::memory_order_relaxed);
        tail_.store(initial, std::memory_order_relaxed);
    }

    ~LockFreeQueue() {
        // Drain remaining items
        while (Pop() != nullptr) {}

        // Free dummy node
        TaggedPtr head = head_.load(std::memory_order_relaxed);
        delete static_cast<Node*>(head.Ptr());
    }

    // Non-copyable, non-movable
    LockFreeQueue(const LockFreeQueue&) = delete;
    LockFreeQueue& operator=(const LockFreeQueue&) = delete;
    LockFreeQueue(LockFreeQueue&&) = delete;
    LockFreeQueue& operator=(LockFreeQueue&&) = delete;

    /**
     * @brief Push an item to the back of the queue (lock-free)
     *
     * Multiple threads can call this concurrently.
     *
     * @param item Pointer to item (ownership transferred to queue)
     */
    void Push(T* item) {
        Node* new_node = new Node(item);

        while (true) {
            TaggedPtr tail = tail_.load(std::memory_order_acquire);
            Node* tail_node = static_cast<Node*>(tail.Ptr());
            TaggedPtr next = tail_node->next.load(std::memory_order_acquire);

            // Check if tail is still consistent
            if (tail == tail_.load(std::memory_order_acquire)) {
                if (next.Ptr() == nullptr) {
                    // Tail is pointing to last node, try to link new node
                    TaggedPtr new_next(new_node, next.Tag() + 1);
                    if (tail_node->next.compare_exchange_weak(next, new_next, std::memory_order_release,
                                                              std::memory_order_relaxed)) {
                        // Successfully linked, try to advance tail
                        TaggedPtr new_tail(new_node, tail.Tag() + 1);
                        tail_.compare_exchange_strong(tail, new_tail, std::memory_order_release,
                                                      std::memory_order_relaxed);
                        return;
                    }
                } else {
                    // Tail is lagging, help advance it
                    TaggedPtr new_tail(next.Ptr(), tail.Tag() + 1);
                    tail_.compare_exchange_strong(tail, new_tail, std::memory_order_release, std::memory_order_relaxed);
                }
            }
        }
    }

    /**
     * @brief Pop an item from the front of the queue (lock-free)
     *
     * In MPSC mode, only one thread should call this.
     *
     * @return Pointer to item, or nullptr if queue is empty
     */
    T* Pop() {
        while (true) {
            TaggedPtr head = head_.load(std::memory_order_acquire);
            TaggedPtr tail = tail_.load(std::memory_order_acquire);
            Node* head_node = static_cast<Node*>(head.Ptr());
            TaggedPtr next = head_node->next.load(std::memory_order_acquire);

            // Check consistency
            if (head == head_.load(std::memory_order_acquire)) {
                if (head.Ptr() == tail.Ptr()) {
                    // Queue appears empty or tail is lagging
                    if (next.Ptr() == nullptr) {
                        // Queue is actually empty
                        return nullptr;
                    }
                    // Tail is lagging, help advance it
                    TaggedPtr new_tail(next.Ptr(), tail.Tag() + 1);
                    tail_.compare_exchange_strong(tail, new_tail, std::memory_order_release, std::memory_order_relaxed);
                } else {
                    // Queue is not empty, read value before CAS
                    Node* next_node = static_cast<Node*>(next.Ptr());
                    T* result = next_node->data;

                    // Try to advance head
                    TaggedPtr new_head(next.Ptr(), head.Tag() + 1);
                    if (head_.compare_exchange_weak(head, new_head, std::memory_order_release,
                                                    std::memory_order_relaxed)) {
                        // Successfully dequeued, free old dummy node
                        delete head_node;
                        return result;
                    }
                }
            }
        }
    }

    /**
     * @brief Check if queue appears empty (approximate)
     *
     * Note: This is a point-in-time check that may be stale by the time
     * the caller acts on it. Use for optimization hints only.
     */
    bool Empty() const noexcept {
        TaggedPtr head = head_.load(std::memory_order_acquire);
        TaggedPtr tail = tail_.load(std::memory_order_acquire);
        Node* head_node = static_cast<Node*>(head.Ptr());
        TaggedPtr next = head_node->next.load(std::memory_order_acquire);

        return (head.Ptr() == tail.Ptr()) && (next.Ptr() == nullptr);
    }

private:
    // Pad to avoid false sharing between head and tail
    alignas(CACHE_LINE_SIZE) std::atomic<TaggedPtr> head_;
    alignas(CACHE_LINE_SIZE) std::atomic<TaggedPtr> tail_;
};

/**
 * @brief Sharded priority queue using lock-free FIFO queues per tier
 *
 * Provides priority scheduling without complex lock-free priority queue.
 * Requests are routed to tier-specific queues on push, and dequeued
 * in priority order (premium > standard > batch).
 */
template <typename T> class ShardedPriorityQueue {
public:
    /**
     * @brief Push item to appropriate tier queue
     *
     * @param item Item to enqueue
     * @param tier Priority tier ("premium", "standard", "batch")
     *
     * NOTE: Mutex synchronization added for robust cross-platform thread-safety.
     * Lock-free was causing visibility issues on some platforms (WSL, ARM).
     */
    void Push(T* item, const std::string& tier) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (tier == "premium") {
            premium_.push(item);
        } else if (tier == "batch") {
            batch_.push(item);
        } else {
            standard_.push(item);
        }
        size_++;
    }

    /**
     * @brief Pop highest priority item (checks tiers in order)
     *
     * @return Item pointer, or nullptr if all queues empty
     */
    T* Pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        // Check tiers in priority order
        if (!premium_.empty()) {
            T* item = premium_.front();
            premium_.pop();
            size_--;
            return item;
        }
        if (!standard_.empty()) {
            T* item = standard_.front();
            standard_.pop();
            size_--;
            return item;
        }
        if (!batch_.empty()) {
            T* item = batch_.front();
            batch_.pop();
            size_--;
            return item;
        }
        return nullptr;
    }

    /**
     * @brief Check if all queues appear empty
     */
    bool Empty() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return premium_.empty() && standard_.empty() && batch_.empty();
    }

    /**
     * @brief Get queue size
     */
    size_t Size() const noexcept {
        std::lock_guard<std::mutex> lock(mutex_);
        return size_;
    }

private:
    mutable std::mutex mutex_;  // For thread-safe access
    std::queue<T*> premium_;    // Tier 0: highest priority
    std::queue<T*> standard_;   // Tier 1: default
    std::queue<T*> batch_;      // Tier 2: lowest priority
    size_t size_{0};
};

}  // namespace densecore

#endif  // DENSECORE_LOCKFREE_QUEUE_H
