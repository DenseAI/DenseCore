/**
 * @file device_interface.h
 * @brief Hardware Abstraction Layer (HAL) interfaces for devices, streams, and events
 *
 * This file is part of DenseCore Public API.
 * Licensed under Apache 2.0 (Open Source) or Commercial License.
 *
 * Design Goals:
 * - Minimal, stable ABI for vendor backends
 * - Explicit ownership and lifecycle control
 * - Compatible with CPU-only backends and future NPU/GPU backends
 */

#ifndef DENSECORE_HAL_DEVICE_INTERFACE_H
#define DENSECORE_HAL_DEVICE_INTERFACE_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "macros.h"

namespace densecore {

/**
 * @brief Memory copy direction
 */
enum class MemcpyKind : uint8_t { HostToDevice = 0, DeviceToHost = 1, DeviceToDevice = 2 };

class Event;

/**
 * @brief Device allocator interface
 */
class Allocator {
public:
    virtual ~Allocator() = default;

    /**
     * @brief Allocate device memory
     * @param bytes Number of bytes
     * @return Device pointer
     */
    virtual void* Allocate(size_t bytes) = 0;

    /**
     * @brief Free device memory
     * @param ptr Pointer previously returned by Allocate
     */
    virtual void Free(void* ptr) = 0;

    /**
     * @brief Asynchronous or synchronous memory copy
     * @param dst Destination pointer
     * @param src Source pointer
     * @param size_bytes Number of bytes
     * @param kind Copy direction
     */
    virtual void Memcpy(void* dst, const void* src, size_t size_bytes, MemcpyKind kind) = 0;
};

/**
 * @brief Stream interface (async execution queue)
 */
class Stream {
public:
    virtual ~Stream() = default;

    /**
     * @brief Block host until all queued work completes
     */
    virtual void Synchronize() = 0;

    /**
     * @brief Record an event in this stream
     */
    virtual void RecordEvent(Event* event) = 0;

    /**
     * @brief Make this stream wait on an event
     */
    virtual void WaitEvent(Event* event) = 0;
};

/**
 * @brief Event interface (synchronization point)
 */
class Event {
public:
    virtual ~Event() = default;

    /**
     * @brief Block host until event is reached
     */
    virtual void Synchronize() = 0;

    /**
     * @brief Elapsed time between two events (milliseconds)
     */
    virtual float ElapsedTime(const Event* start, const Event* end) const = 0;
};

/**
 * @brief Device interface for creating streams and events
 */
class Device {
public:
    virtual ~Device() = default;

    /**
     * @brief Get allocator for this device
     */
    virtual Allocator* GetAllocator() = 0;

    /**
     * @brief Create a new execution stream
     */
    virtual Stream* CreateStream() = 0;

    /**
     * @brief Create a new event
     * @param enable_timing If true, enable timing for profiling
     */
    virtual Event* CreateEvent(bool enable_timing = true) = 0;

    /**
     * @brief Destroy a stream created by this device
     */
    virtual void DestroyStream(Stream* stream) = 0;

    /**
     * @brief Destroy an event created by this device
     */
    virtual void DestroyEvent(Event* event) = 0;
};

// =============================================================================
// RAII Wrappers for Stream/Event (Prevents Resource Leaks)
// =============================================================================

/**
 * @brief Custom deleter for Stream that calls Device::DestroyStream
 */
class StreamDeleter {
    Device* device_;

public:
    explicit StreamDeleter(Device* d = nullptr) : device_(d) {}

    void operator()(Stream* s) const {
        if (device_ && s) {
            device_->DestroyStream(s);
        }
    }

    Device* GetDevice() const { return device_; }
};

/**
 * @brief Custom deleter for Event that calls Device::DestroyEvent
 */
class EventDeleter {
    Device* device_;

public:
    explicit EventDeleter(Device* d = nullptr) : device_(d) {}

    void operator()(Event* e) const {
        if (device_ && e) {
            device_->DestroyEvent(e);
        }
    }

    Device* GetDevice() const { return device_; }
};

/**
 * @brief RAII wrapper for Stream - automatically destroys on scope exit
 *
 * Usage:
 * @code
 * StreamPtr stream = MakeStream(device);
 * stream->Synchronize();
 * // Automatically destroyed when stream goes out of scope
 * @endcode
 */
using StreamPtr = std::unique_ptr<Stream, StreamDeleter>;

/**
 * @brief RAII wrapper for Event - automatically destroys on scope exit
 *
 * Usage:
 * @code
 * EventPtr event = MakeEvent(device);
 * stream->RecordEvent(event.get());
 * event->Synchronize();
 * // Automatically destroyed when event goes out of scope
 * @endcode
 */
using EventPtr = std::unique_ptr<Event, EventDeleter>;

/**
 * @brief Create a Stream with RAII ownership
 * @param device Device to create stream on (must outlive the returned StreamPtr)
 * @return RAII-managed Stream pointer
 */
inline StreamPtr MakeStream(Device* device) {
    assert(device != nullptr && "MakeStream requires non-null device");
    if (!device) {
        return StreamPtr(nullptr, StreamDeleter(nullptr));
    }
    return StreamPtr(device->CreateStream(), StreamDeleter(device));
}

/**
 * @brief Create an Event with RAII ownership
 * @param device Device to create event on (must outlive the returned EventPtr)
 * @param enable_timing If true, enable timing for profiling
 * @return RAII-managed Event pointer
 */
inline EventPtr MakeEvent(Device* device, bool enable_timing = true) {
    assert(device != nullptr && "MakeEvent requires non-null device");
    if (!device) {
        return EventPtr(nullptr, EventDeleter(nullptr));
    }
    return EventPtr(device->CreateEvent(enable_timing), EventDeleter(device));
}

}  // namespace densecore

// =============================================================================
// C-Compatible Plugin ABI
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

#define DENSECORE_PLUGIN_ABI_VERSION 1u

typedef void* DenseCoreDeviceHandle;
typedef void* DenseCoreStreamHandle;
typedef void* DenseCoreEventHandle;
typedef void* DenseCoreAllocatorHandle;

typedef enum DenseCoreMemcpyKind {
    DENSECORE_MEMCPY_HOST_TO_DEVICE = 0,
    DENSECORE_MEMCPY_DEVICE_TO_HOST = 1,
    DENSECORE_MEMCPY_DEVICE_TO_DEVICE = 2
} DenseCoreMemcpyKind;

/**
 * @brief Vendor plugin registration table
 *
 * Vendors should export a function returning this table, e.g.:
 *   const DenseCorePlugin* DenseCoreGetPlugin();
 */
typedef struct DenseCorePlugin {
    uint32_t abi_version;
    const char* name;
    uint32_t vendor_id;
    uint32_t device_type;

    DenseCoreDeviceHandle (*CreateDevice)(int device_ordinal);
    void (*DestroyDevice)(DenseCoreDeviceHandle device);

    DenseCoreAllocatorHandle (*GetAllocator)(DenseCoreDeviceHandle device);
    void* (*Allocate)(DenseCoreAllocatorHandle allocator, size_t bytes);
    void (*Free)(DenseCoreAllocatorHandle allocator, void* ptr);
    void (*Memcpy)(DenseCoreAllocatorHandle allocator, void* dst, const void* src, size_t size_bytes,
                   DenseCoreMemcpyKind kind);

    DenseCoreStreamHandle (*CreateStream)(DenseCoreDeviceHandle device);
    void (*DestroyStream)(DenseCoreDeviceHandle device, DenseCoreStreamHandle stream);
    void (*StreamSynchronize)(DenseCoreStreamHandle stream);
    void (*StreamRecordEvent)(DenseCoreStreamHandle stream, DenseCoreEventHandle event);
    void (*StreamWaitEvent)(DenseCoreStreamHandle stream, DenseCoreEventHandle event);

    DenseCoreEventHandle (*CreateEvent)(DenseCoreDeviceHandle device, int enable_timing);
    void (*DestroyEvent)(DenseCoreDeviceHandle device, DenseCoreEventHandle event);
    void (*EventSynchronize)(DenseCoreEventHandle event);
    float (*EventElapsedTime)(DenseCoreEventHandle start, DenseCoreEventHandle end);
} DenseCorePlugin;

typedef const DenseCorePlugin* (*DenseCoreGetPluginFn)();

#ifdef __cplusplus
}
#endif

#endif  // DENSECORE_HAL_DEVICE_INTERFACE_H
