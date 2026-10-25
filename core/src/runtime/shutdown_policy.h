#pragma once
#include <chrono>
#include <cstdint>
#include <stdexcept>

using EngineShutdownClock = std::chrono::steady_clock;
inline EngineShutdownClock::time_point EngineShutdownDeadline(
    int64_t grace_ms, EngineShutdownClock::time_point now = EngineShutdownClock::now()) {
    const auto maximum = std::chrono::duration_cast<std::chrono::milliseconds>(EngineShutdownClock::duration::max()).count();
    if (grace_ms < 0 || grace_ms > maximum) throw std::invalid_argument("shutdown grace_timeout_ms is out of range");
    const auto duration = std::chrono::duration_cast<EngineShutdownClock::duration>(std::chrono::milliseconds(grace_ms));
    if (now > EngineShutdownClock::time_point::max() - duration)
        throw std::invalid_argument("shutdown deadline overflows the monotonic clock");
    return now + duration;
}
inline bool EngineShutdownGraceExpired(EngineShutdownClock::time_point deadline,
                                       EngineShutdownClock::time_point now = EngineShutdownClock::now()) {
    return now >= deadline;
}
