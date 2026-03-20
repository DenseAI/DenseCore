#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <utility>

#ifdef DENSECORE_USE_SPDLOG
#include <spdlog/async.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace densecore {
namespace utils {

inline void InitLogging() {
    try {
        // Create async logger with a queue size of 8192 items
        // Overflow policy: Block producer (slow down instead of dropping logs)
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        spdlog::init_thread_pool(8192, 1);
        auto logger = std::make_shared<spdlog::async_logger>("densecore", console_sink, spdlog::thread_pool(),
                                                             spdlog::async_overflow_policy::block);

        // Pattern: [Time] [Thread] [Level] Message
        logger->set_pattern("[%H:%M:%S.%e] [t%t] [%^%l%$] %v");

        spdlog::set_default_logger(logger);
        spdlog::set_level(spdlog::level::info);
        spdlog::flush_every(std::chrono::seconds(3));
    } catch (const spdlog::spdlog_ex& ex) {
        std::cerr << "Log init failed: " << ex.what() << std::endl;
    }
}

}  // namespace utils
}  // namespace densecore

// Macros for high-performance logging
#define LOG_TRACE(...) SPDLOG_TRACE(__VA_ARGS__)
#define LOG_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)
#define LOG_INFO(...) SPDLOG_INFO(__VA_ARGS__)
#define LOG_WARN(...) SPDLOG_WARN(__VA_ARGS__)
#define LOG_ERROR(...) SPDLOG_ERROR(__VA_ARGS__)
#define LOG_CRITICAL(...) SPDLOG_CRITICAL(__VA_ARGS__)


#else

// Fallback to std::cout if spdlog is disabled

namespace densecore {
namespace utils {
inline void InitLogging() {}

template <typename... Args> void LogMessage(std::ostream& out, const char* prefix, Args&&... args) {
    out << prefix;
    ((out << std::forward<Args>(args) << " "), ...);
    out << std::endl;
}
}  // namespace utils
}  // namespace densecore

#define LOG_INFO(...) densecore::utils::LogMessage(std::cout, "[INFO] ", __VA_ARGS__)
#define LOG_WARN(...) densecore::utils::LogMessage(std::cerr, "[WARN] ", __VA_ARGS__)
#define LOG_ERROR(...) densecore::utils::LogMessage(std::cerr, "[ERROR] ", __VA_ARGS__)
#define LOG_TRACE(...)
#define LOG_DEBUG(...)
#define LOG_CRITICAL(...) densecore::utils::LogMessage(std::cerr, "[CRITICAL] ", __VA_ARGS__)

#endif
