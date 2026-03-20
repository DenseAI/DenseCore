#include "densecore/utils/logging.h"
#include <chrono>
#include <gtest/gtest.h>
#include <thread>

TEST(LoggingTest, BasicLogging) {
    // Initialize logging
    densecore::utils::InitLogging();

    // Test different log levels
    LOG_INFO("This is an INFO log from LoggingTest");
    LOG_WARN("This is a WARN log from LoggingTest");
    LOG_ERROR("This is an ERROR log from LoggingTest");

    // Verify async logging works (wait a bit for flush)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Check tracing (might not be enabled by default level)
    LOG_TRACE("This is a TRACE log (should be hidden by default)");
    LOG_DEBUG("This is a DEBUG log (should be hidden by default)");

    SUCCEED();
}
