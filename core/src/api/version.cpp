/**
 * @file version.cpp
 * @brief DenseCore version API implementation
 *
 * Implements GetLibraryVersion() and GetLibraryVersionString() using
 * version information injected by CMake at build time.
 */

#include "densecore.h"
#include "densecore_version.h"

// Static version info structure (never changes after compile)
static const DenseCoreVersionInfo g_version_info = {.major = DENSECORE_VERSION_MAJOR,
                                                    .minor = DENSECORE_VERSION_MINOR,
                                                    .patch = DENSECORE_VERSION_PATCH,
                                                    .version = DENSECORE_VERSION_STRING,
                                                    .commit = DENSECORE_GIT_COMMIT,
                                                    .build_time = DENSECORE_BUILD_TIMESTAMP,
                                                    .full = DENSECORE_VERSION_FULL};

extern "C" {

DENSECORE_API const DenseCoreVersionInfo* GetLibraryVersion(void) {
    return &g_version_info;
}

DENSECORE_API const char* GetLibraryVersionString(void) {
    return DENSECORE_VERSION_STRING;
}

}  // extern "C"
