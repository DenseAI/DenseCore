#pragma once
#include "llm/speculative/sha256.h"
#include <array>
#include <cerrno>
#include <cstdio>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#if defined(__linux__) || defined(__APPLE__)
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace densecore::llm::speculative::qualification {
struct FileStamp {
    uint64_t device=0, inode=0, bytes=0;
    int64_t modified_seconds=0, modified_nanoseconds=0, changed_seconds=0, changed_nanoseconds=0;
    bool valid = false;
    bool operator==(const FileStamp& other) const {
        return valid && other.valid && device==other.device && inode==other.inode && bytes==other.bytes &&
            modified_seconds==other.modified_seconds && modified_nanoseconds==other.modified_nanoseconds &&
            changed_seconds==other.changed_seconds && changed_nanoseconds==other.changed_nanoseconds;
    }
    bool operator!=(const FileStamp& other) const { return !(*this == other); }
};
#if defined(__linux__) || defined(__APPLE__)
inline std::optional<FileStamp> StampFromStat(const struct stat& status) {
    if (!S_ISREG(status.st_mode) || status.st_size < 0) return std::nullopt;
    FileStamp result;
    result.device = status.st_dev; result.inode = status.st_ino; result.bytes = status.st_size;
#if defined(__APPLE__)
    result.modified_seconds = status.st_mtimespec.tv_sec; result.modified_nanoseconds = status.st_mtimespec.tv_nsec;
    result.changed_seconds = status.st_ctimespec.tv_sec; result.changed_nanoseconds = status.st_ctimespec.tv_nsec;
#else
    result.modified_seconds = status.st_mtim.tv_sec; result.modified_nanoseconds = status.st_mtim.tv_nsec;
    result.changed_seconds = status.st_ctim.tv_sec; result.changed_nanoseconds = status.st_ctim.tv_nsec;
#endif
    result.valid = true;
    return result;
}
#endif
// Capture before and after GGUF load. AUTO is eligible only if both snapshots
// agree; pass that unchanged loaded snapshot into HashLoadedModel afterwards.
inline std::optional<FileStamp> CaptureFileStamp(const std::string& path, std::string* error = nullptr) {
#if defined(__linux__) || defined(__APPLE__)
    struct stat status{};
    if (::stat(path.c_str(), &status) == 0) {
        auto result = StampFromStat(status);
        if (result) return result;
        if (error) *error = "model identity requires a regular file";
    } else if (error) *error = std::string("model stat failed: ") + std::strerror(errno);
#else
    (void)path;
    if (error) *error = "stable model file identity is unavailable on this platform";
#endif
    return std::nullopt;
}
struct ModelIdentity {
    bool stable_single_file = false;
    bool matches_loaded_file = false;
    std::string sha256;
    FileStamp stamp;
    std::string reason;
};
// Split GGUFs are unsupported for AUTO until an exact ordered multi-file
// identity contract is qualified. No filename-derived model qualification.
inline ModelIdentity HashLoadedModel(const std::string& path, const FileStamp& loaded,
                                     bool split_model, const std::function<void(uint64_t)>& progress = {}) {
    ModelIdentity result;
    const auto fail = [&](std::string reason) { result.reason = std::move(reason); return result; };
    if (split_model) return fail("automatic MTP does not qualify split model identities");
    if (!loaded.valid) return fail("loaded model file identity was not captured");
#if defined(__linux__) || defined(__APPLE__)
    std::unique_ptr<FILE,decltype(&std::fclose)> file(std::fopen(path.c_str(), "rb"), &std::fclose);
    if (!file) return fail(std::string("model open failed: ") + std::strerror(errno));
    struct stat before_status{};
    if (::fstat(::fileno(file.get()), &before_status) != 0) return fail("opened model stat failed");
    const auto before = StampFromStat(before_status);
    const auto named_before = CaptureFileStamp(path);
    if (!before || !named_before || *before != loaded || *named_before != loaded) {
        return fail("model file differs from the loaded snapshot");
    }
    try {
        Sha256 hash;
        std::array<uint8_t,64 * 1024> buffer;
        uint64_t bytes = 0;
        for (;;) {
            const size_t count = std::fread(buffer.data(), 1, buffer.size(), file.get());
            if (count) {
                hash.Update(buffer.data(), count);
                bytes += count;
                if (progress) progress(bytes);
            }
            if (count != buffer.size()) {
                if (std::ferror(file.get())) return fail("model identity read failed");
                if (std::feof(file.get())) break;
            }
        }
        struct stat after_status{};
        if (::fstat(::fileno(file.get()), &after_status) != 0) return fail("hashed model stat failed");
        const auto after = StampFromStat(after_status);
        const auto named_after = CaptureFileStamp(path);
        if (!after || !named_after || *after != loaded || *named_after != loaded || bytes != loaded.bytes) {
            return fail("model file changed while hashing");
        }
        result.sha256 = hash.HexDigest(); result.stamp = loaded;
        result.stable_single_file = true; result.matches_loaded_file = true;
        return result;
    } catch (const std::exception& error) { return fail(std::string("model identity failed: ") + error.what()); }
#else
    (void)path; (void)progress;
    return fail("stable model file identity is unavailable on this platform");
#endif
}
// Initialization is serial. Reuse across c=8..1 planning so a candidate model
// is read at most once; an empty production registry never calls Get().
class LazyModelIdentity {
public:
    explicit LazyModelIdentity(std::function<ModelIdentity()> provider) : provider_(std::move(provider)) {}
    const ModelIdentity& Get() {
        if (!value_) {
            try {
                value_ = provider_ ? provider_() : ModelIdentity{};
                if (!provider_) value_->reason = "model identity provider unavailable";
            } catch (const std::exception& error) {
                value_ = ModelIdentity{}; value_->reason = std::string("model identity provider failed: ") + error.what();
            }
        }
        return *value_;
    }
private:
    std::function<ModelIdentity()> provider_;
    std::optional<ModelIdentity> value_;
};
} // namespace densecore::llm::speculative::qualification
