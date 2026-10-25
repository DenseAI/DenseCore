#pragma once

#include <algorithm>
#include <cmath>
#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#if defined(__linux__)
#include <sched.h>
#endif

namespace densecore::runtime {
// A finite zero is exhausted; it must never be interpreted as unlimited/unknown.
enum class ResourceState : uint32_t { Unknown = 0, Finite = 1, Unlimited = 2 };
struct ResourceAmount {
    ResourceState state = ResourceState::Unknown;
    uint64_t bytes = 0;
};
struct ResourceBudget {
    ResourceAmount memory;
    ResourceAmount cgroup_memory_limit;
    ResourceState cpu_quota_state = ResourceState::Unknown;
    double cpu_quota = 0;
    int effective_threads = 1;
};
struct RuntimeMemoryPlan {
    uint64_t resource_capacity_bytes = 0; // Physical headroom before subtracting other committed plans.
    uint64_t available_bytes = 0;
    uint64_t kv_bytes = 0;
    uint64_t graph_bytes = 0;
    uint64_t repack_bytes = 0;
    uint64_t recurrent_bytes = 0;
    uint64_t prefix_snapshot_bytes = 0;
    uint64_t mtp_checkpoint_bytes = 0;
    uint64_t allocator_bytes = 0;
    uint64_t RequiredBytes() const {
        uint64_t total = 0;
        for (uint64_t bytes : {kv_bytes,graph_bytes,repack_bytes,recurrent_bytes,prefix_snapshot_bytes,mtp_checkpoint_bytes,allocator_bytes}) {
            if (bytes > UINT64_MAX-total) return UINT64_MAX;
            total += bytes;
        }
        return total;
    }
    bool Fits() const { return RequiredBytes() <= available_bytes; }
};
using ResourceReader = std::function<bool(const std::string&, std::string&)>;
inline bool ReadResourceFile(const std::string& path, std::string& text) {
    std::ifstream input(path);
    if (!input) return false;
    text.assign(std::istreambuf_iterator<char>(input), {});
    return !input.bad();
}
inline std::string ResourceWord(const std::string& text) {
    std::istringstream input(text); std::string word; input >> word; return word;
}
inline bool ResourceUnsigned(const std::string& text, uint64_t& value) {
    const auto word = ResourceWord(text);
    if (word.empty() || word[0] == '-') return false;
    try { size_t end = 0; value = std::stoull(word, &end); return end == word.size(); }
    catch (...) { return false; }
}
inline int ResourceCPUCount(const std::string& text) {
    uint64_t total = 0;
    std::istringstream ranges(ResourceWord(text)); std::string range;
    while (std::getline(ranges, range, ',')) {
        const auto dash = range.find('-'); uint64_t first = 0, last = 0;
        if (!ResourceUnsigned(range.substr(0, dash), first)) return 0;
        last = first;
        if (dash != std::string::npos && !ResourceUnsigned(range.substr(dash + 1), last)) return 0;
        if (last < first || last - first > 1000000) return 0;
        total += last - first + 1;
    }
    return total > 0 && total <= 1000000 ? static_cast<int>(total) : 0;
}
inline std::string ResourceUnescape(std::string s) {
    for (size_t p = 0; p + 3 < s.size(); ++p) {
        if (s[p] == '\\' && s[p+1] >= '0' && s[p+1] <= '7' && s[p+2] >= '0' && s[p+2] <= '7' && s[p+3] >= '0' && s[p+3] <= '7') {
            const char c = char((s[p+1]-'0')*64 + (s[p+2]-'0')*8 + s[p+3]-'0'); s.replace(p, 4, 1, c);
        }
    }
    return s;
}
inline bool ResourceController(const std::string& list, const std::string& controller) {
    return ("," + list + ",").find("," + controller + ",") != std::string::npos;
}
inline bool ResourceControllerWords(const std::string& list, const std::string& controller) {
    std::istringstream input(list); std::string word;
    while (input >> word) if (word == controller) return true;
    return false;
}
inline ResourceBudget DetectResourceBudget(const ResourceReader& read, int affinity_threads) {
    ResourceBudget result;
    result.effective_threads = std::max(1, affinity_threads);
    std::string text;
    if (read("/proc/meminfo", text)) {
        std::istringstream input(text); std::string line;
        while (std::getline(input, line)) {
            if (line.rfind("MemAvailable:", 0) == 0) {
                uint64_t kb = 0;
                if (ResourceUnsigned(line.substr(13), kb) && kb <= UINT64_MAX / 1024)
                    result.memory = {ResourceState::Finite, kb * 1024};
            }
        }
    }
    struct Membership { std::string controllers, path; bool v2; };
    std::vector<Membership> memberships;
    const bool membership_read = read("/proc/self/cgroup", text);
    bool membership_malformed = false;
    if (membership_read) {
        std::istringstream input(text); std::string line;
        while (std::getline(input, line)) {
            if (ResourceWord(line).empty()) continue;
            auto a = line.find(':'), b = line.find(':', a == std::string::npos ? 0 : a+1);
            uint64_t hierarchy = 0;
            if (a == std::string::npos || b == std::string::npos || b+1 >= line.size() ||
                line[b+1] != '/' || !ResourceUnsigned(line.substr(0,a),hierarchy)) {
                membership_malformed = true;
                continue;
            }
            memberships.push_back({line.substr(a+1,b-a-1), line.substr(b+1), b == a+1});
        }
    }
    struct Mount { std::string root, path, type, controllers; };
    std::vector<Mount> mounts;
    const bool mounts_read = read("/proc/self/mountinfo", text);
    bool mounts_malformed = false;
    bool saw_mount_record = false;
    if (mounts_read) {
        std::istringstream input(text); std::string line;
        while (std::getline(input, line)) {
            if (ResourceWord(line).empty()) continue;
            auto separator = line.find(" - ");
            if (separator == std::string::npos) { mounts_malformed = true; continue; }
            std::istringstream before(line.substr(0,separator)), after(line.substr(separator+3));
            std::string id, parent, device, root, path, options, type, source, super;
            if (!(before >> id >> parent >> device >> root >> path >> options) ||
                !(after >> type >> source >> super) || root.empty() || path.empty() || path[0] != '/') {
                mounts_malformed = true;
                continue;
            }
            saw_mount_record = true;
            if (type == "cgroup" || type == "cgroup2") {
                if (root[0] != '/') { mounts_malformed = true; continue; }
                mounts.push_back({ResourceUnescape(root), ResourceUnescape(path),type,super});
            }
        }
    }
    const bool empty_membership_with_cgroup_mount = memberships.empty() && !mounts.empty();
    const bool discovery_error = !membership_read || !mounts_read || membership_malformed || mounts_malformed ||
                                 !saw_mount_record || empty_membership_with_cgroup_mount;
    bool memory_error = discovery_error;
    bool cpu_error = discovery_error;
    const auto v1_has = [&](const char* controller) {
        return std::any_of(memberships.begin(),memberships.end(),[&](const auto& entry) {
            return !entry.v2 && ResourceController(entry.controllers,controller);
        });
    };
    const bool v1_memory = v1_has("memory"), v1_cpu = v1_has("cpu"), v1_cpuset = v1_has("cpuset");
    const bool has_v2 = std::any_of(memberships.begin(),memberships.end(),[](const auto& entry) { return entry.v2; });
    bool saw_v2_mount = false;
    bool needs_memory = v1_memory;
    bool needs_cpu = v1_cpu;
    bool observed_memory = false;
    bool observed_cpu = false;
    for (const auto& membership : memberships) for (const auto& mount : mounts) {
        if (membership.v2 != (mount.type == "cgroup2")) continue;
        std::string available_controllers;
        const bool has_controller_list = membership.v2 && read((std::filesystem::path(mount.path)/"cgroup.controllers").string(),available_controllers);
        if (membership.v2) saw_v2_mount = true;
        const auto v2_available = [&](const char* controller) {
            return !has_controller_list || ResourceControllerWords(available_controllers,controller);
        };
        // A controller belongs to only one hierarchy. A hybrid host's empty
        // v2 hierarchy must not mask its authoritative v1 CPU/memory limits.
        const bool memory = membership.v2 ? (!v1_memory && v2_available("memory"))
            : ResourceController(membership.controllers,"memory") && ResourceController(mount.controllers,"memory");
        const bool cpu = membership.v2 ? (!v1_cpu && v2_available("cpu"))
            : ResourceController(membership.controllers,"cpu") && ResourceController(mount.controllers,"cpu");
        const bool cpuset = membership.v2 ? (!v1_cpuset && v2_available("cpuset"))
            : ResourceController(membership.controllers,"cpuset") && ResourceController(mount.controllers,"cpuset");
        needs_memory = needs_memory || memory;
        needs_cpu = needs_cpu || cpu;
        if (!memory && !cpu && !cpuset) continue;
        std::string relative;
        if (mount.root == "/") relative = membership.path;
        else if (membership.path == mount.root) relative = "/";
        else if (membership.path.rfind(mount.root + "/",0) == 0) relative = membership.path.substr(mount.root.size());
        else if (membership.path == "/") relative = "/"; // cgroup namespace rooted at this mount
        else continue;
        const auto boundary = std::filesystem::path(mount.path).lexically_normal();
        auto path = (boundary / std::filesystem::path(relative).relative_path()).lexically_normal();
        if (path.string() != boundary.string() && path.string().rfind(boundary.string()+"/",0) != 0) continue;
        for (;;) {
            auto get = [&](const char* file, std::string& value) { errno = 0; return read((path/file).string(), value); };
            const auto unavailable_here = [&](const char* controller, int read_error) {
                std::string controllers;
                return membership.v2 && read_error == ENOENT && get("cgroup.controllers",controllers) &&
                       !ResourceControllerWords(controllers,controller);
            };
            if (memory) {
                observed_memory = true;
                std::string limit_text, current_text; uint64_t limit = 0, current = 0;
                if (!get(membership.v2 ? "memory.max" : "memory.limit_in_bytes",limit_text)) {
                    // The actual hierarchy root has no controller limit files.
                    // ENOENT is distinct from EACCES and malformed constrained ancestors.
                    const int read_error = errno;
                    if ((membership.v2 && path == boundary && mount.root == "/" && read_error == ENOENT)) {
                        if (result.cgroup_memory_limit.state == ResourceState::Unknown)
                            result.cgroup_memory_limit.state = ResourceState::Unlimited;
                    } else if (!unavailable_here("memory",read_error)) memory_error = true;
                } else if (ResourceWord(limit_text) == "max" || (!membership.v2 && ResourceUnsigned(limit_text,limit) && limit >= (uint64_t{1}<<60))) {
                    if (result.cgroup_memory_limit.state == ResourceState::Unknown) result.cgroup_memory_limit.state = ResourceState::Unlimited;
                } else if (!ResourceUnsigned(limit_text,limit) || !get(membership.v2 ? "memory.current" : "memory.usage_in_bytes",current_text) || !ResourceUnsigned(current_text,current)) memory_error = true;
                else {
                    if (result.cgroup_memory_limit.state != ResourceState::Finite || limit < result.cgroup_memory_limit.bytes) result.cgroup_memory_limit = {ResourceState::Finite,limit};
                    const uint64_t remaining = current >= limit ? 0 : limit-current;
                    if (result.memory.state != ResourceState::Finite || remaining < result.memory.bytes) result.memory = {ResourceState::Finite,remaining};
                }
            }
            if (cpu) {
                observed_cpu = true;
                std::string quota_text, period_text;
                bool ok = get(membership.v2 ? "cpu.max" : "cpu.cfs_quota_us", quota_text);
                const int quota_read_error = errno;
                std::istringstream fields(quota_text); std::string quota, period;
                fields >> quota;
                if (membership.v2) fields >> period; else if (get("cpu.cfs_period_us",period_text)) period = ResourceWord(period_text);
                uint64_t q = 0, p = 0;
                if (ok && (quota == "max" || quota == "-1")) {
                    if (result.cpu_quota_state == ResourceState::Unknown) result.cpu_quota_state = ResourceState::Unlimited;
                } else if (!ok && membership.v2 && path == boundary && mount.root == "/" && quota_read_error == ENOENT) {
                    if (result.cpu_quota_state == ResourceState::Unknown) result.cpu_quota_state = ResourceState::Unlimited;
                } else if (ok && ResourceUnsigned(quota,q) && ResourceUnsigned(period,p) && p > 0) {
                    double ratio = double(q)/double(p);
                    if (result.cpu_quota_state != ResourceState::Finite || ratio < result.cpu_quota) result.cpu_quota = ratio;
                    result.cpu_quota_state = ResourceState::Finite;
                } else if (ok || !unavailable_here("cpu",quota_read_error)) cpu_error = true;
            }
            if (cpuset) {
                std::string cpus;
                if (get(membership.v2 ? "cpuset.cpus.effective" : "cpuset.cpus",cpus)) {
                    int count = ResourceCPUCount(cpus);
                    if (count > 0) result.effective_threads = std::min(result.effective_threads,count);
                }
            }
            if (path == boundary || path == path.parent_path()) break;
            path = path.parent_path();
        }
    }
    if (has_v2 && !saw_v2_mount) {
        if (!v1_memory) memory_error = true;
        if (!v1_cpu) cpu_error = true;
    }
    // A missing finite constraint cannot safely be replaced by host availability.
    if (memory_error || (needs_memory && !observed_memory)) {
        if (result.memory.state != ResourceState::Finite || result.memory.bytes != 0)
            result.memory = {ResourceState::Unknown,0};
        if (result.cgroup_memory_limit.state != ResourceState::Finite || result.cgroup_memory_limit.bytes != 0)
            result.cgroup_memory_limit = {ResourceState::Unknown,0};
    }
    if (!memory_error && !needs_memory) result.cgroup_memory_limit = {ResourceState::Unlimited,0};
    if (!cpu_error && !needs_cpu) result.cpu_quota_state = ResourceState::Unlimited;
    if (needs_cpu && !observed_cpu) cpu_error = true;
    if (result.cpu_quota_state == ResourceState::Finite)
        result.effective_threads = std::min(result.effective_threads, static_cast<int>(std::min<double>(1000000,std::max(1.0,std::ceil(result.cpu_quota)))));
    if (cpu_error) {
        result.cpu_quota_state = ResourceState::Unknown;
        result.cpu_quota = 0;
        result.effective_threads = 1;
    }
    return result;
}
inline ResourceBudget DetectResourceBudget() {
    int threads = static_cast<int>(std::max(1u,std::thread::hardware_concurrency()));
#if defined(__linux__)
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0,sizeof(mask),&mask) == 0) threads = std::max(1,CPU_COUNT(&mask));
#endif
    return DetectResourceBudget(ReadResourceFile, threads);
}
} // namespace densecore::runtime
