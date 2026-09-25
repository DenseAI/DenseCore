/**
 * @file densecore/backend/hardware_topology.h
 * @brief Hardware topology detection and NUMA-aware thread affinity using hwloc
 *
 * Provides:
 * - NUMA node and core topology detection
 * - Physical vs hyper-thread core identification
 * - Thread pinning with Scatter/Compact policies
 * - Thread pool affinity management for GEMM workers
 */

#ifndef DENSECORE_HARDWARE_TOPOLOGY_H
#define DENSECORE_HARDWARE_TOPOLOGY_H

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef DENSECORE_USE_HWLOC
#include <hwloc.h>
#endif

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace densecore {

/**
 * Information about a single CPU core
 */
struct CoreInfo {
    int logical_id;       ///< OS-visible core ID (0-indexed)
    int physical_id;      ///< Physical core ID (same for SMT siblings)
    int numa_node;        ///< NUMA node this core belongs to
    bool is_hyperthread;  ///< true if this is an SMT sibling (not primary thread)
};

/**
 * Cache hierarchy information for adaptive GEMM tile sizing
 *
 * Used by batched GEMM kernels to compute optimal tile dimensions:
 * - TILE_K × TILE_N weight tile should fit in L2 per core
 * - TILE_M activation rows × TILE_K should fit in L1
 * - Full working set should fit in L3 per socket
 */
struct CacheHierarchy {
    int l1d_size_bytes = 32 * 1024;        ///< L1 data cache per core (default: 32KB)
    int l2_size_bytes = 256 * 1024;        ///< L2 cache per core (default: 256KB)
    int l3_size_bytes = 16 * 1024 * 1024;  ///< L3 cache per socket (default: 16MB)
    int cache_line_bytes = 64;             ///< Cache line size (default: 64B)
    int cores_sharing_l3 = 0;              ///< Physical cores sharing L3 (0 = unknown)
};

/**
 * Thread pinning policy for multi-core systems
 */
enum class PinningPolicy {
    COMPACT,  ///< Pack threads on adjacent cores (share L2 cache)
    SCATTER   ///< Spread across physical cores (maximize L3 utilization)
};

/**
 * Hardware topology singleton using hwloc for precise core/NUMA detection
 *
 * Falls back to sysfs-based detection when hwloc is not available.
 */
class HardwareTopology {
public:
    /**
     * Get singleton instance (thread-safe, lazy initialization)
     */
    static HardwareTopology& GetInstance() {
        static HardwareTopology instance;
        return instance;
    }

    // Non-copyable
    HardwareTopology(const HardwareTopology&) = delete;
    HardwareTopology& operator=(const HardwareTopology&) = delete;

    // =========================================================================
    // Topology Queries
    // =========================================================================

    /**
     * Get number of NUMA nodes in the system
     * @return Number of NUMA nodes (minimum 1)
     */
    int GetNumaNodeCount() const { return numa_node_count_; }

    /**
     * Get total number of logical CPUs
     */
    int GetLogicalCoreCount() const { return static_cast<int>(cores_.size()); }

    /**
     * Get number of physical cores (excluding hyper-threads)
     * @param numa_node Specific NUMA node (-1 for all nodes)
     */
    int GetPhysicalCoreCount(int numa_node = -1) const {
        int count = 0;
        for (const auto& core : cores_) {
            if (!core.is_hyperthread) {
                if (numa_node < 0 || core.numa_node == numa_node) {
                    count++;
                }
            }
        }
        return count;
    }

    /**
     * Get all core info for a NUMA node
     * @param numa_node NUMA node ID (0-indexed)
     * @return Vector of CoreInfo for cores in that node
     */
    std::vector<CoreInfo> GetCoresInNumaNode(int numa_node) const {
        std::vector<CoreInfo> result;
        for (const auto& core : cores_) {
            if (core.numa_node == numa_node) {
                result.push_back(core);
            }
        }
        return result;
    }

    /**
     * Get physical core IDs only (excludes hyper-threads)
     * @param numa_node Specific NUMA node (-1 for all)
     * @return Vector of logical core IDs for physical cores
     */
    std::vector<int> GetPhysicalCoreIds(int numa_node = -1) const {
        std::vector<int> result;
        for (const auto& core : cores_) {
            if (!core.is_hyperthread) {
                if (numa_node < 0 || core.numa_node == numa_node) {
                    result.push_back(core.logical_id);
                }
            }
        }
        return result;
    }

    /**
     * Get every physical core in the machine, ordered so that consecutive
     * entries alternate NUMA nodes (node0-core0, node1-core0, node0-core1, ...).
     *
     * GetPhysicalCoreIds(-1) returns cores grouped by node, so a thread count
     * smaller than the machine's core count would land entirely on the first
     * node. Interleaving keeps any prefix of the list spread across sockets,
     * which is what a compute pool sized below the core count needs.
     */
    std::vector<int> GetPhysicalCoreIdsInterleavedByNode() const {
        std::vector<std::vector<int>> per_node(static_cast<size_t>(std::max(1, numa_node_count_)));
        std::vector<int> unknown_node;
        for (const auto& core : cores_) {
            if (core.is_hyperthread) {
                continue;
            }
            if (core.numa_node >= 0 && static_cast<size_t>(core.numa_node) < per_node.size()) {
                per_node[static_cast<size_t>(core.numa_node)].push_back(core.logical_id);
            } else {
                unknown_node.push_back(core.logical_id);
            }
        }
        std::vector<int> result;
        size_t max_per_node = unknown_node.size();
        for (const auto& node_cores : per_node) {
            max_per_node = std::max(max_per_node, node_cores.size());
        }
        for (size_t slot = 0; slot < max_per_node; ++slot) {
            for (const auto& node_cores : per_node) {
                if (slot < node_cores.size()) {
                    result.push_back(node_cores[slot]);
                }
            }
            if (slot < unknown_node.size()) {
                result.push_back(unknown_node[slot]);
            }
        }
        return result;
    }

    /**
     * NUMA node a logical core belongs to, or -1 when the core is unknown.
     */
    int GetNumaNodeOfCore(int logical_id) const {
        for (const auto& core : cores_) {
            if (core.logical_id == logical_id) {
                return core.numa_node;
            }
        }
        return -1;
    }

    // =========================================================================
    // Thread Pinning
    // =========================================================================

    /**
     * Pin the calling thread to a specific CPU core
     * @param core_id Logical core ID (0-indexed)
     * @return true on success
     */
    static bool PinCurrentThread(int core_id) {
#if defined(__linux__) && !defined(__ANDROID__)
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) == 0;
#elif defined(_WIN32)
        if (core_id < 0 || core_id >= 64) return false;
        DWORD_PTR mask = 1ULL << core_id;
        return SetThreadAffinityMask(GetCurrentThread(), mask) != 0;
#else
        (void)core_id;
        return false;
#endif
    }

    /**
     * Pin the calling thread to a NUMA node using specified policy
     * @param numa_node Target NUMA node
     * @param policy SCATTER spreads across physical cores, COMPACT packs densely
     * @return true if pinned successfully
     */
    bool PinCurrentThreadToNumaNode(int numa_node, PinningPolicy policy = PinningPolicy::SCATTER) {
        auto cores = GetPhysicalCoreIds(numa_node);
        if (cores.empty()) return false;

        // For single-thread pinning, use first available physical core
        int target_core = cores[0];
        if (policy == PinningPolicy::SCATTER && cores.size() > 1) {
            // Use a simple round-robin for multiple callers
            static std::atomic<int> scatter_idx{0};
            int idx = scatter_idx.fetch_add(1) % cores.size();
            target_core = cores[idx];
        }

        return PinCurrentThread(target_core);
    }

    /**
     * Pin a thread pool to cores within a NUMA node
     *
     * SCATTER: Distributes threads across physical cores to maximize L3 sharing
     * COMPACT: Packs threads on adjacent cores for L2 sharing
     *
     * @param threads Vector of threads to pin (must be joinable)
     * @param numa_node Target NUMA node (-1 for node 0)
     * @param policy Pinning strategy
     */
    void PinThreadPool(std::vector<std::thread>& threads, int numa_node,
                       PinningPolicy policy = PinningPolicy::SCATTER) {
        int target_node = (numa_node >= 0) ? numa_node : 0;
        auto physical_cores = GetPhysicalCoreIds(target_node);
        if (physical_cores.empty()) return;

        size_t num_threads = threads.size();
        size_t num_cores = physical_cores.size();

        for (size_t i = 0; i < num_threads; ++i) {
            if (!threads[i].joinable()) continue;

            int target_core;
            if (policy == PinningPolicy::SCATTER) {
                // Spread: thread i -> core i % num_cores
                target_core = physical_cores[i % num_cores];
            } else {
                // Compact: pack threads on same cores if more threads than cores
                target_core = physical_cores[std::min(i, num_cores - 1)];
            }

            // Pin using native handle
            PinThreadByHandle(threads[i].native_handle(), target_core);
        }
    }

    /**
     * Pin a thread by its native handle
     * Used for GGML thread pool integration
     */
    static bool PinThreadByHandle(std::thread::native_handle_type handle, int core_id) {
#if defined(__linux__) && !defined(__ANDROID__)
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core_id, &cpuset);
        return pthread_setaffinity_np(handle, sizeof(cpu_set_t), &cpuset) == 0;
#elif defined(_WIN32)
        if (core_id < 0 || core_id >= 64) return false;
        DWORD_PTR mask = 1ULL << core_id;
        return SetThreadAffinityMask((HANDLE)handle, mask) != 0;
#else
        (void)handle;
        (void)core_id;
        return false;
#endif
    }

    /**
     * Execute a function on each thread in a pool with NUMA-aware pinning
     * Useful for late binding when threads are created externally (e.g., GGML)
     *
     * @param thread_init_fn Function called on each thread with (thread_idx,
     * core_id)
     * @param num_threads Number of threads to configure
     * @param numa_node Target NUMA node
     * @param policy Pinning strategy
     */
    void ConfigureThreadPoolAffinity(std::function<void(int, int)> thread_init_fn, int num_threads, int numa_node,
                                     PinningPolicy policy = PinningPolicy::SCATTER) {
        int target_node = (numa_node >= 0) ? numa_node : 0;
        auto physical_cores = GetPhysicalCoreIds(target_node);
        if (physical_cores.empty()) return;

        size_t num_cores = physical_cores.size();

        for (int i = 0; i < num_threads; ++i) {
            int target_core;
            if (policy == PinningPolicy::SCATTER) {
                target_core = physical_cores[i % num_cores];
            } else {
                target_core = physical_cores[std::min(static_cast<size_t>(i), num_cores - 1)];
            }
            thread_init_fn(i, target_core);
        }
    }

    // =========================================================================
    // Compute Thread Affinity (for GGML thread pool integration)
    // =========================================================================

    /**
     * Setup compute thread affinity mapping for GGML workers
     *
     * Call this before compute operations. It pre-computes the core assignment
     * for each thread index so workers can pin themselves on first use.
     *
     * @param numa_node Target NUMA node
     * @param n_threads Number of compute threads (from ggml n_threads setting)
     * @param policy SCATTER spreads threads across physical cores
     */
    void SetupComputeThreadAffinity(int numa_node, int n_threads, PinningPolicy policy = PinningPolicy::SCATTER) {
        std::lock_guard<std::mutex> lock(compute_affinity_mu_);
        if (n_threads <= 0) return;

        // A negative numa_node means "no node was requested" and must NOT collapse
        // to node 0. Callers size n_threads from the machine-wide physical core
        // count, so pinning that many threads into a single node's cores
        // oversubscribes every core on that node (2 threads/core on a 2-socket
        // box) while the other socket's cores stay completely idle for the whole
        // run. Spread machine-wide instead, and only honour a node restriction
        // when one was actually asked for.
        const bool machine_wide = numa_node < 0;
        auto physical_cores = machine_wide ? GetPhysicalCoreIdsInterleavedByNode() : GetPhysicalCoreIds(numa_node);
        if (physical_cores.empty()) return;

        compute_thread_cores_.resize(n_threads);
        compute_thread_nodes_.resize(n_threads);
        size_t num_cores = physical_cores.size();

        for (int i = 0; i < n_threads; ++i) {
            if (policy == PinningPolicy::SCATTER) {
                compute_thread_cores_[i] = physical_cores[i % num_cores];
            } else {
                compute_thread_cores_[i] = physical_cores[std::min(static_cast<size_t>(i), num_cores - 1)];
            }
            compute_thread_nodes_[i] = GetNumaNodeOfCore(compute_thread_cores_[i]);
        }

        compute_affinity_configured_ = true;
        compute_affinity_numa_node_ = machine_wide ? -1 : numa_node;

        if (machine_wide) {
            fprintf(stderr,
                    "[HardwareTopology] Compute thread affinity configured: "
                    "%d threads across %d NUMA node(s), %zu physical cores (%s policy)\n",
                    n_threads, numa_node_count_, num_cores, (policy == PinningPolicy::SCATTER) ? "SCATTER" : "COMPACT");
        } else {
            fprintf(stderr,
                    "[HardwareTopology] Compute thread affinity configured: "
                    "%d threads on NUMA node %d, %zu physical cores (%s policy)\n",
                    n_threads, numa_node, num_cores, (policy == PinningPolicy::SCATTER) ? "SCATTER" : "COMPACT");
        }
    }

    void SetupComputeThreadAffinityFromCoreIds(const std::vector<int>& core_ids, int n_threads,
                                               PinningPolicy policy = PinningPolicy::SCATTER) {
        std::lock_guard<std::mutex> lock(compute_affinity_mu_);
        if (core_ids.empty() || n_threads <= 0) return;

        std::vector<int> sorted_ids = core_ids;
        std::sort(sorted_ids.begin(), sorted_ids.end());
        sorted_ids.erase(std::unique(sorted_ids.begin(), sorted_ids.end()), sorted_ids.end());
        if (sorted_ids.empty()) return;

        compute_thread_cores_.resize(n_threads);
        compute_thread_nodes_.resize(n_threads);
        const size_t num_cores = sorted_ids.size();
        for (int i = 0; i < n_threads; ++i) {
            if (policy == PinningPolicy::SCATTER) {
                compute_thread_cores_[i] = sorted_ids[static_cast<size_t>(i) % num_cores];
            } else {
                compute_thread_cores_[i] = sorted_ids[std::min(static_cast<size_t>(i), num_cores - 1)];
            }
            compute_thread_nodes_[i] = GetNumaNodeOfCore(compute_thread_cores_[i]);
        }

        compute_affinity_configured_ = true;
        compute_affinity_numa_node_ = -1;

        fprintf(stderr,
                "[HardwareTopology] Compute thread affinity configured: "
                "%d threads on explicit core set (%zu cores, %s policy)\n",
                n_threads, sorted_ids.size(), (policy == PinningPolicy::SCATTER) ? "SCATTER" : "COMPACT");
    }

    /**
     * Get the assigned core for a compute thread index
     *
     * @param thread_idx Thread index (ith from GGML callback)
     * @return Core ID to pin to, or -1 if not configured
     */
    int GetAssignedCore(int thread_idx) const {
        std::lock_guard<std::mutex> lock(compute_affinity_mu_);
        if (!compute_affinity_configured_ || thread_idx < 0 ||
            thread_idx >= static_cast<int>(compute_thread_cores_.size())) {
            return -1;
        }
        return compute_thread_cores_[thread_idx];
    }

    /**
     * NUMA node a compute thread index was pinned to, or -1 when affinity is not
     * configured or the core's node is unknown.
     */
    int GetAssignedNumaNode(int thread_idx) const {
        std::lock_guard<std::mutex> lock(compute_affinity_mu_);
        if (!compute_affinity_configured_ || thread_idx < 0 ||
            thread_idx >= static_cast<int>(compute_thread_nodes_.size())) {
            return -1;
        }
        return compute_thread_nodes_[thread_idx];
    }

    /**
     * Pin the current thread based on its GGML thread index
     *
     * Call this from within GGML callbacks (cb_int4_gemm, etc.) on first
     * invocation. Uses thread-local flag to avoid re-pinning on every call.
     *
     * @param thread_idx Thread index (ith from GGML callback)
     * @return true if pinned successfully (or already pinned)
     */
    bool PinComputeThread(int thread_idx) {
        int core_id = GetAssignedCore(thread_idx);
        if (core_id < 0) {
            return false;  // Affinity not configured
        }

        // GGML clears the caller's affinity after every graph while its worker
        // threads stay alive. A boolean TLS therefore becomes stale for task 0.
        // The current CPU is not sufficient proof because a widened mask may
        // still happen to be executing on core_id; verify the singleton mask.
        thread_local int pinned_core = -1;
#if defined(__linux__) && !defined(__ANDROID__)
        if (pinned_core == core_id) {
            cpu_set_t current_mask;
            CPU_ZERO(&current_mask);
            if (pthread_getaffinity_np(pthread_self(), sizeof(current_mask), &current_mask) == 0 &&
                CPU_COUNT(&current_mask) == 1 && CPU_ISSET(core_id, &current_mask)) {
                return true;
            }
        }
#else
        if (pinned_core == core_id) {
            return true;
        }
#endif
        if (PinCurrentThread(core_id)) {
            pinned_core = core_id;
            return true;
        }
        return false;
    }

    /**
     * Check if compute thread affinity is configured
     */
    bool IsComputeAffinityConfigured() const {
        std::lock_guard<std::mutex> lock(compute_affinity_mu_);
        return compute_affinity_configured_;
    }

    // =========================================================================
    // Cache Hierarchy Queries (for adaptive GEMM tile sizing)
    // =========================================================================

    /**
     * Get cache hierarchy information for the system
     *
     * Detection priority:
     * 1. hwloc (most accurate, works on all platforms)
     * 2. sysfs /sys/devices/system/cpu/cpu0/cache/ (Linux)
     * 3. Conservative defaults (safe for any x86/ARM)
     */
    const CacheHierarchy& GetCacheHierarchy() const { return cache_hierarchy_; }

private:
#if defined(__APPLE__)
    static bool ReadSysctlInt(const char* name, int* out_value) {
        if (!name || !out_value) return false;
        int value = 0;
        size_t size = sizeof(value);
        if (sysctlbyname(name, &value, &size, nullptr, 0) != 0 || size != sizeof(value)) {
            return false;
        }
        *out_value = value;
        return true;
    }

    static bool ReadSysctlSize(const char* name, size_t* out_value) {
        if (!name || !out_value) return false;
        uint64_t value = 0;
        size_t size = sizeof(value);
        if (sysctlbyname(name, &value, &size, nullptr, 0) != 0) {
            return false;
        }
        if (size == sizeof(uint32_t)) {
            *out_value = static_cast<size_t>(static_cast<uint32_t>(value));
        } else {
            *out_value = static_cast<size_t>(value);
        }
        return true;
    }
#endif

    HardwareTopology() { Initialize(); }

    ~HardwareTopology() {
#ifdef DENSECORE_USE_HWLOC
        if (topology_initialized_) {
            hwloc_topology_destroy(topology_);
        }
#endif
    }

    void Initialize() {
#ifdef DENSECORE_USE_HWLOC
        InitializeWithHwloc();
#else
        InitializeWithSysfs();
#endif
        DetectCacheHierarchy();
    }

#ifdef DENSECORE_USE_HWLOC
    void InitializeWithHwloc() {
        if (hwloc_topology_init(&topology_) < 0) {
            InitializeWithSysfs();
            return;
        }

        if (hwloc_topology_load(topology_) < 0) {
            hwloc_topology_destroy(topology_);
            InitializeWithSysfs();
            return;
        }

        topology_initialized_ = true;

        // Count NUMA nodes
        int depth = hwloc_get_type_depth(topology_, HWLOC_OBJ_NUMANODE);
        if (depth != HWLOC_TYPE_DEPTH_UNKNOWN) {
            numa_node_count_ = hwloc_get_nbobjs_by_depth(topology_, depth);
        }
        if (numa_node_count_ < 1) numa_node_count_ = 1;

        // Enumerate PUs (processing units = logical cores)
        int num_pus = hwloc_get_nbobjs_by_type(topology_, HWLOC_OBJ_PU);
        cores_.reserve(num_pus);

        for (int i = 0; i < num_pus; ++i) {
            hwloc_obj_t pu = hwloc_get_obj_by_type(topology_, HWLOC_OBJ_PU, i);
            if (!pu) continue;

            CoreInfo info;
            info.logical_id = pu->os_index;

            // Find parent core to get physical_id and hyper-thread status
            hwloc_obj_t core = hwloc_get_ancestor_obj_by_type(topology_, HWLOC_OBJ_CORE, pu);
            if (core) {
                info.physical_id = core->logical_index;
                // First PU in core is primary, others are hyper-threads
                info.is_hyperthread = (pu != core->first_child);
            } else {
                info.physical_id = info.logical_id;
                info.is_hyperthread = false;
            }

            // Find NUMA node.
            //
            // NOT via hwloc_get_ancestor_obj_by_type(HWLOC_OBJ_NUMANODE, pu):
            // hwloc 2.0 moved NUMA nodes out of the CPU hierarchy and attached
            // them as memory children of their local object, so that lookup
            // returns NULL for every PU on hwloc >= 2. The old code then fell
            // into its "no NUMA node" fallback and labelled EVERY core node 0.
            // A 2-socket host still reported numa_node_count_ == 2 (that count
            // comes from a separate query and stayed correct), so the mislabel
            // was invisible in the node count while GetPhysicalCoreIds(1) came
            // back empty -- which silently turned every node-1 pin into a no-op.
            //
            // Match the PU against each NUMA node's cpuset instead, which is the
            // supported hwloc 2 idiom and still correct on hwloc 1.
            info.numa_node = -1;
            hwloc_obj_t numa = nullptr;
            while ((numa = hwloc_get_next_obj_by_type(topology_, HWLOC_OBJ_NUMANODE, numa)) != nullptr) {
                if (numa->cpuset && hwloc_bitmap_isset(numa->cpuset, pu->os_index)) {
                    info.numa_node = numa->logical_index;
                    break;
                }
            }
            if (info.numa_node < 0) {
                // Walk up to the nearest object that carries a nodeset, then map
                // that node's OS index back to its logical index.
                hwloc_obj_t ancestor = pu;
                while (ancestor && (!ancestor->nodeset || hwloc_bitmap_iszero(ancestor->nodeset))) {
                    ancestor = ancestor->parent;
                }
                if (ancestor && ancestor->nodeset) {
                    const int os_node = hwloc_bitmap_first(ancestor->nodeset);
                    hwloc_obj_t probe = nullptr;
                    while ((probe = hwloc_get_next_obj_by_type(topology_, HWLOC_OBJ_NUMANODE, probe)) != nullptr) {
                        if (static_cast<int>(probe->os_index) == os_node) {
                            info.numa_node = probe->logical_index;
                            break;
                        }
                    }
                }
            }
            if (info.numa_node < 0) {
                // Genuinely no NUMA information (memory reported at package level).
                info.numa_node = 0;
            }

            cores_.push_back(info);
        }

        // Sort by logical_id for consistent ordering
        std::sort(cores_.begin(), cores_.end(),
                  [](const CoreInfo& a, const CoreInfo& b) { return a.logical_id < b.logical_id; });
    }
#endif  // DENSECORE_USE_HWLOC

    void InitializeWithSysfs() {
        // Fallback: basic detection without hwloc
        numa_node_count_ = 1;

#if defined(__linux__)
        // Count NUMA nodes
        for (int i = 0; i < 256; ++i) {
            char path[64];
            snprintf(path, sizeof(path), "/sys/devices/system/node/node%d", i);
            if (access(path, F_OK) == 0) {
                numa_node_count_ = i + 1;
            } else {
                break;
            }
        }

        // Get number of CPUs
        int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
        if (num_cpus <= 0) {
            num_cpus = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        }
        cores_.reserve(num_cpus);

        // sysconf() counts logical CPUs.  When hwloc is unavailable, recover
        // physical-core identity from Linux' topology files so SMT siblings do
        // not inflate the decode worker cap.
        std::vector<std::pair<int, int>> physical_core_keys;
        for (int i = 0; i < num_cpus; ++i) {
            CoreInfo info;
            info.logical_id = i;
            info.numa_node = 0;

            int package_id = -1;
            int core_id = -1;
            {
                char package_path[128];
                char core_path[128];
                snprintf(package_path, sizeof(package_path),
                         "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", i);
                snprintf(core_path, sizeof(core_path), "/sys/devices/system/cpu/cpu%d/topology/core_id", i);
                std::ifstream(package_path) >> package_id;
                std::ifstream(core_path) >> core_id;
            }

            const std::pair<int, int> key{package_id, core_id};
            const auto known_core = std::find(physical_core_keys.begin(), physical_core_keys.end(), key);
            const bool is_known_physical_core = known_core != physical_core_keys.end();
            if (package_id >= 0 && core_id >= 0) {
                if (!is_known_physical_core) {
                    physical_core_keys.push_back(key);
                    info.physical_id = static_cast<int>(physical_core_keys.size() - 1);
                } else {
                    info.physical_id = static_cast<int>(known_core - physical_core_keys.begin());
                }
            } else {
                info.physical_id = i;
            }

            // The first logical CPU carrying a package/core pair is its
            // primary thread.  CPU ids are enumerated in ascending order, so
            // this also avoids parsing topology list ranges such as "0-1".
            info.is_hyperthread = package_id >= 0 && core_id >= 0 && is_known_physical_core;

            // Try to detect NUMA node from sysfs
            for (int n = 0; n < numa_node_count_; ++n) {
                char path[128];
                snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpu%d", n, i);
                if (access(path, F_OK) == 0) {
                    info.numa_node = n;
                    break;
                }
            }

            cores_.push_back(info);
        }
#elif defined(_WIN32)
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        int num_cpus = sysinfo.dwNumberOfProcessors;

        cores_.reserve(num_cpus);
        for (int i = 0; i < num_cpus; ++i) {
            CoreInfo info;
            info.logical_id = i;
            info.physical_id = i;
            info.numa_node = 0;
            info.is_hyperthread = false;
            cores_.push_back(info);
        }

        // Try to get NUMA node count on Windows
        ULONG highest_node = 0;
        if (GetNumaHighestNodeNumber(&highest_node)) {
            numa_node_count_ = static_cast<int>(highest_node) + 1;
        }
#elif defined(__APPLE__)
        int logical_cpus = 0;
        int physical_cpus = 0;
        if (!ReadSysctlInt("hw.logicalcpu", &logical_cpus)) {
            ReadSysctlInt("hw.logicalcpu_max", &logical_cpus);
        }
        if (!ReadSysctlInt("hw.physicalcpu", &physical_cpus)) {
            ReadSysctlInt("hw.physicalcpu_max", &physical_cpus);
        }
        if (logical_cpus <= 0) {
            logical_cpus = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        }
        if (physical_cpus <= 0 || physical_cpus > logical_cpus) {
            physical_cpus = logical_cpus;
        }

        cores_.reserve(logical_cpus);
        for (int i = 0; i < logical_cpus; ++i) {
            CoreInfo info;
            info.logical_id = i;
            info.physical_id = (physical_cpus > 0) ? (i % physical_cpus) : i;
            info.numa_node = 0;
            info.is_hyperthread = physical_cpus > 0 && i >= physical_cpus;
            cores_.push_back(info);
        }
#endif
    }

    /**
     * Detect L1/L2/L3 cache sizes from hardware
     * Tries hwloc first, then sysfs, then uses conservative defaults.
     */
    void DetectCacheHierarchy() {
#ifdef DENSECORE_USE_HWLOC
        if (topology_initialized_) {
            // L1 data cache
            int l1_depth = hwloc_get_type_depth(topology_, HWLOC_OBJ_L1CACHE);
            if (l1_depth != HWLOC_TYPE_DEPTH_UNKNOWN) {
                hwloc_obj_t l1 = hwloc_get_obj_by_depth(topology_, l1_depth, 0);
                if (l1 && l1->attr && l1->attr->cache.size > 0) {
                    cache_hierarchy_.l1d_size_bytes = static_cast<int>(l1->attr->cache.size);
                    if (l1->attr->cache.linesize > 0) {
                        cache_hierarchy_.cache_line_bytes = static_cast<int>(l1->attr->cache.linesize);
                    }
                }
            }
            // L2 cache
            int l2_depth = hwloc_get_type_depth(topology_, HWLOC_OBJ_L2CACHE);
            if (l2_depth != HWLOC_TYPE_DEPTH_UNKNOWN) {
                hwloc_obj_t l2 = hwloc_get_obj_by_depth(topology_, l2_depth, 0);
                if (l2 && l2->attr && l2->attr->cache.size > 0) {
                    cache_hierarchy_.l2_size_bytes = static_cast<int>(l2->attr->cache.size);
                }
            }
            // L3 cache
            int l3_depth = hwloc_get_type_depth(topology_, HWLOC_OBJ_L3CACHE);
            if (l3_depth != HWLOC_TYPE_DEPTH_UNKNOWN) {
                hwloc_obj_t l3 = hwloc_get_obj_by_depth(topology_, l3_depth, 0);
                if (l3 && l3->attr && l3->attr->cache.size > 0) {
                    cache_hierarchy_.l3_size_bytes = static_cast<int>(l3->attr->cache.size);
                }
                // Count cores sharing L3 (number of PUs under first L3)
                if (l3_depth != HWLOC_TYPE_DEPTH_UNKNOWN) {
                    hwloc_obj_t l3_obj = hwloc_get_obj_by_depth(topology_, l3_depth, 0);
                    if (l3_obj) {
                        int core_depth = hwloc_get_type_depth(topology_, HWLOC_OBJ_CORE);
                        if (core_depth != HWLOC_TYPE_DEPTH_UNKNOWN) {
                            int cores_in_l3 =
                                hwloc_get_nbobjs_inside_cpuset_by_depth(topology_, l3_obj->cpuset, core_depth);
                            cache_hierarchy_.cores_sharing_l3 = cores_in_l3;
                        }
                    }
                }
            }
            return;
        }
#endif  // DENSECORE_USE_HWLOC

#if defined(__linux__)
        // sysfs fallback: read from /sys/devices/system/cpu/cpu0/cache/
        auto read_sysfs_cache = [](const char* path) -> int {
            FILE* f = fopen(path, "r");
            if (!f) return 0;
            char buf[64];
            if (!fgets(buf, sizeof(buf), f)) {
                fclose(f);
                return 0;
            }
            fclose(f);
            long val = strtol(buf, nullptr, 10);
            // sysfs reports in KB
            return static_cast<int>(val * 1024);
        };

        // Index 0 = L1d (data), Index 2 = L2, Index 3 = L3 (typical layout)
        // But verify via type file
        for (int idx = 0; idx < 8; ++idx) {
            char type_path[128];
            snprintf(type_path, sizeof(type_path), "/sys/devices/system/cpu/cpu0/cache/index%d/type", idx);
            FILE* tf = fopen(type_path, "r");
            if (!tf) break;
            char type_buf[32];
            if (!fgets(type_buf, sizeof(type_buf), tf)) {
                fclose(tf);
                continue;
            }
            fclose(tf);

            char level_path[128];
            snprintf(level_path, sizeof(level_path), "/sys/devices/system/cpu/cpu0/cache/index%d/level", idx);
            FILE* lf = fopen(level_path, "r");
            if (!lf) continue;
            char level_buf[8];
            if (!fgets(level_buf, sizeof(level_buf), lf)) {
                fclose(lf);
                continue;
            }
            fclose(lf);
            int level = atoi(level_buf);

            char size_path[128];
            snprintf(size_path, sizeof(size_path), "/sys/devices/system/cpu/cpu0/cache/index%d/size", idx);

            bool is_data = (strncmp(type_buf, "Data", 4) == 0 || strncmp(type_buf, "Unified", 7) == 0);
            int size = read_sysfs_cache(size_path);
            if (size <= 0) continue;

            if (level == 1 && is_data) {
                cache_hierarchy_.l1d_size_bytes = size;
            } else if (level == 2 && is_data) {
                cache_hierarchy_.l2_size_bytes = size;
            } else if (level == 3) {
                cache_hierarchy_.l3_size_bytes = size;
            }
        }

        // Detect cores sharing L3: physical cores per NUMA node
        if (numa_node_count_ > 0) {
            cache_hierarchy_.cores_sharing_l3 = GetPhysicalCoreCount(0);
        }
#elif defined(__APPLE__)
        size_t cache_line = 0;
        size_t l1d = 0;
        size_t l2 = 0;
        size_t l3 = 0;
        if (ReadSysctlSize("hw.cachelinesize", &cache_line) && cache_line > 0) {
            cache_hierarchy_.cache_line_bytes = static_cast<int>(cache_line);
        }
        if (ReadSysctlSize("hw.l1dcachesize", &l1d) && l1d > 0) {
            cache_hierarchy_.l1d_size_bytes = static_cast<int>(l1d);
        }
        if (ReadSysctlSize("hw.l2cachesize", &l2) && l2 > 0) {
            cache_hierarchy_.l2_size_bytes = static_cast<int>(l2);
        }
        if (ReadSysctlSize("hw.l3cachesize", &l3) && l3 > 0) {
            cache_hierarchy_.l3_size_bytes = static_cast<int>(l3);
        }

        int perf_cluster_cores = 0;
        if (ReadSysctlInt("hw.perflevel0.physicalcpu", &perf_cluster_cores) && perf_cluster_cores > 0) {
            cache_hierarchy_.cores_sharing_l3 = perf_cluster_cores;
        } else {
            cache_hierarchy_.cores_sharing_l3 = GetPhysicalCoreCount();
        }
#endif  // __linux__

        fprintf(stderr, "[HardwareTopology] Cache: L1d=%dKB L2=%dKB L3=%dMB (%d cores/L3)\n",
                cache_hierarchy_.l1d_size_bytes / 1024, cache_hierarchy_.l2_size_bytes / 1024,
                cache_hierarchy_.l3_size_bytes / (1024 * 1024), cache_hierarchy_.cores_sharing_l3);
    }

#ifdef DENSECORE_USE_HWLOC
    hwloc_topology_t topology_;
    bool topology_initialized_ = false;
#endif

    std::vector<CoreInfo> cores_;
    int numa_node_count_ = 1;
    CacheHierarchy cache_hierarchy_;

    // Compute thread affinity state
    mutable std::mutex compute_affinity_mu_;
    std::vector<int> compute_thread_cores_;
    std::vector<int> compute_thread_nodes_;
    bool compute_affinity_configured_ = false;
    int compute_affinity_numa_node_ = -1;
};

}  // namespace densecore

#endif  // DENSECORE_HARDWARE_TOPOLOGY_H
