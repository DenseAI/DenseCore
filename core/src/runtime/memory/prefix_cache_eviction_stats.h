#pragma once

#include <cstdint>

namespace densecore::runtime {

// Process-cumulative allocator reclamation, excluding cache replacements and
// explicit unregister/cleanup. Payload bytes count logical recurrent float data
// removed from the snapshot map; active leases can retain independent copies.
struct PrefixCacheEvictionStats {
    uint64_t kv_blocks = 0;
    uint64_t hybrid_snapshots = 0;
    uint64_t hybrid_payload_bytes = 0;
};

PrefixCacheEvictionStats PrefixCacheEvictionStatsSnapshot();

}  // namespace densecore::runtime
