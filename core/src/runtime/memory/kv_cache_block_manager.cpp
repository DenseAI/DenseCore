#include "runtime/memory/kv_cache_internal.h"

#include <atomic>
#include <cstdlib>

BlockManager::BlockManager(int num_blocks, int block_size) : num_blocks(num_blocks), block_size(block_size) {
    shards.reserve(NUM_SHARDS);
    for (int i = 0; i < NUM_SHARDS; ++i) {
        shards.push_back(std::make_unique<Shard>());
    }

    blocks.resize(num_blocks);
    for (int i = 0; i < num_blocks; i++) {
        blocks[i].id = i;
        blocks[i].ref_count = 0;
        blocks[i].num_filled_slots = 0;
        blocks[i].content_hash = 0;
        shards[i % NUM_SHARDS]->free_blocks.push_back(i);
    }
}

std::vector<int> BlockManager::Allocate(int n) {
    if (n <= 0) return {};
    if (n == 1) {
        int id = AllocateSingle();
        if (id >= 0) return {id};
        return {};
    }

    std::vector<int> allocated;
    allocated.reserve(n);

    int start_shard = rand() % NUM_SHARDS;
    for (int i = 0; i < NUM_SHARDS; ++i) {
        int shard_idx = (start_shard + i) % NUM_SHARDS;
        auto& shard = *shards[shard_idx];

        std::unique_lock<std::mutex> lock(shard.mu, std::try_to_lock);
        if (!lock.owns_lock()) continue;

        if (static_cast<int>(shard.free_blocks.size()) >= n) {
            for (int k = 0; k < n; ++k) {
                int block_id = shard.free_blocks.back();
                shard.free_blocks.pop_back();

                blocks[block_id].ref_count = 1;
                blocks[block_id].num_filled_slots = 0;
                blocks[block_id].content_hash = 0;
                allocated.push_back(block_id);
            }
            return allocated;
        }
    }

    for (int i = 0; i < NUM_SHARDS && static_cast<int>(allocated.size()) < n; ++i) {
        int shard_idx = (start_shard + i) % NUM_SHARDS;
        auto& shard = *shards[shard_idx];

        std::lock_guard<std::mutex> lock(shard.mu);
        while (!shard.free_blocks.empty() && static_cast<int>(allocated.size()) < n) {
            int block_id = shard.free_blocks.back();
            shard.free_blocks.pop_back();

            blocks[block_id].ref_count = 1;
            blocks[block_id].num_filled_slots = 0;
            blocks[block_id].content_hash = 0;
            allocated.push_back(block_id);
        }
    }

    if (static_cast<int>(allocated.size()) < n) {
        Free(allocated);
        return {};
    }

    return allocated;
}

int BlockManager::AllocateSingle() {
    static std::atomic<uint32_t> rr_counter{0};
    int start_shard = static_cast<int>(rr_counter.fetch_add(1, std::memory_order_relaxed) % NUM_SHARDS);

    for (int i = 0; i < NUM_SHARDS; ++i) {
        int shard_idx = (start_shard + i) % NUM_SHARDS;
        auto& shard = *shards[shard_idx];

        std::unique_lock<std::mutex> lock(shard.mu, std::try_to_lock);
        if (!lock.owns_lock()) continue;

        if (!shard.free_blocks.empty()) {
            int block_id = shard.free_blocks.back();
            shard.free_blocks.pop_back();

            blocks[block_id].ref_count = 1;
            blocks[block_id].num_filled_slots = 0;
            blocks[block_id].content_hash = 0;
            return block_id;
        }
    }

    for (int i = 0; i < NUM_SHARDS; ++i) {
        auto& shard = *shards[i];
        std::lock_guard<std::mutex> lock(shard.mu);
        if (!shard.free_blocks.empty()) {
            int block_id = shard.free_blocks.back();
            shard.free_blocks.pop_back();

            blocks[block_id].ref_count = 1;
            blocks[block_id].num_filled_slots = 0;
            blocks[block_id].content_hash = 0;
            return block_id;
        }
    }

    return -1;
}

void BlockManager::Free(const std::vector<int>& block_ids) {
    for (int id : block_ids) {
        FreeSingle(id);
    }
}

void BlockManager::FreeSingle(int block_id) {
    if (block_id < 0 || block_id >= num_blocks) return;

    int shard_idx = block_id % NUM_SHARDS;
    auto& shard = *shards[shard_idx];

    {
        std::lock_guard<std::mutex> lock(shard.mu);
        blocks[block_id].ref_count--;
        if (blocks[block_id].ref_count > 0) return;
    }

    std::lock_guard<std::mutex> p_lock(prefix_mu);
    std::lock_guard<std::mutex> s_lock(shard.mu);
    if (blocks[block_id].ref_count > 0) {
        return;
    }

    if (blocks[block_id].content_hash != 0) {
        prefix_cache.erase(blocks[block_id].content_hash);
        blocks[block_id].content_hash = 0;
    }
    block_tokens.erase(block_id);
    block_hybrid_ssm_snapshots.erase(block_id);
    shard.free_blocks.push_back(block_id);
}

int BlockManager::GetFreeBlockCount() {
    int total = 0;
    for (auto& shard : shards) {
        std::lock_guard<std::mutex> lock(shard->mu);
        total += shard->free_blocks.size();
    }
    return total;
}

int BlockManager::GetUsedBlockCount() {
    return num_blocks - GetFreeBlockCount();
}

int BlockManager::Fork(int block_id) {
    if (block_id < 0 || block_id >= num_blocks) return -1;

    int shard_idx = block_id % NUM_SHARDS;
    auto& shard = *shards[shard_idx];
    std::lock_guard<std::mutex> lock(shard.mu);

    if (blocks[block_id].ref_count <= 0) {
        return -1;
    }

    blocks[block_id].ref_count++;
    return block_id;
}

int BlockManager::CopyOnWrite(int block_id, const std::function<void(int src, int dst)>& copy_callback) {
    if (block_id < 0 || block_id >= num_blocks) return -1;

    int shard_idx = block_id % NUM_SHARDS;
    auto& shard = *shards[shard_idx];

    {
        std::lock_guard<std::mutex> lock(shard.mu);
        if (blocks[block_id].ref_count <= 1) {
            return block_id;
        }
    }

    int new_block_id = AllocateSingle();
    if (new_block_id < 0) {
        return -1;
    }

    copy_callback(block_id, new_block_id);

    {
        std::lock_guard<std::mutex> lock(shard.mu);
        blocks[block_id].ref_count--;
    }

    blocks[new_block_id].num_filled_slots = blocks[block_id].num_filled_slots;
    return new_block_id;
}

bool BlockManager::IsShared(int block_id) const {
    if (block_id < 0 || block_id >= num_blocks) return false;

    int shard_idx = block_id % NUM_SHARDS;
    auto& shard = *shards[shard_idx];
    std::lock_guard<std::mutex> lock(shard.mu);
    return blocks[block_id].ref_count > 1;
}

int BlockManager::GetRefCount(int block_id) const {
    if (block_id < 0 || block_id >= num_blocks) return 0;

    int shard_idx = block_id % NUM_SHARDS;
    auto& shard = *shards[shard_idx];
    std::lock_guard<std::mutex> lock(shard.mu);
    return blocks[block_id].ref_count;
}

void BlockManager::SetFilledSlots(int block_id, int num_slots) {
    if (block_id < 0 || block_id >= num_blocks) return;

    int shard_idx = block_id % NUM_SHARDS;
    auto& shard = *shards[shard_idx];
    std::lock_guard<std::mutex> lock(shard.mu);
    blocks[block_id].num_filled_slots = std::max(0, std::min(num_slots, block_size));
}

int BlockManager::GetFilledSlots(int block_id) const {
    if (block_id < 0 || block_id >= num_blocks) return 0;

    int shard_idx = block_id % NUM_SHARDS;
    auto& shard = *shards[shard_idx];
    std::lock_guard<std::mutex> lock(shard.mu);
    return blocks[block_id].num_filled_slots;
}

int BlockManager::FindCachedBlock(uint64_t hash) {
    if (hash == 0) return -1;

    std::lock_guard<std::mutex> p_lock(prefix_mu);
    auto it = prefix_cache.find(hash);
    if (it == prefix_cache.end()) {
        return -1;
    }

    int block_id = it->second;
    int shard_idx = block_id % NUM_SHARDS;
    auto& shard = *shards[shard_idx];
    std::lock_guard<std::mutex> s_lock(shard.mu);
    if (blocks[block_id].ref_count <= 0) {
        return -1;
    }

    blocks[block_id].ref_count++;
    return block_id;
}

int BlockManager::FindCachedBlockWithVerification(uint64_t hash, const int* tokens, int n_tokens) {
    if (hash == 0 || !tokens || n_tokens <= 0) {
        return -1;
    }

    std::lock_guard<std::mutex> p_lock(prefix_mu);
    auto it = prefix_cache.find(hash);
    if (it == prefix_cache.end()) {
        return -1;
    }

    int block_id = it->second;
    auto tokens_it = block_tokens.find(block_id);
    if (tokens_it == block_tokens.end()) {
        return -1;
    }

    const std::vector<int>& stored_tokens = tokens_it->second;
    if (static_cast<int>(stored_tokens.size()) != n_tokens) {
        return -1;
    }
    if (!std::equal(stored_tokens.begin(), stored_tokens.end(), tokens)) {
        return -1;
    }

    int shard_idx = block_id % NUM_SHARDS;
    auto& shard = *shards[shard_idx];
    std::lock_guard<std::mutex> s_lock(shard.mu);
    if (blocks[block_id].ref_count <= 0) {
        return -1;
    }

    blocks[block_id].ref_count++;
    return block_id;
}

BlockManager::PrefixCacheMatch BlockManager::FindLongestCachedPrefixWithVerification(const int* tokens, int n_tokens,
                                                                                     bool require_hybrid_ssm_snapshot) {
    PrefixCacheMatch match;
    if (!tokens || n_tokens <= BLOCK_SIZE) {
        return match;
    }

    const int max_reusable_tokens = ((n_tokens - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    if (max_reusable_tokens <= 0) {
        return match;
    }

    std::vector<int> acquired_blocks;
    acquired_blocks.reserve(max_reusable_tokens / BLOCK_SIZE);

    for (int offset = 0; offset < max_reusable_tokens; offset += BLOCK_SIZE) {
        const uint64_t hash = ComputeTokenHash(tokens + offset, BLOCK_SIZE);
        const int block_id = FindCachedBlockWithVerification(hash, tokens + offset, BLOCK_SIZE);
        if (block_id < 0) {
            break;
        }

        if (require_hybrid_ssm_snapshot &&
            block_hybrid_ssm_snapshots.find(block_id) == block_hybrid_ssm_snapshots.end()) {
            FreeSingle(block_id);
            break;
        }

        acquired_blocks.push_back(block_id);
    }

    match.cached_block_ids = std::move(acquired_blocks);
    match.cached_tokens = static_cast<int>(match.cached_block_ids.size()) * BLOCK_SIZE;
    return match;
}

void BlockManager::RegisterPrefixBlock(int block_id, uint64_t hash) {
    if (block_id < 0 || block_id >= num_blocks || hash == 0) return;

    std::lock_guard<std::mutex> p_lock(prefix_mu);
    int shard_idx = block_id % NUM_SHARDS;
    std::lock_guard<std::mutex> s_lock(shards[shard_idx]->mu);

    blocks[block_id].content_hash = hash;
    prefix_cache[hash] = block_id;
}

void BlockManager::RegisterPrefixBlockWithTokens(
    int block_id, uint64_t hash, const int* tokens, int n_tokens,
    const std::vector<TransformerModel::SSMSequenceRuntimeState>* hybrid_ssm_snapshot) {
    if (block_id < 0 || block_id >= num_blocks || hash == 0) return;

    std::lock_guard<std::mutex> p_lock(prefix_mu);
    int shard_idx = block_id % NUM_SHARDS;
    std::lock_guard<std::mutex> s_lock(shards[shard_idx]->mu);

    blocks[block_id].content_hash = hash;
    prefix_cache[hash] = block_id;

    if (tokens && n_tokens > 0) {
        block_tokens[block_id] = std::vector<int>(tokens, tokens + n_tokens);
    }
    if (hybrid_ssm_snapshot && !hybrid_ssm_snapshot->empty()) {
        block_hybrid_ssm_snapshots[block_id] = *hybrid_ssm_snapshot;
    }
}

bool BlockManager::LoadHybridSSMSnapshotForBlock(int block_id,
                                                 std::vector<TransformerModel::SSMSequenceRuntimeState>* out_snapshot) {
    if (!out_snapshot || block_id < 0 || block_id >= num_blocks) {
        return false;
    }

    std::lock_guard<std::mutex> p_lock(prefix_mu);
    auto it = block_hybrid_ssm_snapshots.find(block_id);
    if (it == block_hybrid_ssm_snapshots.end()) {
        return false;
    }
    *out_snapshot = it->second;
    return true;
}

void BlockManager::UnregisterPrefixBlock(int block_id) {
    if (block_id < 0 || block_id >= num_blocks) return;

    std::lock_guard<std::mutex> p_lock(prefix_mu);
    int shard_idx = block_id % NUM_SHARDS;
    std::lock_guard<std::mutex> s_lock(shards[shard_idx]->mu);

    if (blocks[block_id].content_hash != 0) {
        prefix_cache.erase(blocks[block_id].content_hash);
        blocks[block_id].content_hash = 0;
    }
    block_tokens.erase(block_id);
    block_hybrid_ssm_snapshots.erase(block_id);
}

uint64_t BlockManager::ComputeTokenHash(const int* tokens, int n_tokens) {
    uint64_t hash = 14695981039346656037ULL ^ GetTokenHashSalt();
    for (int i = 0; i < n_tokens; i++) {
        hash ^= static_cast<uint64_t>(tokens[i]);
        hash *= 1099511628211ULL;
    }
    return hash;
}
