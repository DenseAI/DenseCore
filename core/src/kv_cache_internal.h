#ifndef DENSECORE_KV_CACHE_INTERNAL_H
#define DENSECORE_KV_CACHE_INTERNAL_H

#include "kv_cache.h"

uint64_t GetTokenHashSalt();
bool UseKVBulkSlotPath();
void RecordKVReadBulkUsage(int num_slots);
void RecordKVWriteBulkUsage(int num_slots);

#endif  // DENSECORE_KV_CACHE_INTERNAL_H
