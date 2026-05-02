#ifndef DENSECORE_KV_CACHE_INTERNAL_H
#define DENSECORE_KV_CACHE_INTERNAL_H

#include "densecore/memory/kv_cache.h"

uint64_t GetTokenHashSalt();
bool UseKVBulkSlotPath();
void RecordKVReadSingleSlotUsage();
void RecordKVWriteSingleSlotUsage();
void RecordKVReadBulkUsage(int num_slots);
void RecordKVWriteBulkUsage(int num_slots);
void RecordKVReadSlotFallbackUsage();
void RecordKVWriteSlotFallbackUsage();
void RecordKVScratchGrow();

#endif  // DENSECORE_KV_CACHE_INTERNAL_H
