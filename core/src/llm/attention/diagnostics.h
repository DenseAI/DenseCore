#pragma once
bool IsDebugPagedAttentionReferenceEnabled();
bool IsDebugPagedAttentionEagerReferenceEnabled();
bool ShouldRunPagedAttentionReferenceProbe(int layer, int token_idx);
bool ShouldRunPagedAttentionEagerReferenceProbe(int layer, int token_idx);
bool IsAttentionDecodeProfilingEnabled();
bool IsDecodeAttentionPathLoggingEnabled();
