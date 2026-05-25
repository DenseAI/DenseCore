package api

// Metrics handler ownership: Prometheus exposition for API, queue, prompt-cache, and engine metrics.
import (
	"descore-server/internal/promptcache"
	agenttools "descore-server/internal/tools"
	"fmt"
	"net/http"
	"strings"
)

// MetricsHandler outputs Prometheus-format metrics.
// Error checking for fmt.Fprintf is intentionally omitted per Prometheus exposition pattern.
//
//nolint:errcheck // Prometheus exposition format - write errors are non-recoverable
func (h *Handler) MetricsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		sendError(w, "Method not allowed", "invalid_request_error", ErrCodeMethodNotAllowed, http.StatusMethodNotAllowed)
		return
	}

	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	_, _ = w.Write([]byte(h.RenderMetrics()))
}

// RenderMetrics renders DenseCore-specific Prometheus metrics for the shared DenseCloud endpoint.

// RenderMetrics renders DenseCore-specific Prometheus metrics for the shared DenseCloud endpoint.
func (h *Handler) RenderMetrics() string {
	engine := h.modelService.GetEngine()
	var w strings.Builder
	metric := func(name string) string {
		return h.metricsNamespace + "_" + name
	}

	// Runtime/profile baseline metrics for all profiles.
	fmt.Fprintf(&w, "# HELP %s Runtime health status (1=up)\n", metric("up"))
	fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("up"))
	fmt.Fprintf(&w, "%s 1\n\n", metric("up"))

	fmt.Fprintf(&w, "# HELP %s Workload profile info\n", metric("profile_info"))
	fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("profile_info"))
	fmt.Fprintf(&w, "%s{profile=%q} 1\n\n", metric("profile_info"), h.workloadProfile)

	llmEnabled := 0
	if h.llmAPIEnabled {
		llmEnabled = 1
	}
	fmt.Fprintf(&w, "# HELP %s LLM API enabled status (1=enabled)\n", metric("llm_api_enabled"))
	fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("llm_api_enabled"))
	fmt.Fprintf(&w, "%s %d\n\n", metric("llm_api_enabled"), llmEnabled)

	modelLoaded := 0
	if engine != nil {
		modelLoaded = 1
	}
	fmt.Fprintf(&w, "# HELP %s Model loaded status (1=loaded)\n", metric("model_loaded"))
	fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("model_loaded"))
	fmt.Fprintf(&w, "%s %d\n\n", metric("model_loaded"), modelLoaded)

	toolMetrics := agenttools.SnapshotMetrics()
	fmt.Fprintf(&w, "# TYPE %s counter\n%s %d\n\n", metric("agent_tool_parse_success_total"), metric("agent_tool_parse_success_total"), toolMetrics.ToolParseSuccessTotal)
	fmt.Fprintf(&w, "# TYPE %s counter\n%s %d\n\n", metric("agent_tool_parse_failure_total"), metric("agent_tool_parse_failure_total"), toolMetrics.ToolParseFailureTotal)
	fmt.Fprintf(&w, "# TYPE %s counter\n%s %d\n\n", metric("agent_tool_parse_recovery_total"), metric("agent_tool_parse_recovery_total"), toolMetrics.ToolParseRecoveryTotal)
	fmt.Fprintf(&w, "# TYPE %s counter\n%s %d\n\n", metric("agent_tool_call_emitted_total"), metric("agent_tool_call_emitted_total"), toolMetrics.ToolCallEmittedTotal)
	fmt.Fprintf(&w, "# TYPE %s counter\n%s %d\n\n", metric("agent_reasoning_blocks_total"), metric("agent_reasoning_blocks_total"), toolMetrics.ReasoningBlocksTotal)

	cacheMetrics := promptcache.SnapshotMetrics()
	fmt.Fprintf(&w, "# TYPE %s counter\n%s %d\n\n", metric("prompt_cache_hit_total"), metric("prompt_cache_hit_total"), cacheMetrics.PromptCacheHitTotal)
	fmt.Fprintf(&w, "# TYPE %s counter\n%s %d\n\n", metric("prompt_cache_miss_total"), metric("prompt_cache_miss_total"), cacheMetrics.PromptCacheMissTotal)
	fmt.Fprintf(&w, "# TYPE %s counter\n%s %d\n\n", metric("prompt_cache_disabled_total"), metric("prompt_cache_disabled_total"), cacheMetrics.PromptCacheDisabledTotal)
	fmt.Fprintf(&w, "# TYPE %s counter\n%s %d\n\n", metric("prompt_cache_evictions_total"), metric("prompt_cache_evictions_total"), cacheMetrics.PromptCacheEvictionsTotal)
	fmt.Fprintf(&w, "# TYPE %s counter\n%s %d\n\n", metric("prompt_cache_tokens_reused_total"), metric("prompt_cache_tokens_reused_total"), cacheMetrics.PromptCacheTokensReusedTotal)
	fmt.Fprintf(&w, "# TYPE %s counter\n%s %d\n\n", metric("prompt_cache_prefill_tokens_skipped_total"), metric("prompt_cache_prefill_tokens_skipped_total"), cacheMetrics.PromptCachePrefillSkippedTotal)
	fmt.Fprintf(&w, "# TYPE %s counter\n%s %d\n\n", metric("cache_affinity_key_total"), metric("cache_affinity_key_total"), cacheMetrics.CacheAffinityKeyTotal)
	fmt.Fprintf(&w, "# TYPE %s counter\n%s %d\n\n", metric("cache_affinity_explicit_total"), metric("cache_affinity_explicit_total"), cacheMetrics.CacheAffinityExplicitTotal)
	fmt.Fprintf(&w, "# TYPE %s counter\n%s %d\n\n", metric("cache_affinity_fallback_total"), metric("cache_affinity_fallback_total"), cacheMetrics.CacheAffinityFallbackTotal)
	fmt.Fprintf(&w, "# TYPE %s gauge\n%s %.9f\n\n", metric("prompt_cache_restore_seconds"), metric("prompt_cache_restore_seconds"), cacheMetrics.PromptCacheRestoreSeconds)
	fmt.Fprintf(&w, "# TYPE %s gauge\n%s %.9f\n\n", metric("prompt_cache_lookup_seconds"), metric("prompt_cache_lookup_seconds"), cacheMetrics.PromptCacheLookupSeconds)
	fmt.Fprintf(&w, "# TYPE %s counter\n%s %d\n\n", metric("ssm_snapshot_hit_total"), metric("ssm_snapshot_hit_total"), cacheMetrics.SSMSnapshotHitTotal)
	fmt.Fprintf(&w, "# TYPE %s counter\n%s %d\n\n", metric("ssm_snapshot_miss_total"), metric("ssm_snapshot_miss_total"), cacheMetrics.SSMSnapshotMissTotal)
	fmt.Fprintf(&w, "# TYPE %s counter\n%s %d\n\n", metric("ssm_snapshot_restore_failure_total"), metric("ssm_snapshot_restore_failure_total"), cacheMetrics.SSMSnapshotRestoreFailureTotal)

	if h.queueStatsProvider != nil {
		queueStats := h.queueStatsProvider.Stats()

		fmt.Fprintf(&w, "# HELP %s Current request queue size\n", metric("queue_current_size"))
		fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("queue_current_size"))
		fmt.Fprintf(&w, "%s %d\n\n", metric("queue_current_size"), queueStats.CurrentSize)

		fmt.Fprintf(&w, "# HELP %s Maximum request queue size\n", metric("queue_max_size"))
		fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("queue_max_size"))
		fmt.Fprintf(&w, "%s %d\n\n", metric("queue_max_size"), queueStats.MaxSize)

		fmt.Fprintf(&w, "# HELP %s Total enqueued requests\n", metric("queue_total_enqueued"))
		fmt.Fprintf(&w, "# TYPE %s counter\n", metric("queue_total_enqueued"))
		fmt.Fprintf(&w, "%s %d\n\n", metric("queue_total_enqueued"), queueStats.TotalEnqueued)

		fmt.Fprintf(&w, "# HELP %s Total dequeued requests\n", metric("queue_total_dequeued"))
		fmt.Fprintf(&w, "# TYPE %s counter\n", metric("queue_total_dequeued"))
		fmt.Fprintf(&w, "%s %d\n\n", metric("queue_total_dequeued"), queueStats.TotalDequeued)

		fmt.Fprintf(&w, "# HELP %s Total dropped requests due to backpressure\n", metric("queue_total_dropped"))
		fmt.Fprintf(&w, "# TYPE %s counter\n", metric("queue_total_dropped"))
		fmt.Fprintf(&w, "%s %d\n\n", metric("queue_total_dropped"), queueStats.TotalDropped)
	}

	if engine == nil {
		return w.String()
	}

	// Engine-provided LLM inference metrics.
	metrics := engine.GetDetailedMetrics()

	// Request metrics
	fmt.Fprintf(&w, "# HELP %s Number of currently active requests\n", metric("active_requests"))
	fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("active_requests"))
	fmt.Fprintf(&w, "%s %d\n\n", metric("active_requests"), metrics.ActiveRequests)

	fmt.Fprintf(&w, "# HELP %s Number of pending requests in queue\n", metric("pending_requests"))
	fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("pending_requests"))
	fmt.Fprintf(&w, "%s %d\n\n", metric("pending_requests"), metrics.PendingRequests)

	fmt.Fprintf(&w, "# HELP %s Total number of requests received\n", metric("total_requests"))
	fmt.Fprintf(&w, "# TYPE %s counter\n", metric("total_requests"))
	fmt.Fprintf(&w, "%s %d\n\n", metric("total_requests"), metrics.TotalRequests)

	fmt.Fprintf(&w, "# HELP %s Total number of completed requests\n", metric("completed_requests"))
	fmt.Fprintf(&w, "# TYPE %s counter\n", metric("completed_requests"))
	fmt.Fprintf(&w, "%s %d\n\n", metric("completed_requests"), metrics.CompletedRequests)

	fmt.Fprintf(&w, "# HELP %s Total number of failed requests\n", metric("failed_requests"))
	fmt.Fprintf(&w, "# TYPE %s counter\n", metric("failed_requests"))
	fmt.Fprintf(&w, "%s %d\n\n", metric("failed_requests"), metrics.FailedRequests)

	// Token metrics
	fmt.Fprintf(&w, "# HELP %s Total number of tokens generated\n", metric("total_tokens_generated"))
	fmt.Fprintf(&w, "# TYPE %s counter\n", metric("total_tokens_generated"))
	fmt.Fprintf(&w, "%s %d\n\n", metric("total_tokens_generated"), metrics.TotalTokensGenerated)

	fmt.Fprintf(&w, "# HELP %s Total number of prompt tokens processed\n", metric("total_prompt_tokens"))
	fmt.Fprintf(&w, "# TYPE %s counter\n", metric("total_prompt_tokens"))
	fmt.Fprintf(&w, "%s %d\n\n", metric("total_prompt_tokens"), metrics.TotalPromptTokens)

	fmt.Fprintf(&w, "# HELP %s Average tokens generated per second\n", metric("tokens_per_second"))
	fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("tokens_per_second"))
	fmt.Fprintf(&w, "%s %.2f\n\n", metric("tokens_per_second"), metrics.TokensPerSecond)

	// TTFT metrics
	fmt.Fprintf(&w, "# HELP %s Time to first token latency\n", metric("time_to_first_token_seconds"))
	fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("time_to_first_token_seconds"))
	fmt.Fprintf(&w, "%s{quantile=\"0.5\"} %.6f\n", metric("time_to_first_token_seconds"), metrics.P50TimeToFirstToken/1000.0)
	fmt.Fprintf(&w, "%s{quantile=\"0.9\"} %.6f\n", metric("time_to_first_token_seconds"), metrics.P90TimeToFirstToken/1000.0)
	fmt.Fprintf(&w, "%s{quantile=\"0.99\"} %.6f\n", metric("time_to_first_token_seconds"), metrics.P99TimeToFirstToken/1000.0)
	fmt.Fprintf(&w, "%s{quantile=\"avg\"} %.6f\n\n", metric("time_to_first_token_seconds"), metrics.AvgTimeToFirstToken/1000.0)

	// ITL metrics
	fmt.Fprintf(&w, "# HELP %s Inter-token latency\n", metric("inter_token_latency_seconds"))
	fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("inter_token_latency_seconds"))
	fmt.Fprintf(&w, "%s{quantile=\"0.5\"} %.6f\n", metric("inter_token_latency_seconds"), metrics.P50InterTokenLatency/1000.0)
	fmt.Fprintf(&w, "%s{quantile=\"0.9\"} %.6f\n", metric("inter_token_latency_seconds"), metrics.P90InterTokenLatency/1000.0)
	fmt.Fprintf(&w, "%s{quantile=\"0.99\"} %.6f\n", metric("inter_token_latency_seconds"), metrics.P99InterTokenLatency/1000.0)
	fmt.Fprintf(&w, "%s{quantile=\"avg\"} %.6f\n\n", metric("inter_token_latency_seconds"), metrics.AvgInterTokenLatency/1000.0)

	// Queue wait time
	fmt.Fprintf(&w, "# HELP %s Request queue wait time\n", metric("queue_wait_time_seconds"))
	fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("queue_wait_time_seconds"))
	fmt.Fprintf(&w, "%s{quantile=\"0.99\"} %.6f\n", metric("queue_wait_time_seconds"), metrics.P99QueueWaitTime/1000.0)
	fmt.Fprintf(&w, "%s{quantile=\"avg\"} %.6f\n\n", metric("queue_wait_time_seconds"), metrics.AvgQueueWaitTime/1000.0)

	// KV Cache metrics
	fmt.Fprintf(&w, "# HELP %s Number of KV cache blocks in use\n", metric("kv_cache_usage_blocks"))
	fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("kv_cache_usage_blocks"))
	fmt.Fprintf(&w, "%s %d\n\n", metric("kv_cache_usage_blocks"), metrics.KVCacheUsageBlocks)

	fmt.Fprintf(&w, "# HELP %s Total number of KV cache blocks\n", metric("kv_cache_total_blocks"))
	fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("kv_cache_total_blocks"))
	fmt.Fprintf(&w, "%s %d\n\n", metric("kv_cache_total_blocks"), metrics.KVCacheTotalBlocks)

	fmt.Fprintf(&w, "# HELP %s KV cache usage percentage\n", metric("kv_cache_usage_percent"))
	fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("kv_cache_usage_percent"))
	fmt.Fprintf(&w, "%s %.2f\n\n", metric("kv_cache_usage_percent"), metrics.KVCacheUsagePercent)

	// Batch metrics
	fmt.Fprintf(&w, "# HELP %s Current batch size\n", metric("current_batch_size"))
	fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("current_batch_size"))
	fmt.Fprintf(&w, "%s %d\n\n", metric("current_batch_size"), metrics.CurrentBatchSize)

	fmt.Fprintf(&w, "# HELP %s Average batch size\n", metric("avg_batch_size"))
	fmt.Fprintf(&w, "# TYPE %s gauge\n", metric("avg_batch_size"))
	fmt.Fprintf(&w, "%s %.2f\n\n", metric("avg_batch_size"), metrics.AvgBatchSize)

	// Error metrics
	fmt.Fprintf(&w, "# HELP %s Total number of OOM errors\n", metric("oom_errors"))
	fmt.Fprintf(&w, "# TYPE %s counter\n", metric("oom_errors"))
	fmt.Fprintf(&w, "%s %d\n\n", metric("oom_errors"), metrics.OOMErrors)

	fmt.Fprintf(&w, "# HELP %s Total number of timeout errors\n", metric("timeout_errors"))
	fmt.Fprintf(&w, "# TYPE %s counter\n", metric("timeout_errors"))
	fmt.Fprintf(&w, "%s %d\n", metric("timeout_errors"), metrics.TimeoutErrors)

	return w.String()
}

func sanitizeMetricsNamespace(ns string) string {
	ns = strings.ToLower(strings.TrimSpace(ns))
	if ns == "" {
		return DefaultMetricsNamespace
	}

	var b strings.Builder
	for i, r := range ns {
		valid := (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == '_'
		if !valid {
			r = '_'
		}
		// Prometheus metric names cannot start with a digit.
		if i == 0 && r >= '0' && r <= '9' {
			b.WriteByte('_')
		}
		b.WriteRune(r)
	}

	out := b.String()
	// Keep leading underscore (valid in Prometheus) so namespaced metrics never start with a digit.
	out = strings.TrimRight(out, "_")
	if strings.Trim(out, "_") == "" {
		return DefaultMetricsNamespace
	}
	if out[0] >= '0' && out[0] <= '9' {
		out = "_" + out
	}
	return out
}
