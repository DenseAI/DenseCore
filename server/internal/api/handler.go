package api

// Handler ownership: shared API handler wiring, options, and constructor.
import (
	"descore-server/internal/domain"
	"descore-server/internal/queue"
	"descore-server/internal/service"
	"strings"
)

// API constants for consistent behavior and easier maintenance
const (
	// DefaultMaxTokens is the default maximum tokens when not specified
	DefaultMaxTokens = 100
	// MaxAllowedTokens is the hard limit for max_tokens parameter
	MaxAllowedTokens = 32000

	// KVCacheCriticalThreshold is the KV cache usage percent above which
	// the service is considered degraded and readiness probe fails.
	// Set to 90% to allow headroom before actual capacity issues.
	KVCacheCriticalThreshold = 90.0

	// StreamChannelBufferSize buffers callback events so SSE writes stay off the engine hot path.
	StreamChannelBufferSize = 4096

	// DefaultMetricsNamespace is the default Prometheus metric namespace.
	DefaultMetricsNamespace = "densecore"
)

type Handler struct {
	chatService          *service.ChatService
	modelService         domain.ModelService
	workloadProfile      string
	metricsNamespace     string
	modelLifecycleProbes bool
	llmAPIEnabled        bool
	queueStatsProvider   QueueStatsProvider
	runtimeTuning        RuntimeTuningProfile
}

type RuntimeTuningProfile struct {
	BenchmarkProfile         string `json:"benchmark_profile"`
	EngineThreads            int    `json:"engine_threads"`
	GoWorkers                int    `json:"go_workers"`
	ServerInflight           int    `json:"server_inflight"`
	KVType                   string `json:"kv_type"`
	MaxNumSeqs               int    `json:"max_num_seqs,omitempty"`
	MaxSeqLen                int    `json:"max_seq_len,omitempty"`
	KVTargetMB               int    `json:"kv_target_mb,omitempty"`
	TokenIDSubmit            bool   `json:"token_id_submit_default"`
	PrefixCacheAuto          bool   `json:"prefix_cache_auto"`
	SSMSnapshotAuto          bool   `json:"ssm_snapshot_restore_auto"`
	PrefillArenaReuse        bool   `json:"prefill_arena_reuse"`
	DecodeGraphCacheMaxBatch int    `json:"decode_graph_cache_max_batch,omitempty"`
	MoEDequantCacheMB        int    `json:"moe_dequant_cache_mb,omitempty"`
}

// QueueStatsProvider exposes queue statistics for metrics and status endpoints.
type QueueStatsProvider interface {
	Stats() queue.QueueStats
}

// HandlerOption configures the API handler.
type HandlerOption func(*Handler)

func WithWorkloadProfile(profile string) HandlerOption {
	return func(h *Handler) {
		if p := strings.TrimSpace(strings.ToLower(profile)); p != "" {
			h.workloadProfile = p
		}
	}
}

func WithMetricsNamespace(namespace string) HandlerOption {
	return func(h *Handler) {
		h.metricsNamespace = sanitizeMetricsNamespace(namespace)
	}
}

func WithModelLifecycleProbes(enabled bool) HandlerOption {
	return func(h *Handler) {
		h.modelLifecycleProbes = enabled
	}
}

func WithLLMAPIEnabled(enabled bool) HandlerOption {
	return func(h *Handler) {
		h.llmAPIEnabled = enabled
	}
}

func WithQueueStatsProvider(provider QueueStatsProvider) HandlerOption {
	return func(h *Handler) {
		h.queueStatsProvider = provider
	}
}

func WithRuntimeTuningProfile(profile RuntimeTuningProfile) HandlerOption {
	return func(h *Handler) {
		h.runtimeTuning = profile
	}
}

func NewHandler(chatService *service.ChatService, modelService domain.ModelService, opts ...HandlerOption) *Handler {
	h := &Handler{
		chatService:          chatService,
		modelService:         modelService,
		workloadProfile:      "llm",
		metricsNamespace:     DefaultMetricsNamespace,
		modelLifecycleProbes: true,
		llmAPIEnabled:        true,
	}
	for _, opt := range opts {
		opt(h)
	}
	h.metricsNamespace = sanitizeMetricsNamespace(h.metricsNamespace)
	return h
}
