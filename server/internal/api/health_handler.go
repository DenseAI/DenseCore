package api

// Health handler ownership: basic health, runtime profile, and Kubernetes probe responses.
import (
	"encoding/json"
	"log/slog"
	"net/http"
	"time"

	"github.com/DenseAI/DenseCore/server/internal/buildinfo"
	"github.com/DenseAI/DenseCore/server/internal/domain"
	"github.com/DenseAI/DenseCore/server/internal/service"
)

// HealthHandler - Basic health check (legacy)
func (h *Handler) HealthHandler(w http.ResponseWriter, r *http.Request) {
	response := map[string]interface{}{
		"status":  "ok",
		"version": buildinfo.Version,
		"engine":  "densecore",
		"profile": h.workloadProfile,
		"llm_api": h.llmAPIEnabled,
		"metrics": h.metricsNamespace,
		"probes_v2": map[string]bool{
			"model_lifecycle": h.modelLifecycleProbes,
		},
	}
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		slog.Error("failed to encode response", slog.String("error", err.Error()))
	}
}

// RuntimeProfileHandler returns runtime profile/capability information.

// RuntimeProfileHandler returns runtime profile/capability information.
func (h *Handler) RuntimeProfileHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		sendError(w, "Method not allowed", "invalid_request_error", ErrCodeMethodNotAllowed, http.StatusMethodNotAllowed)
		return
	}

	response := map[string]interface{}{
		"profile":                        h.workloadProfile,
		"llm_api_enabled":                h.llmAPIEnabled,
		"model_lifecycle_probes_enabled": h.modelLifecycleProbes,
		"metrics_namespace":              h.metricsNamespace,
		"timestamp":                      time.Now().UTC().Format(time.RFC3339),
		"configured_env":                 h.runtimeTuning,
		"runtime_tuning":                 h.runtimeTuning,
	}
	if effective, ok := h.effectiveRuntimeOptimizationState(); ok {
		response["effective_runtime_state"] = effective
	}
	if h.queueStatsProvider != nil {
		stats := h.queueStatsProvider.Stats()
		response["queue"] = map[string]interface{}{
			"current_size":   stats.CurrentSize,
			"max_size":       stats.MaxSize,
			"total_enqueued": stats.TotalEnqueued,
			"total_dequeued": stats.TotalDequeued,
			"total_dropped":  stats.TotalDropped,
		}
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		slog.Debug("failed to encode response", slog.String("error", err.Error()))
	}
}

func (h *Handler) effectiveRuntimeOptimizationState() (domain.RuntimeOptimizationState, bool) {
	if h == nil || h.modelService == nil {
		return domain.RuntimeOptimizationState{}, false
	}
	engine, releaseEngine := service.AcquireRequestEngine(h.modelService)
	defer releaseEngine()
	if engine == nil {
		return domain.RuntimeOptimizationState{}, false
	}
	provider, ok := engine.(domain.RuntimeOptimizationStateProvider)
	if !ok {
		return domain.RuntimeOptimizationState{}, false
	}
	state, err := provider.GetRuntimeOptimizationState()
	if err != nil {
		slog.Debug("failed to read effective runtime optimization state", slog.String("error", err.Error()))
		return domain.RuntimeOptimizationState{}, false
	}
	return state, true
}

// LivenessHandler - K8s liveness probe
// Returns 200 if the process is alive

// LivenessHandler - K8s liveness probe
// Returns 200 if the process is alive
func (h *Handler) LivenessHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    "ok",
		"timestamp": time.Now().UTC().Format(time.RFC3339),
	}); err != nil {
		slog.Debug("failed to encode response", slog.String("error", err.Error()))
	}
}

// ReadinessHandler - K8s readiness probe
// Returns 200 only if the model is loaded and ready to serve
// Checks: loading status, engine availability, and KV cache utilization

// ReadinessHandler - K8s readiness probe
// Returns 200 only if the model is loaded and ready to serve
// Checks: loading status, engine availability, and KV cache utilization
func (h *Handler) ReadinessHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	if !h.modelLifecycleProbes {
		response := map[string]interface{}{
			"status":    "ok",
			"profile":   h.workloadProfile,
			"timestamp": time.Now().UTC().Format(time.RFC3339),
		}
		if h.queueStatsProvider != nil {
			response["queue_size"] = h.queueStatsProvider.Stats().CurrentSize
		}
		if err := json.NewEncoder(w).Encode(response); err != nil {
			slog.Debug("failed to encode response", slog.String("error", err.Error()))
		}
		return
	}

	// Check loading status first
	if h.modelService.IsLoading() {
		w.WriteHeader(http.StatusServiceUnavailable)
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "not_ready",
			"reason": "model_loading",
		}); err != nil {
			slog.Debug("failed to encode response", slog.String("error", err.Error()))
		}
		return
	}

	engine, releaseEngine := service.AcquireRequestEngine(h.modelService)
	defer releaseEngine()
	if engine == nil {
		w.WriteHeader(http.StatusServiceUnavailable)
		reason := "model_not_loaded"
		if h.modelService.GetLoadingStatus() == domain.StatusFailed {
			reason = "model_load_failed"
		}
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"status": "not_ready",
			"reason": reason,
		}); err != nil {
			slog.Debug("failed to encode response", slog.String("error", err.Error()))
		}
		return
	}

	// Check if engine is healthy by getting metrics
	metrics := engine.GetDetailedMetrics()

	response := map[string]interface{}{
		"status":       "ok",
		"model":        h.modelService.GetCurrentModel(),
		"kv_cache_pct": metrics.KVCacheUsagePercent,
		"timestamp":    time.Now().UTC().Format(time.RFC3339),
	}

	// Mark as not ready if KV cache is critically full
	// Using 90% threshold to allow headroom before actual capacity issues
	if metrics.KVCacheUsagePercent > KVCacheCriticalThreshold {
		response["status"] = "degraded"
		response["reason"] = "kv_cache_full"
	}

	if err := json.NewEncoder(w).Encode(response); err != nil {
		slog.Debug("failed to encode response", slog.String("error", err.Error()))
	}
}

// StartupHandler - K8s startup probe
// Returns 200 once the initial model loading is complete
// Returns 503 if loading is in progress or failed

// StartupHandler - K8s startup probe
// Returns 200 once the initial model loading is complete
// Returns 503 if loading is in progress or failed
func (h *Handler) StartupHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	if !h.modelLifecycleProbes {
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"status":   "ok",
			"progress": 100,
			"profile":  h.workloadProfile,
		}); err != nil {
			slog.Debug("failed to encode response", slog.String("error", err.Error()))
		}
		return
	}

	status := h.modelService.GetLoadingStatus()

	switch status {
	case domain.StatusReady:
		// Model loaded successfully
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"status":   "ok",
			"progress": 100,
			"model":    h.modelService.GetCurrentModel(),
		}); err != nil {
			slog.Debug("failed to encode response", slog.String("error", err.Error()))
		}

	case domain.StatusLoading:
		// Model is still loading
		w.WriteHeader(http.StatusServiceUnavailable)
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"status":   "loading",
			"progress": 50,
		}); err != nil {
			slog.Debug("failed to encode response", slog.String("error", err.Error()))
		}

	case domain.StatusFailed:
		// Model loading failed - return error for debugging
		w.WriteHeader(http.StatusServiceUnavailable)
		response := map[string]interface{}{
			"status":   "failed",
			"progress": 0,
		}
		if loadErr := h.modelService.GetLoadingError(); loadErr != nil {
			response["error"] = loadErr.Error()
		}
		if err := json.NewEncoder(w).Encode(response); err != nil {
			slog.Debug("failed to encode response", slog.String("error", err.Error()))
		}

	default: // StatusIdle - no model path configured
		engine, releaseEngine := service.AcquireRequestEngine(h.modelService)
		defer releaseEngine()
		if engine != nil {
			// An initialized engine is ready even if status publication lagged.
			if err := json.NewEncoder(w).Encode(map[string]interface{}{
				"status":   "ok",
				"progress": 100,
				"model":    h.modelService.GetCurrentModel(),
			}); err != nil {
				slog.Debug("failed to encode response", slog.String("error", err.Error()))
			}
		} else {
			// The MVP server has a fixed model lifecycle. Recovery requires restart.
			w.WriteHeader(http.StatusServiceUnavailable)
			if err := json.NewEncoder(w).Encode(map[string]interface{}{
				"status":   "idle",
				"progress": 0,
				"message":  "model unavailable; restart the server with a configured model",
			}); err != nil {
				slog.Debug("failed to encode response", slog.String("error", err.Error()))
			}
		}
	}
}

// MetricsHandler outputs Prometheus-format metrics.
// Error checking for fmt.Fprintf is intentionally omitted per Prometheus exposition pattern.
//
//nolint:errcheck // Prometheus exposition format - write errors are non-recoverable
