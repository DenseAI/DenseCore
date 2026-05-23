package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"descore-server/internal/domain"
	"descore-server/internal/queue"
)

type staticQueueStatsProvider struct {
	stats queue.QueueStats
}

func (p staticQueueStatsProvider) Stats() queue.QueueStats { return p.stats }

func TestRuntimeProfileHandlerIncludesRuntimeTuning(t *testing.T) {
	handler := NewHandler(
		nil,
		nil,
		WithWorkloadProfile("llm"),
		WithRuntimeTuningProfile(RuntimeTuningProfile{
			BenchmarkProfile:         "single-e2e",
			EngineThreads:            16,
			GoWorkers:                1,
			ServerInflight:           1,
			KVType:                   "q8_0",
			MaxNumSeqs:               16,
			MaxSeqLen:                16384,
			KVTargetMB:               4096,
			TokenIDSubmit:            true,
			PrefixCacheAuto:          true,
			SSMSnapshotAuto:          true,
			DecodeGraphCacheMaxBatch: 16,
			MoEDequantCacheMB:        512,
		}),
		WithQueueStatsProvider(staticQueueStatsProvider{
			stats: queue.QueueStats{
				CurrentSize:   0,
				MaxSize:       1,
				TotalEnqueued: 3,
				TotalDequeued: 2,
				TotalDropped:  1,
			},
		}),
	)

	req := httptest.NewRequest(http.MethodGet, "/v1/runtime/profile", nil)
	w := httptest.NewRecorder()
	handler.RuntimeProfileHandler(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var response struct {
		RuntimeTuning RuntimeTuningProfile `json:"runtime_tuning"`
		Queue         struct {
			MaxSize int `json:"max_size"`
		} `json:"queue"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}

	if response.RuntimeTuning.BenchmarkProfile != "single-e2e" {
		t.Fatalf("BenchmarkProfile = %q, want single-e2e", response.RuntimeTuning.BenchmarkProfile)
	}
	if response.RuntimeTuning.KVType != "q8_0" {
		t.Fatalf("KVType = %q, want q8_0", response.RuntimeTuning.KVType)
	}
	if !response.RuntimeTuning.TokenIDSubmit {
		t.Fatal("expected token-id submit default to be reported")
	}
	if response.RuntimeTuning.DecodeGraphCacheMaxBatch != 16 {
		t.Fatalf("DecodeGraphCacheMaxBatch = %d, want 16", response.RuntimeTuning.DecodeGraphCacheMaxBatch)
	}
	if response.Queue.MaxSize != 1 {
		t.Fatalf("queue.max_size = %d, want 1", response.Queue.MaxSize)
	}
}

func TestRuntimeProfileHandlerDistinguishesConfiguredAndEffectiveRuntimeState(t *testing.T) {
	modelService := NewMockModelService()
	modelService.engine.runtimeState = domain.RuntimeOptimizationState{
		TokenIDSubmitSupported:          true,
		PrefixCacheReuseEnabled:         true,
		HybridSSMSnapshotRestoreEnabled: true,
		PrefillGraphCacheEnabled:        false,
		PrefillArenaReuseEnabled:        true,
		DecodeGraphCacheEnabled:         true,
		DecodeGraphCacheMaxBatch:        8,
		DecodeGraphCacheLRUSize:         32,
		MoEDequantCacheMB:               256,
		ActiveThreadPolicyLabel:         "model_aware_thread_policy",
	}
	handler := NewHandler(
		nil,
		modelService,
		WithRuntimeTuningProfile(RuntimeTuningProfile{
			DecodeGraphCacheMaxBatch: 16,
			MoEDequantCacheMB:        512,
		}),
	)

	req := httptest.NewRequest(http.MethodGet, "/v1/runtime/profile", nil)
	w := httptest.NewRecorder()
	handler.RuntimeProfileHandler(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}

	var response struct {
		ConfiguredEnv RuntimeTuningProfile            `json:"configured_env"`
		Effective     domain.RuntimeOptimizationState `json:"effective_runtime_state"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v", err)
	}

	if response.ConfiguredEnv.DecodeGraphCacheMaxBatch != 16 {
		t.Fatalf("configured decode max batch = %d, want 16", response.ConfiguredEnv.DecodeGraphCacheMaxBatch)
	}
	if response.Effective.DecodeGraphCacheMaxBatch != 8 {
		t.Fatalf("effective decode max batch = %d, want 8", response.Effective.DecodeGraphCacheMaxBatch)
	}
	if response.Effective.MoEDequantCacheMB != 256 {
		t.Fatalf("effective MoE cache MB = %d, want 256", response.Effective.MoEDequantCacheMB)
	}
	if response.Effective.ActiveThreadPolicyLabel != "model_aware_thread_policy" {
		t.Fatalf("effective thread policy = %q", response.Effective.ActiveThreadPolicyLabel)
	}
}
