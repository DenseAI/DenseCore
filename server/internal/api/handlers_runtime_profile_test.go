package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

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
			BenchmarkProfile: "single-e2e",
			EngineThreads:    16,
			GoWorkers:        1,
			ServerInflight:   1,
			KVType:           "q8_0",
			MaxSeqLen:        16384,
			KVTargetMB:       4096,
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
	if response.Queue.MaxSize != 1 {
		t.Fatalf("queue.max_size = %d, want 1", response.Queue.MaxSize)
	}
}
