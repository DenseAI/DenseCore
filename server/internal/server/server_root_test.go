package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestMakeRootHandlerDoesNotMutateConfiguredEndpoints(t *testing.T) {
	baseEndpoints := map[string]string{
		"health":  "/health",
		"runtime": "/v1/runtime/profile",
	}

	handler := makeRootHandler(rootHandlerConfig{
		ServiceName:                 "DenseCore",
		ServiceDescription:          "test",
		WorkloadProfile:             "generic",
		LLMAPIEnabled:               false,
		ModelLifecycleProbesEnabled: false,
		AuthEnabled:                 false,
		MetricsEnabled:              true,
		MetricsPath:                 "/metrics",
		Endpoints:                   baseEndpoints,
	})

	for i := 0; i < 2; i++ {
		req := httptest.NewRequest(http.MethodGet, "/", nil)
		w := httptest.NewRecorder()
		handler(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("expected status 200, got %d", w.Code)
		}

		var response struct {
			Endpoints map[string]string `json:"endpoints"`
		}
		if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
			t.Fatalf("failed to decode response: %v", err)
		}
		if got := response.Endpoints["metrics"]; got != "/metrics" {
			t.Fatalf("expected metrics endpoint in response, got %q", got)
		}
	}

	if _, exists := baseEndpoints["metrics"]; exists {
		t.Fatalf("base endpoints map should not be mutated with metrics key")
	}
	if len(baseEndpoints) != 2 {
		t.Fatalf("base endpoints map mutated, expected len=2 got len=%d", len(baseEndpoints))
	}
}

func TestBuildEndpointsIncludesCompletionsWhenLLMAPIEnabled(t *testing.T) {
	endpoints := buildEndpoints(true)
	if got := endpoints["completions"]; got != "/v1/completions" {
		t.Fatalf("expected completions endpoint, got %q", got)
	}
}
