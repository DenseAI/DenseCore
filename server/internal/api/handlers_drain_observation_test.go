package api

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/DenseAI/DenseCore/server/internal/domain"
)

type drainingObservationEngine struct{ *MockEngine }

func (e *drainingObservationEngine) GetDetailedMetrics() *domain.DetailedMetrics {
	return &domain.DetailedMetrics{ActiveRequests: 1}
}

type drainingObservationService struct {
	*MockModelService
	observed domain.Engine
	released atomic.Int32
}

// Inference no longer acquires an engine, while diagnostics retain an owner.
func (s *drainingObservationService) GetEngine() domain.Engine { return nil }
func (s *drainingObservationService) AcquireEngineObservationLease() (domain.Engine, func()) {
	return s.observed, func() { s.released.Add(1) }
}

type releaseCheckingResponse struct {
	*httptest.ResponseRecorder
	released *atomic.Int32
	t        *testing.T
}

func (w *releaseCheckingResponse) Write(body []byte) (int, error) {
	if w.released.Load() != 1 {
		w.t.Error("HTTP body write retained the diagnostic engine lease")
	}
	return w.ResponseRecorder.Write(body)
}
func TestDrainDiagnosticsKeepNativeActiveSnapshotAndReleaseBeforeNetworkWrite(t *testing.T) {
	for _, route := range []string{"metrics", "profile"} {
		t.Run(route, func(t *testing.T) {
			engine := &drainingObservationEngine{MockEngine: &MockEngine{runtimeState: domain.RuntimeOptimizationState{ActiveThreadPolicyLabel: "native-draining"}}}
			svc := &drainingObservationService{MockModelService: NewMockModelService(), observed: engine}
			handler := NewHandler(nil, svc, WithModelLifecycleProbes(true))
			ready := httptest.NewRecorder()
			handler.ReadinessHandler(ready, httptest.NewRequest(http.MethodGet, "/health/ready", nil))
			if ready.Code != http.StatusServiceUnavailable {
				t.Fatalf("request admission readiness=%d", ready.Code)
			}
			writer := &releaseCheckingResponse{ResponseRecorder: httptest.NewRecorder(), released: &svc.released, t: t}
			if route == "metrics" {
				handler.MetricsHandler(writer, httptest.NewRequest(http.MethodGet, "/metrics", nil))
				if !strings.Contains(writer.Body.String(), "densecore_active_requests 1\n") {
					t.Fatalf("native active snapshot missing: %s", writer.Body.String())
				}
			} else {
				handler.RuntimeProfileHandler(writer, httptest.NewRequest(http.MethodGet, "/v1/runtime/profile", nil))
				if !strings.Contains(writer.Body.String(), "native-draining") {
					t.Fatalf("effective native profile missing: %s", writer.Body.String())
				}
			}
			if writer.Code != http.StatusOK {
				t.Fatalf("diagnostic status=%d", writer.Code)
			}
			if got := svc.released.Load(); got != 1 {
				t.Fatalf("observation release count=%d", got)
			}
		})
	}
}
