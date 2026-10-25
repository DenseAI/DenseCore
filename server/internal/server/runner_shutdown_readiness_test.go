package server

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	cloudserver "github.com/DenseAI/DenseCloud/go/server"
	"github.com/DenseAI/DenseCore/server/internal/config"
)

// Exercise the real SDK runner/runtime order while an accepted handler and the
// product's native-drain pre-hook remain blocked. No wall-clock drain sleeps.
func TestRunnerWithdrawsReadinessBeforeBlockingProductDrain(t *testing.T) {
	var closing atomic.Bool
	accepted := make(chan struct{})
	release := make(chan struct{})
	drained := make(chan struct{})
	enteredDrain := make(chan struct{})
	var releaseOnce sync.Once
	defer releaseOnce.Do(func() { close(release) })
	root := http.NewServeMux()
	root.HandleFunc("/accepted", func(w http.ResponseWriter, r *http.Request) {
		close(accepted)
		select {
		case <-release:
			_, _ = io.WriteString(w, "accepted complete")
		case <-r.Context().Done():
			t.Error("accepted request context cancelled before drain completed")
		}
		close(drained)
	})
	root.HandleFunc("/submit", func(w http.ResponseWriter, r *http.Request) {
		if closing.Load() {
			http.Error(w, "engine draining", http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	})
	runtime, err := cloudserver.NewHTTPRuntime(cloudserver.HTTPRuntimeConfig{ServiceName: "drain-test", RootMux: root, DisableRegisteredExtensions: true})
	if err != nil {
		t.Fatal(err)
	}
	if err = runtime.Startup(context.Background()); err != nil {
		t.Fatal(err)
	}
	transport := httptest.NewServer(runtime.Handler())
	defer transport.Close()
	client := transport.Client()
	client.Timeout = 2 * time.Second
	status := func(method, path string) int {
		t.Helper()
		request, err := http.NewRequest(method, transport.URL+path, nil)
		if err != nil {
			t.Fatal(err)
		}
		response, err := client.Do(request)
		if err != nil {
			t.Fatal(err)
		}
		defer response.Body.Close()
		if _, err := io.Copy(io.Discard, response.Body); err != nil {
			t.Fatal(err)
		}
		return response.StatusCode
	}
	if got := status(http.MethodGet, "/health/ready"); got != http.StatusOK {
		t.Fatalf("initial readiness=%d", got)
	}
	requestDone := make(chan string, 1)
	go func() {
		response, err := client.Get(transport.URL + "/accepted")
		if err != nil {
			requestDone <- err.Error()
			return
		}
		defer response.Body.Close()
		body, err := io.ReadAll(response.Body)
		if err != nil {
			requestDone <- err.Error()
			return
		}
		requestDone <- string(body)
	}()
	select {
	case <-accepted:
	case <-time.After(2 * time.Second):
		t.Fatal("handler did not start")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	hook := func(hookCtx context.Context) error {
		if hookCtx != ctx {
			t.Error("product drain did not receive the overall shutdown context")
		}
		closing.Store(true)
		close(enteredDrain)
		select {
		case <-drained:
			return nil
		case <-hookCtx.Done():
			return hookCtx.Err()
		}
	}
	options := denseCloudRunnerOptions(config.DefaultConfig(), httpRuntimeBootstrap{runtime: runtime, server: transport.Config}, grpcBootstrap{}, []cloudserver.ShutdownHook{hook}, nil)
	runner, err := cloudserver.NewRunner(options)
	if err != nil {
		t.Fatal(err)
	}
	shutdownDone := make(chan error, 1)
	go func() { shutdownDone <- runner.Shutdown(ctx) }()
	defer func() {
		releaseOnce.Do(func() { close(release) })
		select {
		case err := <-shutdownDone:
			if err != nil {
				t.Errorf("shutdown: %v", err)
			}
		case <-time.After(2 * time.Second):
			t.Error("shutdown owner did not finish")
		}
	}()
	select {
	case <-enteredDrain:
	case <-time.After(2 * time.Second):
		t.Fatal("product drain did not start")
	}
	// The native pre-hook is still active, so transport remains available and
	// readiness must be an explicit 503, rather than a reset after the stream ends.
	if got := status(http.MethodGet, "/health/ready"); got != http.StatusServiceUnavailable {
		t.Fatalf("readiness during blocked native drain=%d, want 503", got)
	}
	if got := status(http.MethodPost, "/submit"); got != http.StatusServiceUnavailable {
		t.Fatalf("admission during blocked native drain=%d, want 503", got)
	}
	select {
	case <-requestDone:
		t.Fatal("accepted handler completed before release")
	default:
	}
	client.CloseIdleConnections()
	releaseOnce.Do(func() { close(release) })
	select {
	case body := <-requestDone:
		if body != "accepted complete" {
			t.Fatalf("accepted response=%q", body)
		}
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
}
