package server

import (
	"context"
	"net/http"
	"testing"
	"time"

	cloudserver "github.com/DenseAI/DenseCloud/go/server"

	"github.com/DenseAI/DenseCore/server/internal/config"
)

func TestLoadBootstrapConfigAppliesCLIThreads(t *testing.T) {
	t.Setenv("THREADS", "")

	cfg, _, _, err := loadBootstrapConfig(&Options{Threads: 7})
	if err != nil {
		t.Fatalf("loadBootstrapConfig returned error: %v", err)
	}
	if cfg.Threads != 7 {
		t.Fatalf("cfg.Threads = %d, want 7", cfg.Threads)
	}
}

func TestDenseCloudRunnerOptionsUseConfiguredShutdownTimeout(t *testing.T) {
	cfg := config.DefaultConfig()
	cfg.ShutdownTimeout = 12 * time.Second

	options := denseCloudRunnerOptions(
		cfg,
		httpRuntimeBootstrap{
			runtime: &cloudserver.HTTPRuntime{},
			server:  &http.Server{},
		},
		grpcBootstrap{},
		nil,
		nil,
	)

	if options.ShutdownTimeout != 12*time.Second {
		t.Fatalf("ShutdownTimeout = %v, want %v", options.ShutdownTimeout, 12*time.Second)
	}
}

func TestDenseCloudRunnerOptionsPreservePreShutdownHooks(t *testing.T) {
	cfg := config.DefaultConfig()
	preHook := func(context.Context) error { return nil }

	options := denseCloudRunnerOptions(
		cfg,
		httpRuntimeBootstrap{runtime: &cloudserver.HTTPRuntime{}, server: &http.Server{}},
		grpcBootstrap{},
		[]cloudserver.ShutdownHook{preHook},
		nil,
	)

	if len(options.PreShutdownHooks) != 1 {
		t.Fatalf("PreShutdownHooks length = %d, want 1", len(options.PreShutdownHooks))
	}
}
