package server

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"os"
	"strings"
	"time"

	"github.com/DenseAI/DenseCore/server/internal/api"
	"github.com/DenseAI/DenseCore/server/internal/buildinfo"
)

const (
	Banner = `
==========================================
  DenseCore Server v%s
  Cloud-Native CPU Inference Engine
  OpenAI-Compatible REST + optional gRPC
  Configurable auth, health, and shutdown
==========================================
`
	// envTrue is the expected value for boolean environment variables
	envTrue = "true"
	// defaultMetricsPath is used when DENSECORE_ENT_METRICS_PATH is unset.
	defaultMetricsPath = "/metrics/enterprise"
	// startupRollbackTimeout bounds rollback cleanup on startup failures.
	startupRollbackTimeout = 30 * time.Second
)

// Options for starting the server.
type Options struct {
	Host         string
	Port         int
	ModelPath    string
	Threads      int
	Verbose      bool
	LogOutput    io.Writer     // If nil, defaults to os.Stdout. Use io.Discard to silence logs.
	ShowBanner   bool          // Whether to print the ASCII banner
	Background   bool          // If true, don't wait for shutdown signals (caller manages lifecycle)
	ShutdownChan chan struct{} // Channel to signal shutdown in background mode
	AuthEnabled  bool          // Enable API key authentication

	// gRPC options. If GRPCEnabled is nil, environment config is used.
	GRPCEnabled *bool
	GRPCPort    int
}

// ServerInstance represents a running server instance for external control (TUI mode).
type ServerInstance struct {
	shutdownChan chan struct{}
	done         chan error
}

type denseCoreMetricsCollector struct {
	handler *api.Handler
}

func (c denseCoreMetricsCollector) AppendPrometheus(builder *strings.Builder) {
	if builder == nil || c.handler == nil {
		return
	}
	builder.WriteString(c.handler.RenderMetrics())
}

// Shutdown gracefully shuts down the server.
func (s *ServerInstance) Shutdown(ctx context.Context) error {
	if s == nil {
		return nil
	}
	close(s.shutdownChan)
	select {
	case err := <-s.done:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Start initializes and starts the server in background mode, returning a controllable instance.
func Start(opts *Options) (*ServerInstance, error) {
	if opts == nil {
		opts = &Options{}
	}
	opts.Background = true
	shutdownChan := make(chan struct{})
	opts.ShutdownChan = shutdownChan

	done := make(chan error, 1)
	go func() {
		done <- Run(opts)
	}()

	return &ServerInstance{
		shutdownChan: shutdownChan,
		done:         done,
	}, nil
}

// Run starts the DenseCore server.
func Run(opts *Options) error {
	if opts == nil {
		opts = &Options{}
	}

	rollback := newStartupRollback()
	defer rollback.RunIfArmed()

	logOutput := opts.LogOutput
	if logOutput == nil {
		logOutput = os.Stdout
	}

	cfg, profile, logLevel, err := loadBootstrapConfig(opts)
	if err != nil {
		slog.Error("failed to load config", slog.String("error", err.Error()))
		return err
	}
	initServerTelemetry(logLevel, logOutput)
	if err := cfg.Validate(); err != nil {
		slog.Error("invalid config", slog.String("error", err.Error()))
		return err
	}
	if err := validateFixedModelServeConfig(cfg); err != nil {
		slog.Error("invalid fixed-model serve config", slog.String("error", err.Error()))
		return err
	}

	runtimeSetup, err := startRuntimeExtensions(cfg, rollback.Register)
	if err != nil {
		return err
	}
	otelProvider, err := startOTelProvider(rollback.Register)
	if err != nil {
		return err
	}

	if opts.ShowBanner {
		fmt.Printf(Banner, buildinfo.Version)
	}
	slog.Info("configuration loaded", slog.String("config", cfg.String()))

	tuning := configureRuntimeTuning(cfg)
	services, err := assembleServices(cfg, tuning, runtimeSetup.usageRecorder, rollback.Register)
	if err != nil {
		slog.Error("failed to initialize fixed startup model", slog.String("error", err.Error()))
		return err
	}
	metrics := buildHTTPMetrics(cfg, services.handler)
	apiKeyStore, authEnabled, err := setupAPIKeyStore(opts)
	if err != nil {
		return err
	}
	rateLimiter := setupRateLimiter(cfg)

	httpSetup, err := assembleHTTPRuntime(cfg, runtimeSetup, services, metrics.http, apiKeyStore, rateLimiter, authEnabled)
	if err != nil {
		return err
	}
	grpcSetup := createGRPCServer(opts, cfg, profile, services, authEnabled, apiKeyStore, metrics.grpc)
	shutdownHooks := buildShutdownHooks(services, otelProvider, runtimeSetup)
	preShutdownHooks := buildPreShutdownHooks(services)

	ctx, err := runContext(opts)
	if err != nil {
		return err
	}

	runner, err := createDenseCloudRunner(cfg, httpSetup, grpcSetup, preShutdownHooks, shutdownHooks)
	if err != nil {
		return err
	}
	rollback.Disarm()

	slog.Info("starting DenseCore server",
		slog.String("address", cfg.Address()),
		slog.Bool("auth_enabled", authEnabled),
		slog.Bool("grpc_enabled", grpcSetup.config.Enabled),
	)

	return runner.RunBlocking(ctx)
}

func runContext(opts *Options) (context.Context, error) {
	if !opts.Background {
		return context.Background(), nil
	}
	if opts.ShutdownChan == nil {
		return nil, fmt.Errorf("background mode requires ShutdownChan")
	}

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		<-opts.ShutdownChan
		cancel()
	}()
	return ctx, nil
}
