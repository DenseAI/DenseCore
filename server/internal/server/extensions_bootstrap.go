package server

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	cloudserver "github.com/DenseAI/DenseCloud/go/server"

	"descore-server/internal/config"
	"descore-server/internal/extensions"
)

// Runtime extension bootstrap ownership: extension startup, route hooks, and startup rollback.
type startupRollback struct {
	hooks []cloudserver.ShutdownHook
	armed bool
}

func newStartupRollback() *startupRollback {
	return &startupRollback{armed: true}
}

func (r *startupRollback) Register(name string, hook cloudserver.ShutdownHook) {
	r.hooks = append(r.hooks, func(ctx context.Context) error {
		if err := hook(ctx); err != nil {
			return fmt.Errorf("%s: %w", name, err)
		}
		return nil
	})
}

func (r *startupRollback) Disarm() {
	r.armed = false
}

func (r *startupRollback) RunIfArmed() {
	if r == nil || !r.armed {
		return
	}
	runStartupRollbackHooks(r.hooks)
}

type runtimeExtensionsBootstrap struct {
	runtime    extensions.Runtime
	chassisExt []cloudserver.RuntimeExtension
}

func startRuntimeExtensions(cfg *config.ServerConfig, registerRollback func(string, cloudserver.ShutdownHook)) (runtimeExtensionsBootstrap, error) {
	runtimeExt, err := extensions.NewRuntime(slog.Default())
	if err != nil {
		return runtimeExtensionsBootstrap{}, fmt.Errorf("failed to initialize runtime extensions: %w", err)
	}
	if err := validateMetricsPathConflict(cfg.MetricsEnabled, cfg.MetricsPath, runtimeExt.Enabled()); err != nil {
		return runtimeExtensionsBootstrap{}, err
	}
	if err := runtimeExt.Startup(context.Background()); err != nil {
		return runtimeExtensionsBootstrap{}, fmt.Errorf("runtime extensions startup failed: %w", err)
	}
	registerRollback("runtime extensions", runtimeExt.Shutdown)

	chassisExtensions := cloudserver.RuntimeExtensions()
	for _, ext := range chassisExtensions {
		extension := ext
		if err := extension.Startup(context.Background()); err != nil {
			return runtimeExtensionsBootstrap{}, fmt.Errorf("extension startup failed (%s): %w", extension.Name(), err)
		}
		registerRollback(fmt.Sprintf("extension %s", extension.Name()), extension.Shutdown)
	}

	return runtimeExtensionsBootstrap{runtime: runtimeExt, chassisExt: chassisExtensions}, nil
}

func runStartupRollbackHooks(hooks []cloudserver.ShutdownHook) {
	if len(hooks) == 0 {
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), startupRollbackTimeout)
	defer cancel()

	for i := len(hooks) - 1; i >= 0; i-- {
		hook := hooks[i]
		if hook == nil {
			continue
		}
		if err := hook(ctx); err != nil {
			slog.Warn("startup rollback hook failed", slog.String("error", err.Error()))
		}
	}
}

func validateMetricsPathConflict(coreMetricsEnabled bool, coreMetricsPath string, runtimeEnabled bool) error {
	if !coreMetricsEnabled || !runtimeEnabled {
		return nil
	}

	entMetricsPath := strings.TrimSpace(getEnvOrDefault("DENSECORE_ENT_METRICS_PATH", ""))
	if entMetricsPath == "" {
		entMetricsPath = defaultMetricsPath
	}
	coreMetricsPath = strings.TrimSpace(coreMetricsPath)
	if entMetricsPath == coreMetricsPath {
		return fmt.Errorf(
			"invalid metrics configuration: DENSECORE_ENT_METRICS_PATH (%q) conflicts with METRICS_PATH (%q)",
			entMetricsPath, coreMetricsPath,
		)
	}
	return nil
}
