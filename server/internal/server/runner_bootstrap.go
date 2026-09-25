package server

import (
	"fmt"
	"time"

	"github.com/DenseAI/DenseCore/server/internal/config"

	cloudserver "github.com/DenseAI/DenseCloud/go/server"
)

const defaultRunnerShutdownTimeout = 30 * time.Second

// Runner bootstrap ownership: DenseCloud Runner wiring and lifecycle hook order.
func createDenseCloudRunner(cfg *config.ServerConfig, httpSetup httpRuntimeBootstrap, grpcSetup grpcBootstrap,
	preShutdownHooks, shutdownHooks []cloudserver.ShutdownHook) (*cloudserver.Runner, error) {
	runner, err := cloudserver.NewRunner(denseCloudRunnerOptions(cfg, httpSetup, grpcSetup, preShutdownHooks, shutdownHooks))
	if err != nil {
		return nil, fmt.Errorf("failed to create server runner: %w", err)
	}
	return runner, nil
}

func denseCloudRunnerOptions(cfg *config.ServerConfig, httpSetup httpRuntimeBootstrap, grpcSetup grpcBootstrap,
	preShutdownHooks, shutdownHooks []cloudserver.ShutdownHook) cloudserver.Options {
	var cloudGRPC cloudserver.GRPCServer
	if grpcSetup.server != nil {
		cloudGRPC = grpcSetup.server
	}

	return cloudserver.Options{
		HTTPServer:       httpSetup.server,
		GRPCServer:       cloudGRPC,
		EnableGRPC:       grpcSetup.config.Enabled,
		ShutdownTimeout:  effectiveRunnerShutdownTimeout(cfg.ShutdownTimeout),
		PreShutdownHooks: preShutdownHooks,
		StartupHooks: []cloudserver.StartupHook{
			httpSetup.runtime.Startup,
		},
		ShutdownHooks: append([]cloudserver.ShutdownHook{
			httpSetup.runtime.Shutdown,
		}, shutdownHooks...),
	}
}

func effectiveRunnerShutdownTimeout(timeout time.Duration) time.Duration {
	if timeout <= 0 {
		return defaultRunnerShutdownTimeout
	}
	return timeout
}
