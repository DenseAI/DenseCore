package server

import (
	"fmt"
	"time"

	cloudserver "github.com/DenseAI/DenseCloud/go/server"
)

// Runner bootstrap ownership: DenseCloud Runner wiring and lifecycle hook order.
func createDenseCloudRunner(httpSetup httpRuntimeBootstrap, grpcSetup grpcBootstrap, shutdownHooks []cloudserver.ShutdownHook) (*cloudserver.Runner, error) {
	var cloudGRPC cloudserver.GRPCServer
	if grpcSetup.server != nil {
		cloudGRPC = grpcSetup.server
	}

	runner, err := cloudserver.NewRunner(cloudserver.Options{
		HTTPServer:      httpSetup.server,
		GRPCServer:      cloudGRPC,
		EnableGRPC:      grpcSetup.config.Enabled,
		ShutdownTimeout: 30 * time.Second,
		StartupHooks: []cloudserver.StartupHook{
			httpSetup.runtime.Startup,
		},
		ShutdownHooks: append([]cloudserver.ShutdownHook{
			httpSetup.runtime.Shutdown,
		}, shutdownHooks...),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create server runner: %w", err)
	}
	return runner, nil
}
