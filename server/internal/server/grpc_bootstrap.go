package server

import (
	"log/slog"
	"os"

	"github.com/DenseAI/DenseCloud/go/telemetry"

	"github.com/DenseAI/DenseCore/server/internal/config"
	densecoregrpc "github.com/DenseAI/DenseCore/server/internal/grpc"
	"github.com/DenseAI/DenseCore/server/internal/middleware"
	"github.com/DenseAI/DenseCore/server/internal/workload"
)

// gRPC bootstrap ownership: gRPC server config resolution and server creation.
type grpcBootstrap struct {
	server *densecoregrpc.Server
	config densecoregrpc.Config
}

func createGRPCServer(
	opts *Options,
	cfg *config.ServerConfig,
	profile workload.Profile,
	services servicesBootstrap,
	authEnabled bool,
	apiKeyStore middleware.APIKeyStore,
	grpcMetrics *telemetry.GRPCMetrics,
) grpcBootstrap {
	grpcConfig := densecoregrpc.DefaultConfig()
	grpcConfig.Port = getEnvOrDefaultInt("GRPC_PORT", 50051)
	if opts.GRPCPort > 0 {
		grpcConfig.Port = opts.GRPCPort
	}
	grpcConfig.Enabled = profile.EnableGRPC && cfg.LLMAPIEnabled
	if os.Getenv("GRPC_ENABLED") != "" {
		grpcConfig.Enabled = parseEnvBool("GRPC_ENABLED", grpcConfig.Enabled)
	}
	if opts.GRPCEnabled != nil {
		grpcConfig.Enabled = *opts.GRPCEnabled
	}
	if !cfg.LLMAPIEnabled && grpcConfig.Enabled {
		slog.Warn("disabling gRPC because LLM API is disabled for current profile")
		grpcConfig.Enabled = false
	}
	grpcConfig.AuthEnabled = authEnabled
	grpcConfig.APIKeyStore = apiKeyStore
	grpcConfig.TLSEnabled = parseEnvBool("GRPC_TLS_ENABLED", false)
	grpcConfig.TLSCertFile = getEnvOrDefault("GRPC_TLS_CERT_FILE", "")
	grpcConfig.TLSKeyFile = getEnvOrDefault("GRPC_TLS_KEY_FILE", "")
	grpcConfig.TLSClientCAFile = getEnvOrDefault("GRPC_TLS_CLIENT_CA_FILE", "")
	grpcConfig.TLSRequireClientCert = parseEnvBool("GRPC_TLS_REQUIRE_CLIENT_CERT", false)
	grpcConfig.SharedMetrics = grpcMetrics

	var grpcServer *densecoregrpc.Server
	if cfg.LLMAPIEnabled {
		grpcServer = densecoregrpc.NewServer(services.chatService, services.modelService, grpcConfig)
	}

	return grpcBootstrap{server: grpcServer, config: grpcConfig}
}
