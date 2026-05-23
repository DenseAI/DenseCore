package server

import (
	"fmt"
	"io"
	"log/slog"
	"os"
	"strconv"

	cloudmw "github.com/DenseAI/DenseCloud/go/middleware"
	cloudserver "github.com/DenseAI/DenseCloud/go/server"
	"github.com/DenseAI/DenseCloud/go/telemetry"

	"descore-server/internal/api"
	"descore-server/internal/buildinfo"
	"descore-server/internal/config"
)

// Telemetry bootstrap ownership: logger, OpenTelemetry, and HTTP/gRPC metrics setup.
func initServerTelemetry(logLevel string, output io.Writer) {
	telemetry.Init(telemetry.Config{
		ServiceName: "densecore",
		Version:     buildinfo.Version,
		Level:       logLevel,
		Output:      output,
	})
}

func startOTelProvider(registerRollback func(string, cloudserver.ShutdownHook)) (*telemetry.OTelProvider, error) {
	cloudmw.InitOTelPropagator()
	otelProvider, err := initOTelProviderFromEnv()
	if err != nil {
		return nil, err
	}
	if otelProvider != nil {
		registerRollback("otel provider", otelProvider.Shutdown)
	}
	return otelProvider, nil
}

type telemetryBootstrap struct {
	grpc *telemetry.GRPCMetrics
	http *telemetry.HTTPMetrics
}

func buildHTTPMetrics(cfg *config.ServerConfig, handler *api.Handler) telemetryBootstrap {
	grpcMetrics := telemetry.NewGRPCMetrics(telemetry.GRPCMetricsConfig{ServiceName: "densecore"})
	var httpMetrics *telemetry.HTTPMetrics
	if cfg.MetricsEnabled {
		httpMetrics = telemetry.NewHTTPMetrics(telemetry.HTTPMetricsConfig{
			ServiceName: "densecore",
			IgnorePaths: []string{
				"/health",
				"/health/live",
				"/health/ready",
				"/health/startup",
				cfg.MetricsPath,
			},
			Collectors: []telemetry.PrometheusCollector{
				grpcMetrics,
				denseCoreMetricsCollector{handler: handler},
			},
		})
	}
	return telemetryBootstrap{grpc: grpcMetrics, http: httpMetrics}
}

func initOTelProviderFromEnv() (*telemetry.OTelProvider, error) {
	enabled := parseEnvBool("OTEL_ENABLED", false)
	if !enabled {
		return nil, nil
	}

	cfg := telemetry.DefaultOTelConfig()
	cfg.Enabled = true
	cfg.Endpoint = getEnvOrDefault("OTEL_ENDPOINT", cfg.Endpoint)
	cfg.ServiceName = getEnvOrDefault("OTEL_SERVICE_NAME", "densecore-server")
	cfg.ServiceVersion = buildinfo.Version
	cfg.Insecure = parseEnvBool("OTEL_INSECURE", cfg.Insecure)

	if sampling := os.Getenv("OTEL_SAMPLING_RATE"); sampling != "" {
		if v, err := strconv.ParseFloat(sampling, 64); err == nil {
			cfg.SamplingRate = v
		} else {
			slog.Warn("invalid OTEL_SAMPLING_RATE, using default",
				slog.String("value", sampling),
				slog.Float64("default", cfg.SamplingRate),
				slog.String("error", err.Error()))
		}
	}

	provider, err := telemetry.InitOTelTracer(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize OpenTelemetry: %w", err)
	}
	return provider, nil
}
