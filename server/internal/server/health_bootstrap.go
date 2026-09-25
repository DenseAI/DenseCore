package server

import (
	"context"
	"errors"

	cloudmw "github.com/DenseAI/DenseCloud/go/middleware"
	cloudserver "github.com/DenseAI/DenseCloud/go/server"

	"github.com/DenseAI/DenseCore/server/internal/config"
	"github.com/DenseAI/DenseCore/server/internal/domain"
	"github.com/DenseAI/DenseCore/server/internal/middleware"
)

// Health bootstrap ownership: DenseCloud health registry and dependency checks.
func buildDenseCoreHealthRegistry(cfg *config.ServerConfig, modelService domain.ModelService, apiKeyStore middleware.APIKeyStore, rateLimiter cloudmw.RateLimiterInterface) *cloudserver.HealthRegistry {
	registry := cloudserver.NewHealthRegistry()

	if cfg.ModelLifecycleProbesEnabled {
		registry.RegisterStartup("model_loaded", func(context.Context) error {
			switch modelService.GetLoadingStatus() {
			case domain.StatusReady:
				return nil
			case domain.StatusLoading:
				return errors.New("model loading")
			case domain.StatusFailed:
				return modelLoadingErrorOrDefault(modelService, "model load failed")
			default:
				if modelService.GetEngine() != nil {
					return nil
				}
				return errors.New("no model configured")
			}
		})
		registry.RegisterReadiness("model_ready", func(context.Context) error {
			if modelService.IsLoading() {
				return errors.New("model loading")
			}
			if modelService.GetEngine() == nil {
				return modelLoadingErrorOrDefault(modelService, "model not loaded")
			}
			return nil
		})
	}

	registerHealthDependency(registry, "api_key_store", apiKeyStore)
	registerHealthDependency(registry, "rate_limiter", rateLimiter)
	return registry
}

func modelLoadingErrorOrDefault(modelService domain.ModelService, defaultMessage string) error {
	if err := modelService.GetLoadingError(); err != nil {
		return err
	}
	return errors.New(defaultMessage)
}

func registerHealthDependency(registry *cloudserver.HealthRegistry, name string, dependency any) {
	if registry == nil || dependency == nil {
		return
	}
	healthDependency, ok := dependency.(interface {
		HealthCheck(context.Context) error
	})
	if !ok {
		return
	}
	registry.RegisterDependency(name, healthDependency, cloudserver.HealthPhaseStartup, cloudserver.HealthPhaseReady)
}
