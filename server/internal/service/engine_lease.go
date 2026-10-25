package service

import (
	"errors"

	"github.com/DenseAI/DenseCore/server/internal/domain"
)

type engineLeaseProvider interface {
	AcquireEngineLease() *engineLease
}

func AcquireRequestEngine(modelService domain.ModelService) (domain.Engine, func()) {
	engine, release, _ := acquireRequestEngineWithError(modelService)
	return engine, release
}

func acquireRequestEngineWithError(modelService domain.ModelService) (domain.Engine, func(), error) {
	if provider, ok := modelService.(interface {
		AcquireRequestEngine() (domain.Engine, func(), error)
	}); ok {
		engine, release, err := provider.AcquireRequestEngine()
		if engine == nil && err == nil {
			err = errors.New("no model loaded")
		}
		return engine, release, err
	}
	if modelService == nil {
		return nil, func() {}, errors.New("no model loaded")
	}
	if provider, ok := modelService.(engineLeaseProvider); ok {
		lease := provider.AcquireEngineLease()
		if lease == nil {
			return nil, func() {}, errors.New("no model loaded")
		}
		return lease.Engine(), lease.Close, nil
	}
	engine := modelService.GetEngine()
	if engine == nil {
		return nil, func() {}, errors.New("no model loaded")
	}
	return engine, func() {}, nil
}

func acquireRequestEngine(modelService domain.ModelService) (domain.Engine, func()) {
	return AcquireRequestEngine(modelService)
}

// AcquireObservationEngine is restricted to diagnostic readers. Implementations
// without the optional observation contract retain the request-lease policy.
func AcquireObservationEngine(modelService domain.ModelService) (domain.Engine, func()) {
	if provider, ok := modelService.(interface {
		AcquireEngineObservationLease() (domain.Engine, func())
	}); ok {
		return provider.AcquireEngineObservationLease()
	}
	return AcquireRequestEngine(modelService)
}
