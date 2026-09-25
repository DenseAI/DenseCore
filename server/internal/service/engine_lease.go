package service

import "github.com/DenseAI/DenseCore/server/internal/domain"

type engineLeaseProvider interface {
	AcquireEngineLease() *engineLease
}

func AcquireRequestEngine(modelService domain.ModelService) (domain.Engine, func()) {
	if modelService == nil {
		return nil, func() {}
	}
	if provider, ok := modelService.(engineLeaseProvider); ok {
		lease := provider.AcquireEngineLease()
		if lease == nil {
			return nil, func() {}
		}
		return lease.Engine(), lease.Close
	}
	engine := modelService.GetEngine()
	return engine, func() {}
}

func acquireRequestEngine(modelService domain.ModelService) (domain.Engine, func()) {
	return AcquireRequestEngine(modelService)
}
