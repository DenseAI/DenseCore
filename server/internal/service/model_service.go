package service

import (
	"log"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"

	"descore-server/internal/domain"
	"descore-server/internal/engine"
)

type ModelService struct {
	currentEngine    domain.Engine
	currentModelPath string
	mu               sync.RWMutex

	// Atomic status for non-blocking startup probe
	loadingStatus atomic.Int32
	loadingError  atomic.Value // stores error
}

func NewModelService() *ModelService {
	return &ModelService{}
}

// LoadModel loads a model with Blue/Green deployment strategy.
//
// If force=false (default): Loads new engine first, then swaps atomically, then closes old.
//   - Zero downtime during model updates
//   - Requires 2x memory temporarily
//
// If force=true: Unloads old engine first, then loads new one.
//   - Has downtime window
//   - Safe when memory is constrained
func (s *ModelService) LoadModel(mainModelPath, draftModelPath string, threads int) error {
	return s.LoadModelWithOptions(mainModelPath, draftModelPath, threads, false)
}

// LoadModelWithOptions provides explicit control over the loading strategy.
func (s *ModelService) LoadModelWithOptions(mainModelPath, draftModelPath string, threads int, force bool) error {
	s.loadingStatus.Store(int32(domain.StatusLoading))
	s.loadingError.Store(error(nil))

	var err error
	if force {
		err = s.loadModelForce(mainModelPath, draftModelPath, threads)
	} else {
		err = s.loadModelBlueGreen(mainModelPath, draftModelPath, threads)
	}

	if err != nil {
		s.loadingStatus.Store(int32(domain.StatusFailed))
		s.loadingError.Store(err)
		return err
	}

	s.loadingStatus.Store(int32(domain.StatusReady))
	return nil
}

// loadModelBlueGreen implements zero-downtime Blue/Green deployment.
// 1. Load new engine (old continues serving)
// 2. Atomically swap the engine pointer
// 3. Close old engine
func (s *ModelService) loadModelBlueGreen(mainModelPath, draftModelPath string, threads int) error {
	log.Printf("[ModelService] Blue/Green loading: %s", mainModelPath)

	// Step 1: Initialize new engine (old engine still serves requests)
	newEngine, err := engine.NewDenseEngine(mainModelPath, draftModelPath, threads)
	if err != nil {
		log.Printf("[ModelService] Failed to load new engine, falling back to force mode: %v", err)
		// If Blue/Green fails (e.g., OOM), fall back to force mode
		return s.loadModelForce(mainModelPath, draftModelPath, threads)
	}

	// Step 2: Atomically swap engine (critical section - very short)
	s.mu.Lock()
	oldEngine := s.currentEngine
	s.currentEngine = newEngine
	s.currentModelPath = mainModelPath
	s.mu.Unlock()

	log.Printf("[ModelService] Engine swapped successfully")

	// Step 3: Close old engine (outside of lock, non-blocking for new requests)
	if oldEngine != nil {
		log.Printf("[ModelService] Closing old engine...")
		oldEngine.Close()
		log.Printf("[ModelService] Old engine closed")
	}

	return nil
}

// loadModelForce implements the original behavior with downtime.
// Unloads old engine first, then loads new one.
func (s *ModelService) loadModelForce(mainModelPath, draftModelPath string, threads int) error {
	log.Printf("[ModelService] Force loading (with downtime): %s", mainModelPath)

	s.mu.Lock()
	defer s.mu.Unlock()

	// Unload existing if any
	if s.currentEngine != nil {
		log.Printf("[ModelService] Unloading current engine...")
		s.currentEngine.Close()
		s.currentEngine = nil
	}

	// Initialize new engine
	eng, err := engine.NewDenseEngine(mainModelPath, draftModelPath, threads)
	if err != nil {
		return err
	}

	s.currentEngine = eng
	s.currentModelPath = mainModelPath
	return nil
}

func (s *ModelService) UnloadModel() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.currentEngine != nil {
		s.currentEngine.Close()
		s.currentEngine = nil
	}
	s.currentModelPath = ""
	s.loadingStatus.Store(int32(domain.StatusIdle))
	return nil
}

func (s *ModelService) GetCurrentModel() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.currentModelPath
}

// GetModelIdentity returns user-facing identifiers for the loaded model.
func (s *ModelService) GetModelIdentity() (id, ownedBy, root string) {
	root = s.GetCurrentModel()
	if root == "" {
		return "densecore-v1", "densecore", ""
	}

	base := strings.TrimSuffix(filepath.Base(root), filepath.Ext(root))
	if base == "" {
		base = "densecore-v1"
	}

	return base, "local", root
}

func (s *ModelService) GetEngine() domain.Engine {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.currentEngine
}

// GetLoadingStatus returns the current loading status for startup probes.
func (s *ModelService) GetLoadingStatus() domain.LoadingStatus {
	return domain.LoadingStatus(s.loadingStatus.Load())
}

// GetLoadingError returns the last loading error, if any.
func (s *ModelService) GetLoadingError() error {
	if err := s.loadingError.Load(); err != nil {
		return err.(error)
	}
	return nil
}

// IsLoading returns true if a model is currently being loaded.
func (s *ModelService) IsLoading() bool {
	return s.GetLoadingStatus() == domain.StatusLoading
}
