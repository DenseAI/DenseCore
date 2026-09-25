package service

import (
	"context"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/DenseAI/DenseCore/server/internal/domain"
	"github.com/DenseAI/DenseCore/server/internal/engine"
)

type modelLoadStrategy int

const (
	modelLoadStrategyAuto modelLoadStrategy = iota
	modelLoadStrategyBlueGreen
	modelLoadStrategyForce
)

type engineSlot struct {
	engine    domain.Engine
	closing   bool
	refCount  int
	wg        sync.WaitGroup
	closeOnce sync.Once
}

type engineLease struct {
	engine domain.Engine
	slot   *engineSlot
	owner  *ModelService
	once   sync.Once
}

func (l *engineLease) Engine() domain.Engine {
	if l == nil {
		return nil
	}
	return l.engine
}

func (l *engineLease) Close() {
	if l == nil || l.owner == nil || l.slot == nil {
		return
	}
	l.once.Do(func() {
		l.owner.releaseEngineLease(l.slot)
	})
}

type ModelService struct {
	currentSlot      *engineSlot
	currentModelPath string
	mu               sync.RWMutex

	// Atomic status for non-blocking startup probe
	loadingStatus atomic.Int32
	loadingError  atomic.Pointer[loadErrorValue]
}

type loadErrorValue struct {
	err error
}

func NewModelService() *ModelService {
	return &ModelService{}
}

// LoadModel loads a model using the configured deployment strategy.
//
// If force=false and DENSECORE_MODEL_LOAD_STRATEGY=blue_green:
// Loads new engine first, then swaps atomically, then closes old.
//   - Zero downtime during model updates
//   - Requires 2x memory temporarily
//
// If force=false and DENSECORE_MODEL_LOAD_STRATEGY=auto (default):
// Uses Blue/Green only when basic memory headroom checks indicate that the new
// model can be loaded alongside the current one. Otherwise it uses force mode.
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
	s.loadingError.Store(nil)

	var err error
	strategy := resolveModelLoadStrategy(os.Getenv("DENSECORE_MODEL_LOAD_STRATEGY"))
	if force || strategy == modelLoadStrategyForce ||
		(strategy == modelLoadStrategyAuto && s.shouldForceModelLoadForMemory(mainModelPath, draftModelPath)) {
		err = s.loadModelForce(mainModelPath, draftModelPath, threads)
	} else {
		err = s.loadModelBlueGreen(mainModelPath, draftModelPath, threads)
	}

	if err != nil {
		s.loadingStatus.Store(int32(domain.StatusFailed))
		s.loadingError.Store(&loadErrorValue{err: err})
		return err
	}

	s.loadingError.Store(nil)
	s.loadingStatus.Store(int32(domain.StatusReady))
	return nil
}

func resolveModelLoadStrategy(value string) modelLoadStrategy {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", "auto":
		return modelLoadStrategyAuto
	case "blue_green", "blue-green", "bluegreen":
		return modelLoadStrategyBlueGreen
	case "force":
		return modelLoadStrategyForce
	default:
		log.Printf("[ModelService] Unknown DENSECORE_MODEL_LOAD_STRATEGY=%q, using auto", value)
		return modelLoadStrategyAuto
	}
}

func (s *ModelService) hasCurrentEngine() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.currentSlot != nil && s.currentSlot.engine != nil
}

func (s *ModelService) shouldForceModelLoadForMemory(mainModelPath, draftModelPath string) bool {
	if !s.hasCurrentEngine() {
		return false
	}

	requiredBytes, ok := estimateModelFileBytes(mainModelPath, draftModelPath)
	if !ok || requiredBytes == 0 {
		return false
	}
	availableBytes, ok := linuxMemAvailableBytes("/proc/meminfo")
	if !ok {
		return false
	}

	if availableBytes < requiredBytes {
		log.Printf(
			"[ModelService] Auto load strategy selected force mode: mem_available_mb=%d estimated_new_model_file_mb=%d",
			availableBytes/(1024*1024), requiredBytes/(1024*1024),
		)
		return true
	}
	return false
}

func estimateModelFileBytes(paths ...string) (uint64, bool) {
	var total uint64
	var sawPath bool
	for _, path := range paths {
		if strings.TrimSpace(path) == "" {
			continue
		}
		info, err := os.Stat(path)
		if err != nil || info.IsDir() || info.Size() <= 0 {
			return 0, false
		}
		total += uint64(info.Size())
		sawPath = true
	}
	return total, sawPath
}

func linuxMemAvailableBytes(path string) (uint64, bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, false
	}
	for _, line := range strings.Split(string(data), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 2 || fields[0] != "MemAvailable:" {
			continue
		}
		kb, err := strconv.ParseUint(fields[1], 10, 64)
		if err != nil {
			return 0, false
		}
		return kb * 1024, true
	}
	return 0, false
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
	oldSlot := s.currentSlot
	if oldSlot != nil {
		oldSlot.closing = true
	}
	s.currentSlot = &engineSlot{engine: newEngine}
	s.currentModelPath = mainModelPath
	s.mu.Unlock()

	log.Printf("[ModelService] Engine swapped successfully")

	// Step 3: Close old engine (outside of lock, non-blocking for new requests)
	if oldSlot != nil {
		log.Printf("[ModelService] Closing old engine...")
		if err := s.closeEngineSlot(context.Background(), oldSlot); err != nil {
			return err
		}
		log.Printf("[ModelService] Old engine closed")
	}

	return nil
}

// loadModelForce implements the original behavior with downtime.
// Unloads old engine first, then loads new one.
func (s *ModelService) loadModelForce(mainModelPath, draftModelPath string, threads int) error {
	log.Printf("[ModelService] Force loading (with downtime): %s", mainModelPath)

	s.mu.Lock()
	oldSlot := s.currentSlot
	if oldSlot != nil {
		log.Printf("[ModelService] Unloading current engine...")
		oldSlot.closing = true
	}
	s.currentSlot = nil
	s.currentModelPath = ""
	s.mu.Unlock()

	if err := s.closeEngineSlot(context.Background(), oldSlot); err != nil {
		return err
	}

	// Initialize new engine
	eng, err := engine.NewDenseEngine(mainModelPath, draftModelPath, threads)
	if err != nil {
		return err
	}

	s.mu.Lock()
	s.currentSlot = &engineSlot{engine: eng}
	s.currentModelPath = mainModelPath
	s.mu.Unlock()
	return nil
}

func (s *ModelService) UnloadModel() error {
	return s.UnloadModelContext(context.Background())
}

func (s *ModelService) UnloadModelContext(ctx context.Context) error {
	s.mu.Lock()
	oldSlot := s.currentSlot
	if oldSlot != nil {
		oldSlot.closing = true
	}
	s.currentSlot = nil
	s.currentModelPath = ""
	s.mu.Unlock()

	if err := s.closeEngineSlot(ctx, oldSlot); err != nil {
		return err
	}

	s.loadingError.Store(nil)
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
	if s.currentSlot == nil {
		return nil
	}
	return s.currentSlot.engine
}

// GetLoadingStatus returns the current loading status for startup probes.
func (s *ModelService) GetLoadingStatus() domain.LoadingStatus {
	return domain.LoadingStatus(s.loadingStatus.Load())
}

// GetLoadingError returns the last loading error, if any.
func (s *ModelService) GetLoadingError() error {
	if err := s.loadingError.Load(); err != nil {
		return err.err
	}
	return nil
}

// IsLoading returns true if a model is currently being loaded.
func (s *ModelService) IsLoading() bool {
	return s.GetLoadingStatus() == domain.StatusLoading
}

func (s *ModelService) AcquireEngineLease() *engineLease {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.currentSlot == nil || s.currentSlot.engine == nil || s.currentSlot.closing {
		return nil
	}

	s.currentSlot.refCount++
	s.currentSlot.wg.Add(1)
	return &engineLease{
		engine: s.currentSlot.engine,
		slot:   s.currentSlot,
		owner:  s,
	}
}

// BeginEngineShutdown stops new leases and asks the native engine to resolve
// accepted requests before the HTTP server waits for handlers to drain.
func (s *ModelService) BeginEngineShutdown() {
	s.mu.Lock()
	slot := s.currentSlot
	if slot != nil {
		slot.closing = true
	}
	var eng domain.Engine
	if slot != nil {
		eng = slot.engine
	}
	s.mu.Unlock()

	if shutdowner, ok := eng.(interface{ BeginShutdown() }); ok {
		shutdowner.BeginShutdown()
	}
}

func (s *ModelService) releaseEngineLease(slot *engineSlot) {
	if slot == nil {
		return
	}

	s.mu.Lock()
	slot.refCount--
	s.mu.Unlock()
	slot.wg.Done()
}

func (s *ModelService) closeEngineSlot(ctx context.Context, slot *engineSlot) error {
	if slot == nil || slot.engine == nil {
		return nil
	}

	done := make(chan struct{})
	go func() {
		slot.wg.Wait()
		close(done)
	}()

	if ctx == nil {
		<-done
	} else {
		select {
		case <-done:
		case <-ctx.Done():
			go func() { _ = s.closeEngineSlot(context.Background(), slot) }()
			return ctx.Err()
		}
	}

	slot.closeOnce.Do(func() {
		slot.engine.Close()
	})
	return nil
}
