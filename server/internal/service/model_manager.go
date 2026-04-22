// Package service provides business logic services for DenseCore.
// Multi-Model Manager는 런타임에 여러 모델을 관리하고 LRU 정책으로 메모리를 최적화합니다.
package service

import (
	"container/list"
	"errors"
	"fmt"
	"log/slog"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"

	"descore-server/internal/domain"
	"descore-server/internal/engine"
)

// ============================================================================
// Error Definitions
// ============================================================================

var (
	// ErrModelNotLoaded indicates no model is currently loaded
	ErrModelNotLoaded = errors.New("model not loaded")

	// ErrModelNotFound indicates the requested model was not found
	ErrModelNotFound = errors.New("model not found")

	// ErrModelIDRequired indicates a model ID or path must be provided
	ErrModelIDRequired = errors.New("model id or path is required")

	// ErrMaxModelsLoaded indicates the maximum number of models is loaded
	ErrMaxModelsLoaded = errors.New("maximum number of models loaded")

	// ErrNoEvictableModels indicates all loaded models are in use and cannot be evicted
	ErrNoEvictableModels = errors.New("all loaded models are in use; cannot evict any models")

	// ErrModelInUse indicates the model is currently in use
	ErrModelInUse = errors.New("model is currently in use")
)

// ============================================================================
// Loaded Model
// ============================================================================

// LoadedModel represents a loaded model with metadata.
// 로드된 모델과 메타데이터를 나타냅니다.
type LoadedModel struct {
	// ID is the unique identifier for the model
	ID string

	// Path is the file path to the model
	Path string

	// DraftPath is the optional draft model path for speculative decoding
	DraftPath string

	// Engine is the underlying inference engine
	Engine domain.Engine

	// LoadedAt is the time when the model was loaded
	LoadedAt time.Time

	// LastUsed is the time when the model was last used
	LastUsed time.Time

	// RefCount is the number of active references to this model
	RefCount int32

	// Threads is the number of threads used for inference
	Threads int
}

// IncrRef increments the reference count.
// 참조 카운트를 증가시킵니다.
func (m *LoadedModel) IncrRef() {
	atomic.AddInt32(&m.RefCount, 1)
	m.LastUsed = time.Now()
}

// DecrRef decrements the reference count.
// 참조 카운트를 감소시킵니다.
func (m *LoadedModel) DecrRef() {
	atomic.AddInt32(&m.RefCount, -1)
}

// GetRefCount returns the current reference count.
// 현재 참조 카운트를 반환합니다.
func (m *LoadedModel) GetRefCount() int32 {
	return atomic.LoadInt32(&m.RefCount)
}

// ============================================================================
// Model Manager
// ============================================================================

// ModelManagerConfig holds configuration for ModelManager.
// ModelManager의 설정을 보관합니다.
type ModelManagerConfig struct {
	// MaxModels is the maximum number of models that can be loaded simultaneously
	MaxModels int

	// DefaultThreads is the default number of threads for inference
	DefaultThreads int

	// EvictionPolicy determines how models are evicted (currently only LRU)
	EvictionPolicy string
}

// DefaultModelManagerConfig returns the default configuration.
// 기본 설정을 반환합니다.
func DefaultModelManagerConfig() ModelManagerConfig {
	return ModelManagerConfig{
		MaxModels:      2,
		DefaultThreads: 0,
		EvictionPolicy: "lru",
	}
}

// ModelManager manages multiple loaded models with LRU eviction.
// 여러 로드된 모델을 LRU 정책으로 관리합니다.
type ModelManager struct {
	mu          sync.RWMutex
	models      map[string]*LoadedModel
	lruOrder    *list.List
	lruElements map[string]*list.Element
	config      ModelManagerConfig
	defaultID   string

	// Statistics
	totalLoads  int64
	totalEvicts int64
	cacheHits   int64
	cacheMisses int64
}

// NewModelManager creates a new ModelManager.
// 새 ModelManager를 생성합니다.
func NewModelManager(cfg ModelManagerConfig) *ModelManager {
	if cfg.MaxModels <= 0 {
		cfg.MaxModels = 2
	}

	return &ModelManager{
		models:      make(map[string]*LoadedModel),
		lruOrder:    list.New(),
		lruElements: make(map[string]*list.Element),
		config:      cfg,
	}
}

// ============================================================================
// Model Operations
// ============================================================================

// GetModel retrieves a loaded model by ID.
// ID로 로드된 모델을 가져옵니다. 참조 카운트가 증가되므로 사용 후 ReleaseModel을 호출해야 합니다.
func (m *ModelManager) GetModel(modelID string) (*LoadedModel, error) {
	m.mu.RLock()
	model, ok := m.models[modelID]
	m.mu.RUnlock()

	if !ok {
		atomic.AddInt64(&m.cacheMisses, 1)
		return nil, ErrModelNotFound
	}

	atomic.AddInt64(&m.cacheHits, 1)
	model.IncrRef()
	m.touchLRU(modelID)

	return model, nil
}

// ReleaseModel decrements the reference count for a model.
// 모델의 참조 카운트를 감소시킵니다.
func (m *ModelManager) ReleaseModel(model *LoadedModel) {
	if model != nil {
		model.DecrRef()
	}
}

// GetDefaultModel retrieves the default model.
// 기본 모델을 가져옵니다.
func (m *ModelManager) GetDefaultModel() (*LoadedModel, error) {
	m.mu.RLock()
	defaultID := m.defaultID
	m.mu.RUnlock()

	if defaultID != "" {
		return m.GetModel(defaultID)
	}

	// Return any loaded model if no default is set
	m.mu.RLock()
	for _, model := range m.models {
		m.mu.RUnlock()
		model.IncrRef()
		return model, nil
	}
	m.mu.RUnlock()

	return nil, ErrModelNotLoaded
}

// LoadModel loads a model with the given ID and path.
// 주어진 ID와 경로로 모델을 로드합니다.
func (m *ModelManager) LoadModel(modelID, modelPath, draftPath string, threads int) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if modelID == "" {
		if modelPath == "" {
			return ErrModelIDRequired
		}
		modelID = filepath.Clean(modelPath)
	}

	// Check if already loaded
	if existing, exists := m.models[modelID]; exists {
		slog.Info("model already loaded",
			slog.String("model_id", modelID),
			slog.String("path", existing.Path),
		)
		return nil
	}

	// Evict if at capacity
	if len(m.models) >= m.config.MaxModels {
		if err := m.evictLRULocked(); err != nil {
			if errors.Is(err, ErrNoEvictableModels) {
				return ErrNoEvictableModels
			}
			return fmt.Errorf("failed to evict model: %w", err)
		}
	}

	// Use default threads if not specified
	if threads <= 0 {
		threads = m.config.DefaultThreads
	}

	// Load new model
	slog.Info("loading model",
		slog.String("model_id", modelID),
		slog.String("path", modelPath),
		slog.Int("threads", threads),
	)

	eng, err := engine.NewDenseEngine(modelPath, draftPath, threads)
	if err != nil {
		return fmt.Errorf("failed to load engine: %w", err)
	}

	now := time.Now()
	model := &LoadedModel{
		ID:        modelID,
		Path:      modelPath,
		DraftPath: draftPath,
		Engine:    eng,
		LoadedAt:  now,
		LastUsed:  now,
		Threads:   threads,
	}

	m.models[modelID] = model
	elem := m.lruOrder.PushFront(modelID)
	m.lruElements[modelID] = elem

	// Set as default if first model
	if m.defaultID == "" {
		m.defaultID = modelID
	}

	atomic.AddInt64(&m.totalLoads, 1)
	slog.Info("model loaded successfully",
		slog.String("model_id", modelID),
		slog.Int("total_models", len(m.models)),
	)

	return nil
}

// UnloadModel unloads a specific model.
// 특정 모델을 언로드합니다.
func (m *ModelManager) UnloadModel(modelID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	model, ok := m.models[modelID]
	if !ok {
		return ErrModelNotFound
	}

	// Check if model is in use
	if model.GetRefCount() > 0 {
		return ErrModelInUse
	}

	// Close engine
	model.Engine.Close()
	delete(m.models, modelID)

	// Remove from LRU
	if elem, ok := m.lruElements[modelID]; ok {
		m.lruOrder.Remove(elem)
		delete(m.lruElements, modelID)
	}

	// Update default if necessary
	if m.defaultID == modelID {
		m.defaultID = ""
		for id := range m.models {
			m.defaultID = id
			break
		}
	}

	slog.Info("model unloaded",
		slog.String("model_id", modelID),
		slog.Int("remaining_models", len(m.models)),
	)

	return nil
}

// UnloadAll unloads all models.
// 모든 모델을 언로드합니다.
func (m *ModelManager) UnloadAll() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	for id, model := range m.models {
		if model.GetRefCount() > 0 {
			slog.Warn("model in use during shutdown, forcing unload",
				slog.String("model_id", id),
				slog.Int("ref_count", int(model.GetRefCount())),
			)
		}
		model.Engine.Close()
	}

	m.models = make(map[string]*LoadedModel)
	m.lruOrder = list.New()
	m.lruElements = make(map[string]*list.Element)
	m.defaultID = ""

	slog.Info("all models unloaded")
	return nil
}

// ============================================================================
// Model Information
// ============================================================================

// ListModels returns information about all loaded models.
// 모든 로드된 모델의 정보를 반환합니다.
func (m *ModelManager) ListModels() []LoadedModel {
	m.mu.RLock()
	defer m.mu.RUnlock()

	models := make([]LoadedModel, 0, len(m.models))
	for _, model := range m.models {
		models = append(models, LoadedModel{
			ID:        model.ID,
			Path:      model.Path,
			DraftPath: model.DraftPath,
			LoadedAt:  model.LoadedAt,
			LastUsed:  model.LastUsed,
			RefCount:  model.GetRefCount(),
			Threads:   model.Threads,
		})
	}
	return models
}

// GetModelCount returns the number of loaded models.
// 로드된 모델 수를 반환합니다.
func (m *ModelManager) GetModelCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.models)
}

// SetDefaultModel sets the default model ID.
// 기본 모델 ID를 설정합니다.
func (m *ModelManager) SetDefaultModel(modelID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, ok := m.models[modelID]; !ok {
		return ErrModelNotFound
	}
	m.defaultID = modelID
	slog.Info("default model set", slog.String("model_id", modelID))
	return nil
}

// GetDefaultModelID returns the default model ID.
// 기본 모델 ID를 반환합니다.
func (m *ModelManager) GetDefaultModelID() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.defaultID
}

// ============================================================================
// Statistics
// ============================================================================

// ModelManagerStats holds statistics about the model manager.
type ModelManagerStats struct {
	TotalModels int
	MaxModels   int
	TotalLoads  int64
	TotalEvicts int64
	CacheHits   int64
	CacheMisses int64
	HitRate     float64
}

// GetStats returns statistics about the model manager.
// ModelManager의 통계를 반환합니다.
func (m *ModelManager) GetStats() ModelManagerStats {
	m.mu.RLock()
	defer m.mu.RUnlock()

	hits := atomic.LoadInt64(&m.cacheHits)
	misses := atomic.LoadInt64(&m.cacheMisses)
	total := hits + misses

	var hitRate float64
	if total > 0 {
		hitRate = float64(hits) / float64(total)
	}

	return ModelManagerStats{
		TotalModels: len(m.models),
		MaxModels:   m.config.MaxModels,
		TotalLoads:  atomic.LoadInt64(&m.totalLoads),
		TotalEvicts: atomic.LoadInt64(&m.totalEvicts),
		CacheHits:   hits,
		CacheMisses: misses,
		HitRate:     hitRate,
	}
}

// ============================================================================
// Internal Methods
// ============================================================================

// touchLRU moves a model to the front of the LRU list.
func (m *ModelManager) touchLRU(modelID string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if elem, ok := m.lruElements[modelID]; ok {
		m.lruOrder.MoveToFront(elem)
	}
}

// evictLRULocked evicts the least recently used model (must hold write lock).
func (m *ModelManager) evictLRULocked() error {
	// Find LRU model that's not in use
	for e := m.lruOrder.Back(); e != nil; e = e.Prev() {
		modelID := e.Value.(string)
		model := m.models[modelID]

		if model.GetRefCount() == 0 {
			slog.Info("evicting LRU model",
				slog.String("model_id", modelID),
				slog.Duration("age", time.Since(model.LoadedAt)),
			)

			model.Engine.Close()
			delete(m.models, modelID)
			m.lruOrder.Remove(e)
			delete(m.lruElements, modelID)

			if m.defaultID == modelID {
				m.defaultID = ""
				for id := range m.models {
					m.defaultID = id
					break
				}
			}

			atomic.AddInt64(&m.totalEvicts, 1)
			return nil
		}
	}

	return ErrNoEvictableModels
}
