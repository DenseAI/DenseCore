package service

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/DenseAI/DenseCore/server/internal/domain"
	"github.com/DenseAI/DenseCore/server/internal/queue"
)

type boundGenerationTestEngine struct {
	chatServiceRenderTestEngine
	onRender func()
	done     chan struct{}
	submits  atomic.Int32
	closes   atomic.Int32
}

func (e *boundGenerationTestEngine) RenderChatPrompt(messages []domain.Message, thinking, preserve *bool) (*domain.RenderedChatPrompt, error) {
	if e.onRender != nil {
		e.onRender()
	}
	return e.chatServiceRenderTestEngine.RenderChatPrompt(messages, thinking, preserve)
}
func (e *boundGenerationTestEngine) SubmitGeneration(context.Context, domain.GenerationRequest) (*domain.Generation, error) {
	e.submits.Add(1)
	events := make(chan domain.StreamEvent, 1)
	events <- domain.NewTerminalEvent(nil)
	close(events)
	return &domain.Generation{Events: events, Done: e.done, Cancel: func() {}}, nil
}
func (e *boundGenerationTestEngine) Close() { e.closes.Add(1) }

func TestGenerationKeepsPreparedEngineLeaseThroughCallbackCleanup(t *testing.T) {
	old := &boundGenerationTestEngine{done: make(chan struct{})}
	old.renderedPrompt = "prepared prompt"
	replacement := &boundGenerationTestEngine{done: make(chan struct{})}
	var finish sync.Once
	finishOld := func() { finish.Do(func() { close(old.done) }) }
	defer finishOld()
	models := NewModelService()
	slot := &engineSlot{engine: old}
	models.currentSlot = slot
	old.onRender = func() {
		models.mu.Lock()
		models.currentSlot = &engineSlot{engine: replacement}
		models.mu.Unlock()
	}
	q := queue.NewRequestQueue(2)
	worker := NewQueueProcessor(q, models)
	worker.Start(1)
	defer func() { finishOld(); worker.Stop() }()
	service := NewChatService(models, q)
	if _, _, _, err := service.GenerateSync(context.Background(), domain.ChatCompletionRequest{Messages: []domain.Message{{Role: "user", Content: "hello"}}, MaxTokens: 8}); err != nil {
		t.Fatal(err)
	}
	if old.submits.Load() != 1 || replacement.submits.Load() != 0 {
		t.Fatal("queued generation used a different engine from prompt preparation")
	}
	models.mu.RLock()
	refs := slot.refCount
	models.mu.RUnlock()
	if refs != 1 {
		t.Fatalf("active callback lease count=%d, want 1", refs)
	}
	closed := make(chan struct{})
	go func() { _ = models.closeEngineSlot(context.Background(), slot); close(closed) }()
	select {
	case <-closed:
		t.Fatal("engine closed before callback cleanup")
	default:
	}
	finishOld()
	select {
	case <-closed:
	case <-time.After(time.Second):
		t.Fatal("engine lease not released after callback cleanup")
	}
	if old.closes.Load() != 1 {
		t.Fatal("engine must close exactly once")
	}
}

func TestGenerationQueueRejectionReleasesPreparedLease(t *testing.T) {
	eng := &boundGenerationTestEngine{}
	eng.renderedPrompt = "prepared prompt"
	models := NewModelService()
	slot := &engineSlot{engine: eng}
	models.currentSlot = slot
	q := queue.NewRequestQueue(1)
	q.Close()
	svc := NewChatService(models, q)
	if _, _, _, err := svc.GenerateSync(context.Background(), domain.ChatCompletionRequest{Messages: []domain.Message{{Role: "user", Content: "hello"}}, MaxTokens: 8}); err != domain.ErrServiceBusy {
		t.Fatalf("queue rejection: %v", err)
	}
	models.mu.RLock()
	refs := slot.refCount
	models.mu.RUnlock()
	if refs != 0 || eng.submits.Load() != 0 {
		t.Fatalf("rejected request retained lease or submitted: refs=%d submits=%d", refs, eng.submits.Load())
	}
}
