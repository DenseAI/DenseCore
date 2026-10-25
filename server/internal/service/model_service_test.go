package service

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/DenseAI/DenseCore/server/internal/domain"
)

type modelServiceTestEngine struct {
	chatServiceRenderTestEngine
	closeCount         int32
	beginShutdownCount int32
	closeCh            chan struct{}
	beginShutdownCh    chan struct{}
	embeddingStartedCh chan struct{}
	embeddingErr       error
}

func (e *modelServiceTestEngine) BeginShutdown() {
	if atomic.AddInt32(&e.beginShutdownCount, 1) == 1 && e.beginShutdownCh != nil {
		close(e.beginShutdownCh)
	}
}

func (e *modelServiceTestEngine) GetEmbeddingsWithOptions(string, string, *bool) ([]float32, error) {
	if e.embeddingStartedCh != nil {
		close(e.embeddingStartedCh)
	}
	if e.beginShutdownCh != nil {
		<-e.beginShutdownCh
	}
	return nil, e.embeddingErr
}

func (e *modelServiceTestEngine) Close() {
	if atomic.AddInt32(&e.closeCount, 1) == 1 && e.closeCh != nil {
		close(e.closeCh)
	}
}

func waitForCondition(t *testing.T, timeout time.Duration, cond func() bool, message string) {
	t.Helper()

	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	if !cond() {
		t.Fatal(message)
	}
}

func TestResolveModelLoadStrategy(t *testing.T) {
	tests := []struct {
		name  string
		value string
		want  modelLoadStrategy
	}{
		{name: "default", value: "", want: modelLoadStrategyAuto},
		{name: "auto", value: "auto", want: modelLoadStrategyAuto},
		{name: "blue_green", value: "blue_green", want: modelLoadStrategyBlueGreen},
		{name: "blue-green", value: "blue-green", want: modelLoadStrategyBlueGreen},
		{name: "force", value: "force", want: modelLoadStrategyForce},
		{name: "invalid", value: "nonsense", want: modelLoadStrategyAuto},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := resolveModelLoadStrategy(tt.value); got != tt.want {
				t.Fatalf("resolveModelLoadStrategy(%q) = %v, want %v", tt.value, got, tt.want)
			}
		})
	}
}

func TestEstimateModelFileBytes(t *testing.T) {
	dir := t.TempDir()
	mainPath := filepath.Join(dir, "main.gguf")
	draftPath := filepath.Join(dir, "draft.gguf")
	if err := os.WriteFile(mainPath, []byte("12345"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(draftPath, []byte("123"), 0o600); err != nil {
		t.Fatal(err)
	}

	got, ok := estimateModelFileBytes(mainPath, draftPath)
	if !ok {
		t.Fatalf("estimateModelFileBytes returned ok=false")
	}
	if got != 8 {
		t.Fatalf("estimateModelFileBytes = %d, want 8", got)
	}

	if _, ok := estimateModelFileBytes(filepath.Join(dir, "missing.gguf")); ok {
		t.Fatalf("estimateModelFileBytes missing path returned ok=true")
	}
}

func TestLinuxMemAvailableBytes(t *testing.T) {
	path := filepath.Join(t.TempDir(), "meminfo")
	if err := os.WriteFile(path, []byte("MemTotal:       1000 kB\nMemAvailable:    42 kB\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	got, ok := linuxMemAvailableBytes(path)
	if !ok {
		t.Fatalf("linuxMemAvailableBytes returned ok=false")
	}
	if got != 42*1024 {
		t.Fatalf("linuxMemAvailableBytes = %d, want %d", got, 42*1024)
	}
}

func TestUnloadModelContextWaitsForActiveLease(t *testing.T) {
	engine := &modelServiceTestEngine{closeCh: make(chan struct{})}
	svc := NewModelService()
	svc.currentSlot = &engineSlot{engine: engine}
	svc.currentModelPath = "/tmp/test.gguf"

	lease := svc.AcquireEngineLease()
	if lease == nil {
		t.Fatal("expected active engine lease")
	}

	done := make(chan error, 1)
	go func() {
		done <- svc.UnloadModelContext(context.Background())
	}()

	waitForCondition(t, 2*time.Second, func() bool {
		return svc.GetEngine() == nil
	}, "engine slot was not detached during unload")

	lease.Close()

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("UnloadModelContext returned error: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("UnloadModelContext did not finish after lease release")
	}

	select {
	case <-engine.closeCh:
	case <-time.After(2 * time.Second):
		t.Fatal("engine was not closed after lease release")
	}
}

func TestBeginEngineShutdownReleasesEmbeddingLeaseBeforeClose(t *testing.T) {
	terminalErr := errors.New("engine stopped")
	engine := &modelServiceTestEngine{
		closeCh:            make(chan struct{}),
		beginShutdownCh:    make(chan struct{}),
		embeddingStartedCh: make(chan struct{}),
		embeddingErr:       terminalErr,
	}
	svc := NewModelService()
	svc.currentSlot = &engineSlot{engine: engine}
	svc.currentModelPath = "/tmp/test.gguf"
	chatSvc := NewChatService(svc, nil)

	requestDone := make(chan error, 1)
	go func() {
		_, err := chatSvc.GetEmbeddings(domain.EmbeddingRequest{Input: "shutdown"})
		requestDone <- err
	}()
	select {
	case <-engine.embeddingStartedCh:
	case <-time.After(2 * time.Second):
		t.Fatal("embedding request did not reach the engine")
	}

	svc.BeginEngineShutdown()
	select {
	case err := <-requestDone:
		if !errors.Is(err, terminalErr) {
			t.Fatalf("embedding terminal error = %v, want %v", err, terminalErr)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("embedding request did not release its engine lease during pre-shutdown")
	}
	if lease := svc.AcquireEngineLease(); lease != nil {
		lease.Close()
		t.Fatal("new engine lease admitted after pre-shutdown")
	}

	if err := svc.UnloadModelContext(context.Background()); err != nil {
		t.Fatalf("UnloadModelContext returned error: %v", err)
	}
	select {
	case <-engine.closeCh:
	case <-time.After(2 * time.Second):
		t.Fatal("engine did not close after embedding lease released")
	}
	if got := atomic.LoadInt32(&engine.beginShutdownCount); got != 1 {
		t.Fatalf("BeginShutdown count = %d, want 1", got)
	}
}

func TestBeginEngineShutdownReleasesRerankBatchLease(t *testing.T) {
	terminalErr := errors.New("engine stopped")
	engine := &modelServiceTestEngine{
		closeCh:            make(chan struct{}),
		beginShutdownCh:    make(chan struct{}),
		embeddingStartedCh: make(chan struct{}),
		embeddingErr:       terminalErr,
	}
	svc := NewModelService()
	svc.currentSlot = &engineSlot{engine: engine}
	chatSvc := NewChatService(svc, nil)

	requestDone := make(chan error, 1)
	go func() {
		_, err := chatSvc.GetBatchEmbeddings([]string{"query", "document"})
		requestDone <- err
	}()
	select {
	case <-engine.embeddingStartedCh:
	case <-time.After(2 * time.Second):
		t.Fatal("rerank embedding batch did not reach the engine")
	}

	svc.BeginEngineShutdown()
	select {
	case err := <-requestDone:
		if !errors.Is(err, terminalErr) {
			t.Fatalf("rerank terminal error = %v, want wrapped %v", err, terminalErr)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("rerank request did not release its engine lease during pre-shutdown")
	}
	if err := svc.UnloadModelContext(context.Background()); err != nil {
		t.Fatalf("UnloadModelContext returned error: %v", err)
	}
}

func TestUnloadModelContextDefersCloseAfterTimeout(t *testing.T) {
	engine := &modelServiceTestEngine{closeCh: make(chan struct{})}
	svc := NewModelService()
	svc.currentSlot = &engineSlot{engine: engine}
	svc.currentModelPath = "/tmp/test.gguf"

	lease := svc.AcquireEngineLease()
	if lease == nil {
		t.Fatal("expected active engine lease")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	done := make(chan error, 1)
	go func() {
		done <- svc.UnloadModelContext(ctx)
	}()

	waitForCondition(t, 2*time.Second, func() bool {
		return svc.GetEngine() == nil
	}, "engine slot was not detached during timed unload")

	err := <-done
	if err == nil {
		t.Fatal("expected unload timeout while lease is active")
	}

	select {
	case <-engine.closeCh:
		t.Fatal("engine closed before deferred lease release")
	default:
	}

	lease.Close()

	select {
	case <-engine.closeCh:
	case <-time.After(2 * time.Second):
		t.Fatal("engine was not closed after deferred lease release")
	}
}

type contextShutdownTestEngine struct {
	modelServiceTestEngine
	selected chan context.Context
	release  chan struct{}
}

func (e *contextShutdownTestEngine) BeginShutdownContext(ctx context.Context) error {
	e.selected <- ctx
	select {
	case <-e.release:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
func TestPreShutdownPropagatesOriginalContextAndClosesLeasesFirst(t *testing.T) {
	engine := &contextShutdownTestEngine{selected: make(chan context.Context, 1), release: make(chan struct{})}
	svc := NewModelService()
	svc.currentSlot = &engineSlot{engine: engine}
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(30*time.Second))
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- svc.BeginEngineShutdownContext(ctx) }()
	selected := <-engine.selected
	if selected != ctx {
		t.Fatal("shutdown replaced the runner context")
	}
	if lease := svc.AcquireEngineLease(); lease != nil {
		lease.Close()
		t.Fatal("lease admitted while native shutdown draining")
	}
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("shutdown=%v", err)
	}
}

type blockingCloseTestEngine struct {
	modelServiceTestEngine
	entered, release chan struct{}
}

func (e *blockingCloseTestEngine) Close() {
	close(e.entered)
	<-e.release
	e.modelServiceTestEngine.Close()
}
func TestModelCloseContextReturnsWithoutFreeingAnInUseCleanup(t *testing.T) {
	engine := &blockingCloseTestEngine{modelServiceTestEngine: modelServiceTestEngine{closeCh: make(chan struct{})}, entered: make(chan struct{}), release: make(chan struct{})}
	svc := NewModelService()
	slot := &engineSlot{engine: engine}
	svc.currentSlot = slot
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- svc.UnloadModelContext(ctx) }()
	<-engine.entered
	cancel()
	if err := <-done; !errors.Is(err, context.Canceled) {
		t.Fatalf("close=%v", err)
	}
	if got := atomic.LoadInt32(&engine.closeCount); got != 0 {
		t.Fatalf("early free=%d", got)
	}
	close(engine.release)
	if err := svc.closeEngineSlot(context.Background(), slot); err != nil {
		t.Fatal(err)
	}
	if got := atomic.LoadInt32(&engine.closeCount); got != 1 {
		t.Fatalf("free count=%d", got)
	}
}

type blockingLegacyShutdownEngine struct {
	modelServiceTestEngine
	entered, release          chan struct{}
	beginReturned, earlyClose atomic.Bool
}

func (e *blockingLegacyShutdownEngine) Close() {
	if !e.beginReturned.Load() {
		e.earlyClose.Store(true)
	}
	e.modelServiceTestEngine.Close()
}

func (e *blockingLegacyShutdownEngine) BeginShutdown() {
	if atomic.AddInt32(&e.beginShutdownCount, 1) == 1 {
		close(e.entered)
	}
	<-e.release
	e.beginReturned.Store(true)
}
func TestLegacyBeginRetainsModelUntilCleanupReturnsAfterContextExpiry(t *testing.T) {
	engine := &blockingLegacyShutdownEngine{modelServiceTestEngine: modelServiceTestEngine{closeCh: make(chan struct{})}, entered: make(chan struct{}), release: make(chan struct{})}
	svc := NewModelService()
	slot := &engineSlot{engine: engine}
	svc.currentSlot = slot
	ctx, cancel := context.WithCancel(context.Background())
	started := make(chan error, 1)
	go func() { started <- svc.BeginEngineShutdownContext(ctx) }()
	<-engine.entered
	cancel()
	if err := <-started; !errors.Is(err, context.Canceled) {
		t.Fatalf("begin=%v", err)
	}
	if err := svc.BeginEngineShutdownContext(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("repeat=%v", err)
	}
	if err := svc.UnloadModelContext(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("unload=%v", err)
	}
	if got := atomic.LoadInt32(&engine.closeCount); got != 0 {
		t.Fatalf("close raced legacy begin: %d", got)
	}
	select {
	case <-engine.closeCh:
		t.Fatal("legacy close completed before BeginShutdown returned")
	case <-time.After(20 * time.Millisecond):
	}
	close(engine.release)
	if err := svc.closeEngineSlot(context.Background(), slot); err != nil {
		t.Fatal(err)
	}
	if engine.earlyClose.Load() {
		t.Fatal("Close raced in-flight legacy BeginShutdown")
	}
	if got := atomic.LoadInt32(&engine.closeCount); got != 1 {
		t.Fatalf("close=%d", got)
	}
	if got := atomic.LoadInt32(&engine.beginShutdownCount); got != 1 {
		t.Fatalf("begin=%d", got)
	}
}
