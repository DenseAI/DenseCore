package service

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	nativeengine "github.com/DenseAI/DenseCore/server/internal/engine"

	"github.com/DenseAI/DenseCore/server/internal/domain"
	"github.com/DenseAI/DenseCore/server/internal/queue"
)

type queueSubmitterTestEngine struct {
	admissionEntered chan struct{}
	admission        nativeengine.GenerationAdmission
	submitMu         sync.Mutex
	chatServiceRenderTestEngine
	submitCount             int32
	activeCount             int32
	firstSubmitted          chan struct{}
	submitStarted           chan struct{}
	allowCompletion         chan struct{}
	renderedPromptSubmitted string
	renderedChatSubmitted   string
	prefixCacheReuseEnabled bool
	snapshotRestoreEnabled  bool
	runtimeStateError       error
	completeOnCancel        bool
	submitDelay             time.Duration
	submitInFlight          int32
	maxSubmitInFlight       int32
}

func (e *queueSubmitterTestEngine) GenerateStreamWithSamplingAwaitable(ctx context.Context, prompt string, maxTokens int, loraAdapter string, jsonMode bool, temperature float64, topP float64, topK int, repetitionPenalty float64, stop []string, allowedTokenIDs []int, allowedTokensStrict bool, disallowedTokenIDs []int, outputChan chan domain.StreamEvent) (<-chan struct{}, error) {
	return e.awaitSubmit(ctx)
}

func (e *queueSubmitterTestEngine) GenerateStreamTokensWithSamplingAwaitable(ctx context.Context, inputIDs []int, maxTokens int, loraAdapter string, jsonMode bool, temperature float64, topP float64, topK int, repetitionPenalty float64, stop []string, allowedTokenIDs []int, allowedTokensStrict bool, disallowedTokenIDs []int, outputChan chan domain.StreamEvent) (<-chan struct{}, error) {
	return e.awaitSubmit(ctx)
}

func (e *queueSubmitterTestEngine) GenerateStreamRenderedTokensWithSamplingAwaitable(ctx context.Context, renderedPrompt string, inputIDs []int, maxTokens int, loraAdapter string, jsonMode bool, temperature float64, topP float64, topK int, repetitionPenalty float64, stop []string, allowedTokenIDs []int, allowedTokensStrict bool, disallowedTokenIDs []int, outputChan chan domain.StreamEvent) (<-chan struct{}, error) {
	e.renderedPromptSubmitted = renderedPrompt
	return e.awaitSubmit(ctx)
}

func (e *queueSubmitterTestEngine) GenerateStreamRenderedChatWithSamplingAwaitable(ctx context.Context, renderedPrompt string, maxTokens int, loraAdapter string, jsonMode bool, temperature float64, topP float64, topK int, repetitionPenalty float64, stop []string, allowedTokenIDs []int, allowedTokensStrict bool, disallowedTokenIDs []int, outputChan chan domain.StreamEvent) (<-chan struct{}, error) {
	e.renderedChatSubmitted = renderedPrompt
	return e.awaitSubmit(ctx)
}

func (e *queueSubmitterTestEngine) awaitSubmit(ctx context.Context) (<-chan struct{}, error) {
	inFlight := atomic.AddInt32(&e.submitInFlight, 1)
	defer atomic.AddInt32(&e.submitInFlight, -1)
	for {
		maximum := atomic.LoadInt32(&e.maxSubmitInFlight)
		if inFlight <= maximum || atomic.CompareAndSwapInt32(&e.maxSubmitInFlight, maximum, inFlight) {
			break
		}
	}
	if e.submitDelay > 0 {
		time.Sleep(e.submitDelay)
	}
	count := atomic.AddInt32(&e.submitCount, 1)
	atomic.AddInt32(&e.activeCount, 1)
	if count == 1 && e.firstSubmitted != nil {
		close(e.firstSubmitted)
	}
	if e.submitStarted != nil {
		e.submitStarted <- struct{}{}
	}
	done := make(chan struct{})
	go func() {
		defer atomic.AddInt32(&e.activeCount, -1)
		if e.completeOnCancel {
			select {
			case <-e.allowCompletion:
			case <-ctx.Done():
			}
		} else {
			<-e.allowCompletion
		}
		close(done)
	}()
	return done, nil
}

func (e *queueSubmitterTestEngine) GetRuntimeOptimizationState() (domain.RuntimeOptimizationState, error) {
	return domain.RuntimeOptimizationState{
		PrefixCacheReuseEnabled:         e.prefixCacheReuseEnabled,
		HybridSSMSnapshotRestoreEnabled: e.snapshotRestoreEnabled,
	}, e.runtimeStateError
}

func waitForSubmitCount(t *testing.T, engine *queueSubmitterTestEngine, want int32) {
	t.Helper()

	deadline := time.After(2 * time.Second)
	for atomic.LoadInt32(&engine.submitCount) < want {
		select {
		case <-deadline:
			t.Fatalf("submit_count=%d, want at least %d", atomic.LoadInt32(&engine.submitCount), want)
		default:
			time.Sleep(10 * time.Millisecond)
		}
	}
}

func waitForActiveCount(t *testing.T, engine *queueSubmitterTestEngine, want int32) {
	t.Helper()

	deadline := time.After(2 * time.Second)
	for atomic.LoadInt32(&engine.activeCount) != want {
		select {
		case <-deadline:
			t.Fatalf("active_count=%d, want %d", atomic.LoadInt32(&engine.activeCount), want)
		default:
			time.Sleep(10 * time.Millisecond)
		}
	}
}

func TestQueueProcessorSubmitsRenderedChatBeforeRenderedTokenIDs(t *testing.T) {
	engine := &queueSubmitterTestEngine{
		firstSubmitted:  make(chan struct{}),
		allowCompletion: make(chan struct{}),
	}
	modelService := &chatServiceRenderTestModelService{engine: engine, model: "/tmp/test.gguf"}
	requestQueue := queue.NewRequestQueue(1)
	processor := NewQueueProcessor(requestQueue, modelService)
	processor.Start(1)
	defer processor.Stop()
	defer close(engine.allowCompletion)

	req := &queue.QueuedRequest{
		Engine: engine,
		GenerationRequest: domain.GenerationRequest{Input: domain.GenerationInput{Kind: domain.GenerationInputRenderedText, Text: "<|turn>user\nhello<turn|>\n<|turn>model\n", TokenIDs: []int{10, 20, 30}},

			MaxTokens: 1,
		},

		ID:         "req-rendered-chat",
		Context:    context.Background(),
		ResultChan: make(chan queue.GenerationResult, 1),
	}
	if !requestQueue.Enqueue(req) {
		t.Fatal("failed to enqueue test request")
	}
	select {
	case <-engine.firstSubmitted:
	case <-time.After(2 * time.Second):
		t.Fatal("request was not submitted")
	}
	if engine.renderedChatSubmitted != req.Input.Text {
		t.Fatalf("rendered chat submitted=%q want %q", engine.renderedChatSubmitted, req.Input.Text)
	}
	if engine.renderedPromptSubmitted != "" {
		t.Fatalf("expected rendered-token fallback to stay unused, got %q", engine.renderedPromptSubmitted)
	}
}

func TestQueueProcessorSubmitsRenderedPromptWithTokenIDs(t *testing.T) {
	engine := &queueSubmitterTestEngine{
		firstSubmitted:  make(chan struct{}),
		allowCompletion: make(chan struct{}),
	}
	modelService := &chatServiceRenderTestModelService{engine: engine, model: "/tmp/test.gguf"}
	requestQueue := queue.NewRequestQueue(1)
	processor := NewQueueProcessor(requestQueue, modelService)
	processor.Start(1)
	defer processor.Stop()
	defer close(engine.allowCompletion)

	req := &queue.QueuedRequest{
		Engine: engine,
		GenerationRequest: domain.GenerationRequest{Input: domain.GenerationInput{Kind: domain.GenerationInputRenderedTokenIDs, Text: "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n", TokenIDs: []int{10, 20, 30}},

			MaxTokens: 1,
		},

		ID:         "req-rendered",
		Context:    context.Background(),
		ResultChan: make(chan queue.GenerationResult, 1),
	}
	if !requestQueue.Enqueue(req) {
		t.Fatal("failed to enqueue test request")
	}
	select {
	case <-engine.firstSubmitted:
	case <-time.After(2 * time.Second):
		t.Fatal("request was not submitted")
	}
	if engine.renderedPromptSubmitted != req.Input.Text {
		t.Fatalf("rendered prompt submitted=%q want %q", engine.renderedPromptSubmitted, req.Input.Text)
	}
}

func TestQueueProcessorWorkerReturnsAfterSubmitBeforeCompletion(t *testing.T) {
	engine := &queueSubmitterTestEngine{
		firstSubmitted:  make(chan struct{}),
		allowCompletion: make(chan struct{}),
	}
	modelService := &chatServiceRenderTestModelService{engine: engine, model: "/tmp/test.gguf"}
	requestQueue := queue.NewRequestQueue(4)
	processor := NewQueueProcessor(requestQueue, modelService)
	processor.Start(1)
	defer processor.Stop()
	defer close(engine.allowCompletion)

	req1 := &queue.QueuedRequest{
		Engine: engine,
		GenerationRequest: domain.GenerationRequest{Input: domain.GenerationInput{Kind: domain.GenerationInputTokenIDs, TokenIDs: []int{1}},

			MaxTokens: 1,
		},

		ID:         "req-1",
		Context:    context.Background(),
		ResultChan: make(chan queue.GenerationResult, 1),
	}
	req2 := &queue.QueuedRequest{
		Engine: engine,
		GenerationRequest: domain.GenerationRequest{Input: domain.GenerationInput{Kind: domain.GenerationInputTokenIDs, TokenIDs: []int{2}},

			MaxTokens: 1,
		},

		ID:         "req-2",
		Context:    context.Background(),
		ResultChan: make(chan queue.GenerationResult, 1),
	}

	if !requestQueue.Enqueue(req1) || !requestQueue.Enqueue(req2) {
		t.Fatal("failed to enqueue test requests")
	}
	select {
	case <-engine.firstSubmitted:
	case <-time.After(2 * time.Second):
		t.Fatal("first request was not submitted")
	}

	waitForSubmitCount(t, engine, 2)
}

func TestQueueProcessorRequestsWithoutPrefixSnapshotsRemainConcurrentlyAdmitted(t *testing.T) {
	for _, tc := range []struct {
		name            string
		prefixReuse     bool
		snapshotRestore bool
		stateError      error
	}{
		{name: "snapshot disabled", prefixReuse: true},
		{name: "prefix reuse disabled", snapshotRestore: true},
		{name: "both disabled"},
		{name: "runtime state error preserves async admission", prefixReuse: true, snapshotRestore: true, stateError: errors.New("runtime state unavailable")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			engine := &queueSubmitterTestEngine{
				firstSubmitted:          make(chan struct{}),
				allowCompletion:         make(chan struct{}),
				submitDelay:             20 * time.Millisecond,
				prefixCacheReuseEnabled: tc.prefixReuse,
				snapshotRestoreEnabled:  tc.snapshotRestore,
				runtimeStateError:       tc.stateError,
			}
			modelService := &chatServiceRenderTestModelService{engine: engine, model: "/tmp/test.gguf"}
			requestQueue := queue.NewRequestQueue(4)
			processor := NewQueueProcessor(requestQueue, modelService)
			processor.Start(4)
			defer processor.Stop()
			defer close(engine.allowCompletion)

			for i := 0; i < 4; i++ {
				req := &queue.QueuedRequest{
					Engine: engine,
					GenerationRequest: domain.GenerationRequest{Input: domain.GenerationInput{Kind: domain.GenerationInputTokenIDs, TokenIDs: []int{i + 1}},

						MaxTokens: 1,
					},

					ID:         fmt.Sprintf("req-%d", i),
					Context:    context.Background(),
					ResultChan: make(chan queue.GenerationResult, 1),
				}
				if !requestQueue.Enqueue(req) {
					t.Fatalf("failed to enqueue request %d", i)
				}
			}

			select {
			case <-engine.firstSubmitted:
			case <-time.After(2 * time.Second):
				t.Fatal("first request was not submitted")
			}
			waitForSubmitCount(t, engine, 4)
			if got := atomic.LoadInt32(&engine.activeCount); got != 4 {
				t.Fatalf("active_count=%d, want 4 before any completion", got)
			}
			if got := atomic.LoadInt32(&engine.maxSubmitInFlight); got != 1 {
				t.Fatalf("max_submit_in_flight=%d, want serialized C++ submission", got)
			}
		})
	}
}

func TestQueueProcessorSnapshotEnabledRequestsRemainCompletionScoped(t *testing.T) {
	engine := &queueSubmitterTestEngine{
		firstSubmitted:          make(chan struct{}),
		allowCompletion:         make(chan struct{}),
		prefixCacheReuseEnabled: true,
		snapshotRestoreEnabled:  true,
	}
	modelService := &chatServiceRenderTestModelService{engine: engine, model: "/tmp/test.gguf"}
	requestQueue := queue.NewRequestQueue(2)
	processor := NewQueueProcessor(requestQueue, modelService)
	processor.Start(2)
	defer processor.Stop()

	for i := 0; i < 2; i++ {
		req := &queue.QueuedRequest{
			Engine: engine,
			GenerationRequest: domain.GenerationRequest{Input: domain.GenerationInput{Kind: domain.GenerationInputTokenIDs, TokenIDs: []int{i + 1}},

				MaxTokens: 1,
			},

			ID:         fmt.Sprintf("req-%d", i),
			Context:    context.Background(),
			ResultChan: make(chan queue.GenerationResult, 1),
		}
		if !requestQueue.Enqueue(req) {
			t.Fatalf("failed to enqueue request %d", i)
		}
	}

	select {
	case <-engine.firstSubmitted:
	case <-time.After(2 * time.Second):
		t.Fatal("first request was not submitted")
	}
	time.Sleep(50 * time.Millisecond)
	if got := atomic.LoadInt32(&engine.submitCount); got != 1 {
		t.Fatalf("submit_count=%d, want 1 before first completion", got)
	}
	close(engine.allowCompletion)
	waitForSubmitCount(t, engine, 2)
}

func TestQueueProcessorStopAndWaitTracksHybridSSMRequestCompletion(t *testing.T) {
	engine := &queueSubmitterTestEngine{
		firstSubmitted:          make(chan struct{}),
		allowCompletion:         make(chan struct{}),
		prefixCacheReuseEnabled: true,
		snapshotRestoreEnabled:  true,
	}
	modelService := &chatServiceRenderTestModelService{engine: engine, model: "/tmp/test.gguf"}
	requestQueue := queue.NewRequestQueue(4)
	processor := NewQueueProcessor(requestQueue, modelService)
	processor.Start(1)
	var closed int32
	closeAllowCompletion := func() {
		if atomic.CompareAndSwapInt32(&closed, 0, 1) {
			close(engine.allowCompletion)
		}
	}
	defer closeAllowCompletion()

	req1 := &queue.QueuedRequest{
		Engine: engine,
		GenerationRequest: domain.GenerationRequest{Input: domain.GenerationInput{Kind: domain.GenerationInputTokenIDs, TokenIDs: []int{1}},

			MaxTokens: 1,
		},

		ID:         "req-1",
		Context:    context.Background(),
		ResultChan: make(chan queue.GenerationResult, 1),
	}
	if !requestQueue.Enqueue(req1) {
		t.Fatal("failed to enqueue test request")
	}

	select {
	case <-engine.firstSubmitted:
	case <-time.After(2 * time.Second):
		t.Fatal("first request was not submitted")
	}

	stopDone := make(chan error, 1)
	go func() {
		stopDone <- processor.StopAndWait(context.Background())
	}()

	select {
	case err := <-stopDone:
		t.Fatalf("StopAndWait returned before in-flight completion: %v", err)
	case <-time.After(50 * time.Millisecond):
	}

	closeAllowCompletion()

	select {
	case err := <-stopDone:
		if err != nil {
			t.Fatalf("StopAndWait returned error: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("StopAndWait did not finish after in-flight completion")
	}
}

func TestQueueProcessorStopPreventsDequeuedRequestSubmission(t *testing.T) {
	engine := &queueSubmitterTestEngine{
		allowCompletion: make(chan struct{}),
	}
	modelService := &chatServiceRenderTestModelService{engine: engine, model: "/tmp/test.gguf"}
	requestQueue := queue.NewRequestQueue(1)
	processor := NewQueueProcessor(requestQueue, modelService)
	processor.lifecycleMu.Lock()
	processor.Start(1)

	req := &queue.QueuedRequest{
		Engine: engine,
		GenerationRequest: domain.GenerationRequest{Input: domain.GenerationInput{Kind: domain.GenerationInputTokenIDs, TokenIDs: []int{1}},

			MaxTokens: 1,
		},

		ID:         "dequeued-before-stop",
		Context:    context.Background(),
		ResultChan: make(chan queue.GenerationResult, 1),
	}
	if !requestQueue.Enqueue(req) {
		processor.lifecycleMu.Unlock()
		t.Fatal("failed to enqueue request")
	}

	deadline := time.Now().Add(2 * time.Second)
	for requestQueue.Len() != 0 {
		if time.Now().After(deadline) {
			processor.lifecycleMu.Unlock()
			t.Fatal("worker did not dequeue request")
		}
		runtime.Gosched()
	}

	stopDone := make(chan error, 1)
	go func() {
		stopDone <- processor.StopAndWait(context.Background())
	}()
	deadline = time.Now().Add(2 * time.Second)
	for processor.state.Load() == queueProcessorRunning {
		if time.Now().After(deadline) {
			processor.lifecycleMu.Unlock()
			t.Fatal("shutdown did not begin")
		}
		runtime.Gosched()
	}
	processor.lifecycleMu.Unlock()

	select {
	case result := <-req.ResultChan:
		err := result.Err
		ok := err != nil
		if !ok || !errors.Is(err, errQueueProcessorStopping) {
			t.Fatalf("dequeued shutdown result = %v, want %v", result, errQueueProcessorStopping)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("dequeued request was not terminally rejected")
	}

	if err := <-stopDone; err != nil {
		t.Fatalf("StopAndWait returned error: %v", err)
	}
	if got := atomic.LoadInt32(&engine.submitCount); got != 0 {
		t.Fatalf("submit_count=%d, want 0", got)
	}
	close(engine.allowCompletion)
}

func TestQueueProcessorStopAndWaitHonorsContextWhileRequestInFlight(t *testing.T) {
	engine := &queueSubmitterTestEngine{
		firstSubmitted:  make(chan struct{}),
		allowCompletion: make(chan struct{}),
	}
	modelService := &chatServiceRenderTestModelService{engine: engine, model: "/tmp/test.gguf"}
	requestQueue := queue.NewRequestQueue(1)
	processor := NewQueueProcessor(requestQueue, modelService)
	processor.Start(1)
	var closed int32
	closeAllowCompletion := func() {
		if atomic.CompareAndSwapInt32(&closed, 0, 1) {
			close(engine.allowCompletion)
		}
	}
	defer closeAllowCompletion()

	req := &queue.QueuedRequest{
		Engine: engine,
		GenerationRequest: domain.GenerationRequest{Input: domain.GenerationInput{Kind: domain.GenerationInputTokenIDs, TokenIDs: []int{1}},

			MaxTokens: 1,
		},

		ID:         "req-timeout",
		Context:    context.Background(),
		ResultChan: make(chan queue.GenerationResult, 1),
	}
	if !requestQueue.Enqueue(req) {
		t.Fatal("failed to enqueue request")
	}

	select {
	case <-engine.firstSubmitted:
	case <-time.After(2 * time.Second):
		t.Fatal("request was not submitted")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	if err := processor.StopAndWait(ctx); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("StopAndWait error = %v, want %v", err, context.DeadlineExceeded)
	}

	closeAllowCompletion()

	if err := processor.StopAndWait(context.Background()); err != nil {
		t.Fatalf("StopAndWait retry returned error: %v", err)
	}
}

func TestQueueProcessorStopAndWaitAcrossRepeatedCycles(t *testing.T) {
	for i := 0; i < 5; i++ {
		engine := &queueSubmitterTestEngine{
			firstSubmitted:  make(chan struct{}),
			submitStarted:   make(chan struct{}, 2),
			allowCompletion: make(chan struct{}),
		}
		modelService := &chatServiceRenderTestModelService{engine: engine, model: "/tmp/test.gguf"}
		requestQueue := queue.NewRequestQueue(2)
		processor := NewQueueProcessor(requestQueue, modelService)
		processor.Start(1)

		req := &queue.QueuedRequest{
			Engine: engine,
			GenerationRequest: domain.GenerationRequest{Input: domain.GenerationInput{Kind: domain.GenerationInputTokenIDs, TokenIDs: []int{1}},

				MaxTokens: 1,
			},

			ID:         "cycle-req",
			Context:    context.Background(),
			ResultChan: make(chan queue.GenerationResult, 1),
		}
		if !requestQueue.Enqueue(req) {
			t.Fatalf("cycle %d: failed to enqueue request", i)
		}

		select {
		case <-engine.firstSubmitted:
		case <-time.After(2 * time.Second):
			t.Fatalf("cycle %d: request was not submitted", i)
		}

		close(engine.allowCompletion)

		if err := processor.StopAndWait(context.Background()); err != nil {
			t.Fatalf("cycle %d: StopAndWait returned error: %v", i, err)
		}
		waitForActiveCount(t, engine, 0)
		if !requestQueue.Enqueue(req) {
			continue
		}
		t.Fatalf("cycle %d: queue accepted request after shutdown", i)
	}
}

func TestQueueProcessorStopAndWaitJoinsMultipleWorkers(t *testing.T) {
	const workers = 3
	engine := &queueSubmitterTestEngine{
		submitStarted:   make(chan struct{}, workers),
		allowCompletion: make(chan struct{}),
	}
	modelService := &chatServiceRenderTestModelService{engine: engine, model: "/tmp/test.gguf"}
	requestQueue := queue.NewRequestQueue(6)
	processor := NewQueueProcessor(requestQueue, modelService)
	processor.Start(workers)

	for i := 0; i < workers; i++ {
		req := &queue.QueuedRequest{
			Engine: engine,
			GenerationRequest: domain.GenerationRequest{Input: domain.GenerationInput{Kind: domain.GenerationInputTokenIDs, TokenIDs: []int{i + 1}},

				MaxTokens: 1,
			},

			ID:         "worker-req",
			Context:    context.Background(),
			ResultChan: make(chan queue.GenerationResult, 1),
		}
		if !requestQueue.Enqueue(req) {
			t.Fatalf("failed to enqueue request %d", i)
		}
	}

	for i := 0; i < workers; i++ {
		select {
		case <-engine.submitStarted:
		case <-time.After(2 * time.Second):
			t.Fatalf("worker %d did not submit", i)
		}
	}
	waitForActiveCount(t, engine, workers)

	stopDone := make(chan error, 1)
	go func() {
		stopDone <- processor.StopAndWait(context.Background())
	}()

	select {
	case err := <-stopDone:
		t.Fatalf("StopAndWait returned early: %v", err)
	case <-time.After(100 * time.Millisecond):
	}

	close(engine.allowCompletion)

	select {
	case err := <-stopDone:
		if err != nil {
			t.Fatalf("StopAndWait returned error: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("StopAndWait did not finish after releasing multiple workers")
	}

	waitForActiveCount(t, engine, 0)
}

func TestQueueProcessorStopAndWaitCompletesAfterCancellationDuringShutdown(t *testing.T) {
	engine := &queueSubmitterTestEngine{
		firstSubmitted:   make(chan struct{}),
		allowCompletion:  make(chan struct{}),
		completeOnCancel: true,
	}
	modelService := &chatServiceRenderTestModelService{engine: engine, model: "/tmp/test.gguf"}
	requestQueue := queue.NewRequestQueue(1)
	processor := NewQueueProcessor(requestQueue, modelService)
	processor.Start(1)

	reqCtx, cancelReq := context.WithCancel(context.Background())
	defer cancelReq()

	req := &queue.QueuedRequest{
		Engine: engine,
		GenerationRequest: domain.GenerationRequest{Input: domain.GenerationInput{Kind: domain.GenerationInputTokenIDs, TokenIDs: []int{1}},

			MaxTokens: 1,
		},

		ID:         "cancel-on-shutdown",
		Context:    reqCtx,
		ResultChan: make(chan queue.GenerationResult, 1),
	}
	if !requestQueue.Enqueue(req) {
		t.Fatal("failed to enqueue request")
	}

	select {
	case <-engine.firstSubmitted:
	case <-time.After(2 * time.Second):
		t.Fatal("request was not submitted")
	}

	stopDone := make(chan error, 1)
	go func() {
		stopDone <- processor.StopAndWait(context.Background())
	}()

	cancelReq()

	select {
	case err := <-stopDone:
		if err != nil {
			t.Fatalf("StopAndWait returned error: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("StopAndWait did not finish after request cancellation")
	}

	waitForActiveCount(t, engine, 0)
}

func TestQueueProcessorStopAndWaitLeavesNoActiveSubmitters(t *testing.T) {
	engine := &queueSubmitterTestEngine{
		submitStarted:   make(chan struct{}, 2),
		allowCompletion: make(chan struct{}),
	}
	modelService := &chatServiceRenderTestModelService{engine: engine, model: "/tmp/test.gguf"}
	requestQueue := queue.NewRequestQueue(2)
	processor := NewQueueProcessor(requestQueue, modelService)
	processor.Start(2)

	for i := 0; i < 2; i++ {
		req := &queue.QueuedRequest{
			Engine: engine,
			GenerationRequest: domain.GenerationRequest{Input: domain.GenerationInput{Kind: domain.GenerationInputTokenIDs, TokenIDs: []int{i + 1}},

				MaxTokens: 1,
			},

			ID:         "join-req",
			Context:    context.Background(),
			ResultChan: make(chan queue.GenerationResult, 1),
		}
		if !requestQueue.Enqueue(req) {
			t.Fatalf("failed to enqueue request %d", i)
		}
	}

	for i := 0; i < 2; i++ {
		select {
		case <-engine.submitStarted:
		case <-time.After(2 * time.Second):
			t.Fatalf("submit %d did not start", i)
		}
	}

	close(engine.allowCompletion)

	if err := processor.StopAndWait(context.Background()); err != nil {
		t.Fatalf("StopAndWait returned error: %v", err)
	}

	waitForActiveCount(t, engine, 0)
}

func (e *queueSubmitterTestEngine) SubmitGeneration(ctx context.Context, req domain.GenerationRequest) (*domain.Generation, error) {
	if e.admissionEntered != nil {
		e.admissionEntered <- struct{}{}
	}
	release, err := e.admission.Acquire(ctx, e.runtimeStateError == nil && e.prefixCacheReuseEnabled && e.snapshotRestoreEnabled)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(ctx)
	e.submitMu.Lock()
	switch req.Input.Kind {
	case domain.GenerationInputRenderedText:
		e.renderedChatSubmitted = req.Input.Text
	case domain.GenerationInputRenderedTokenIDs:
		e.renderedPromptSubmitted = req.Input.Text
	}
	nativeDone, err := e.awaitSubmit(ctx)
	e.submitMu.Unlock()
	if err != nil {
		release()
		cancel()
		return nil, err
	}
	done := make(chan struct{})
	output := make(chan domain.StreamEvent)
	go func() { <-nativeDone; close(output); release(); close(done) }()
	return &domain.Generation{Events: output, Done: done, Cancel: cancel}, nil
}

func TestQueueProcessorAdmissionWaitKeepsBacklogBoundedAndCancelable(t *testing.T) {
	eng := &queueSubmitterTestEngine{
		allowCompletion:         make(chan struct{}),
		prefixCacheReuseEnabled: true,
		snapshotRestoreEnabled:  true,
		admissionEntered:        make(chan struct{}, 4),
	}
	q := queue.NewRequestQueue(1)
	processor := NewQueueProcessor(q, nil)
	processor.Start(1)
	var complete sync.Once
	finish := func() { complete.Do(func() { close(eng.allowCompletion) }) }
	defer func() { finish(); processor.Stop() }()
	makeRequest := func(ctx context.Context) *queue.QueuedRequest {
		return &queue.QueuedRequest{Engine: eng, Context: ctx, GenerationRequest: domain.GenerationRequest{Input: domain.GenerationInput{Kind: domain.GenerationInputText, Text: "hello"}, MaxTokens: 8}, ResultChan: make(chan queue.GenerationResult, 1)}
	}
	first := makeRequest(context.Background())
	if !q.Enqueue(first) {
		t.Fatal("first enqueue")
	}
	<-eng.admissionEntered
	waitForSubmitCount(t, eng, 1)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	second := makeRequest(ctx)
	if !q.Enqueue(second) {
		t.Fatal("second enqueue")
	}
	select {
	case <-eng.admissionEntered:
	case <-time.After(time.Second):
		t.Fatal("worker did not enter admission wait")
	}
	third := makeRequest(context.Background())
	if !q.Enqueue(third) {
		t.Fatal("bounded waiting slot rejected")
	}
	if q.Enqueue(makeRequest(context.Background())) {
		t.Fatal("admission moved waiting requests into an unbounded backlog")
	}
	cancel()
	select {
	case result := <-second.ResultChan:
		if !errors.Is(result.Err, context.Canceled) {
			t.Fatalf("waiting request cancellation: %v", result.Err)
		}
	case <-time.After(time.Second):
		t.Fatal("admission wait ignored request cancellation")
	}
	if got := atomic.LoadInt32(&eng.submitCount); got != 1 {
		t.Fatalf("native submissions=%d before exclusive completion", got)
	}
	finish()
	select {
	case result := <-third.ResultChan:
		if result.Err != nil {
			t.Fatal(result.Err)
		}
	case <-time.After(time.Second):
		t.Fatal("backlog did not resume after completion")
	}
}
