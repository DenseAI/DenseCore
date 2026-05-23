package service

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"descore-server/internal/domain"
	"descore-server/internal/queue"
)

func TestProxyStreamStopsBlockingWhenDestinationIsAbandoned(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	src := make(chan domain.StreamEvent, 1)
	dst := make(chan domain.StreamEvent)
	done := make(chan struct{})

	src <- domain.StreamEvent{Token: "partial"}

	go func() {
		proxyStream(ctx, "trace-1", "queue-1", src, dst)
		close(done)
	}()

	cancel()
	close(src)

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("proxyStream remained blocked after destination cancellation")
	}
}

func TestProxyStreamSynthesizesTerminalErrorWhenSourceClosesWithoutTerminal(t *testing.T) {
	ctx := context.Background()
	src := make(chan domain.StreamEvent, 1)
	dst := make(chan domain.StreamEvent, 2)

	src <- domain.StreamEvent{Token: "partial"}
	close(src)

	proxyStream(ctx, "trace-1", "queue-1", src, dst)

	first, ok := <-dst
	if !ok || first.Token != "partial" {
		t.Fatalf("expected partial token event, got %+v ok=%v", first, ok)
	}
	terminal, ok := <-dst
	if !ok {
		t.Fatal("expected synthesized terminal event")
	}
	if !terminal.Terminal {
		t.Fatalf("expected terminal event, got %+v", terminal)
	}
	if err := terminal.TerminalError(); err == nil {
		t.Fatalf("expected synthesized terminal error, got %+v", terminal)
	}
}

func TestProxyStreamContextCancellationEmitsTerminalAndReturns(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	src := make(chan domain.StreamEvent)
	dst := make(chan domain.StreamEvent, 1)
	done := make(chan struct{})

	go func() {
		proxyStream(ctx, "trace-ctx-cancel", "queue-ctx-cancel", src, dst)
		close(done)
	}()

	cancel()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("proxyStream did not terminate after context cancellation")
	}

	event, ok := <-dst
	if !ok {
		t.Fatal("expected synthesized terminal event before close")
	}
	if !event.Terminal {
		t.Fatalf("expected terminal event, got %+v", event)
	}
	if err := event.TerminalError(); err == nil {
		t.Fatalf("expected terminal cancellation error, got %+v", event)
	}
}

type queueSubmitterTestEngine struct {
	chatServiceRenderTestEngine
	submitCount             int32
	firstSubmitted          chan struct{}
	allowCompletion         chan struct{}
	renderedPromptSubmitted string
}

func (e *queueSubmitterTestEngine) GenerateStreamWithSamplingAwaitable(ctx context.Context, prompt string, maxTokens int, loraAdapter string, jsonMode bool, temperature float64, topP float64, topK int, repetitionPenalty float64, stop []string, allowedTokenIDs []int, allowedTokensStrict bool, disallowedTokenIDs []int, outputChan chan domain.StreamEvent) (<-chan struct{}, error) {
	return e.awaitSubmit()
}

func (e *queueSubmitterTestEngine) GenerateStreamTokensWithSamplingAwaitable(ctx context.Context, inputIDs []int, maxTokens int, loraAdapter string, jsonMode bool, temperature float64, topP float64, topK int, repetitionPenalty float64, stop []string, allowedTokenIDs []int, allowedTokensStrict bool, disallowedTokenIDs []int, outputChan chan domain.StreamEvent) (<-chan struct{}, error) {
	return e.awaitSubmit()
}

func (e *queueSubmitterTestEngine) GenerateStreamRenderedTokensWithSamplingAwaitable(ctx context.Context, renderedPrompt string, inputIDs []int, maxTokens int, loraAdapter string, jsonMode bool, temperature float64, topP float64, topK int, repetitionPenalty float64, stop []string, allowedTokenIDs []int, allowedTokensStrict bool, disallowedTokenIDs []int, outputChan chan domain.StreamEvent) (<-chan struct{}, error) {
	e.renderedPromptSubmitted = renderedPrompt
	return e.awaitSubmit()
}

func (e *queueSubmitterTestEngine) awaitSubmit() (<-chan struct{}, error) {
	count := atomic.AddInt32(&e.submitCount, 1)
	if count == 1 {
		close(e.firstSubmitted)
	}
	done := make(chan struct{})
	go func() {
		<-e.allowCompletion
		close(done)
	}()
	return done, nil
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
		ID:         "req-rendered",
		Context:    context.Background(),
		Prompt:     "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n",
		InputIDs:   []int{10, 20, 30},
		MaxTokens:  1,
		ResultChan: make(chan interface{}, 1),
		OutputChan: make(chan domain.StreamEvent, 1),
		DoneChan:   make(chan struct{}),
	}
	if !requestQueue.Enqueue(req) {
		t.Fatal("failed to enqueue test request")
	}
	select {
	case <-engine.firstSubmitted:
	case <-time.After(2 * time.Second):
		t.Fatal("request was not submitted")
	}
	if engine.renderedPromptSubmitted != req.Prompt {
		t.Fatalf("rendered prompt submitted=%q want %q", engine.renderedPromptSubmitted, req.Prompt)
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
		ID:         "req-1",
		Context:    context.Background(),
		InputIDs:   []int{1},
		MaxTokens:  1,
		ResultChan: make(chan interface{}, 1),
		OutputChan: make(chan domain.StreamEvent, 1),
		DoneChan:   make(chan struct{}),
	}
	req2 := &queue.QueuedRequest{
		ID:         "req-2",
		Context:    context.Background(),
		InputIDs:   []int{2},
		MaxTokens:  1,
		ResultChan: make(chan interface{}, 1),
		OutputChan: make(chan domain.StreamEvent, 1),
		DoneChan:   make(chan struct{}),
	}

	if !requestQueue.Enqueue(req1) || !requestQueue.Enqueue(req2) {
		t.Fatal("failed to enqueue test requests")
	}
	select {
	case <-engine.firstSubmitted:
	case <-time.After(2 * time.Second):
		t.Fatal("first request was not submitted")
	}

	deadline := time.After(2 * time.Second)
	for atomic.LoadInt32(&engine.submitCount) < 2 {
		select {
		case <-deadline:
			t.Fatalf("worker waited for first completion before submitting second request; submit_count=%d", atomic.LoadInt32(&engine.submitCount))
		default:
			time.Sleep(10 * time.Millisecond)
		}
	}
}
