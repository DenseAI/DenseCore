package service

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"descore-server/internal/domain"
	"descore-server/internal/queue"
)

// QueueProcessor manages the worker pool that consumes requests from the priority queue.
type QueueProcessor struct {
	queue        *queue.RequestQueue
	modelService domain.ModelService
}

type awaitableEngine interface {
	GenerateStreamWithSamplingAwaitable(ctx context.Context, prompt string, maxTokens int, loraAdapter string, jsonMode bool, temperature float64, topP float64, topK int, repetitionPenalty float64, stop []string, allowedTokenIDs []int, allowedTokensStrict bool, disallowedTokenIDs []int, outputChan chan domain.StreamEvent) (<-chan struct{}, error)
	GenerateStreamTokensWithSamplingAwaitable(ctx context.Context, inputIDs []int, maxTokens int, loraAdapter string, jsonMode bool, temperature float64, topP float64, topK int, repetitionPenalty float64, stop []string, allowedTokenIDs []int, allowedTokensStrict bool, disallowedTokenIDs []int, outputChan chan domain.StreamEvent) (<-chan struct{}, error)
}

type renderedTokenAwaitableEngine interface {
	GenerateStreamRenderedTokensWithSamplingAwaitable(ctx context.Context, renderedPrompt string, inputIDs []int, maxTokens int, loraAdapter string, jsonMode bool, temperature float64, topP float64, topK int, repetitionPenalty float64, stop []string, allowedTokenIDs []int, allowedTokensStrict bool, disallowedTokenIDs []int, outputChan chan domain.StreamEvent) (<-chan struct{}, error)
}

type renderedChatAwaitableEngine interface {
	GenerateStreamRenderedChatWithSamplingAwaitable(ctx context.Context, renderedPrompt string, maxTokens int, loraAdapter string, jsonMode bool, temperature float64, topP float64, topK int, repetitionPenalty float64, stop []string, allowedTokenIDs []int, allowedTokensStrict bool, disallowedTokenIDs []int, outputChan chan domain.StreamEvent) (<-chan struct{}, error)
}

type generationResult struct {
	OutputChan     chan domain.StreamEvent
	CompletionChan <-chan struct{}
	QueueWaitMS    float64
	EngineSubmitMS float64
}

// NewQueueProcessor creates a new processor.
func NewQueueProcessor(q *queue.RequestQueue, modelService domain.ModelService) *QueueProcessor {
	return &QueueProcessor{
		queue:        q,
		modelService: modelService,
	}
}

// Start launches N worker goroutines to process requests.
func (p *QueueProcessor) Start(workers int) {
	slog.Info("starting worker pool", slog.Int("workers", workers))
	for i := 0; i < workers; i++ {
		go p.workerLoop(i)
	}
}

// Stop signals the worker pool to stop by closing the request queue.
func (p *QueueProcessor) Stop() {
	slog.Info("stopping worker pool")
	p.queue.Close()
}

// workerLoop continuously processes requests from the queue. Workers are
// submitters: they hold a slot only until C++ accepts the request, then return
// to dequeue more work so the runtime scheduler can form decode batches.
func (p *QueueProcessor) workerLoop(workerID int) {
	ctx := context.Background() // Long-running context for the worker itself
	activeExperts := make(map[int]struct{})
	const maxExperts = 8

	for {
		// 1. Dequeue request (blocks until available)
		var req *queue.QueuedRequest
		var ok bool
		if len(activeExperts) > 0 {
			req, ok = p.queue.MoEAwareDequeue(ctx, activeExperts, maxExperts)
		} else {
			req, ok = p.queue.Dequeue(ctx)
		}
		if !ok {
			// Queue closed or context canceled
			slog.Debug("worker stopping", slog.Int("worker_id", workerID))
			return
		}
		activeExperts = make(map[int]struct{})
		for _, expertID := range req.ExpertCluster {
			activeExperts[expertID] = struct{}{}
		}

		slog.Debug("worker picked request",
			slog.Int("worker_id", workerID),
			slog.String("trace_id", req.TraceID),
			slog.String("req_id", req.ID),
			slog.String("priority", fmtPriority(req.Priority)),
		)
		queueWaitMS := durationMillis(time.Since(req.EnqueueTime))

		outputChan := req.OutputChan
		if outputChan == nil {
			outputChan = make(chan domain.StreamEvent, defaultStreamEventBufferSize())
		}

		// 3. Get Engine (Dynamic)
		engine := p.modelService.GetEngine()
		if engine == nil {
			slog.Error("request failed: no model loaded", slog.String("req_id", req.ID))
			select {
			case req.ResultChan <- fmt.Errorf("no model loaded"):
			default:
			}
			continue
		}

		// 4. Submit to Engine
		submitStart := time.Now()
		completionCh, err := p.submitRequestToEngine(engine, req, outputChan)
		engineSubmitMS := durationMillis(time.Since(submitStart))

		if err != nil {
			slog.Error("engine submission failed",
				slog.String("trace_id", req.TraceID),
				slog.String("req_id", req.ID),
				slog.String("error", err.Error()))

			// Send error to ChatService
			select {
			case req.ResultChan <- err:
			default:
				slog.Warn("request result channel abandoned during error report", slog.String("req_id", req.ID))
			}
			continue
		}

		trackerStarted := false
		startTracker := func(abandoned bool) {
			if trackerStarted {
				return
			}
			trackerStarted = true
			go p.trackCompletion(req, outputChan, completionCh, abandoned)
		}

		// Submission succeeded, so give ChatService the callback channel directly.
		// Completion/cancellation cleanup is supervised outside the worker slot.
		select {
		case req.ResultChan <- generationResult{
			OutputChan:     outputChan,
			CompletionChan: completionCh,
			QueueWaitMS:    queueWaitMS,
			EngineSubmitMS: engineSubmitMS,
		}:
			if envFlagEnabled("DENSECORE_DEBUG_REQUEST_LIFECYCLE") {
				slog.Info("request lifecycle: engine submission succeeded",
					slog.String("trace_id", req.TraceID),
					slog.String("queue_request_id", req.ID),
					slog.Int("worker_id", workerID),
				)
			}
			startTracker(false)
		default:
			slog.Warn("request result channel abandoned after submission", slog.String("req_id", req.ID))
			startTracker(true)
		}

		slog.Debug("worker submitted request", slog.Int("worker_id", workerID), slog.String("req_id", req.ID))
	}
}

func (p *QueueProcessor) submitRequestToEngine(engine domain.Engine, req *queue.QueuedRequest, outputChan chan domain.StreamEvent) (<-chan struct{}, error) {
	if awaitable, ok := engine.(awaitableEngine); ok {
		if req.RenderedChatSubmit && req.Prompt != "" {
			if renderedChat, ok := engine.(renderedChatAwaitableEngine); ok {
				return renderedChat.GenerateStreamRenderedChatWithSamplingAwaitable(
					req.Context,
					req.Prompt,
					req.MaxTokens,
					req.LoraAdapter,
					req.JSONMode,
					req.Temperature,
					req.TopP,
					req.TopK,
					req.RepetitionPenalty,
					req.StopSequences,
					req.AllowedTokenIDs,
					req.AllowedTokensStrict,
					req.DisallowedTokenIDs,
					outputChan,
				)
			}
		}
		if len(req.InputIDs) > 0 {
			if renderedAwaitable, ok := engine.(renderedTokenAwaitableEngine); ok && req.Prompt != "" {
				return renderedAwaitable.GenerateStreamRenderedTokensWithSamplingAwaitable(
					req.Context,
					req.Prompt,
					req.InputIDs,
					req.MaxTokens,
					req.LoraAdapter,
					req.JSONMode,
					req.Temperature,
					req.TopP,
					req.TopK,
					req.RepetitionPenalty,
					req.StopSequences,
					req.AllowedTokenIDs,
					req.AllowedTokensStrict,
					req.DisallowedTokenIDs,
					outputChan,
				)
			}
			return awaitable.GenerateStreamTokensWithSamplingAwaitable(
				req.Context,
				req.InputIDs,
				req.MaxTokens,
				req.LoraAdapter,
				req.JSONMode,
				req.Temperature,
				req.TopP,
				req.TopK,
				req.RepetitionPenalty,
				req.StopSequences,
				req.AllowedTokenIDs,
				req.AllowedTokensStrict,
				req.DisallowedTokenIDs,
				outputChan,
			)
		}
		return awaitable.GenerateStreamWithSamplingAwaitable(
			req.Context,
			req.Prompt,
			req.MaxTokens,
			req.LoraAdapter,
			req.JSONMode,
			req.Temperature,
			req.TopP,
			req.TopK,
			req.RepetitionPenalty,
			req.StopSequences,
			req.AllowedTokenIDs,
			req.AllowedTokensStrict,
			req.DisallowedTokenIDs,
			outputChan,
		)
	}

	if len(req.InputIDs) > 0 {
		if err := engine.GenerateStreamTokensWithSampling(
			req.Context,
			req.InputIDs,
			req.MaxTokens,
			req.LoraAdapter,
			req.JSONMode,
			req.Temperature,
			req.TopP,
			req.TopK,
			req.RepetitionPenalty,
			req.StopSequences,
			req.AllowedTokenIDs,
			req.AllowedTokensStrict,
			req.DisallowedTokenIDs,
			outputChan,
		); err != nil {
			return nil, err
		}
	} else if err := engine.GenerateStreamWithSampling(
		req.Context,
		req.Prompt,
		req.MaxTokens,
		req.LoraAdapter,
		req.JSONMode,
		req.Temperature,
		req.TopP,
		req.TopK,
		req.RepetitionPenalty,
		req.StopSequences,
		req.AllowedTokenIDs,
		req.AllowedTokensStrict,
		req.DisallowedTokenIDs,
		outputChan,
	); err != nil {
		return nil, err
	}
	done := make(chan struct{})
	close(done)
	return done, nil
}

func (p *QueueProcessor) trackCompletion(req *queue.QueuedRequest, outputChan <-chan domain.StreamEvent, completionCh <-chan struct{}, abandoned bool) {
	if abandoned && outputChan != nil {
		go func() {
			for range outputChan {
			}
		}()
	}
	p.waitForCompletion(req, completionCh)
	if envFlagEnabled("DENSECORE_DEBUG_REQUEST_LIFECYCLE") {
		slog.Info("request lifecycle: completion tracked",
			slog.String("trace_id", req.TraceID),
			slog.String("queue_request_id", req.ID),
			slog.Bool("abandoned", abandoned),
		)
	}
}

func (p *QueueProcessor) waitForCompletion(req *queue.QueuedRequest, completionCh <-chan struct{}) {
	if completionCh == nil {
		return
	}
	select {
	case <-completionCh:
	case <-req.Context.Done():
		select {
		case <-completionCh:
		case <-req.DoneChan:
		case <-time.After(5 * time.Second):
			slog.Warn("request completion wait timed out",
				slog.String("trace_id", req.TraceID),
				slog.String("req_id", req.ID),
				slog.String("error", req.Context.Err().Error()),
			)
		}
	}
}

// proxyStream forwards events from source to dest until source closes.
// Legacy/test-only: production chat generation passes QueueRequest.OutputChan
// directly into the engine and should not route hot-path tokens through this
// worker-level proxy.
func proxyStream(ctx context.Context, traceID string, queueReqID string, src <-chan domain.StreamEvent, dst chan<- domain.StreamEvent) {
	defer close(dst)
	dstAbandoned := false
	sawPartial := false
	terminalSeen := false

	sendEvent := func(event domain.StreamEvent) {
		if dstAbandoned && !event.Terminal {
			return
		}
		if dstAbandoned {
			select {
			case dst <- event:
			default:
			}
		} else {
			dst <- event
		}
		if envFlagEnabled("DENSECORE_DEBUG_REQUEST_LIFECYCLE") && (event.Token != "" || event.Terminal) {
			slog.Info("request lifecycle: forwarded stream event",
				slog.String("trace_id", traceID),
				slog.String("queue_request_id", queueReqID),
				slog.Bool("terminal", event.Terminal),
				slog.Bool("finished", event.IsFinished),
				slog.Bool("canceled", event.Canceled),
				slog.Bool("has_error", event.TerminalError() != nil),
				slog.String("token_preview", previewText(event.Token, 64)),
			)
		}
	}

	logAbandonment := func(err error) {
		if dstAbandoned || !envFlagEnabled("DENSECORE_DEBUG_REQUEST_LIFECYCLE") || err == nil {
			return
		}
		state := "before_generation"
		if sawPartial {
			state = "after_partial_output"
		}
		slog.Warn("request lifecycle: stream destination abandoned",
			slog.String("trace_id", traceID),
			slog.String("queue_request_id", queueReqID),
			slog.String("state", state),
			slog.String("error", err.Error()),
		)
	}

	for {
		select {
		case event, ok := <-src:
			if !ok {
				if !terminalSeen {
					err := domain.ErrStreamClosedWithoutTerminal
					if ctxErr := ctx.Err(); ctxErr != nil {
						err = ctxErr
					}
					sendEvent(domain.NewTerminalEvent(err))
				}
				return
			}
			if event.Token != "" {
				sawPartial = true
			}
			if event.Terminal {
				terminalSeen = true
				sendEvent(event)
				continue
			}
			if dstAbandoned {
				continue
			}
			select {
			case dst <- event:
				if envFlagEnabled("DENSECORE_DEBUG_REQUEST_LIFECYCLE") && event.Token != "" {
					slog.Info("request lifecycle: forwarded stream event",
						slog.String("trace_id", traceID),
						slog.String("queue_request_id", queueReqID),
						slog.Bool("terminal", false),
						slog.Bool("finished", event.IsFinished),
						slog.Bool("has_error", false),
						slog.String("token_preview", previewText(event.Token, 64)),
					)
				}
			case <-ctx.Done():
				logAbandonment(ctx.Err())
				dstAbandoned = true
				if !terminalSeen {
					sendEvent(domain.NewTerminalEvent(ctx.Err()))
					terminalSeen = true
				}
				return
			}
		case <-ctx.Done():
			logAbandonment(ctx.Err())
			dstAbandoned = true
			if !terminalSeen {
				sendEvent(domain.NewTerminalEvent(ctx.Err()))
				terminalSeen = true
			}
			return
		}
	}
}

func fmtPriority(p queue.RequestPriority) string {
	return fmt.Sprintf("%d", p)
}
