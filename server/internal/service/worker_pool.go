package service

import (
	"context"
	"fmt"
	"log/slog"

	"descore-server/internal/domain"
	"descore-server/internal/queue"
)

// QueueProcessor manages the worker pool that consumes requests from the priority queue.
type QueueProcessor struct {
	queue        *queue.RequestQueue
	modelService domain.ModelService
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

// workerLoop continuously processes requests from the queue.
// It acts as a proxy, holding the "slot" until the request is fully streamed.
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

		// 2. Prepare channels
		// workerChan: receives tokens from Engine (C++)
		// userChan: receives tokens forwarded by Worker (sent to ChatService)
		workerChan := make(chan domain.StreamEvent, 100)
		userChan := make(chan domain.StreamEvent, 100)

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
		var err error
		if len(req.InputIDs) > 0 {
			err = engine.GenerateStreamTokensWithSampling(
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
				workerChan,
			)
		} else {
			err = engine.GenerateStreamWithSampling(
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
				workerChan,
			)
		}

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

		// 4. Send userChan to ChatService so it can start listening
		// Submission succeeded, so we give them the channel to read tokens.
		select {
		case req.ResultChan <- userChan:
			if envFlagEnabled("DENSECORE_DEBUG_REQUEST_LIFECYCLE") {
				slog.Info("request lifecycle: engine submission succeeded",
					slog.String("trace_id", req.TraceID),
					slog.String("queue_request_id", req.ID),
					slog.Int("worker_id", workerID),
				)
			}
			// 5. Proxy Loop (The Monitor)
			// This keeps the worker "busy" until generation finishes.
			proxyStream(req.Context, req.TraceID, req.ID, workerChan, userChan)
		default:
			slog.Warn("request result channel abandoned after submission", slog.String("req_id", req.ID))
			// We just drain and exit.
			// context cancellation handled by engine
			go func() {
				for range workerChan {
				}
			}()
			close(userChan)
		}

		slog.Debug("worker finished request", slog.Int("worker_id", workerID), slog.String("req_id", req.ID))
	}
}

// proxyStream forwards events from source to dest until source closes.
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
