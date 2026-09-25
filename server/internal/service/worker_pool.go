package service

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/DenseAI/DenseCore/server/internal/domain"
	"github.com/DenseAI/DenseCore/server/internal/queue"
)

var errQueueProcessorStopping = errors.New("request rejected: server shutting down")

// QueueProcessor manages the worker pool that consumes requests from the priority queue.
type QueueProcessor struct {
	queue       *queue.RequestQueue
	lifecycleMu sync.RWMutex
	stopOnce    sync.Once
	state       atomic.Int32
	workerWG    sync.WaitGroup
	trackerWG   sync.WaitGroup
	stopContext context.Context
	stopCancel  context.CancelFunc
}

const (
	queueProcessorRunning int32 = iota
	queueProcessorStopping
	queueProcessorStopped
)

// NewQueueProcessor creates a new processor.
func NewQueueProcessor(q *queue.RequestQueue, _ domain.ModelService) *QueueProcessor {
	ctx, cancel := context.WithCancel(context.Background())
	return &QueueProcessor{
		stopContext: ctx, stopCancel: cancel,
		queue: q,
	}
}

// Start launches N worker goroutines to process requests.
func (p *QueueProcessor) Start(workers int) {
	slog.Info("starting worker pool", slog.Int("workers", workers))
	for i := 0; i < workers; i++ {
		p.workerWG.Add(1)
		go p.workerLoop(i)
	}
}

// Stop signals the worker pool to stop by closing the request queue.
func (p *QueueProcessor) Stop() {
	_ = p.StopAndWait(context.Background())
}

func (p *QueueProcessor) StopAndWait(ctx context.Context) error {
	p.stopOnce.Do(func() {
		slog.Info("stopping worker pool")
		p.state.Store(queueProcessorStopping)
		p.stopCancel()
		p.lifecycleMu.Lock()
		p.queue.Close()
		p.rejectPendingRequests(p.queue.Drain(), errQueueProcessorStopping)
		p.lifecycleMu.Unlock()
	})

	if err := waitGroupWithContext(ctx, &p.workerWG); err != nil {
		return err
	}

	if err := waitGroupWithContext(ctx, &p.trackerWG); err != nil {
		return err
	}

	p.state.Store(queueProcessorStopped)
	return nil
}

// workerLoop continuously processes requests from the queue. Workers are
// submitters: they hold a slot only until C++ accepts the request, then return
// to dequeue more work so the runtime scheduler can form decode batches.
func (p *QueueProcessor) workerLoop(workerID int) {
	defer p.workerWG.Done()

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
		if envFlagEnabled("DENSECORE_DEBUG_REQUEST_LIFECYCLE") {
			slog.Info("request lifecycle: queue_dequeue",
				slog.String("trace_id", req.TraceID),
				slog.String("queue_request_id", req.ID),
				slog.Int("worker_id", workerID),
				slog.String("priority", fmtPriority(req.Priority)),
				slog.Int("max_tokens", req.MaxTokens),
				slog.Int("prompt_len", len(req.Input.Text)),
				slog.Int("input_ids", len(req.Input.TokenIDs)),
			)
		}
		queueWaitMS := durationMillis(time.Since(req.EnqueueTime))
		p.lifecycleMu.RLock()
		if p.state.Load() != queueProcessorRunning {
			p.lifecycleMu.RUnlock()
			p.rejectPendingRequests([]*queue.QueuedRequest{req}, errQueueProcessorStopping)
			continue
		}

		p.lifecycleMu.RUnlock()
		if err := req.Context.Err(); err != nil {
			p.rejectPendingRequests([]*queue.QueuedRequest{req}, err)
			continue
		}
		eng, release := req.Engine, req.ReleaseEngine
		if eng == nil { // Direct queue users must bind the engine before preparation.
			p.rejectPendingRequests([]*queue.QueuedRequest{req}, fmt.Errorf("no request engine bound"))
			continue
		}
		if release == nil {
			release = func() {}
		}
		submitContext, cancel := context.WithCancel(req.Context)
		stopWaiting := context.AfterFunc(p.stopContext, cancel)
		submitStart := time.Now()
		generation, err := eng.SubmitGeneration(submitContext, req.GenerationRequest)
		stopWaiting()
		if err != nil {
			cancel()
			release()
			select {
			case req.ResultChan <- queue.GenerationResult{Err: err}:
			default:
			}
			continue
		}
		p.trackerWG.Add(1)
		go func() {
			defer p.trackerWG.Done()
			<-generation.Done
			cancel()
			release()
		}()
		select {
		case req.ResultChan <- queue.GenerationResult{
			Generation:     generation,
			QueueWaitMS:    queueWaitMS,
			EngineSubmitMS: durationMillis(time.Since(submitStart)),
		}:
		default:
			generation.Cancel()
		}
	}
}

func fmtPriority(p queue.RequestPriority) string {
	return fmt.Sprintf("%d", p)
}

func (p *QueueProcessor) rejectPendingRequests(requests []*queue.QueuedRequest, err error) {
	for _, req := range requests {
		if req == nil {
			continue
		}
		if req.ReleaseEngine != nil {
			req.ReleaseEngine()
		}
		select {
		case req.ResultChan <- queue.GenerationResult{Err: err}:
		default:
			slog.Warn("pending request result channel abandoned during shutdown",
				slog.String("trace_id", req.TraceID),
				slog.String("req_id", req.ID),
			)
		}
	}
}

func waitGroupWithContext(ctx context.Context, wg *sync.WaitGroup) error {
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	if ctx == nil {
		<-done
		return nil
	}

	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
