package decision

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// ServeOptions bounds work admitted by the HTTP endpoint. Zero values retain
// immediate rejection while a single request is running.
type ServeOptions struct {
	BatchSize     int
	BatchWait     time.Duration
	QueueCapacity int
}

func (o ServeOptions) normalized(b Backend) (ServeOptions, error) {
	if o.BatchSize == 0 {
		o.BatchSize = 1
	}
	if o.BatchSize < 1 || o.BatchSize > 64 || o.QueueCapacity < 0 || o.QueueCapacity > 65536 || o.BatchWait < 0 || o.BatchWait > time.Second {
		return o, fmt.Errorf("batch size must be 1..64, queue capacity 0..65536, and batch wait 0..1s")
	}
	if o.BatchSize > 1 {
		if _, ok := b.(BatchBackend); !ok {
			return o, fmt.Errorf("backend does not support native batching")
		}
	}
	return o, nil
}

type BatchInput struct {
	IDs, Markers []int32
	QType        int
}
type BatchOutput struct{ Logits, Action []float32 }

// BatchBackend must execute independent sequences in a native batch. A serial
// loop over Infer is deliberately not a batching implementation.
type BatchBackend interface {
	InferBatch([]BatchInput) ([]BatchOutput, error)
}

type decisionResult struct {
	out map[string]any
	err error
}
type decisionJob struct {
	ctx    context.Context
	req    Request
	result chan decisionResult
}
type scheduler struct {
	jobs    chan decisionJob
	stop    chan struct{}
	done    chan struct{}
	once    sync.Once
	options ServeOptions
	handler *Handler
}

func (s *scheduler) run() {
	defer close(s.done)
	stopping := false
	for {
		var first decisionJob
		if stopping {
			select {
			case first = <-s.jobs:
			default:
				return
			}
		} else {
			select {
			case first = <-s.jobs:
			case <-s.stop:
				stopping = true
				continue
			}
		}
		batch := []decisionJob{first}
		if !stopping && s.options.BatchSize > 1 && s.options.BatchWait > 0 {
			timer := time.NewTimer(s.options.BatchWait)
		gather:
			for len(batch) < s.options.BatchSize {
				select {
				case j := <-s.jobs:
					batch = append(batch, j)
				case <-timer.C:
					break gather
				case <-s.stop:
					stopping = true
					break gather
				}
			}
			timer.Stop()
		}
		for len(batch) < s.options.BatchSize {
			select {
			case j := <-s.jobs:
				batch = append(batch, j)
			default:
				goto execute
			}
		}
	execute:
		live := batch[:0]
		for _, j := range batch {
			if err := j.ctx.Err(); err != nil {
				j.result <- decisionResult{err: err}
				<-s.handler.busy
				s.handler.active.Done()
			} else {
				live = append(live, j)
			}
		}
		if len(live) == 0 {
			continue
		}
		if s.options.BatchSize == 1 {
			out, err := Predict(live[0].ctx, s.handler.backend, live[0].req)
			live[0].result <- decisionResult{out, err}
			<-s.handler.busy
			s.handler.active.Done()
		} else {
			requests := make([]Request, len(live))
			contexts := make([]context.Context, len(live))
			for i, j := range live {
				requests[i] = j.req
				contexts[i] = j.ctx
			}
			results := predictBatch(contexts, s.handler.backend, requests)
			for i, j := range live {
				j.result <- results[i]
				<-s.handler.busy
				s.handler.active.Done()
			}
		}
	}
}
