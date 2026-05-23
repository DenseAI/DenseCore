package queue

import (
	"container/heap"
	"context"
	"sync"
	"time"

	"descore-server/internal/domain"
)

// RequestPriority defines request priority levels
type RequestPriority int

// QueuedRequest represents a request in the queue
type QueuedRequest struct {
	ID                  string
	TraceID             string
	Priority            RequestPriority
	EnqueueTime         time.Time
	MaxTokens           int
	JSONMode            bool
	Prompt              string
	InputIDs            []int
	RenderedChatSubmit  bool
	LoraAdapter         string
	StopSequences       []string
	Temperature         float64
	TopP                float64
	TopK                int
	RepetitionPenalty   float64
	AllowedTokenIDs     []int
	AllowedTokensStrict bool
	DisallowedTokenIDs  []int
	Context             context.Context
	ResultChan          chan interface{}
	OutputChan          chan domain.StreamEvent
	DoneChan            chan struct{}

	// MoE optimization: predicted or last-used expert IDs for this request
	// Used by MoEAwareDequeue to group requests with similar expert affinity
	ExpertCluster []int

	// Internal heap index
	index int
}

// ============================================================================
// Priority Queue Implementation
// ============================================================================

// requestHeap implements heap.Interface for priority queue
type requestHeap []*QueuedRequest

func (h requestHeap) Len() int { return len(h) }

func (h requestHeap) Less(i, j int) bool {
	// Higher priority first
	if h[i].Priority != h[j].Priority {
		return h[i].Priority > h[j].Priority
	}
	// FIFO within same priority
	return h[i].EnqueueTime.Before(h[j].EnqueueTime)
}

func (h requestHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
	h[i].index = i
	h[j].index = j
}

func (h *requestHeap) Push(x interface{}) {
	n := len(*h)
	req := x.(*QueuedRequest)
	req.index = n
	*h = append(*h, req)
}

func (h *requestHeap) Pop() interface{} {
	old := *h
	n := len(old)
	req := old[n-1]
	old[n-1] = nil
	req.index = -1
	*h = old[0 : n-1]
	return req
}

// ============================================================================
// Request Queue (Channel-Based Signaling)
// ============================================================================

// RequestQueue manages incoming requests with priority and backpressure.
// Uses channel-based signaling for zero-latency dequeue operations.
type RequestQueue struct {
	mu      sync.Mutex
	heap    requestHeap
	maxSize int
	closed  bool
	signal  chan struct{} // Signals when items are enqueued

	// Stats
	totalEnqueued int64
	totalDequeued int64
	totalDropped  int64
}

// NewRequestQueue creates a new priority queue with the specified max size.
func NewRequestQueue(maxSize int) *RequestQueue {
	q := &RequestQueue{
		heap:    make(requestHeap, 0),
		maxSize: maxSize,
		signal:  make(chan struct{}, 1), // Buffered to avoid blocking on signal
	}
	heap.Init(&q.heap)
	return q
}

// Enqueue adds a request to the queue.
// Returns false if queue is full (backpressure) or closed.
func (q *RequestQueue) Enqueue(req *QueuedRequest) bool {
	q.mu.Lock()

	if q.closed {
		q.mu.Unlock()
		return false
	}

	// Backpressure: reject if full
	if len(q.heap) >= q.maxSize {
		q.totalDropped++
		q.mu.Unlock()
		return false
	}

	req.EnqueueTime = time.Now()
	heap.Push(&q.heap, req)
	q.totalEnqueued++
	q.mu.Unlock()

	// Signal waiting dequeue (non-blocking)
	select {
	case q.signal <- struct{}{}:
	default:
		// Signal already pending, no need to send another
	}

	return true
}

// Dequeue removes and returns the highest priority request.
// Blocks until a request is available or context is canceled.
// Uses select for zero-latency context cancellation handling.
func (q *RequestQueue) Dequeue(ctx context.Context) (*QueuedRequest, bool) {
	for {
		q.mu.Lock()

		// Check if we have items
		if len(q.heap) > 0 {
			req := heap.Pop(&q.heap).(*QueuedRequest)
			q.totalDequeued++
			q.mu.Unlock()
			return req, true
		}

		// Check if closed
		if q.closed {
			q.mu.Unlock()
			return nil, false
		}

		q.mu.Unlock()

		// Wait for signal OR context cancellation (zero-latency)
		select {
		case <-q.signal:
			// Item may have been enqueued, loop and check
			continue
		case <-ctx.Done():
			return nil, false
		}
	}
}

// TryDequeue attempts to dequeue without blocking
func (q *RequestQueue) TryDequeue() (*QueuedRequest, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()

	if len(q.heap) == 0 || q.closed {
		return nil, false
	}

	req := heap.Pop(&q.heap).(*QueuedRequest)
	q.totalDequeued++
	return req, true
}

// MoEAwareDequeue dequeues a request preferring those with overlapping expert clusters.
// This minimizes the expert working set for L3 cache/NUMA locality.
//
// Parameters:
//   - ctx: Context for cancellation
//   - currentExperts: Set of currently active expert IDs in the batch
//   - maxExperts: Maximum number of unique experts allowed in batch
//
// Returns the best matching request, prioritizing:
// 1. Requests whose experts fully overlap with currentExperts
// 2. Requests that add fewest new experts
// 3. Falls back to priority-based dequeue if no MoE info available
func (q *RequestQueue) MoEAwareDequeue(ctx context.Context, currentExperts map[int]struct{}, maxExperts int) (*QueuedRequest, bool) {
	q.mu.Lock()

	if len(q.heap) == 0 {
		if q.closed {
			q.mu.Unlock()
			return nil, false
		}
		q.mu.Unlock()

		// Wait for signal
		select {
		case <-q.signal:
			return q.MoEAwareDequeue(ctx, currentExperts, maxExperts)
		case <-ctx.Done():
			return nil, false
		}
	}

	// Find best request based on expert overlap
	bestIdx := -1
	bestScore := -1 // Higher = better (more overlap)

	for i, req := range q.heap {
		if len(req.ExpertCluster) == 0 {
			// No MoE info - treat as neutral
			if bestIdx == -1 {
				bestIdx = i
				bestScore = 0
			}
			continue
		}

		// Count how many experts overlap with current active set
		overlap := 0
		newExperts := 0
		for _, e := range req.ExpertCluster {
			if _, ok := currentExperts[e]; ok {
				overlap++
			} else {
				newExperts++
			}
		}

		// If adding this request would exceed maxExperts, skip it (unless it's the only option)
		if len(currentExperts)+newExperts > maxExperts && bestIdx != -1 {
			continue
		}

		// Score = overlap count (prefer higher overlap)
		score := overlap*10 - newExperts // Penalize new experts
		if score > bestScore {
			bestScore = score
			bestIdx = i
		}
	}

	if bestIdx == -1 {
		// Fallback to standard priority dequeue
		req := heap.Pop(&q.heap).(*QueuedRequest)
		q.totalDequeued++
		q.mu.Unlock()
		return req, true
	}

	// Remove the selected request from heap
	req := q.heap[bestIdx]
	heap.Remove(&q.heap, bestIdx)
	q.totalDequeued++
	q.mu.Unlock()

	return req, true
}

// Len returns current queue length
func (q *RequestQueue) Len() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.heap)
}

// Close closes the queue and wakes up all waiting goroutines
func (q *RequestQueue) Close() {
	q.mu.Lock()
	if !q.closed {
		q.closed = true
		close(q.signal) // Closing channel wakes all waiters
	}
	q.mu.Unlock()
}

// Stats returns queue statistics
func (q *RequestQueue) Stats() QueueStats {
	q.mu.Lock()
	defer q.mu.Unlock()
	return QueueStats{
		CurrentSize:   len(q.heap),
		MaxSize:       q.maxSize,
		TotalEnqueued: q.totalEnqueued,
		TotalDequeued: q.totalDequeued,
		TotalDropped:  q.totalDropped,
	}
}

// QueueStats holds queue statistics
type QueueStats struct {
	CurrentSize   int   `json:"current_size"`
	MaxSize       int   `json:"max_size"`
	TotalEnqueued int64 `json:"total_enqueued"`
	TotalDequeued int64 `json:"total_dequeued"`
	TotalDropped  int64 `json:"total_dropped"`
}

// ============================================================================
// Concurrency Limiter
// ============================================================================

// ConcurrencyLimiter limits concurrent operations
type ConcurrencyLimiter struct {
	sem chan struct{}
}

// NewConcurrencyLimiter creates a new limiter with the specified max concurrency.
func NewConcurrencyLimiter(maxConcurrency int) *ConcurrencyLimiter {
	return &ConcurrencyLimiter{
		sem: make(chan struct{}, maxConcurrency),
	}
}

// Acquire acquires a slot, blocking if necessary
func (l *ConcurrencyLimiter) Acquire(ctx context.Context) bool {
	select {
	case l.sem <- struct{}{}:
		return true
	case <-ctx.Done():
		return false
	}
}

// TryAcquire tries to acquire a slot without blocking
func (l *ConcurrencyLimiter) TryAcquire() bool {
	select {
	case l.sem <- struct{}{}:
		return true
	default:
		return false
	}
}

// Release releases a slot
func (l *ConcurrencyLimiter) Release() {
	<-l.sem
}

// Available returns number of available slots
func (l *ConcurrencyLimiter) Available() int {
	return cap(l.sem) - len(l.sem)
}

// ============================================================================
// Request Coalescer
// ============================================================================

// RequestCoalescer groups similar requests
type RequestCoalescer struct {
	mu      sync.Mutex
	pending map[string][]*QueuedRequest
	timeout time.Duration
}

// NewRequestCoalescer creates a new request coalescer.
func NewRequestCoalescer(timeout time.Duration) *RequestCoalescer {
	return &RequestCoalescer{
		pending: make(map[string][]*QueuedRequest),
		timeout: timeout,
	}
}

// Add adds a request to the coalescer
// Returns the coalesce key
func (c *RequestCoalescer) Add(key string, req *QueuedRequest) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.pending[key] = append(c.pending[key], req)
}

// Flush returns and removes all requests for a key
func (c *RequestCoalescer) Flush(key string) []*QueuedRequest {
	c.mu.Lock()
	defer c.mu.Unlock()

	requests := c.pending[key]
	delete(c.pending, key)
	return requests
}

// Size returns total pending requests
func (c *RequestCoalescer) Size() int {
	c.mu.Lock()
	defer c.mu.Unlock()

	count := 0
	for _, reqs := range c.pending {
		count += len(reqs)
	}
	return count
}
