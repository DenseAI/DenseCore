package engine

import (
	"context"
	"sync"
	"time"
)

const legacyShutdownGrace = 5 * time.Second

// One native cleanup owns the handle until it finishes, even if its caller's
// context expires while a kernel or user callback is still returning.
type shutdownLifecycle struct {
	beginOnce sync.Once
	done      chan struct{}
	err       error
	freeOnce  sync.Once
	freed     chan struct{}
}

func (l *shutdownLifecycle) start(ctx context.Context, stop func(context.Context) error) <-chan struct{} {
	l.beginOnce.Do(func() {
		l.done = make(chan struct{})
		if ctx == nil {
			ctx = context.Background()
		}
		if _, ok := ctx.Deadline(); !ok {
			var cancel context.CancelFunc
			ctx, cancel = context.WithTimeout(ctx, legacyShutdownGrace)
			go func() { defer cancel(); defer close(l.done); l.err = stop(ctx) }()
		} else {
			go func() { defer close(l.done); l.err = stop(ctx) }()
		}
	})
	return l.done
}
func (l *shutdownLifecycle) begin(ctx context.Context, stop func(context.Context) error) error {
	if ctx == nil {
		ctx = context.Background()
	}
	done := l.start(ctx, stop)
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-done:
		if err := ctx.Err(); err != nil {
			return err
		}
		return l.err
	}
}
func (l *shutdownLifecycle) close(ctx context.Context, stop func(context.Context) error, free func()) error {
	if ctx == nil {
		ctx = context.Background()
	}
	l.freeOnce.Do(func() {
		l.freed = make(chan struct{})
		done := l.start(ctx, stop)
		go func() { defer close(l.freed); <-done; free() }()
	})
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-l.freed:
		if err := ctx.Err(); err != nil {
			return err
		}
		return l.err
	}
}

// Recompute immediately before entering C, including time spent waiting for a
// handle mutex. Round positive fractional milliseconds up without overflow.
func remainingShutdownMillis(ctx context.Context, now time.Time) int64 {
	if ctx.Err() != nil {
		return 0
	}
	deadline, ok := ctx.Deadline()
	if !ok {
		return legacyShutdownGrace.Milliseconds()
	}
	remaining := deadline.Sub(now)
	if remaining <= 0 {
		return 0
	}
	value := remaining.Milliseconds()
	if remaining%time.Millisecond != 0 {
		value++
	}
	return value
}
