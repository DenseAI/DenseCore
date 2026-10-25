package engine

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestRemainingShutdownMillisUsesOneDeadlineAndRoundsSafely(t *testing.T) {
	now := time.Unix(100, 0)
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(time.Hour))
	defer cancel()
	deadline, _ := ctx.Deadline()
	if got := remainingShutdownMillis(ctx, deadline.Add(-30*time.Second)); got != 30000 {
		t.Fatalf("30s=%d", got)
	}
	if got := remainingShutdownMillis(ctx, deadline.Add(-7*time.Second)); got != 7000 {
		t.Fatalf("remaining7s=%d", got)
	}
	if got := remainingShutdownMillis(ctx, deadline.Add(-time.Nanosecond)); got != 1 {
		t.Fatalf("roundup=%d", got)
	}
	if got := remainingShutdownMillis(ctx, deadline.Add(time.Second)); got != 0 {
		t.Fatalf("expired=%d", got)
	}
	if got := remainingShutdownMillis(context.Background(), now); got != 5000 {
		t.Fatalf("legacy=%d", got)
	}
	cancel()
	if got := remainingShutdownMillis(ctx, now); got != 0 {
		t.Fatalf("cancelled=%d", got)
	}
}

func TestShutdownContextReturnsWhileOneCleanupStillOwnsHandle(t *testing.T) {
	var lifecycle shutdownLifecycle
	entered, release := make(chan struct{}), make(chan struct{})
	var stops, frees atomic.Int32
	var stopReturned, earlyFree atomic.Bool
	stop := func(ctx context.Context) error {
		stops.Add(1)
		close(entered)
		<-release
		stopReturned.Store(true)
		return nil
	}
	ctx, cancel := context.WithCancel(context.Background())
	first := make(chan error, 1)
	go func() { first <- lifecycle.begin(ctx, stop) }()
	<-entered
	cancel()
	if err := <-first; !errors.Is(err, context.Canceled) {
		t.Fatalf("caller=%v", err)
	}
	closed := make(chan error, 1)
	go func() {
		closed <- lifecycle.close(context.Background(), stop, func() {
			if !stopReturned.Load() {
				earlyFree.Store(true)
			}
			frees.Add(1)
		})
	}()
	if err := lifecycle.begin(ctx, stop); !errors.Is(err, context.Canceled) {
		t.Fatalf("repeat=%v", err)
	}
	select {
	case <-closed:
		t.Fatal("freed handle before native join completed")
	default:
	}
	if frees.Load() != 0 {
		t.Fatal("early free")
	}
	close(release)
	if err := <-closed; err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = lifecycle.close(context.Background(), stop, func() {
				if !stopReturned.Load() {
					earlyFree.Store(true)
				}
				frees.Add(1)
			})
		}()
	}
	wg.Wait()
	if earlyFree.Load() {
		t.Fatal("handle freed while native cleanup still owned it")
	}
	if stops.Load() != 1 || frees.Load() != 1 {
		t.Fatalf("stops=%d frees=%d", stops.Load(), frees.Load())
	}
}

func TestShutdownPreservesFirstConfiguredDeadlineAcrossCloseRace(t *testing.T) {
	var lifecycle shutdownLifecycle
	entered, release := make(chan context.Context, 1), make(chan struct{})
	deadline := time.Now().Add(30 * time.Second)
	ctx, cancel := context.WithDeadline(context.Background(), deadline)
	defer cancel()
	stop := func(selected context.Context) error { entered <- selected; <-release; return nil }
	done := make(chan error, 1)
	go func() { done <- lifecycle.begin(ctx, stop) }()
	selected := <-entered
	actual, ok := selected.Deadline()
	if !ok || !actual.Equal(deadline) {
		t.Fatalf("deadline replaced: %v", actual)
	}
	closed := make(chan error, 1)
	go func() { closed <- lifecycle.close(context.Background(), stop, func() {}) }()
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if err := <-closed; err != nil {
		t.Fatal(err)
	}
}
