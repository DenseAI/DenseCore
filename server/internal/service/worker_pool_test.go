package service

import (
	"context"
	"testing"
	"time"

	"descore-server/internal/domain"
)

func TestProxyStreamStopsBlockingWhenDestinationIsAbandoned(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	src := make(chan domain.StreamEvent, 1)
	dst := make(chan domain.StreamEvent)
	done := make(chan struct{})

	src <- domain.StreamEvent{Token: "partial", IsFinished: false}

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
