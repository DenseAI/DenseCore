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
