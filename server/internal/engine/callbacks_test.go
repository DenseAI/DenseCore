package engine

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/DenseAI/DenseCore/server/internal/domain"
)

func TestCleanupStaleChannelsKeepsActiveRequestChannels(t *testing.T) {
	requestMap := NewRequestChannelMap()
	ch := make(chan domain.StreamEvent, 1)
	const reqID uintptr = 42

	requestMap.Store(reqID, ch)
	requestMap.mu.Lock()
	item := requestMap.channels[reqID]
	item.createdAt = time.Now().Add(-10 * time.Minute)
	requestMap.channels[reqID] = item
	requestMap.mu.Unlock()

	completionCh := completionChannels.Register(reqID)
	defer func() {
		completionChannels.Signal(reqID)
		_ = completionCh
	}()

	requestMap.cleanupStaleChannels(5 * time.Minute)

	if _, ok := requestMap.Load(reqID); !ok {
		t.Fatal("active request channel was cleaned up despite an outstanding completion channel")
	}
}

func TestCleanupStaleChannelsRemovesInactiveChannel(t *testing.T) {
	requestMap := NewRequestChannelMap()
	ch := make(chan domain.StreamEvent, 1)
	const reqID uintptr = 77

	requestMap.Store(reqID, ch)
	requestMap.mu.Lock()
	item := requestMap.channels[reqID]
	item.createdAt = time.Now().Add(-10 * time.Minute)
	requestMap.channels[reqID] = item
	requestMap.mu.Unlock()

	requestMap.cleanupStaleChannels(5 * time.Minute)

	if _, ok := requestMap.Load(reqID); ok {
		t.Fatal("inactive stale request channel was not cleaned up")
	}
}

func TestCallbackStreamEvent(t *testing.T) {
	for _, tc := range []struct {
		name        string
		token       string
		finished    bool
		wantToken   string
		wantErr     error
		wantFailure bool
	}{
		{name: "token", token: "hello", wantToken: "hello"},
		{name: "nonterminal error-like text", token: "Error: example", wantToken: "Error: example"},
		{name: "terminal token", token: " world", finished: true, wantToken: " world"},
		{name: "empty terminal", finished: true},
		{name: "canceled", token: "Error: request canceled", finished: true, wantErr: context.Canceled, wantFailure: true},
		{name: "timeout", token: "Error: request timeout", finished: true, wantErr: context.DeadlineExceeded, wantFailure: true},
		{name: "failure", token: "Error: native failure", finished: true, wantFailure: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			event := callbackStreamEvent(tc.token, 42, tc.finished)
			if event.Token != tc.wantToken || event.TokenID != 42 || event.Terminal != tc.finished {
				t.Fatalf("unexpected callback event: %+v", event)
			}
			if event.IsFinished != (tc.finished && !tc.wantFailure) || (event.TerminalError() != nil) != tc.wantFailure {
				t.Fatalf("unexpected completion status: %+v", event)
			}
			if tc.wantErr != nil && !errors.Is(event.TerminalError(), tc.wantErr) {
				t.Fatalf("error = %v, want %v", event.TerminalError(), tc.wantErr)
			}
			if event.Canceled != errors.Is(tc.wantErr, context.Canceled) {
				t.Fatalf("unexpected cancellation status: %+v", event)
			}
		})
	}
}

func TestSendCallbackEventAttachesCompletionOnlyToTerminal(t *testing.T) {
	completion := &domain.GenerationCompletion{Tokens: 128, FinishReason: "length"}
	ch := make(chan domain.StreamEvent, 2)
	item := requestItem{ch: ch, completion: completion}
	sendCallbackEvent(0, item, domain.StreamEvent{Token: "visible"})
	sendCallbackEvent(0, item, domain.NewTerminalEvent(nil))
	if event := <-ch; event.Completion != nil {
		t.Fatalf("nonterminal callback carried completion: %+v", event)
	}
	if event := <-ch; event.Completion != completion || !event.TerminalSuccess() {
		t.Fatalf("terminal metadata was lost: %+v", event)
	}
}
