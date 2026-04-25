package engine

import (
	"testing"
	"time"

	"descore-server/internal/domain"
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
