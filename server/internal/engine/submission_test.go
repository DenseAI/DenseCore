package engine

import (
	"context"
	"strings"
	"testing"

	"github.com/DenseAI/DenseCore/server/internal/domain"
)

func TestRegisterGenerationTracksAndCleansBothChannels(t *testing.T) {
	output := make(chan domain.StreamEvent, 1)
	registration := registerGeneration(output)

	if got, ok := requestChannels.Load(registration.callbackID); !ok || got != output {
		t.Fatal("request output channel was not registered")
	}
	if got, ok := completionChannels.Load(registration.callbackID); !ok || got != registration.completionCh {
		t.Fatal("completion channel was not registered with the same callback ID")
	}

	registration.abort()
	if _, ok := requestChannels.Load(registration.callbackID); ok {
		t.Fatal("request output channel survived aborted submission")
	}
	if _, ok := completionChannels.Load(registration.callbackID); ok {
		t.Fatal("completion channel survived aborted submission")
	}
}

func TestNativeAdmissionRejectionAbortsWithoutWaitingForCallback(t *testing.T) {
	output := make(chan domain.StreamEvent, 1)
	engine := &DenseEngine{}
	done, requestID, callbackID, err := engine.submitGeneration(context.Background(), output, func(uintptr) int { return -3 })
	if err == nil || !strings.Contains(err.Error(), "-3") || done != nil || requestID != 0 {
		t.Fatalf("rejection result done=%v request=%d err=%v", done, requestID, err)
	}
	if _, ok := requestChannels.Load(callbackID); ok {
		t.Fatal("rejected request retained callback channel")
	}
	if _, ok := completionChannels.Load(callbackID); ok {
		t.Fatal("rejected request retained completion waiter")
	}
}
