package engine

import (
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
