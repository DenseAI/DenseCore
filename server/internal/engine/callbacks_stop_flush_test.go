package engine

import (
	"github.com/DenseAI/DenseCore/server/internal/domain"
	"testing"
)

func TestSyntheticStopPrefixFlushPreservesContentAndTerminalOwnership(t *testing.T) {
	for _, text := range []string{"<ST", "Error:", "끝"} {
		t.Run(text, func(t *testing.T) {
			event := callbackStreamEvent(text, -1, false)
			if event.Token != text || event.TokenID != -1 || event.Terminal || event.IsFinished || event.TerminalError() != nil {
				t.Fatalf("synthetic text misclassified: %+v", event)
			}
			const id uintptr = 19001
			ch := make(chan domain.StreamEvent, 2)
			requestChannels.Store(id, ch)
			done := completionChannels.Register(id)
			defer requestChannels.Delete(id)
			defer completionChannels.Signal(id)
			completion := &domain.GenerationCompletion{Tokens: 1, FinishReason: "length"}
			requestChannels.mu.Lock()
			item := requestChannels.channels[id]
			item.completion = completion
			requestChannels.channels[id] = item
			requestChannels.mu.Unlock()
			dispatchGenerationEvent(id, event)
			select {
			case <-done:
				t.Fatal("text flush finished request")
			default:
			}
			if _, ok := requestChannels.LoadItem(id); !ok {
				t.Fatal("flush removed request registration")
			}
			flush := <-ch
			if flush.Completion != nil || flush.Terminal || flush.Token != text {
				t.Fatalf("flush=%+v", flush)
			}
			dispatchGenerationEvent(id, callbackStreamEvent("", -1, true))
			terminal := <-ch
			if !terminal.TerminalSuccess() || terminal.Completion != completion || terminal.Completion.Tokens != 1 {
				t.Fatalf("terminal=%+v", terminal)
			}
			if _, ok := <-ch; ok {
				t.Fatal("missing channel close")
			}
			select {
			case <-done:
			default:
				t.Fatal("terminal failed to signal completion")
			}
		})
	}
}
