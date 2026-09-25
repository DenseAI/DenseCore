package engine

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/DenseAI/DenseCore/server/internal/domain"
)

func TestGenerationAdmissionCancellationDoesNotStealPermit(t *testing.T) {
	var admission GenerationAdmission
	release, err := admission.Acquire(context.Background(), true)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := admission.Acquire(ctx, true); !errors.Is(err, context.Canceled) {
		t.Fatalf("waiting cancellation: %v", err)
	}
	release()
	release()
	ctx, cancel = context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	release, err = admission.Acquire(ctx, true)
	if err != nil {
		t.Fatal(err)
	}
	release()
}

func TestGenerationAdmissionAsyncDoesNotWaitForExclusivePermit(t *testing.T) {
	var admission GenerationAdmission
	release, err := admission.Acquire(context.Background(), true)
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	asyncRelease, err := admission.Acquire(ctx, false)
	if err != nil {
		t.Fatal(err)
	}
	asyncRelease()
}

func TestCanceledCallbackConsumerCannotBlockTerminal(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	output := make(chan domain.StreamEvent)
	registration := registerGenerationContext(ctx, output)
	defer registration.abort()
	item, _ := requestChannels.LoadItem(registration.callbackID)
	finished := make(chan struct{})
	go func() {
		sendCallbackEvent(registration.callbackID, item, domain.NewTerminalEvent(nil))
		close(finished)
	}()
	cancel()
	select {
	case <-finished:
	case <-time.After(time.Second):
		t.Fatal("terminal blocked after consumer cancellation")
	}
}

func TestGenerationDonePublishesTerminalAndRegistrationCleanup(t *testing.T) {
	for _, canceled := range []bool{false, true} {
		ctx, cancel := context.WithCancel(context.Background())
		output := make(chan domain.StreamEvent, 1)
		registration := registerGenerationContext(ctx, output)
		if canceled {
			cancel()
		}
		dispatchGenerationEvent(registration.callbackID, domain.NewTerminalEvent(nil))
		<-registration.completionCh
		if _, ok := requestChannels.Load(registration.callbackID); ok {
			t.Fatal("Done preceded request registration cleanup")
		}
		if _, ok := completionChannels.Load(registration.callbackID); ok {
			t.Fatal("Done preceded completion registration cleanup")
		}
		events := 0
		for event := range output {
			if !event.Terminal {
				t.Fatal("expected terminal")
			}
			events++
		}
		if !canceled && events != 1 {
			t.Fatalf("terminal count=%d", events)
		}
		dispatchGenerationEvent(registration.callbackID, domain.NewTerminalEvent(nil)) // removed registration ignores late callbacks
		cancel()
	}
}
