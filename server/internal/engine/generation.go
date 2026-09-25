package engine

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"sync"

	"github.com/DenseAI/DenseCore/server/internal/domain"
)

// GenerationAdmission preserves bounded server backlog: the submitting worker
// waits here, rather than creating another unbounded native pending queue.
// It is capacity policy, not the native scheduler's resource-retirement guard.
type GenerationAdmission struct {
	once   sync.Once
	permit chan struct{}
}

func (a *GenerationAdmission) Acquire(ctx context.Context, exclusive bool) (func(), error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !exclusive {
		return func() {}, nil
	}
	a.once.Do(func() { a.permit = make(chan struct{}, 1) })
	select {
	case a.permit <- struct{}{}:
		if err := ctx.Err(); err != nil {
			<-a.permit
			return nil, err
		}
		var once sync.Once
		return func() { once.Do(func() { <-a.permit }) }, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (e *DenseEngine) SubmitGeneration(ctx context.Context, req domain.GenerationRequest) (*domain.Generation, error) {
	e.initializeGenerationPolicy()
	release, err := e.generationAdmission.Acquire(ctx, e.generationExclusive)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(ctx)
	bufferSize := req.EventBufferSize
	if bufferSize <= 0 {
		bufferSize = generationBufferSize()
	}
	output := make(chan domain.StreamEvent, bufferSize)
	// Native entry points serialize request construction per engine. Admission
	// remains context-cancellable and independent of that short native guard.
	if err := ctx.Err(); err != nil {
		cancel()
		release()
		return nil, err
	}
	var done <-chan struct{}
	switch req.Input.Kind {
	case domain.GenerationInputRenderedText:
		done, err = e.GenerateStreamRenderedChatWithSamplingAwaitable(ctx, req.Input.Text, req.MaxTokens, req.LoraAdapter, req.JSONMode, req.Sampling.Temperature, req.Sampling.TopP, req.Sampling.TopK, req.Sampling.RepetitionPenalty, req.Sampling.StopSequences, req.Constraints.AllowedTokenIDs, req.Constraints.AllowedTokensStrict, req.Constraints.DisallowedTokenIDs, output)
	case domain.GenerationInputRenderedTokenIDs:
		done, err = e.GenerateStreamRenderedTokensWithSamplingAwaitable(ctx, req.Input.Text, req.Input.TokenIDs, req.MaxTokens, req.LoraAdapter, req.JSONMode, req.Sampling.Temperature, req.Sampling.TopP, req.Sampling.TopK, req.Sampling.RepetitionPenalty, req.Sampling.StopSequences, req.Constraints.AllowedTokenIDs, req.Constraints.AllowedTokensStrict, req.Constraints.DisallowedTokenIDs, output)
	case domain.GenerationInputTokenIDs:
		done, err = e.GenerateStreamTokensWithSamplingAwaitable(ctx, req.Input.TokenIDs, req.MaxTokens, req.LoraAdapter, req.JSONMode, req.Sampling.Temperature, req.Sampling.TopP, req.Sampling.TopK, req.Sampling.RepetitionPenalty, req.Sampling.StopSequences, req.Constraints.AllowedTokenIDs, req.Constraints.AllowedTokensStrict, req.Constraints.DisallowedTokenIDs, output)
	case domain.GenerationInputText:
		done, err = e.GenerateStreamWithSamplingAwaitable(ctx, req.Input.Text, req.MaxTokens, req.LoraAdapter, req.JSONMode, req.Sampling.Temperature, req.Sampling.TopP, req.Sampling.TopK, req.Sampling.RepetitionPenalty, req.Sampling.StopSequences, req.Constraints.AllowedTokenIDs, req.Constraints.AllowedTokensStrict, req.Constraints.DisallowedTokenIDs, output)
	default:
		err = fmt.Errorf("unknown generation input kind: %d", req.Input.Kind)
	}
	if err != nil {
		cancel()
		release()
		return nil, err
	}
	go func() { <-done; release(); cancel() }()
	return &domain.Generation{Events: output, Done: done, Cancel: cancel}, nil
}

func generationBufferSize() int {
	size, err := strconv.Atoi(os.Getenv("DENSECORE_STREAM_CHANNEL_BUFFER"))
	if err == nil && size > 0 {
		return size
	}
	return 4096
}

func (e *DenseEngine) initializeGenerationPolicy() {
	e.generationPolicyOnce.Do(func() {
		state, stateErr := e.GetRuntimeOptimizationState()
		e.generationExclusive = stateErr == nil && state.PrefixCacheReuseEnabled && state.HybridSSMSnapshotRestoreEnabled
		if stateErr != nil {
			slog.Warn("runtime optimization state unavailable; preserving async admission", "error", stateErr)
		}
	})
}
