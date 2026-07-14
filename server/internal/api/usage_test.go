package api

import (
	"context"
	"testing"
)

type recordingUsageRecorder struct {
	prompt     int
	completion int
}

func (r *recordingUsageRecorder) RecordInferenceUsage(_ context.Context, prompt, completion int) error {
	r.prompt = prompt
	r.completion = completion
	return nil
}

func TestInferenceUsageRecorderReceivesActualCounts(t *testing.T) {
	recorder := &recordingUsageRecorder{}
	handler := &Handler{usageRecorder: recorder}
	if err := handler.recordInferenceUsage(context.Background(), 12, 7); err != nil {
		t.Fatalf("recordInferenceUsage: %v", err)
	}
	if recorder.prompt != 12 || recorder.completion != 7 {
		t.Fatalf("usage = (%d, %d), want (12, 7)", recorder.prompt, recorder.completion)
	}
}
