package decision

import (
	"context"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

const decisionPayload = `{"state":"hello","questions":{"safe":{"type":"noul","instructions":"safe?"}}}`

type blockingBackend struct {
	*fakeBackend
	entered chan struct{}
	release chan struct{}
	mu      sync.Mutex
	calls   int
}

func (b *blockingBackend) Infer(ids, markers []int32, q int) ([]float32, []float32, error) {
	b.mu.Lock()
	b.calls++
	b.mu.Unlock()
	b.entered <- struct{}{}
	<-b.release
	return b.fakeBackend.Infer(ids, markers, q)
}
func waitSlots(t *testing.T, h *Handler, n int) {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for len(h.busy) != n {
		if time.Now().After(deadline) {
			t.Fatalf("admission slots %d, want %d", len(h.busy), n)
		}
		time.Sleep(time.Millisecond)
	}
}
func sendDecision(h *Handler, ctx context.Context) <-chan int {
	done := make(chan int, 1)
	go func() {
		w := httptest.NewRecorder()
		r := httptest.NewRequest("POST", "/v1/systemone", strings.NewReader(decisionPayload)).WithContext(ctx)
		h.ServeHTTP(w, r)
		done <- w.Code
	}()
	return done
}
func TestQueueCancellationOverloadAndDrain(t *testing.T) {
	b := &blockingBackend{fakeBackend: fixtureBackend(), entered: make(chan struct{}, 2), release: make(chan struct{})}
	h, err := NewHandlerWithOptions(b, ServeOptions{QueueCapacity: 1})
	if err != nil {
		t.Fatal(err)
	}
	first := sendDecision(h, context.Background())
	<-b.entered
	ctx, cancel := context.WithCancel(context.Background())
	second := sendDecision(h, ctx)
	waitSlots(t, h, 2)
	if code := <-sendDecision(h, context.Background()); code != 429 {
		t.Fatal(code)
	}
	cancel()
	if code := <-second; code != 408 {
		t.Fatal(code)
	}
	h.Drain()
	if code := <-sendDecision(h, context.Background()); code != 503 {
		t.Fatal(code)
	}
	close(b.release)
	if code := <-first; code != 200 {
		t.Fatal(code)
	}
	waitCtx, stop := context.WithTimeout(context.Background(), 3*time.Second)
	defer stop()
	if err := h.Wait(waitCtx); err != nil {
		t.Fatal(err)
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.calls != 1 {
		t.Fatalf("canceled queued request executed: %d calls", b.calls)
	}
}

type batchFixture struct {
	*fakeBackend
	sizes []int
}

func (b *batchFixture) InferBatch(inputs []BatchInput) ([]BatchOutput, error) {
	b.sizes = append(b.sizes, len(inputs))
	out := make([]BatchOutput, len(inputs))
	for i, input := range inputs {
		out[i].Logits, out[i].Action, _ = b.fakeBackend.Infer(input.IDs, input.Markers, input.QType)
	}
	return out, nil
}
func TestNativeBatchRoutingAndQuestionChunkLimit(t *testing.T) {
	b := &batchFixture{fakeBackend: fixtureBackend()}
	req, err := Parse([]byte(decisionPayload))
	if err != nil {
		t.Fatal(err)
	}
	for len(req.Questions) < 32 {
		req.Questions = append(req.Questions, req.Questions[0])
	}
	results, err := PredictBatch(context.Background(), b, []Request{req, req, req})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(b.sizes, []int{64, 32}) {
		t.Fatal(b.sizes)
	}
	expected, err := Predict(context.Background(), b, req)
	if err != nil {
		t.Fatal(err)
	}
	for _, result := range results {
		if !reflect.DeepEqual(result, expected) {
			t.Fatalf("batch differs: %+v", result)
		}
	}
}
func TestBatchAdmissionRejectsUnsupportedBackend(t *testing.T) {
	if _, err := NewHandlerWithOptions(fixtureBackend(), ServeOptions{BatchSize: 2}); err == nil {
		t.Fatal("accepted serial backend as native batch")
	}
}
func TestBatchSchedulerCollectsAndDrains(t *testing.T) {
	b := &batchFixture{fakeBackend: fixtureBackend()}
	h, err := NewHandlerWithOptions(b, ServeOptions{BatchSize: 2, BatchWait: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	first := sendDecision(h, context.Background())
	waitSlots(t, h, 1)
	second := sendDecision(h, context.Background())
	if code := <-first; code != 200 {
		t.Fatal(code)
	}
	if code := <-second; code != 200 {
		t.Fatal(code)
	}
	h.Drain()
	if err := h.Wait(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(b.sizes, []int{2}) {
		t.Fatal(b.sizes)
	}
}

func TestDrainFlushesPartiallyCollectedBatch(t *testing.T) {
	b := &batchFixture{fakeBackend: fixtureBackend()}
	h, err := NewHandlerWithOptions(b, ServeOptions{BatchSize: 2, BatchWait: time.Second})
	if err != nil {
		t.Fatal(err)
	}
	req, err := Parse([]byte(decisionPayload))
	if err != nil {
		t.Fatal(err)
	}
	result := make(chan decisionResult, 1)
	h.active.Add(1)
	h.busy <- struct{}{}
	h.scheduler.jobs <- decisionJob{ctx: context.Background(), req: req, result: result}
	h.Drain()
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	if err := h.Wait(ctx); err != nil {
		t.Fatalf("drain waited for batch collection timer: %v", err)
	}
	if output := <-result; output.err != nil {
		t.Fatal(output.err)
	}
	if !reflect.DeepEqual(b.sizes, []int{1}) {
		t.Fatal(b.sizes)
	}
}
func TestPrecisionOptions(t *testing.T) {
	for _, precision := range []string{"", "fp32", "q8_0"} {
		if _, err := (OpenOptions{Precision: precision}).precision(); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := (OpenOptions{Precision: "fp16"}).precision(); err == nil {
		t.Fatal("accepted unavailable precision")
	}
}

func TestCanceledActiveInferenceStillDrains(t *testing.T) {
	b := &blockingBackend{fakeBackend: fixtureBackend(), entered: make(chan struct{}, 1), release: make(chan struct{})}
	h, err := NewHandlerWithOptions(b, ServeOptions{QueueCapacity: 1})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	request := sendDecision(h, ctx)
	<-b.entered
	cancel()
	if code := <-request; code != 408 {
		t.Fatal(code)
	}
	h.Drain()
	deadline, stop := context.WithTimeout(context.Background(), 10*time.Millisecond)
	if err := h.Wait(deadline); err != context.DeadlineExceeded {
		t.Fatalf("native call escaped drain: %v", err)
	}
	stop()
	close(b.release)
	deadline, stop = context.WithTimeout(context.Background(), time.Second)
	defer stop()
	if err := h.Wait(deadline); err != nil {
		t.Fatal(err)
	}
}

type blockedResponseWriter struct {
	*httptest.ResponseRecorder
	started chan struct{}
	release chan struct{}
	once    sync.Once
}

func (w *blockedResponseWriter) Write(p []byte) (int, error) {
	w.once.Do(func() { close(w.started) })
	<-w.release
	return w.ResponseRecorder.Write(p)
}
func TestDefaultNativeDrainDoesNotWaitForResponseWriter(t *testing.T) {
	for _, test := range []struct {
		name, payload string
		status, calls int
	}{
		{"success", decisionPayload, 200, 1},
		{"invalid_request", `{"state":"hello","questions":{"q":{"type":"unsupported","instructions":"safe?"}}}`, 422, 0},
		{"oversized_body", strings.Repeat(" ", MaxRequestBytes+1), 413, 0},
	} {
		t.Run(test.name, func(t *testing.T) {
			backend := &countedInference{fakeBackend: fixtureBackend()}
			h := NewHandler(backend)
			writer := &blockedResponseWriter{ResponseRecorder: httptest.NewRecorder(), started: make(chan struct{}), release: make(chan struct{})}
			var release sync.Once
			defer release.Do(func() { close(writer.release) })
			completed := make(chan struct{})
			go func() {
				h.ServeHTTP(writer, httptest.NewRequest("POST", "/v1/systemone", strings.NewReader(test.payload)))
				close(completed)
			}()
			select {
			case <-writer.started:
			case <-time.After(time.Second):
				t.Fatal("response writer not reached")
			}
			second := httptest.NewRecorder()
			h.ServeHTTP(second, httptest.NewRequest("POST", "/v1/systemone", strings.NewReader(decisionPayload)))
			if second.Code != 429 {
				t.Fatalf("default admission expanded while writer blocked: %d", second.Code)
			}
			h.Drain()
			ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
			defer cancel()
			if err := h.Wait(ctx); err != nil {
				t.Fatalf("finished admitted work still waits for HTTP writer: %v", err)
			}
			if len(h.busy) != 1 {
				t.Fatal("default admission slot released before response completion")
			}
			select {
			case <-completed:
				t.Fatal("writer unexpectedly completed before release")
			default:
			}
			release.Do(func() { close(writer.release) })
			select {
			case <-completed:
			case <-time.After(time.Second):
				t.Fatal("handler did not complete")
			}
			if writer.Code != test.status || backend.calls != test.calls {
				t.Fatal(writer.Code, backend.calls)
			}
			if len(h.busy) != 0 {
				t.Fatal("default admission slot retained after response completion")
			}
		})
	}
}
