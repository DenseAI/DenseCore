package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/DenseAI/DenseCore/server/internal/decision"
)

type fixtureBackend struct {
	cfg         decision.Config
	nativeDelay time.Duration
	fail        bool
	calls       int
}

func (b *fixtureBackend) Config() decision.Config { return b.cfg }
func (b *fixtureBackend) Close()                  {}
func (b *fixtureBackend) Tokenize(s string) ([]int32, error) {
	var ids []int32
	for _, r := range s {
		if r != ' ' {
			ids = append(ids, int32(r)%80+10)
		}
	}
	return ids, nil
}
func (b *fixtureBackend) Infer(ids, markers []int32, q int) ([]float32, []float32, error) {
	out, e := b.InferBatch([]decision.BatchInput{{IDs: ids, Markers: markers, QType: q}})
	if e != nil {
		return nil, nil, e
	}
	return out[0].Logits, out[0].Action, nil
}
func (b *fixtureBackend) InferBatch(in []decision.BatchInput) ([]decision.BatchOutput, error) {
	b.calls++
	time.Sleep(b.nativeDelay)
	if b.fail {
		return nil, fmt.Errorf("injected failure")
	}
	out := make([]decision.BatchOutput, len(in))
	for i, v := range in {
		logits := make([]float32, len(v.Markers))
		for j := range logits {
			logits[j] = float32(j)
		}
		out[i] = decision.BatchOutput{Logits: logits, Action: []float32{1, 0}}
	}
	return out, nil
}
func fixture(t *testing.T) (*fixtureBackend, requestCase) {
	t.Helper()
	b := &fixtureBackend{cfg: decision.Config{MaxLen: 128, HeadMaxLen: 96, CLS: 100, SEP: 101, Mask: 102, MaskText: "[MASK]", Temperature: [3]float64{1, 1, 1}}}
	raw := json.RawMessage(`{"state":"news","questions":{"q":{"type":"choice","instructions":"Choose.","criteria":{"a":"one","b":"two"}}}}`)
	req, e := decision.Parse(raw)
	if e != nil {
		t.Fatal(e)
	}
	st, _ := b.Tokenize("news")
	seq, e := decision.Build(b, b.cfg, req.Questions[0], st, false)
	if e != nil {
		t.Fatal(e)
	}
	keys := make([]*string, len(seq.IDs))
	mask := make([]int, len(seq.IDs))
	for i := range keys {
		key := fmt.Sprintf("key-%d", i)
		keys[i] = &key
		mask[i] = 1
	}
	v := false
	return b, requestCase{ID: "case", Raw: raw, Request: req, Truncated: &v, Sequences: []sequence{{IDs: seq.IDs, Keys: keys, Mask: mask, Markers: seq.Markers, QType: req.Questions[0].QType}}, Useful: len(keys), Compute: len(keys)}
}
func TestPipelineIncludesPreparationNativeAndFormattedAnswers(t *testing.T) {
	b, c := fixture(t)
	b.nativeDelay = time.Millisecond
	measured := &measuredBackend{Backend: b}
	out, elapsed, e := execute(context.Background(), measured, []requestCase{c, c})
	if e != nil {
		t.Fatal(e)
	}
	if b.calls != 1 || len(measured.Inputs) != 2 {
		t.Fatal("independent requests must use one native batch")
	}
	if elapsed < measured.InferTime+measured.TokenizeTime || measured.TokenizeTime <= 0 || measured.InferTime < time.Millisecond {
		t.Fatal("pipeline does not cover tokenizer and native execution")
	}
	if len(out) != 2 || out[0]["answers"] == nil {
		t.Fatal("timer must end after calibrated answer map formatting")
	}
	if e = checkReceipts([]requestCase{c, c}, append(measured.Inputs, measured.Inputs[0]), out); e == nil || !strings.Contains(e.Error(), "repeats") {
		t.Fatal("extra internal encoder repeat accepted")
	}
	// Exact token and marker receipts are proof checks, not assumptions from length.
	c.Sequences[0].IDs = append([]int32(nil), c.Sequences[0].IDs...)
	c.Sequences[0].IDs[1]++
	if e = checkReceipts([]requestCase{c, c}, measured.Inputs, out); e == nil || !strings.Contains(e.Error(), "differs") {
		t.Fatal("changed same-length token receipt accepted")
	}
}
func TestPipelineFailureRetainsAllRequestsAndDrain(t *testing.T) {
	b, c := fixture(t)
	b.nativeDelay = 30 * time.Millisecond
	b.fail = true
	records, meta := measure(context.Background(), &measuredBackend{Backend: b}, []requestCase{c}, 4, 10*time.Millisecond, "run")
	if meta["successful_requests"] != 0 || meta["useful_tokens"] != 0 || meta["error_rate"] != float64(1) {
		t.Fatal("failed requests entered throughput numerator")
	}
	if len(records) != 4 || meta["elapsed_s"].(float64) < .03 {
		t.Fatal("failure/drain omitted")
	}
	for i, r := range records {
		if r.Success || r.Error == nil || r.Attempt != i || r.End < .03 || r.Useful != c.Useful {
			t.Fatal("failed request record missing")
		}
	}
}
func TestWorkloadRejectsPaddingAndUsefulKeyDuplication(t *testing.T) {
	_, c := fixture(t)
	w := workload{Schema: "systemone-workload-v1", Requests: []requestCase{c}}
	path := filepath.Join(t.TempDir(), "workload.json")
	write := func() {
		raw, e := json.Marshal(w)
		if e != nil {
			t.Fatal(e)
		}
		if e = os.WriteFile(path, raw, 0600); e != nil {
			t.Fatal(e)
		}
	}
	write()
	if _, e := loadWorkload(path); e != nil {
		t.Fatal(e)
	}
	w.Requests[0].Sequences[0].Mask[1] = 0
	write()
	if _, e := loadWorkload(path); e == nil {
		t.Fatal("masked padding accepted by no-mask native pipeline")
	}
	w.Requests[0].Sequences[0].Mask[1] = 1
	w.Requests[0].Sequences[0].Keys[1] = w.Requests[0].Sequences[0].Keys[0]
	write()
	if _, e := loadWorkload(path); e == nil {
		t.Fatal("duplicate useful token key accepted")
	}
}

func TestPipelineCompositeCanonicalIdentity(t *testing.T) {
	got := pipelineImplementation(strings.Repeat("1", 64), strings.Repeat("2", 64))
	if got != "aaae6b3a041b174b3798064caf55bbb7d5f4629b2d50e277a5a8fe55ff62b179" {
		t.Fatal("canonical implementation digest differs from Python contract")
	}
	if got == pipelineImplementation(strings.Repeat("3", 64), strings.Repeat("2", 64)) {
		t.Fatal("Go binary change did not change pipeline identity")
	}
}
