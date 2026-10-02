package decision

import (
	"context"
	"math"
	"net/http/httptest"
	"strings"
	"testing"
)

type fakeBackend struct{ cfg Config }

func (f *fakeBackend) Config() Config { return f.cfg }
func (f *fakeBackend) Close()         {}
func (f *fakeBackend) Tokenize(s string) ([]int32, error) {
	ids := []int32{}
	for _, r := range s {
		ids = append(ids, int32(r))
	}
	return ids, nil
}
func (f *fakeBackend) Infer(ids, markers []int32, q int) ([]float32, []float32, error) {
	v := make([]float32, len(markers))
	v[0] = 2
	return v, []float32{0, 0}, nil
}
func fixtureBackend() *fakeBackend {
	return &fakeBackend{Config{MaxLen: 256, HeadMaxLen: 96, CLS: 1, SEP: 2, Mask: 3, MaskText: "[MASK]", Temperature: [3]float64{1, 1, 1}}}
}
func TestOrderedCriteriaAndState(t *testing.T) {
	r, e := Parse([]byte(`{"state":{"z":"한국<&","a":[1,true]},"questions":{"q":{"type":"choice","instructions":"route","criteria":{"z":"last","a":"first"}}}}`))
	if e != nil {
		t.Fatal(e)
	}
	if got := render(r.State); got != `{"z": "한국<&", "a": [1, true]}` {
		t.Fatal(got)
	}
	q := r.Questions[0]
	if strings.Join(q.Keys, ",") != "z,a" {
		t.Fatal(q.Keys)
	}
	b := fixtureBackend()
	s, e := Build(b, b.cfg, q, []int32{1000}, false)
	if e != nil {
		t.Fatal(e)
	}
	var text strings.Builder
	for _, v := range s.IDs[int(s.Markers[0])+1 : int(s.Markers[1])] {
		text.WriteRune(rune(v))
	}
	if text.String() != " z: last" {
		t.Fatal(text.String())
	}
}
func TestRequestValidation(t *testing.T) {
	for _, raw := range []string{
		`{"state":"x","questions":{"q":{"type":"bad","instructions":"x"}}}`,
		`{"state":"x","questions":{"q":{"type":"choice","instructions":"x","criteria":[]}}}`,
		`{"state":"x","questions":{"q":{"type":"choice","instructions":"x","criteria":{"a":null,"a":null}}}}`,
		`{"state":"x","questions":{"q":{"type":"choice","instructions":"x","criteria":["a","b"],"option_order":[0,0]}}}`,
		`{"state":"x","questions":{"q":{"type":"noul","instructions":"x","criteria":{"yes":"x"}}}}`,
		`{"state":"x","questions":{"q":{"type":"score","instructions":"x","criteria":[null]}}}`,
		`{"state":"x","questions":{},"lang":"ko"}`, `{} {}`,
	} {
		if _, e := Parse([]byte(raw)); e == nil {
			t.Fatalf("accepted %s", raw)
		}
	}
}
func TestSequenceTruncationAndMask(t *testing.T) {
	b := fixtureBackend()
	b.cfg.MaxLen = 48
	b.cfg.HeadMaxLen = 24
	q := Question{Type: "choice", Instructions: "[MASK]", Options: []string{"a"}, Order: []int{0}}
	state := make([]int32, 100)
	for i := range state {
		state[i] = int32(1000 + i)
	}
	right, e := Build(b, b.cfg, q, state, false)
	if e != nil {
		t.Fatal(e)
	}
	left, e := Build(b, b.cfg, q, state, true)
	if e != nil {
		t.Fatal(e)
	}
	if len(right.IDs) != 48 || right.IDs[46] != state[right.StateUsed-1] || left.IDs[46] != 1099 {
		t.Fatalf("bad truncation: %+v %+v", right, left)
	}
	masks := 0
	for _, v := range right.IDs {
		if v == 3 {
			masks++
		}
	}
	if masks != 1 {
		t.Fatal(masks)
	}
}
func TestDecodePermutationAndTemperature(t *testing.T) {
	c := fixtureBackend().cfg
	c.Buckets = map[string]float64{"choice:2": 2}
	q := Question{Type: "choice", Keys: []string{"a", "b"}, Options: []string{"a", "b"}, Order: []int{1, 0}}
	a, e := Decode(c, q, []float32{2, 0}, []float32{0, 0})
	if e != nil {
		t.Fatal(e)
	}
	if a["choice"] != "b" || a["answer_confidence"] != .7311 {
		t.Fatal(a)
	}
	p := a["probabilities"].(map[string]float64)
	if math.Abs(p["a"]+p["b"]-1) > 1e-4 {
		t.Fatal(p)
	}
	q.Type = "score"
	q.QType = 1
	q.Keys = []string{"0", "1"}
	q.Legend = []string{"low", "high"}
	a, e = Decode(c, q, []float32{0, 0}, []float32{0, 0})
	if e != nil || a["score"] != .5 {
		t.Fatal(a, e)
	}
	q.Type = "noul"
	q.QType = 2
	a, e = Decode(c, q, []float32{0, 0}, []float32{0, 0})
	if e != nil || a["noul"] != .5 || a["confidence"] != .5 {
		t.Fatal(a, e)
	}
}
func TestHTTPDecisionAndAdmission(t *testing.T) {
	h := NewHandler(fixtureBackend())
	payload := `{"state":"hello","questions":{"safe":{"type":"noul","instructions":"safe?"}}}`
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/v1/systemone", strings.NewReader(payload)))
	if w.Code != 200 || !strings.Contains(w.Body.String(), `"noul":0.1192`) {
		t.Fatal(w.Code, w.Body.String())
	}
	h.busy <- struct{}{}
	w = httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/v1/systemone", strings.NewReader(payload)))
	if w.Code != 429 {
		t.Fatal(w.Code)
	}
	<-h.busy
	h.Drain()
	w = httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("GET", "/health", nil))
	if w.Code != 503 {
		t.Fatal(w.Code)
	}
}
func TestHTTPBodyLimitAndCancellation(t *testing.T) {
	h := NewHandler(fixtureBackend())
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/v1/systemone", strings.NewReader(strings.Repeat(" ", MaxRequestBytes+1))))
	if w.Code != 413 {
		t.Fatal(w.Code)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	r, e := Parse([]byte(`{"state":"x","questions":{"q":{"type":"noul","instructions":"x"}}}`))
	if e != nil {
		t.Fatal(e)
	}
	if _, e = Predict(ctx, h.backend, r); e != context.Canceled {
		t.Fatal(e)
	}
}

func TestPythonJSONNumberAndUnicodeSerialization(t *testing.T) {
	v, err := parseJSON([]byte("{\"n\":1e2,\"small\":1e-5,\"float\":2.0,\"text\":\"\\u2028<&\\n\"}"))
	if err != nil {
		t.Fatal(err)
	}
	want := "{\"n\": 100.0, \"small\": 1e-05, \"float\": 2.0, \"text\": \"\u2028<&\\n\"}"
	if got := pythonJSON(v); got != want {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestJSONFiniteNumbers(t *testing.T) {
	v, err := parseJSON([]byte(`[-0, -0.0]`))
	if err != nil {
		t.Fatal(err)
	}
	if got := pythonJSON(v); got != `[0, -0.0]` {
		t.Fatal(got)
	}
	if _, err := parseJSON([]byte(`{"n":1e999}`)); err == nil {
		t.Fatal("accepted overflowing float")
	}
}
