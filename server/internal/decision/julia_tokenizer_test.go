package decision

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"testing"
)

func TestJuliaTokenizerContract(t *testing.T) {
	vocab := map[string]int32{"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "<mask>": 4, "▁": 5, "a": 6, "b": 7, "▁a": 8, "▁ab": 9, "e": 10}
	for b := 0; b < 256; b++ {
		vocab[fmt.Sprintf("<0x%02X>", b)] = int32(1000 + b)
	}
	spec := map[string]any{
		"normalizer":    map[string]any{"type": "Replace", "pattern": map[string]string{"String": " "}, "content": "▁"},
		"pre_tokenizer": map[string]any{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true},
		"model":         map[string]any{"type": "BPE", "vocab": vocab, "merges": [][]string{{"▁", "a"}, {"▁a", "b"}}, "byte_fallback": true, "fuse_unk": true, "unk_token": "<unk>"},
		"added_tokens":  []map[string]any{{"id": 4, "content": "<mask>", "lstrip": true, "normalized": false}},
	}
	raw, err := json.Marshal(spec)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := NewJuliaTokenizer(raw)
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		text string
		ids  []int32
	}{
		{"ab", []int32{9}},
		{"ab<mask>ab", []int32{9, 4, 9}},
		{"é", []int32{5, 1195, 1169}},
		{"e\u0301", []int32{5, 10, 1204, 1129}},
	} {
		ids, err := tok.Encode(test.text)
		if err != nil || !reflect.DeepEqual(ids, test.ids) {
			t.Errorf("%q: got%v want%v error%v", test.text, ids, test.ids, err)
		}
	}
	if _, err := tok.Encode(string([]byte{255})); err == nil {
		t.Fatal("invalid UTF8 accepted")
	}
	spec["normalizer"] = map[string]string{"type": "NFKC"}
	raw, _ = json.Marshal(spec)
	if _, err := NewJuliaTokenizer(raw); err == nil {
		t.Fatal("accepted unsupported compatibility normalization")
	}
}

func TestJuliaTokenizerHFParity(t *testing.T) {
	path := os.Getenv("JULIA_TOKENIZER_JSON")
	if path == "" {
		t.Skip("set JULIA_TOKENIZER_JSON for pinned Rust tokenizer differential")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := NewJuliaTokenizer(raw)
	if err != nil {
		t.Fatal(err)
	}
	fixture, err := os.ReadFile("testdata/julia-tokenizer-cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle struct {
		SHA   string `json:"tokenizer_sha256"`
		Cases []struct {
			Text string  `json:"text"`
			IDs  []int32 `json:"ids"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(fixture, &oracle); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(raw)
	if hex.EncodeToString(sum[:]) != oracle.SHA {
		t.Fatal("oracle tokenizer hash mismatch")
	}
	if len(oracle.Cases) < 12 {
		t.Fatal("insufficient independent tokenizer cases")
	}
	for _, test := range oracle.Cases {
		ids, err := tok.Encode(test.Text)
		if err != nil || !reflect.DeepEqual(ids, test.IDs) {
			t.Errorf("%q\ngot %v\nwant%v\nerror%v", test.Text, ids, test.IDs, err)
		}
	}
}

func TestJuliaStrictInputAndRawProbability(t *testing.T) {
	b := fixtureBackend()
	b.cfg.Model = "julia-1"
	b.cfg.StrictEncoding = true
	b.cfg.MaskText = "<mask>"
	q := Question{Type: "choice", QType: 0, Instructions: "route", Options: []string{"a", "b"}, Keys: []string{"a", "b"}, Order: []int{0, 1}}
	if _, err := Build(b, b.cfg, q, []int32{10}, false); err != nil {
		t.Fatal(err)
	}
	if _, err := Build(b, b.cfg, q, make([]int32, b.cfg.MaxLen), false); err == nil {
		t.Fatal("accepted truncated Julia state")
	}
	long := q
	long.Options = []string{strings.Repeat("x", 49), "b"}
	if _, err := Build(b, b.cfg, long, nil, false); err == nil {
		t.Fatal("accepted truncated Julia option")
	}
	long = q
	long.Instructions = strings.Repeat("x", b.cfg.HeadMaxLen)
	if _, err := Build(b, b.cfg, long, nil, false); err == nil {
		t.Fatal("accepted truncated Julia head")
	}
	marker := q
	marker.Instructions = "literal <mask>"
	if _, err := Build(b, b.cfg, marker, nil, false); err == nil {
		t.Fatal("accepted reserved marker")
	}
	b.cfg.Temperature = [3]float64{5, 5, 5}
	b.cfg.Buckets = map[string]float64{"choice:2": 5}
	decoded, err := Decode(b.cfg, q, []float32{8, 0}, []float32{0, 0})
	if err != nil {
		t.Fatal(err)
	}
	if got := decoded["probabilities"].(map[string]float64)["a"]; got != .9997 {
		t.Fatalf("Julia raw probabilities unexpectedly calibrated/display-thresholded: %v", got)
	}
}

func TestJuliaHTTPModelAndStrictState(t *testing.T) {
	b := fixtureBackend()
	b.cfg.Model = "julia-1"
	b.cfg.MaskText = "<mask>"
	b.cfg.StrictEncoding = true
	h := NewHandler(b)
	for _, path := range []string{"/health", "/v1/models"} {
		w := httptest.NewRecorder()
		h.ServeHTTP(w, httptest.NewRequest("GET", path, nil))
		if w.Code != 200 || !strings.Contains(w.Body.String(), "julia-1") {
			t.Fatal(w.Code, w.Body.String())
		}
	}
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/v1/systemone", strings.NewReader(decisionPayload)))
	if w.Code != 200 || !strings.Contains(w.Body.String(), `"model":"julia-1"`) {
		t.Fatal(w.Code, w.Body.String())
	}
	w = httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest("POST", "/v1/systemone", strings.NewReader(strings.Replace(decisionPayload, "hello", "reserved <mask>", 1))))
	if w.Code != 422 {
		t.Fatal(w.Code, w.Body.String())
	}
}

func TestJuliaNamedDescriptionsAreSharedAndNonmutating(t *testing.T) {
	raw := []byte(`{"state":"hello","questions":{"choice":{"type":"choice","instructions":"route","criteria":{"billing_id":"Billing","tech_id":"Technical support"}},"score":{"type":"score","instructions":"urgency","criteria":["Low","High"]},"noul":{"type":"noul","instructions":"safe","criteria":{"false":"Unsafe","true":"Safe"}},"default":{"type":"noul","instructions":"safe"}}}`)
	req, err := Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	b := &batchFixture{fakeBackend: fixtureBackend()}
	b.cfg.Model = "julia-1"
	b.cfg.StrictEncoding = true
	b.cfg.MaskText = "<mask>"
	want := [][]string{{"Billing", "Technical support"}, {"Low", "High"}, {"Unsafe", "Safe"}, {"false", "true"}}
	original, err := Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	for i, q := range req.Questions {
		prepared, err := PrepareQuestion(b.cfg, q)
		if err != nil {
			t.Fatal(err)
		}
		twice, err := PrepareQuestion(b.cfg, prepared)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(prepared.Options, want[i]) || !reflect.DeepEqual(twice, prepared) {
			t.Fatalf("question %d: %+v", i, prepared)
		}
		direct, err := Build(b, b.cfg, q, nil, false)
		if err != nil {
			t.Fatal(err)
		}
		prepped, err := Build(b, b.cfg, prepared, nil, false)
		if err != nil || !reflect.DeepEqual(direct, prepped) {
			t.Fatal("direct/shared preparation differs", err)
		}
	}
	first, err := Predict(context.Background(), b, req)
	if err != nil {
		t.Fatal(err)
	}
	second, err := Predict(context.Background(), b, req)
	if err != nil {
		t.Fatal(err)
	}
	batch, err := PredictBatch(context.Background(), b, []Request{req, req})
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(req, original) || !reflect.DeepEqual(first, second) || !reflect.DeepEqual(first, batch[0]) || !reflect.DeepEqual(first, batch[1]) {
		t.Fatal("reused request was mutated or produced different result")
	}
	laya := fixtureBackend().cfg
	untouched, err := PrepareQuestion(laya, req.Questions[0])
	if err != nil || !reflect.DeepEqual(untouched, original.Questions[0]) {
		t.Fatal("Laya labels changed")
	}
}

type countedInference struct {
	*fakeBackend
	calls int
}

func (b *countedInference) Infer(ids, markers []int32, q int) ([]float32, []float32, error) {
	b.calls++
	return b.fakeBackend.Infer(ids, markers, q)
}
func TestJuliaNativeContextAdmissionBeforeInference(t *testing.T) {
	b := &countedInference{fakeBackend: fixtureBackend()}
	b.cfg.Model = "julia-1"
	b.cfg.MaskText = "<mask>"
	b.cfg.StrictEncoding = true
	b.cfg.ModelMaxLen = 8192
	b.cfg.MaxLen = 128
	h := NewHandler(b)
	w := httptest.NewRecorder()
	raw := strings.Replace(decisionPayload, "hello", strings.Repeat("x", 256), 1)
	h.ServeHTTP(w, httptest.NewRequest("POST", "/v1/systemone", strings.NewReader(raw)))
	if w.Code != 422 || b.calls != 0 || b.cfg.ModelMaxLen != 8192 || !strings.Contains(w.Body.String(), "effective limit 128") {
		t.Fatal(w.Code, b.calls, w.Body.String())
	}
}

type juliaTokenizerBackend struct {
	*fakeBackend
	tokenizer *Tokenizer
}

func (b *juliaTokenizerBackend) Tokenize(text string) ([]int32, error) {
	return b.tokenizer.Encode(text)
}
func TestJuliaSequenceAndNamedHFParity(t *testing.T) {
	path := os.Getenv("JULIA_TOKENIZER_JSON")
	if path == "" {
		t.Skip("set JULIA_TOKENIZER_JSON for full Julia preprocessing differential")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := NewJuliaTokenizer(raw)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("testdata/julia-sequence-cases.json")
	if err != nil {
		t.Fatal(err)
	}
	type row struct {
		State    json.RawMessage `json:"state"`
		Question string          `json:"question"`
		Options  []string        `json:"options"`
		Type     string          `json:"type"`
	}
	type expected struct {
		IDs, Markers []int32
		QType        int  `json:"qtype"`
		Truncated    bool `json:"truncated"`
	}
	var fixture struct {
		SHA       string `json:"tokenizer_sha256"`
		Sequences []struct {
			Row      row      `json:"row"`
			MaxLen   int      `json:"max_length"`
			HeadLen  int      `json:"head_length"`
			Expected expected `json:"expected"`
			Error    string   `json:"error"`
		} `json:"sequence_cases"`
		Named []struct {
			Request  json.RawMessage `json:"request"`
			MaxLen   int             `json:"max_length"`
			HeadLen  int             `json:"head_length"`
			Rows     []row           `json:"rows"`
			Expected []expected      `json:"expected"`
			Error    string          `json:"error"`
		} `json:"named_cases"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	hash := sha256.Sum256(raw)
	if hex.EncodeToString(hash[:]) != fixture.SHA {
		t.Fatal("sequence oracle tokenizer hash mismatch")
	}
	if len(fixture.Sequences) < 12 || len(fixture.Named) < 12 {
		t.Fatal("insufficient full sequence/named request cases")
	}
	backend := &juliaTokenizerBackend{fakeBackend: fixtureBackend(), tokenizer: tok}
	configure := func(maxLen, headLen int) {
		backend.cfg = Config{Model: "julia-1", StrictEncoding: true, ModelMaxLen: 8192, MaxLen: maxLen, HeadMaxLen: headLen, CLS: 2, SEP: 1, Mask: 4, MaskText: "<mask>", Temperature: [3]float64{1, 1, 1}}
	}
	check := func(t *testing.T, seq Sequence, qtype int, want expected) {
		t.Helper()
		if !reflect.DeepEqual(seq.IDs, want.IDs) || !reflect.DeepEqual(seq.Markers, want.Markers) || qtype != want.QType || (seq.StateUsed < seq.StateTokens) != want.Truncated {
			t.Fatalf("sequence differs\nids %v\nwant%v\nmarkers%v want%v", seq.IDs, want.IDs, seq.Markers, want.Markers)
		}
	}
	for i, test := range fixture.Sequences {
		t.Run(fmt.Sprintf("sequence_%d", i), func(t *testing.T) {
			configure(test.MaxLen, test.HeadLen)
			state, err := parseJSON(test.Row.State)
			if err != nil {
				t.Fatal(err)
			}
			kind := test.Row.Type
			if kind == "" {
				kind = "choice"
			}
			q := Question{ID: "q", Type: kind, Instructions: test.Row.Question, Options: test.Row.Options, QType: map[string]int{"choice": 0, "score": 1, "noul": 2}[kind]}
			for i := range q.Options {
				q.Order = append(q.Order, i)
			}
			prepared, err := prepare(context.Background(), backend, Request{State: state, Questions: []Question{q}})
			if test.Error != "" {
				if err == nil {
					t.Fatal("accepted upstream rejected sequence")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			check(t, prepared.sequences[0], q.QType, test.Expected)
		})
	}
	for i, test := range fixture.Named {
		t.Run(fmt.Sprintf("named_http_%d", i), func(t *testing.T) {
			configure(test.MaxLen, test.HeadLen)
			req, err := Parse(test.Request)
			var prepared preparedRequest
			if err == nil {
				prepared, err = prepare(context.Background(), backend, req)
			}
			if test.Error != "" {
				if err == nil {
					t.Fatal("accepted upstream rejected named request")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if len(prepared.sequences) != len(test.Expected) {
				t.Fatal("named question count differs")
			}
			for i, seq := range prepared.sequences {
				if !reflect.DeepEqual(prepared.request.Questions[i].Options, test.Rows[i].Options) {
					t.Fatalf("named descriptions differ: %v want %v", prepared.request.Questions[i].Options, test.Rows[i].Options)
				}
				check(t, seq, prepared.request.Questions[i].QType, test.Expected[i])
			}
		})
	}
}
