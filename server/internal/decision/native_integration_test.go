//go:build cgo

package decision

import (
	"encoding/json"
	"math"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"testing"
)

// Opt in with DENSECORE_LAYA_MODEL=/absolute/path/to/laya-f16.gguf. The checked-in
// oracle is generated independently by pinned upstream PyTorch, not this runtime.
func TestNativeUpstreamParity(t *testing.T) {
	model := os.Getenv("DENSECORE_LAYA_MODEL")
	if model == "" {
		t.Skip("set DENSECORE_LAYA_MODEL to run native upstream parity")
	}
	raw, err := os.ReadFile("../../../benchmarks/fixtures/laya/parity-inputs.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Cases []struct {
			Name          string          `json:"name"`
			State         json.RawMessage `json:"state"`
			Question      json.RawMessage `json:"question"`
			IDs           []int32         `json:"token_ids"`
			Markers       []int32         `json:"markers"`
			Logits        []float64       `json:"reference_logits"`
			Act           []float64       `json:"reference_act_logits"`
			Probabilities []float64       `json:"reference_probabilities"`
			Temperature   float64         `json:"temperature"`
		}
	}
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) == 0 {
		t.Fatal("empty oracle fixture")
	}
	b, err := Open(model, 4)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			value, err := parseJSON(tc.Question)
			if err != nil {
				t.Fatal(err)
			}
			qobj := value.(object)
			for i := range qobj {
				switch qobj[i].key {
				case "t":
					qobj[i].key = "type"
				case "ins":
					qobj[i].key = "instructions"
				case "crit":
					qobj[i].key = "criteria"
				}
			}
			request := `{"state":` + string(tc.State) + `,"questions":{"q":` + pythonJSON(qobj) + `}}`
			req, err := Parse([]byte(request))
			if err != nil {
				t.Fatal(err)
			}
			q := req.Questions[0]
			state, err := b.Tokenize(strings.ReplaceAll(render(req.State), b.Config().MaskText, " "))
			if err != nil {
				t.Fatal(err)
			}
			_, left := req.State.([]any)
			seq, err := Build(b, b.Config(), q, state, left)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(seq.IDs, tc.IDs) || !reflect.DeepEqual(seq.Markers, tc.Markers) {
				t.Fatalf("Go sequence differs from upstream: ids=%v markers=%v", seq.IDs, seq.Markers)
			}
			logits, act, err := b.Infer(seq.IDs, seq.Markers, q.QType)
			if err != nil {
				t.Fatal(err)
			}
			if len(logits) != len(tc.Logits) || len(act) != len(tc.Act) {
				t.Fatal("output shape mismatch")
			}
			for i, v := range logits {
				if d := math.Abs(float64(v) - tc.Logits[i]); d > 1e-3 {
					t.Errorf("logit[%d] error %g", i, d)
				}
			}
			for i, v := range act {
				if d := math.Abs(float64(v) - tc.Act[i]); d > 1e-3+1e-6*math.Abs(tc.Act[i]) {
					t.Errorf("act[%d] error %g", i, d)
				}
			}
			probs, err := softmax(logits, tc.Temperature)
			if err != nil {
				t.Fatal(err)
			}
			for i, v := range probs {
				if d := math.Abs(v - tc.Probabilities[i]); d > 1e-4 {
					t.Errorf("probability[%d] error %g", i, d)
				}
			}
			decoded, err := Decode(b.Config(), q, logits, act)
			if err != nil {
				t.Fatal(err)
			}
			if q.Type == "noul" {
				if math.Abs(decoded["noul"].(float64)-tc.Probabilities[1]) > 1.5e-4 {
					t.Fatal("noul probability differs", decoded)
				}
			} else {
				p := decoded["probabilities"].(map[string]float64)
				for i, key := range q.Keys {
					if math.Abs(p[key]-tc.Probabilities[i]) > 1.5e-4 {
						t.Fatal("decoded probability differs", key, p[key], tc.Probabilities[i])
					}
				}
			}
			if tc.Name == "choice" {
				w := httptest.NewRecorder()
				NewHandler(b).ServeHTTP(w, httptest.NewRequest("POST", "/v1/systemone", strings.NewReader(request)))
				if w.Code != 200 {
					t.Fatal(w.Code, w.Body.String())
				}
				var response struct {
					Answers map[string]struct {
						Choice     string  `json:"choice"`
						Confidence float64 `json:"answer_confidence"`
					}
				}
				if err = json.Unmarshal(w.Body.Bytes(), &response); err != nil {
					t.Fatal(err)
				}
				if response.Answers["q"].Choice != "billing" || math.Abs(response.Answers["q"].Confidence-tc.Probabilities[0]) > 1.5e-4 {
					t.Fatal(w.Body.String())
				}
			}
		})
	}
}
