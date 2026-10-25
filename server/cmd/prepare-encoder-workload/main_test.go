package main

import (
	"encoding/json"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/DenseAI/DenseCore/server/internal/decision"
)

func TestRealTokenizerAccounting(t *testing.T) {
	path := os.Getenv("LAYA_TOKENIZER_JSON")
	if path == "" {
		t.Skip("real model tokenizer required")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := decision.NewTokenizer(raw)
	if err != nil {
		t.Fatal(err)
	}
	cfg := decision.Config{MaxLen: 512, HeadMaxLen: 192, MaskText: "[MASK]"}
	for name, target := range map[string]*int32{"[CLS]": &cfg.CLS, "[SEP]": &cfg.SEP, "[MASK]": &cfg.Mask} {
		id, ok := tok.TokenID(name)
		if !ok {
			t.Fatal(name)
		}
		*target = id
	}
	for _, tc := range []struct {
		name, state, option string
		truncated           bool
	}{
		{"short", "An invoice needs review.", "billing", false},
		{"literal_mask", "An [MASK] invoice needs [MASK] review.", "billing", false},
		{"long_state", strings.Repeat("A report arrived. ", 1000), "billing", true},
		{"long_option", "An invoice needs review.", strings.Repeat("billing ", 100), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			request := map[string]any{"state": tc.state, "questions": map[string]any{"q": map[string]any{"type": "choice", "instructions": "Choose category.", "criteria": []string{tc.option, "technical", "world", "sports"}}}}
			encoded, _ := json.Marshal(request)
			row := map[string]json.RawMessage{"id": json.RawMessage(`"example"`), "request": encoded}
			out, err := receipt(row, tokenBackend{tok, cfg})
			if err != nil {
				t.Fatal(err)
			}
			if out["truncated"] != tc.truncated {
				t.Fatalf("truncated=%v", out["truncated"])
			}
			ids := out["token_ids"].([]int32)
			parsed, err := decision.Parse(encoded)
			if err != nil {
				t.Fatal(err)
			}
			servingState, err := tok.Encode(strings.ReplaceAll(tc.state, cfg.MaskText, " "))
			if err != nil {
				t.Fatal(err)
			}
			servingSequence, err := decision.Build(tokenBackend{tok, cfg}, cfg, parsed.Questions[0], servingState, false)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(ids, servingSequence.IDs) {
				t.Fatal("receipt differs from serving sequence")
			}
			if string(row["request"]) != string(encoded) {
				t.Fatal("original request changed")
			}

			mask := out["attention_mask"].([]int)
			if len(mask) != len(ids) {
				t.Fatal("mask length mismatch")
			}
			for _, value := range mask {
				if value != 1 {
					t.Fatal("unpadded receipt contains masked token")
				}
			}
			keys := out["token_keys"].([]any)
			if len(ids) != len(keys) || out["compute_input_tokens"] != len(ids) {
				t.Fatal("invalid compute count")
			}
			if out["useful_input_tokens"] != len(ids)-8 {
				t.Fatalf("special markers counted as useful: %v vs %d", out["useful_input_tokens"], len(ids))
			}
			if tc.truncated && out["dropped_tokens"].(int) <= 0 {
				t.Fatal("truncation count missing")
			}
		})
	}
}

func TestJuliaReceiptUsesGenericServingContract(t *testing.T) {
	path := os.Getenv("JULIA_TOKENIZER_JSON")
	if path == "" {
		t.Skip("real Julia tokenizer required")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	tok, cfg, err := configureTokenizer(raw, decision.Config{Model: "julia-1", MaxLen: 8192, HeadMaxLen: 512})
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.StrictEncoding || cfg.MaskText != "<mask>" || cfg.CLS != 2 || cfg.SEP != 1 || cfg.Mask != 4 {
		t.Fatal("wrong Julia configuration")
	}
	for _, tc := range []struct {
		name, question string
		options        []string
		qtype          int
	}{
		{"choice", `{"type":"choice","instructions":"Choose category.","criteria":{"invoice":"billing issue","software":"technical issue","general":"general issue","sports":"sport issue"}}`, []string{"billing issue", "technical issue", "general issue", "sport issue"}, 0},
		{"score", `{"type":"score","instructions":"Rate the invoice.","criteria":["no issue","needs review","serious issue"]}`, []string{"no issue", "needs review", "serious issue"}, 1},
		{"noul", `{"type":"noul","instructions":"Does the invoice need review?"}`, []string{"false", "true"}, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			request := json.RawMessage(`{"state":"A café invoice needs review.","questions":{"q":` + tc.question + `}}`)
			row := map[string]json.RawMessage{"id": json.RawMessage(`"julia-example"`), "request": request}
			out, err := receipt(row, tokenBackend{tok, cfg})
			if err != nil {
				t.Fatal(err)
			}
			if out["truncated"] != false || out["dropped_tokens"] != 0 {
				t.Fatalf("lossless Julia receipt marked truncated: %#v", out)
			}
			if out["question_type"] != tc.qtype {
				t.Fatal("wrong question type")
			}
			parsed, err := decision.Parse(request)
			if err != nil {
				t.Fatal(err)
			}
			prepared, err := decision.PrepareQuestion(cfg, parsed.Questions[0])
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(prepared.Options, tc.options) {
				t.Fatalf("wrong model options: %v", prepared.Options)
			}
			stateIDs, err := tok.Encode("A café invoice needs review.")
			if err != nil {
				t.Fatal(err)
			}
			seq, err := decision.Build(tokenBackend{tok, cfg}, cfg, prepared, stateIDs, false)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(out["token_ids"], seq.IDs) || out["model_name"] != "julia-1" {
				t.Fatal("receipt differs from Julia serving builder")
			}
			if out["compute_input_tokens"] != len(seq.IDs) || out["useful_input_tokens"] != len(seq.IDs)-4-len(seq.Markers) {
				t.Fatal("wrong token accounting")
			}
			if string(row["request"]) != string(request) {
				t.Fatal("source request changed")
			}
		})
	}
	reserved := map[string]json.RawMessage{"id": json.RawMessage(`"julia-reserved"`), "request": json.RawMessage(`{"state":"A <mask> invoice.","questions":{"q":{"type":"choice","instructions":"Choose.","criteria":{"a":"first","b":"second"}}}}`)}
	if _, err = receipt(reserved, tokenBackend{tok, cfg}); err == nil {
		t.Fatal("reserved Julia marker accepted")
	}
	if _, _, err = configureTokenizer(raw, decision.Config{Model: "unknown"}); err == nil {
		t.Fatal("unknown model accepted")
	}
}
