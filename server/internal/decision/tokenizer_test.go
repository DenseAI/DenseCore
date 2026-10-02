package decision

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"
)

func TestGPT2UnicodePieces(t *testing.T) {
	for _, tc := range []struct {
		text string
		want []string
	}{
		{"Hello, world!", []string{"Hello", ",", " world", "!"}},
		{"I'M we're", []string{"I", "'", "M", " we", "'re"}},
		{"abc中文 १२३😀", []string{"abc中文", " १२३", "😀"}},
		{"a\t \n b   ", []string{"a", "\t \n", " b", "   "}},
		{"x\u0301é", []string{"x", "\u0301", "é"}},
	} {
		if got := gpt2Pieces(tc.text); !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%q: got %#v want %#v", tc.text, got, tc.want)
		}
	}
}

// The fixture IDs were produced by Hugging Face tokenizers from the tokenizer
// embedded in zerodegress/laya-gguf revision ff850ac9c08707e926847711a4d1819dc228ecc7.
// Set LAYA_TOKENIZER_JSON to that embedded JSON for the real-vocabulary parity gate.
func TestLayaTokenizerHFParity(t *testing.T) {
	path := os.Getenv("LAYA_TOKENIZER_JSON")
	if path == "" {
		t.Skip("set LAYA_TOKENIZER_JSON for real-vocabulary Hugging Face parity")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := NewTokenizer(raw)
	if err != nil {
		t.Fatal(err)
	}
	casePath := os.Getenv("LAYA_TOKENIZER_CASES")
	if casePath == "" {
		casePath = "testdata/laya-tokenizer-cases.json"
	}
	cases, err := os.ReadFile(casePath)
	if err != nil {
		t.Fatal(err)
	}
	var rows []struct {
		Text string  `json:"text"`
		IDs  []int32 `json:"ids"`
	}
	if err = json.Unmarshal(cases, &rows); err != nil {
		t.Fatal(err)
	}
	for _, row := range rows {
		got, err := tok.Encode(row.Text)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, row.IDs) {
			t.Errorf("%q\ngot  %v\nwant %v", row.Text, got, row.IDs)
		}
	}
	if _, err = tok.Encode(string([]byte{0xff})); err == nil {
		t.Error("invalid UTF-8 accepted")
	}
}

func TestTokenizerRejectsUnsupportedContract(t *testing.T) {
	if _, err := NewTokenizer([]byte(`{"normalizer":{"type":"Lowercase"}}`)); err == nil {
		t.Fatal("unsupported tokenizer accepted")
	}
}
