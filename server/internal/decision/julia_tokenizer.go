package decision

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// NewJuliaTokenizer accepts the pinned Julia BPE contract: literal-space
// replacement, Metaspace pretokenization and UTF-8 byte fallback. In particular,
// it does not apply NFC or compatibility normalization to Julia inputs.
func NewJuliaTokenizer(raw []byte) (*Tokenizer, error) {
	var spec struct {
		Normalizer struct {
			Type    string `json:"type"`
			Pattern struct {
				String string `json:"String"`
			} `json:"pattern"`
			Content string `json:"content"`
		} `json:"normalizer"`
		Pre struct {
			Type        string `json:"type"`
			Replacement string `json:"replacement"`
			Prepend     string `json:"prepend_scheme"`
			Split       bool   `json:"split"`
		} `json:"pre_tokenizer"`
		Added []addedToken `json:"added_tokens"`
		Model struct {
			Type         string            `json:"type"`
			Vocab        map[string]int32  `json:"vocab"`
			Merges       []json.RawMessage `json:"merges"`
			Dropout      *float64          `json:"dropout"`
			Prefix       string            `json:"continuing_subword_prefix"`
			Suffix       string            `json:"end_of_word_suffix"`
			ByteFallback bool              `json:"byte_fallback"`
			IgnoreMerges bool              `json:"ignore_merges"`
			Unk          string            `json:"unk_token"`
			FuseUnk      bool              `json:"fuse_unk"`
		} `json:"model"`
	}
	if err := json.Unmarshal(raw, &spec); err != nil {
		return nil, fmt.Errorf("Julia tokenizer JSON: %w", err)
	}
	if spec.Normalizer.Type != "Replace" || spec.Normalizer.Pattern.String != " " || spec.Normalizer.Content != "▁" || spec.Pre.Type != "Metaspace" || spec.Pre.Replacement != "▁" || spec.Pre.Prepend != "always" || !spec.Pre.Split || spec.Model.Type != "BPE" || spec.Model.Dropout != nil || spec.Model.Prefix != "" || spec.Model.Suffix != "" || !spec.Model.ByteFallback || spec.Model.IgnoreMerges || spec.Model.Unk != "<unk>" || !spec.Model.FuseUnk {
		return nil, fmt.Errorf("unsupported Julia tokenizer contract")
	}
	if len(spec.Model.Vocab) == 0 || len(spec.Model.Merges) == 0 {
		return nil, fmt.Errorf("empty Julia tokenizer")
	}
	t := &Tokenizer{vocab: spec.Model.Vocab, ranks: make(map[[2]string]int, len(spec.Model.Merges)), metaspace: true}
	for rank, raw := range spec.Model.Merges {
		var pair []string
		if json.Unmarshal(raw, &pair) != nil {
			var legacy string
			if json.Unmarshal(raw, &legacy) != nil {
				return nil, fmt.Errorf("invalid Julia BPE merge")
			}
			pair = strings.Split(legacy, " ")
		}
		if len(pair) != 2 || pair[0] == "" || pair[1] == "" {
			return nil, fmt.Errorf("invalid Julia BPE merge %d", rank)
		}
		if _, ok := t.vocab[pair[0]+pair[1]]; !ok {
			return nil, fmt.Errorf("Julia BPE merge missing vocabulary entry")
		}
		t.ranks[[2]string{pair[0], pair[1]}] = rank
	}
	for _, a := range spec.Added {
		if a.Content == "" || a.SingleWord || a.Normalized || a.ID < 0 {
			return nil, fmt.Errorf("unsupported Julia added token")
		}
		if id, ok := t.vocab[a.Content]; ok && id != a.ID {
			return nil, fmt.Errorf("Julia added token ID mismatch")
		}
		t.vocab[a.Content] = a.ID
		t.added[0] = append(t.added[0], a)
	}
	for phase := range t.added {
		sort.SliceStable(t.added[phase], func(i, j int) bool { return len(t.added[phase][i].Content) > len(t.added[phase][j].Content) })
	}
	for token, expected := range map[string]int32{"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "<mask>": 4} {
		if actual, ok := t.vocab[token]; !ok || actual != expected {
			return nil, fmt.Errorf("unsupported Julia token ID for %s", token)
		}
	}
	for b := 0; b < 256; b++ {
		if _, ok := t.vocab[fmt.Sprintf("<0x%02X>", b)]; !ok {
			// An ASCII character present directly in the vocabulary never enters
			// byte fallback. Julia stores TAB directly and omits <0x09>.
			if _, direct := t.vocab[string(rune(b))]; b >= 128 || !direct {
				return nil, fmt.Errorf("Julia tokenizer missing byte fallback %d", b)
			}
		}
	}
	return t, nil
}
func metaspacePieces(s string) []string {
	if s == "" {
		return nil
	}
	if !strings.HasPrefix(s, "▁") {
		s = "▁" + s
	}
	var pieces []string
	start := 0
	for i, r := range s {
		if r == '▁' && i > start {
			pieces = append(pieces, s[start:i])
			start = i
		}
	}
	return append(pieces, s[start:])
}
func (t *Tokenizer) encodeMetaspacePiece(piece string) ([]int32, error) {
	symbols := make([]string, 0, len(piece))
	for _, r := range piece {
		symbol := string(r)
		if _, ok := t.vocab[symbol]; ok {
			symbols = append(symbols, symbol)
			continue
		}
		for _, b := range []byte(symbol) {
			symbols = append(symbols, fmt.Sprintf("<0x%02X>", b))
		}
	}
	return t.encodeSymbols(symbols)
}
