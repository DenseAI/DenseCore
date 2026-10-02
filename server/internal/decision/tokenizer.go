package decision

import (
	"container/heap"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

type addedToken struct {
	ID         int32  `json:"id"`
	Content    string `json:"content"`
	SingleWord bool   `json:"single_word"`
	LStrip     bool   `json:"lstrip"`
	RStrip     bool   `json:"rstrip"`
	Normalized bool   `json:"normalized"`
}

// Tokenizer implements the NFC and byte-level BPE contract embedded in Laya GGUFs.
// Encode does not add CLS/SEP: the decision prompt builder owns those markers.
type Tokenizer struct {
	vocab map[string]int32
	ranks map[[2]string]int
	added [2][]addedToken
	bytes [256]string
}

func NewTokenizer(raw []byte) (*Tokenizer, error) {
	var spec struct {
		Normalizer struct {
			Type string `json:"type"`
		} `json:"normalizer"`
		Pre struct {
			Type   string `json:"type"`
			Prefix bool   `json:"add_prefix_space"`
			Regex  bool   `json:"use_regex"`
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
		} `json:"model"`
	}
	if err := json.Unmarshal(raw, &spec); err != nil {
		return nil, fmt.Errorf("decision tokenizer JSON: %w", err)
	}
	if spec.Normalizer.Type != "NFC" || spec.Pre.Type != "ByteLevel" || spec.Pre.Prefix || !spec.Pre.Regex || spec.Model.Type != "BPE" || spec.Model.Dropout != nil || spec.Model.Prefix != "" || spec.Model.Suffix != "" || spec.Model.ByteFallback || spec.Model.IgnoreMerges {
		return nil, fmt.Errorf("unsupported decision tokenizer contract")
	}
	if len(spec.Model.Vocab) == 0 || len(spec.Model.Merges) == 0 {
		return nil, fmt.Errorf("decision tokenizer has empty vocabulary or merges")
	}
	t := &Tokenizer{vocab: spec.Model.Vocab, ranks: make(map[[2]string]int, len(spec.Model.Merges))}
	for rank, rawPair := range spec.Model.Merges {
		var pair []string
		if err := json.Unmarshal(rawPair, &pair); err != nil {
			var legacy string
			if err := json.Unmarshal(rawPair, &legacy); err != nil {
				return nil, fmt.Errorf("invalid BPE merge %d", rank)
			}
			pair = strings.Split(legacy, " ")
		}
		if len(pair) != 2 || pair[0] == "" || pair[1] == "" {
			return nil, fmt.Errorf("invalid BPE merge %d", rank)
		}
		if _, ok := t.vocab[pair[0]+pair[1]]; !ok {
			return nil, fmt.Errorf("BPE merge %d missing vocabulary entry", rank)
		}
		t.ranks[[2]string{pair[0], pair[1]}] = rank
	}
	for _, a := range spec.Added {
		if a.Content == "" || a.SingleWord || a.ID < 0 {
			return nil, fmt.Errorf("unsupported added token")
		}
		if id, ok := t.vocab[a.Content]; ok && id != a.ID {
			return nil, fmt.Errorf("added token ID mismatch")
		}
		t.vocab[a.Content] = a.ID
		phase := 0
		if a.Normalized {
			phase = 1
		}
		t.added[phase] = append(t.added[phase], a)
	}
	for phase := range t.added {
		sort.SliceStable(t.added[phase], func(i, j int) bool { return len(t.added[phase][i].Content) > len(t.added[phase][j].Content) })
	}
	n := rune(256)
	for b := 0; b < 256; b++ {
		cp := rune(b)
		if !(b >= 33 && b <= 126 || b >= 161 && b <= 172 || b >= 174) {
			cp = n
			n++
		}
		t.bytes[b] = string(cp)
		// Some byte-level vocabularies omit bytes that cannot occur in valid UTF-8.
		if _, ok := t.vocab[t.bytes[b]]; !ok && (b < 192 || b >= 194 && b <= 244) {
			return nil, fmt.Errorf("byte %d missing from tokenizer vocabulary", b)
		}
	}
	return t, nil
}

func (t *Tokenizer) TokenID(token string) (int32, bool) { id, ok := t.vocab[token]; return id, ok }

func (t *Tokenizer) Encode(text string) ([]int32, error) {
	if !utf8.ValidString(text) {
		return nil, fmt.Errorf("decision input is not valid UTF-8")
	}
	result := make([]int32, 0, len(text)/3)
	var split func(string, int) error
	split = func(s string, phase int) error {
		if phase == 1 {
			s = norm.NFC.String(s)
		}
		if phase == 2 {
			for _, piece := range gpt2Pieces(s) {
				ids, err := t.encodePiece(piece)
				if err != nil {
					return err
				}
				result = append(result, ids...)
			}
			return nil
		}
		start := 0
		for i := 0; i < len(s); {
			var found *addedToken
			for k := range t.added[phase] {
				a := &t.added[phase][k]
				if strings.HasPrefix(s[i:], a.Content) {
					found = a
					break
				}
			}
			if found == nil {
				_, size := utf8.DecodeRuneInString(s[i:])
				i += size
				continue
			}
			end := i
			if found.LStrip {
				for end > start {
					r, size := utf8.DecodeLastRuneInString(s[start:end])
					if !unicode.IsSpace(r) {
						break
					}
					end -= size
				}
			}
			if err := split(s[start:end], phase+1); err != nil {
				return err
			}
			result = append(result, found.ID)
			i += len(found.Content)
			if found.RStrip {
				for i < len(s) {
					r, size := utf8.DecodeRuneInString(s[i:])
					if !unicode.IsSpace(r) {
						break
					}
					i += size
				}
			}
			start = i
		}
		return split(s[start:], phase+1)
	}
	if err := split(text, 0); err != nil {
		return nil, err
	}
	return result, nil
}

// Equivalent to GPT-2's case-sensitive Unicode regex:
// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
func gpt2Pieces(text string) []string {
	runes := []rune(text)
	var pieces []string
	category := func(r rune) int {
		if unicode.IsLetter(r) {
			return 1
		}
		if unicode.IsNumber(r) {
			return 2
		}
		if unicode.IsSpace(r) {
			return 0
		}
		return 3
	}
	for i := 0; i < len(runes); {
		start := i
		if runes[i] == '\'' {
			matched := false
			for _, ending := range []string{"s", "t", "re", "ve", "m", "ll", "d"} {
				suffix := []rune(ending)
				if i+1+len(suffix) <= len(runes) && string(runes[i+1:i+1+len(suffix)]) == ending {
					i += 1 + len(suffix)
					matched = true
					break
				}
			}
			if matched {
				pieces = append(pieces, string(runes[start:i]))
				continue
			}
		}
		if runes[i] == ' ' && i+1 < len(runes) && category(runes[i+1]) != 0 {
			i++
		}
		cat := category(runes[i])
		i++
		for i < len(runes) && category(runes[i]) == cat {
			i++
		}
		if cat == 0 && i < len(runes) && i-start > 1 {
			i--
		}
		pieces = append(pieces, string(runes[start:i]))
	}
	return pieces
}

type mergeCandidate struct{ rank, left, right, generation int }
type mergeHeap []mergeCandidate

func (h mergeHeap) Len() int { return len(h) }
func (h mergeHeap) Less(i, j int) bool {
	if h[i].rank != h[j].rank {
		return h[i].rank < h[j].rank
	}
	return h[i].left < h[j].left
}
func (h mergeHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h *mergeHeap) Push(x any)   { *h = append(*h, x.(mergeCandidate)) }
func (h *mergeHeap) Pop() any     { old := *h; x := old[len(old)-1]; *h = old[:len(old)-1]; return x }

func (t *Tokenizer) encodePiece(piece string) ([]int32, error) {
	if piece == "" {
		return nil, nil
	}
	type node struct {
		value                  string
		prev, next, generation int
		alive                  bool
	}
	nodes := make([]node, len(piece))
	for i := range nodes {
		nodes[i] = node{t.bytes[piece[i]], i - 1, i + 1, 0, true}
	}
	nodes[len(nodes)-1].next = -1
	q := &mergeHeap{}
	push := func(i int) {
		if i < 0 || !nodes[i].alive || nodes[i].next < 0 {
			return
		}
		j := nodes[i].next
		if rank, ok := t.ranks[[2]string{nodes[i].value, nodes[j].value}]; ok {
			heap.Push(q, mergeCandidate{rank, i, j, nodes[i].generation})
		}
	}
	for i := range nodes {
		push(i)
	}
	for q.Len() > 0 {
		m := heap.Pop(q).(mergeCandidate)
		a := &nodes[m.left]
		if !a.alive || a.generation != m.generation || a.next != m.right {
			continue
		}
		b := &nodes[m.right]
		a.value += b.value
		a.next = b.next
		a.generation++
		b.alive = false
		if a.next >= 0 {
			nodes[a.next].prev = m.left
		}
		if a.prev >= 0 {
			nodes[a.prev].generation++
			push(a.prev)
		}
		push(m.left)
	}
	ids := make([]int32, 0, len(nodes))
	for i := 0; i >= 0; i = nodes[i].next {
		id, ok := t.vocab[nodes[i].value]
		if !ok {
			return nil, fmt.Errorf("BPE result missing vocabulary entry")
		}
		ids = append(ids, id)
	}
	return ids, nil
}
