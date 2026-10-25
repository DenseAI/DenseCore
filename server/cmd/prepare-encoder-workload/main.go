// Prepare real token receipts using exactly the serving input builder.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/DenseAI/DenseCore/server/internal/decision"
)

type tokenBackend struct {
	tokenizer *decision.Tokenizer
	config    decision.Config
}

func (b tokenBackend) Tokenize(s string) ([]int32, error) { return b.tokenizer.Encode(s) }
func (b tokenBackend) Infer([]int32, []int32, int) ([]float32, []float32, error) {
	return nil, nil, fmt.Errorf("tokenizer only")
}
func (b tokenBackend) Config() decision.Config { return b.config }
func (b tokenBackend) Close()                  {}

func receipt(row map[string]json.RawMessage, b tokenBackend) (map[string]any, error) {
	request, err := decision.Parse(row["request"])
	if err != nil {
		return nil, err
	}
	if len(request.Questions) != 1 {
		return nil, fmt.Errorf("one question required")
	}
	state, ok := request.State.(string)
	if !ok {
		return nil, fmt.Errorf("state must be a source-preserving string")
	}
	if b.config.StrictEncoding && strings.Contains(state, b.config.MaskText) {
		return nil, fmt.Errorf("reserved model marker in state")
	}
	stateIDs, err := b.Tokenize(strings.ReplaceAll(state, b.config.MaskText, " "))
	if err != nil {
		return nil, err
	}
	q, err := decision.PrepareQuestion(b.config, request.Questions[0])
	if err != nil {
		return nil, err
	}
	seq, err := decision.Build(b, b.config, q, stateIDs, false)
	if err != nil {
		return nil, err
	}
	clean := func(s string) string { return strings.ReplaceAll(s, b.config.MaskText, " ") }
	head, err := b.Tokenize(q.Type + " question: " + clean(q.Instructions))
	if err != nil {
		return nil, err
	}
	full := 4 + len(head) + len(stateIDs)
	for _, ix := range q.Order {
		ids, e := b.Tokenize(" " + clean(q.Options[ix]))
		if e != nil {
			return nil, e
		}
		full += 1 + len(ids)
	}
	dropped := full - len(seq.IDs)
	if dropped < 0 {
		return nil, fmt.Errorf("invalid input-builder accounting")
	}
	out := map[string]any{}
	for key, value := range row {
		out[key] = value
	}
	keys := make([]any, len(seq.IDs))
	var sourceID string
	_ = json.Unmarshal(row["id"], &sourceID)
	if sourceID == "" {
		return nil, fmt.Errorf("source id required")
	}
	for i := range keys {
		keys[i] = fmt.Sprintf("%s:%d", sourceID, i)
	}
	keys[0] = nil
	keys[len(keys)-1] = nil
	keys[len(keys)-seq.StateUsed-2] = nil
	if len(seq.Markers) > 0 {
		keys[int(seq.Markers[0])-1] = nil
	}
	for _, marker := range seq.Markers {
		keys[int(marker)] = nil
	}
	useful := 0
	for _, key := range keys {
		if key != nil {
			useful++
		}
	}
	out["model_name"] = b.config.ModelID()
	out["token_keys"] = keys
	out["token_ids"] = seq.IDs
	attentionMask := make([]int, len(seq.IDs))
	for i := range attentionMask {
		attentionMask[i] = 1
	}
	out["attention_mask"] = attentionMask
	out["markers"] = seq.Markers
	out["question_type"] = q.QType
	out["padding_tokens"] = 0
	out["dropped_tokens"] = dropped
	out["truncated"] = dropped > 0
	out["useful_input_tokens"] = useful
	out["compute_input_tokens"] = len(seq.IDs)
	return out, nil
}

func configureTokenizer(raw []byte, cfg decision.Config) (*decision.Tokenizer, decision.Config, error) {
	if cfg.Model == "" {
		cfg.Model = "laya"
	}
	names := []string{"[CLS]", "[SEP]", "[MASK]"}
	var tokenizer *decision.Tokenizer
	var err error
	switch cfg.Model {
	case "laya":
		cfg.MaskText = "[MASK]"
		cfg.StrictEncoding = false
		tokenizer, err = decision.NewTokenizer(raw)
	case "julia-1":
		cfg.MaskText = "<mask>"
		cfg.StrictEncoding = true
		names = []string{"<bos>", "<eos>", "<mask>"}
		tokenizer, err = decision.NewJuliaTokenizer(raw)
	default:
		return nil, cfg, fmt.Errorf("unsupported decision model %q", cfg.Model)
	}
	if err != nil {
		return nil, cfg, err
	}
	if cfg.MaxLen < 16 || cfg.MaxLen > 8192 || cfg.HeadMaxLen < 16 || cfg.HeadMaxLen > cfg.MaxLen {
		return nil, cfg, fmt.Errorf("invalid model context configuration")
	}
	for i, target := range []*int32{&cfg.CLS, &cfg.SEP, &cfg.Mask} {
		id, ok := tokenizer.TokenID(names[i])
		if !ok {
			return nil, cfg, fmt.Errorf("missing marker %s", names[i])
		}
		*target = id
	}
	if cfg.Model == "julia-1" && (cfg.CLS != 2 || cfg.SEP != 1 || cfg.Mask != 4 || cfg.HeadMaxLen+4 >= cfg.MaxLen) {
		return nil, cfg, fmt.Errorf("unsupported Julia tokenizer/context configuration")
	}
	return tokenizer, cfg, nil
}

func run() error {
	tokenizerPath := flag.String("tokenizer", "", "exact tokenizer JSON extracted from measured model")
	configPath := flag.String("config", "", "JSON Config matching measured model metadata")
	input := flag.String("input", "", "normalized source JSONL requests")
	output := flag.String("output", "", "tokenized JSONL")
	flag.Parse()
	if *tokenizerPath == "" || *configPath == "" || *input == "" || *output == "" {
		return fmt.Errorf("tokenizer, config, input and output required")
	}
	raw, err := os.ReadFile(*tokenizerPath)
	if err != nil {
		return err
	}
	cfgRaw, err := os.ReadFile(*configPath)
	if err != nil {
		return err
	}
	var cfg decision.Config
	if err = json.Unmarshal(cfgRaw, &cfg); err != nil {
		return err
	}
	tokenizer, cfg, err := configureTokenizer(raw, cfg)
	if err != nil {
		return err
	}
	in, err := os.Open(*input)
	if err != nil {
		return err
	}
	defer in.Close()
	// Stage in memory so malformed input never leaves a partial workload file.
	var lines [][]byte
	scanner := bufio.NewScanner(in)
	scanner.Buffer(make([]byte, 4096), 16*1024*1024)
	for scanner.Scan() {
		var row map[string]json.RawMessage
		if err = json.Unmarshal(scanner.Bytes(), &row); err != nil {
			return err
		}
		result, e := receipt(row, tokenBackend{tokenizer, cfg})
		if e != nil {
			return e
		}
		encoded, e := json.Marshal(result)
		if e != nil {
			return e
		}
		lines = append(lines, encoded)
	}
	if err = scanner.Err(); err != nil {
		return err
	}
	var data []byte
	for _, line := range lines {
		data = append(data, line...)
		data = append(data, '\n')
	}
	return os.WriteFile(*output, data, 0600)
}
func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
