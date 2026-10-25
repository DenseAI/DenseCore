// Package decision serves native, non-autoregressive typed decisions.
package decision

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
)

type Config struct {
	Model              string
	StrictEncoding     bool
	MaxLen, HeadMaxLen int
	ModelMaxLen        int
	CLS, SEP, Mask     int32
	MaskText           string
	Temperature        [3]float64
	Buckets            map[string]float64
}

func (c Config) ModelID() string {
	if c.Model == "" {
		return "laya"
	}
	return c.Model
}

type Backend interface {
	Tokenize(string) ([]int32, error)
	Infer([]int32, []int32, int) ([]float32, []float32, error)
	Config() Config
	Close()
}
type Question struct {
	ID, Type, Instructions string
	Keys, Options, Legend  []string
	Order                  []int
	QType                  int
	definition             object
}
type Request struct {
	State     any
	Questions []Question
}
type InputError struct{ Err error }

func (e *InputError) Error() string         { return e.Err.Error() }
func invalid(format string, a ...any) error { return &InputError{fmt.Errorf(format, a...)} }
func Parse(raw []byte) (Request, error) {
	v, e := parseJSON(raw)
	if e != nil {
		return Request{}, invalid("invalid JSON: %v", e)
	}
	o, ok := v.(object)
	if !ok {
		return Request{}, invalid("request must be an object")
	}
	for _, m := range o {
		if m.key != "state" && m.key != "questions" {
			return Request{}, invalid("unsupported field %q", m.key)
		}
	}
	r := Request{State: o.get("state")}
	switch r.State.(type) {
	case string, object, []any:
	default:
		return r, invalid("state must be a string, object or array")
	}
	qs, ok := o.get("questions").(object)
	if !ok || len(qs) == 0 || len(qs) > 32 {
		return r, invalid("questions must contain 1 to 32 entries")
	}
	for _, m := range qs {
		q, e := parseQuestion(m)
		if e != nil {
			return r, e
		}
		r.Questions = append(r.Questions, q)
	}
	return r, nil
}
func parseQuestion(m member) (Question, error) {
	q := Question{ID: m.key}
	o, ok := m.value.(object)
	if !ok || strings.TrimSpace(m.key) == "" {
		return q, invalid("question must have a nonempty id and object definition")
	}
	q.definition = o
	for _, f := range o {
		switch f.key {
		case "type", "instructions", "criteria", "labels", "option_order":
		default:
			return q, invalid("question %q: unsupported field %q", m.key, f.key)
		}
	}
	q.Type, _ = o.get("type").(string)
	ins := o.get("instructions")
	if ins == nil {
		return q, invalid("question %q: instructions required", m.key)
	}
	switch v := ins.(type) {
	case object:
		if len(v) == 0 {
			return q, invalid("instructions must not be empty")
		}
	case []any:
		if len(v) == 0 {
			return q, invalid("instructions must not be empty")
		}
	case bool:
		return q, invalid("instructions must be text, a number, object or array")
	}
	q.Instructions = render(ins)
	if strings.TrimSpace(q.Instructions) == "" {
		return q, invalid("empty instructions")
	}
	switch q.Type {
	case "choice":
		q.QType = 0
		switch c := o.get("criteria").(type) {
		case object:
			for _, m := range c {
				opt := m.key
				if m.value != nil && m.value != "" {
					opt += ": " + render(m.value)
				}
				q.Keys = append(q.Keys, m.key)
				q.Options = append(q.Options, opt)
			}
		case []any:
			seen := map[string]bool{}
			for _, v := range c {
				s, ok := v.(string)
				if !ok {
					return q, invalid("choice list labels must be strings; use an object for descriptions")
				}
				if seen[s] {
					return q, invalid("duplicate choice label %q", s)
				}
				seen[s] = true
				q.Keys = append(q.Keys, s)
				q.Options = append(q.Options, s)
			}
		default:
			return q, invalid("choice criteria must be an object or string array")
		}
	case "score":
		q.QType = 1
		c, ok := o.get("criteria").([]any)
		if !ok {
			return q, invalid("score criteria must be an array")
		}
		for i, v := range c {
			if v == nil {
				return q, invalid("score levels cannot be null")
			}
			q.Legend = append(q.Legend, render(v))
			q.Keys = append(q.Keys, strconv.Itoa(i))
			q.Options = append(q.Options, fmt.Sprintf("level %d: %s", i, render(v)))
		}
	case "noul":
		q.QType = 2
		labels := []string{"false", "true"}
		if v := o.get("labels"); v != nil {
			l, ok := v.(object)
			if !ok || len(l) != 2 {
				return q, invalid("labels must map false and true")
			}
			for i, k := range labels {
				s, ok := l.get(k).(string)
				if !ok || strings.TrimSpace(s) == "" {
					return q, invalid("labels must be nonempty strings")
				}
				labels[i] = strings.TrimSpace(s)
			}
			if labels[0] == labels[1] {
				return q, invalid("labels must differ")
			}
		}
		c := object{}
		if v := o.get("criteria"); v != nil {
			var ok bool
			c, ok = v.(object)
			if !ok {
				return q, invalid("noul criteria must be an object")
			}
			for _, m := range c {
				if m.key != "false" && m.key != "true" {
					return q, invalid("noul criteria keys must be false or true")
				}
			}
		}
		for i, k := range []string{"false", "true"} {
			desc := []string{"no, the statement does not hold", "yes, the statement holds"}[i]
			if v := c.get(k); v != nil && v != "" {
				desc = render(v)
			}
			q.Options = append(q.Options, labels[i]+": "+desc)
		}
	default:
		return q, invalid("unknown question type %q", q.Type)
	}
	if q.Type != "noul" && o.get("labels") != nil {
		return q, invalid("labels is only supported for noul")
	}
	if len(q.Options) == 0 || len(q.Options) > 64 {
		return q, invalid("each question needs 1 to 64 options")
	}
	for i := range q.Options {
		q.Order = append(q.Order, i)
	}
	if v := o.get("option_order"); v != nil {
		a, ok := v.([]any)
		if !ok || len(a) != len(q.Options) {
			return q, invalid("option_order must be a permutation")
		}
		seen := map[int]bool{}
		for i, v := range a {
			n, ok := v.(json.Number)
			if !ok {
				return q, invalid("option_order must contain integers")
			}
			ix, e := strconv.Atoi(string(n))
			if e != nil || ix < 0 || ix >= len(a) || seen[ix] {
				return q, invalid("option_order must be a permutation")
			}
			seen[ix] = true
			q.Order[i] = ix
		}
	}
	return q, nil
}

// PrepareQuestion returns model-specific encoder descriptions without changing
// parsed request labels, order or the caller's slices. It is safe to call again.
func PrepareQuestion(c Config, q Question) (Question, error) {
	if c.ModelID() != "julia-1" || q.definition == nil {
		return q, nil
	}
	if _, ok := q.definition.get("instructions").(string); !ok {
		return q, invalid("Julia instructions must be text")
	}
	labels := []string{}
	add := func(value any) error {
		label, ok := value.(string)
		if !ok || label == "" {
			return invalid("Julia option descriptions must be nonempty text")
		}
		labels = append(labels, label)
		return nil
	}
	switch q.Type {
	case "choice":
		criteria, ok := q.definition.get("criteria").(object)
		if !ok {
			return q, invalid("Julia choice criteria must map IDs to descriptions")
		}
		for _, m := range criteria {
			if m.key == "" {
				return q, invalid("Julia choice IDs must be nonempty")
			}
			if err := add(m.value); err != nil {
				return q, err
			}
		}
	case "score":
		criteria, ok := q.definition.get("criteria").([]any)
		if !ok {
			return q, invalid("Julia score criteria must be an ordered rubric")
		}
		for _, value := range criteria {
			if err := add(value); err != nil {
				return q, err
			}
		}
	case "noul":
		value := q.definition.get("criteria")
		if value == nil {
			labels = []string{"false", "true"}
		} else {
			criteria, ok := value.(object)
			if !ok || len(criteria) != 2 {
				return q, invalid("Julia noul criteria must map false and true to descriptions")
			}
			for _, key := range []string{"false", "true"} {
				if err := add(criteria.get(key)); err != nil {
					return q, err
				}
			}
		}
	}
	q.Options = labels
	return q, nil
}

type Sequence struct {
	IDs, Markers                     []int32
	StateTokens, StateUsed, Distinct int
}

func Build(b Backend, c Config, q Question, state []int32, left bool) (Sequence, error) {
	s := Sequence{StateTokens: len(state)}
	var err error
	q, err = PrepareQuestion(c, q)
	if err != nil {
		return s, err
	}
	if c.StrictEncoding {
		if len(q.Options) < 2 || len(q.Options) > 20 {
			return s, invalid("Julia requires 2 to 20 options")
		}
		if strings.Contains(q.Instructions, c.MaskText) {
			return s, invalid("reserved model marker in question")
		}
		for _, option := range q.Options {
			if strings.Contains(option, c.MaskText) {
				return s, invalid("reserved model marker in option")
			}
		}
	}
	clean := func(s string) string { return strings.ReplaceAll(s, c.MaskText, " ") }
	head, e := b.Tokenize(q.Type + " question: " + clean(q.Instructions))
	if e != nil {
		return s, e
	}
	opts := [][]int32{}
	optionLengths := []int{}
	total := 0
	for _, ix := range q.Order {
		tok, e := b.Tokenize(" " + clean(q.Options[ix]))
		if e != nil {
			return s, e
		}
		if c.StrictEncoding && len(tok) > 48 {
			return s, invalid("option exceeds 48-token model contract")
		}
		tok = tok[:min(48, len(tok))]
		optionLengths = append(optionLengths, len(tok))
		tok = append([]int32{c.Mask}, tok...)
		opts = append(opts, tok)
		total += len(tok)
	}
	budget := c.HeadMaxLen - total
	if budget < 16 {
		per := max(4, (c.HeadMaxLen-16)/len(opts))
		total = 0
		for i := range opts {
			opts[i] = opts[i][:min(per, len(opts[i]))]
			total += len(opts[i])
		}
		budget = c.HeadMaxLen - total
	}
	if c.StrictEncoding {
		if len(head) > budget {
			return s, invalid("question/options exceed lossless head budget")
		}
		for i, option := range opts {
			if len(option) != optionLengths[i]+1 {
				return s, invalid("question/options exceed lossless head budget")
			}
		}
	}
	head = head[:min(len(head), max(8, budget))]
	s.IDs = append([]int32{c.CLS}, head...)
	s.IDs = append(s.IDs, c.SEP)
	distinct := map[string]bool{}
	for _, o := range opts {
		s.Markers = append(s.Markers, int32(len(s.IDs)))
		s.IDs = append(s.IDs, o...)
		distinct[fmt.Sprint(o)] = true
	}
	s.Distinct = len(distinct)
	s.IDs = append(s.IDs, c.SEP)
	room := max(0, c.MaxLen-len(s.IDs)-1)
	if c.StrictEncoding && (room < 1 || len(state) > room) {
		return s, invalid("state exceeds lossless context budget (effective limit %d tokens)", c.MaxLen)
	}
	s.StateUsed = min(room, len(state))
	if left {
		state = state[len(state)-s.StateUsed:]
	} else {
		state = state[:s.StateUsed]
	}
	s.IDs = append(s.IDs, state...)
	s.IDs = append(s.IDs, c.SEP)
	if len(s.IDs) > c.MaxLen {
		return s, invalid("question options exceed model context")
	}
	return s, nil
}
func softmax(logits []float32, temp float64) ([]float64, error) {
	if len(logits) == 0 {
		return nil, fmt.Errorf("empty logits")
	}
	p := make([]float64, len(logits))
	mx := math.Inf(-1)
	for _, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("nonfinite model output")
		}
		mx = math.Max(mx, float64(v))
	}
	sum := 0.
	for i, v := range logits {
		p[i] = math.Exp((float64(v) - mx) / temp)
		sum += p[i]
	}
	for i := range p {
		p[i] /= sum
	}
	return p, nil
}
func round(v float64) float64 { return math.RoundToEven(v*1e4) / 1e4 }
func Decode(c Config, q Question, logits, act []float32) (map[string]any, error) {
	k := len(q.Options)
	if len(logits) != k {
		return nil, fmt.Errorf("logit count mismatch")
	}
	bucket := "11+"
	if k <= 2 {
		bucket = "2"
	} else if k <= 5 {
		bucket = "3-5"
	} else if k <= 10 {
		bucket = "6-10"
	}
	temp := c.Temperature[q.QType]
	if v, ok := c.Buckets[q.Type+":"+bucket]; ok {
		temp = v
	}
	if c.Model == "julia-1" {
		temp = 1
	}
	if math.IsNaN(temp) || math.IsInf(temp, 0) {
		temp = 1
	}
	temp = math.Max(.5, math.Min(5, temp))
	p, e := softmax(logits, temp)
	if e != nil {
		return nil, e
	}
	canonical := make([]float64, k)
	for i, ix := range q.Order {
		canonical[ix] = p[i]
	}
	p = canonical
	ap, e := softmax(act, 1)
	if e != nil {
		return nil, e
	}
	best := 0
	entropy := 0.
	for i, v := range p {
		if v > p[best] {
			best = i
		}
		entropy -= v * math.Log(math.Max(v, 1e-12))
	}
	conf := 1.
	if k > 1 {
		conf = math.Max(0, math.Min(1, 1-entropy/math.Log(float64(k))))
	}
	a := map[string]any{"type": q.Type, "confidence": round(conf), "answer_confidence": round(p[best]), "action": map[string]any{"act_probability": round(ap[0])}}
	if q.Type == "noul" {
		a["noul"] = round(p[1])
		a["confidence"] = round(p[best])
		return a, nil
	}
	probs := map[string]float64{}
	for i, v := range p {
		probs[q.Keys[i]] = round(v)
	}
	a["probabilities"] = probs
	if q.Type == "choice" {
		a["choice"] = q.Keys[best]
	} else {
		score := 0.
		legend := map[string]string{}
		for i, v := range p {
			score += float64(i) * v
			legend[strconv.Itoa(i)] = q.Legend[i]
		}
		a["score"] = round(score)
		a["legend"] = legend
	}
	return a, nil
}

type preparedRequest struct {
	request     Request
	config      Config
	stateTokens int
	sequences   []Sequence
}

func prepare(ctx context.Context, b Backend, r Request) (preparedRequest, error) {
	p := preparedRequest{request: r, config: b.Config()}
	if err := ctx.Err(); err != nil {
		return p, err
	}
	state := render(r.State)
	if p.config.StrictEncoding && strings.Contains(state, p.config.MaskText) {
		return p, invalid("reserved model marker in state")
	}
	st, err := b.Tokenize(strings.ReplaceAll(state, p.config.MaskText, " "))
	if err != nil {
		return p, err
	}
	p.stateTokens = len(st)
	p.request.Questions = append([]Question(nil), r.Questions...)
	_, left := r.State.([]any)
	for i, q := range r.Questions {
		var err error
		q, err = PrepareQuestion(p.config, q)
		if err != nil {
			return p, err
		}
		p.request.Questions[i] = q
		if err := ctx.Err(); err != nil {
			return p, err
		}
		seq, err := Build(b, p.config, q, st, left)
		if err != nil {
			return p, err
		}
		p.sequences = append(p.sequences, seq)
	}
	return p, nil
}
func (p preparedRequest) finish(outputs []BatchOutput) (map[string]any, error) {
	if len(outputs) != len(p.sequences) {
		return nil, fmt.Errorf("batch output count mismatch")
	}
	answers := map[string]any{}
	input, used, dropped := 0, 0, 0
	collapsed := map[string]any{}
	for i, q := range p.request.Questions {
		seq := p.sequences[i]
		a, err := Decode(p.config, q, outputs[i].Logits, outputs[i].Action)
		if err != nil {
			return nil, err
		}
		answers[q.ID] = a
		input += len(seq.IDs)
		used += seq.StateUsed
		dropped += seq.StateTokens - seq.StateUsed
		if seq.Distinct < len(q.Options) {
			collapsed[q.ID] = map[string]int{"total": len(q.Options), "distinct": seq.Distinct}
		}
	}
	return map[string]any{"model": p.config.ModelID(), "answers": answers, "usage": map[string]any{"input_tokens": input, "output_tokens": 0, "state_tokens": p.stateTokens * len(p.request.Questions), "state_tokens_used": used, "state_tokens_dropped": dropped, "truncated": dropped > 0, "collapsed_options": collapsed}}, nil
}
func Predict(ctx context.Context, b Backend, r Request) (map[string]any, error) {
	p, err := prepare(ctx, b, r)
	if err != nil {
		return nil, err
	}
	outputs := make([]BatchOutput, len(p.sequences))
	for i, seq := range p.sequences {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		outputs[i].Logits, outputs[i].Action, err = b.Infer(seq.IDs, seq.Markers, r.Questions[i].QType)
		if err != nil {
			return nil, err
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return p.finish(outputs)
}
func predictBatch(contexts []context.Context, b Backend, requests []Request) []decisionResult {
	results := make([]decisionResult, len(requests))
	prepared := make([]preparedRequest, len(requests))
	starts := make([]int, len(requests))
	inputs := []BatchInput{}
	for i, r := range requests {
		p, err := prepare(contexts[i], b, r)
		prepared[i] = p
		results[i].err = err
		starts[i] = len(inputs)
		if err != nil {
			continue
		}
		for j, seq := range p.sequences {
			inputs = append(inputs, BatchInput{seq.IDs, seq.Markers, r.Questions[j].QType})
		}
	}
	if len(inputs) == 0 {
		return results
	}
	outputs := make([]BatchOutput, 0, len(inputs))
	var err error
	for start := 0; start < len(inputs); start += 64 {
		end := min(start+64, len(inputs))
		var chunk []BatchOutput
		chunk, err = b.(BatchBackend).InferBatch(inputs[start:end])
		if err == nil && len(chunk) != end-start {
			err = fmt.Errorf("batch output count mismatch")
		}
		if err != nil {
			break
		}
		outputs = append(outputs, chunk...)
	}
	for i, p := range prepared {
		if results[i].err != nil {
			continue
		}
		if err != nil {
			results[i].err = err
			continue
		}
		if e := contexts[i].Err(); e != nil {
			results[i].err = e
			continue
		}
		results[i].out, results[i].err = p.finish(outputs[starts[i] : starts[i]+len(p.sequences)])
	}
	return results
}

// PredictBatch prepares independent requests, invokes native batches and decodes
// each response. Failed input or inference returns an error for the whole call;
// HTTP scheduling uses per-request results so one invalid request stays isolated.
func PredictBatch(ctx context.Context, b Backend, requests []Request) ([]map[string]any, error) {
	if len(requests) < 1 || len(requests) > 64 {
		return nil, fmt.Errorf("request batch size must be 1..64")
	}
	if _, ok := b.(BatchBackend); !ok {
		return nil, fmt.Errorf("backend does not support native batching")
	}
	contexts := make([]context.Context, len(requests))
	for i := range contexts {
		contexts[i] = ctx
	}
	results := predictBatch(contexts, b, requests)
	out := make([]map[string]any, len(results))
	for i, result := range results {
		if result.err != nil {
			return nil, fmt.Errorf("request %d: %w", i, result.err)
		}
		out[i] = result.out
	}
	return out, nil
}
