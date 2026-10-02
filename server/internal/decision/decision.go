// Package decision serves native, non-autoregressive Laya decisions.
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
	MaxLen, HeadMaxLen int
	CLS, SEP, Mask     int32
	MaskText           string
	Temperature        [3]float64
	Buckets            map[string]float64
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

type Sequence struct {
	IDs, Markers                     []int32
	StateTokens, StateUsed, Distinct int
}

func Build(b Backend, c Config, q Question, state []int32, left bool) (Sequence, error) {
	s := Sequence{StateTokens: len(state)}
	clean := func(s string) string { return strings.ReplaceAll(s, c.MaskText, " ") }
	head, e := b.Tokenize(q.Type + " question: " + clean(q.Instructions))
	if e != nil {
		return s, e
	}
	opts := [][]int32{}
	total := 0
	for _, ix := range q.Order {
		tok, e := b.Tokenize(" " + clean(q.Options[ix]))
		if e != nil {
			return s, e
		}
		tok = tok[:min(48, len(tok))]
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
func Predict(ctx context.Context, b Backend, r Request) (map[string]any, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	c := b.Config()
	st, e := b.Tokenize(strings.ReplaceAll(render(r.State), c.MaskText, " "))
	if e != nil {
		return nil, e
	}
	_, left := r.State.([]any)
	answers := map[string]any{}
	input, used, dropped := 0, 0, 0
	collapsed := map[string]any{}
	for _, q := range r.Questions {
		if e := ctx.Err(); e != nil {
			return nil, e
		}
		s, e := Build(b, c, q, st, left)
		if e != nil {
			return nil, e
		}
		logits, act, e := b.Infer(s.IDs, s.Markers, q.QType)
		if e != nil {
			return nil, e
		}
		a, e := Decode(c, q, logits, act)
		if e != nil {
			return nil, e
		}
		answers[q.ID] = a
		input += len(s.IDs)
		used += s.StateUsed
		dropped += s.StateTokens - s.StateUsed
		if s.Distinct < len(q.Options) {
			collapsed[q.ID] = map[string]int{"total": len(q.Options), "distinct": s.Distinct}
		}
	}
	return map[string]any{"model": "laya", "answers": answers, "usage": map[string]any{"input_tokens": input, "output_tokens": 0, "state_tokens": len(st) * len(r.Questions), "state_tokens_used": used, "state_tokens_dropped": dropped, "truncated": dropped > 0, "collapsed_options": collapsed}}, nil
}
