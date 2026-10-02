package decision

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
)

type member struct {
	key   string
	value any
}
type object []member

func (o object) get(k string) any {
	for _, m := range o {
		if m.key == k {
			return m.value
		}
	}
	return nil
}
func parseJSON(raw []byte) (any, error) {
	d := json.NewDecoder(bytes.NewReader(raw))
	d.UseNumber()
	v, e := readValue(d)
	if e != nil {
		return nil, e
	}
	if _, e = d.Token(); e != io.EOF {
		return nil, fmt.Errorf("expected one JSON value")
	}
	return v, nil
}
func readValue(d *json.Decoder) (any, error) {
	t, e := d.Token()
	if e != nil {
		return nil, e
	}
	switch t {
	case json.Delim('{'):
		o := object{}
		seen := map[string]bool{}
		for d.More() {
			k, e := d.Token()
			if e != nil {
				return nil, e
			}
			s := k.(string)
			if seen[s] {
				return nil, fmt.Errorf("duplicate key %q", s)
			}
			seen[s] = true
			v, e := readValue(d)
			if e != nil {
				return nil, e
			}
			o = append(o, member{s, v})
		}
		_, e = d.Token()
		return o, e
	case json.Delim('['):
		a := []any{}
		for d.More() {
			v, e := readValue(d)
			if e != nil {
				return nil, e
			}
			a = append(a, v)
		}
		_, e = d.Token()
		return a, e
	}
	if n, ok := t.(json.Number); ok && strings.ContainsAny(string(n), ".eE") {
		v, err := strconv.ParseFloat(string(n), 64)
		if err != nil || math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("JSON floating point number is outside the supported finite range")
		}
	}
	return t, nil
}

// Python's json.dumps(..., ensure_ascii=False) separators, preserving input object order.
func pythonJSON(v any) string {
	switch x := v.(type) {
	case object:
		a := []string{}
		for _, m := range x {
			a = append(a, quote(m.key)+": "+pythonJSON(m.value))
		}
		return "{" + strings.Join(a, ", ") + "}"
	case []any:
		a := []string{}
		for _, v := range x {
			a = append(a, pythonJSON(v))
		}
		return "[" + strings.Join(a, ", ") + "]"
	case string:
		return quote(x)
	case json.Number:
		if !strings.ContainsAny(string(x), ".eE") {
			if x == "-0" {
				return "0"
			}
			return string(x)
		}
		f, err := strconv.ParseFloat(string(x), 64)
		if err != nil {
			return string(x)
		}
		format := byte('g')
		if f == 0 || (math.Abs(f) >= 1e-4 && math.Abs(f) < 1e16) {
			format = 'f'
		}
		s := strconv.FormatFloat(f, format, -1, 64)
		if !strings.ContainsAny(s, ".eE") {
			s += ".0"
		}
		return s
	case bool:
		return strconv.FormatBool(x)
	case nil:
		return "null"
	}
	panic("invalid JSON value")
}
func quote(s string) string {
	var b strings.Builder
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"', '\\':
			b.WriteByte('\\')
			b.WriteRune(r)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		default:
			if r < 0x20 {
				fmt.Fprintf(&b, `\u%04x`, r)
			} else {
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
	return b.String()
}
func render(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return pythonJSON(v)
}
