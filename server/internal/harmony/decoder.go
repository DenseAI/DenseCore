// Package harmony decodes text-only GPT-OSS completion channels.
// Protocol reference: https://github.com/openai/harmony/blob/main/docs/format.md
package harmony

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

type Delta struct{ Content, Reasoning string }

// Decoder starts at the open assistant header emitted by the native renderer.
// It buffers only incomplete markers/UTF-8 and the bounded current header.
// Tool recipients are rejected until the server supports Harmony tool routing.
type Decoder struct {
	pending, header, channel string
	inHeader, ended          bool
	err                      error
}

func NewDecoder() *Decoder { return &Decoder{inHeader: true, header: "assistant"} }

var markers = []string{"<|start|>", "<|message|>", "<|channel|>", "<|constrain|>", "<|end|>", "<|return|>", "<|call|>",
	"<|im_start|>", "<|im_sep|>", "<|meta_sep|>", "<|meta_start|>", "<|im_end|>", "<|fim_suffix|>", "<|ghissue|>"}

func (d *Decoder) Write(text string) (Delta, error) { return d.consume(text, false) }
func (d *Decoder) Finish() (Delta, error)           { return d.consume("", true) }

func (d *Decoder) consume(text string, finish bool) (Delta, error) {
	var content, reasoning strings.Builder
	if d.err != nil {
		return Delta{}, d.err
	}
	if d.ended {
		return Delta{}, nil
	}
	d.pending += text
	emit := func(s string) {
		if !utf8.ValidString(s) {
			d.err = fmt.Errorf("invalid UTF-8 in Harmony completion")
			return
		}
		if d.inHeader {
			d.header += s
			if len(d.header) > 4096 {
				d.err = fmt.Errorf("Harmony header exceeds 4096 bytes")
			}
			return
		}
		if d.channel == "analysis" {
			reasoning.WriteString(s)
		} else if d.channel == "final" {
			content.WriteString(s)
		}
	}
	for len(d.pending) > 0 && d.err == nil && !d.ended {
		at, marker := -1, ""
		for _, candidate := range markers {
			if p := strings.Index(d.pending, candidate); p >= 0 && (at < 0 || p < at) {
				at, marker = p, candidate
			}
		}
		if at < 0 {
			keep := 0
			for _, candidate := range markers {
				for n := 1; n < len(candidate) && n <= len(d.pending); n++ {
					if strings.HasSuffix(d.pending, candidate[:n]) && n > keep {
						keep = n
					}
				}
			}
			end := len(d.pending) - keep
			// Reject malformed bytes immediately. Retain at most one incomplete
			// codepoint, rather than buffering a permanently invalid stream.
			for offset := 0; offset < end; {
				r, size := utf8.DecodeRuneInString(d.pending[offset:end])
				if r == utf8.RuneError && size == 1 {
					if utf8.FullRuneInString(d.pending[offset:end]) {
						d.err = fmt.Errorf("invalid UTF-8 in Harmony completion")
					} else {
						end = offset
					}
					break
				}
				offset += size
			}
			if d.err != nil {
				break
			}
			if finish && keep > 0 {
				d.err = fmt.Errorf("incomplete Harmony control marker")
				break
			}
			if finish && end != len(d.pending) {
				d.err = fmt.Errorf("incomplete UTF-8 in Harmony completion")
				break
			}
			emit(d.pending[:end])
			d.pending = d.pending[end:]
			break
		}
		emit(d.pending[:at])
		d.pending = d.pending[at+len(marker):]
		switch marker {
		case "<|start|>", "<|im_start|>":
			d.inHeader = true
			d.header = ""
			d.channel = ""
		case "<|channel|>", "<|meta_sep|>":
			if !d.inHeader {
				d.err = fmt.Errorf("Harmony channel marker outside header")
				break
			}
			d.header += "\x00"
		case "<|constrain|>", "<|meta_start|>":
			d.err = fmt.Errorf("Harmony constrained format is not supported by the text decoder")
		case "<|message|>", "<|im_sep|>":
			if !d.inHeader {
				d.err = fmt.Errorf("Harmony body marker outside header")
				break
			}
			parts := strings.SplitN(d.header, "\x00", 2)
			role := strings.TrimSpace(parts[0])
			if role != "assistant" {
				d.err = fmt.Errorf("unsupported Harmony speaker or recipient %q", role)
				break
			}
			d.channel = "final"
			if len(parts) == 2 {
				d.channel = strings.TrimSpace(parts[1])
			}
			if d.channel != "analysis" && d.channel != "final" && d.channel != "commentary" {
				d.err = fmt.Errorf("unsupported Harmony channel %q", d.channel)
				break
			}
			d.inHeader = false
			d.header = ""
		case "<|end|>", "<|im_end|>":
			if d.inHeader {
				d.err = fmt.Errorf("Harmony message ended inside header")
				break
			}
			d.inHeader = true
			d.header = ""
			d.channel = ""
		case "<|return|>", "<|fim_suffix|>":
			if d.inHeader {
				d.err = fmt.Errorf("Harmony turn ended inside header")
				break
			}
			d.ended = true
			d.pending = ""
		case "<|call|>", "<|ghissue|>":
			d.err = fmt.Errorf("Harmony tool handoff is unsupported by text-only inference")
		}
	}
	if finish && d.inHeader && strings.TrimSpace(d.header) != "" && d.err == nil {
		d.err = fmt.Errorf("incomplete Harmony message header")
	}
	return Delta{content.String(), reasoning.String()}, d.err
}

func Split(text string) (Delta, error) {
	d := NewDecoder()
	a, err := d.Write(text)
	if err != nil {
		return a, err
	}
	b, err := d.Finish()
	return Delta{a.Content + b.Content, a.Reasoning + b.Reasoning}, err
}
