//go:build cgo

package decision

/*
#cgo CFLAGS: -I${SRCDIR}/../../../core/include
#cgo LDFLAGS: -L${SRCDIR}/../../../build -Wl,-rpath,${SRCDIR}/../../../build -ldensecore_decision -lstdc++
#include <stdlib.h>
#include "densecore/decision.h"
*/
import "C"
import (
	"encoding/json"
	"fmt"
	"sync"
	"unsafe"
)

type native struct {
	mu        sync.Mutex
	handle    *C.DenseCoreDecisionHandle
	tokenizer *Tokenizer
	cfg       Config
}

func Open(path string, threads int) (Backend, error) {
	if threads < 1 || threads > 256 {
		return nil, fmt.Errorf("threads must be between 1 and 256")
	}
	p := C.CString(path)
	defer C.free(unsafe.Pointer(p))
	var msg [4096]C.char
	h := C.DenseCoreDecisionLoad(p, C.int(threads), &msg[0], C.size_t(len(msg)))
	if h == nil {
		return nil, fmt.Errorf("load Laya: %s", C.GoString(&msg[0]))
	}
	n := &native{handle: h}
	ok := false
	defer func() {
		if !ok {
			n.Close()
		}
	}()
	var meta map[string]json.RawMessage
	if e := json.Unmarshal([]byte(C.GoString(C.DenseCoreDecisionMetadata(h))), &meta); e != nil {
		return nil, e
	}
	integer := func(key string, def int) int {
		v := def
		if raw, ok := meta[key]; ok {
			_ = json.Unmarshal(raw, &v)
		}
		return v
	}
	c := Config{MaxLen: integer("laya.max_len", 512), HeadMaxLen: integer("laya.head_max_len", 192), MaskText: "[MASK]", Temperature: [3]float64{1, 1, 1}}
	for i := 0; i < 3; i++ {
		if raw, ok := meta[fmt.Sprintf("laya.temperature.%d", i)]; ok {
			if e := json.Unmarshal(raw, &c.Temperature[i]); e != nil {
				return nil, e
			}
		}
	}
	if raw, exists := meta["laya.temperature_by_options"]; exists {
		var encoded string
		if json.Unmarshal(raw, &encoded) == nil {
			raw = json.RawMessage(encoded)
		}
		if e := json.Unmarshal(raw, &c.Buckets); e != nil {
			return nil, e
		}
	}
	var e error
	n.tokenizer, e = NewTokenizer([]byte(C.GoString(C.DenseCoreDecisionTokenizerJSON(h))))
	if e != nil {
		return nil, e
	}
	for _, s := range []struct {
		name string
		out  *int32
	}{{"[CLS]", &c.CLS}, {"[SEP]", &c.SEP}, {"[MASK]", &c.Mask}} {
		v, exists := n.tokenizer.TokenID(s.name)
		if !exists {
			return nil, fmt.Errorf("tokenizer missing %s", s.name)
		}
		*s.out = v
	}
	if c.MaxLen < 16 || c.MaxLen > 8192 || c.HeadMaxLen < 16 || c.HeadMaxLen > c.MaxLen {
		return nil, fmt.Errorf("unsupported Laya context configuration")
	}
	n.cfg = c
	ok = true
	return n, nil
}
func (n *native) Config() Config                     { return n.cfg }
func (n *native) Tokenize(s string) ([]int32, error) { return n.tokenizer.Encode(s) }
func (n *native) Close() {
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.handle != nil {
		C.DenseCoreDecisionFree(n.handle)
		n.handle = nil
	}
}
func (n *native) Infer(ids, markers []int32, qtype int) ([]float32, []float32, error) {
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.handle == nil {
		return nil, nil, fmt.Errorf("decision model is closed")
	}
	if len(ids) == 0 || len(markers) == 0 {
		return nil, nil, fmt.Errorf("empty decision input")
	}
	logits := make([]float32, len(markers))
	act := make([]float32, 2)
	var msg [4096]C.char
	ok := C.DenseCoreDecisionPredict(n.handle, (*C.int32_t)(unsafe.Pointer(&ids[0])), C.size_t(len(ids)), C.int(qtype), (*C.int32_t)(unsafe.Pointer(&markers[0])), C.size_t(len(markers)), (*C.float)(unsafe.Pointer(&logits[0])), (*C.float)(unsafe.Pointer(&act[0])), &msg[0], C.size_t(len(msg)))
	if ok == 0 {
		return nil, nil, fmt.Errorf("Laya inference: %s", C.GoString(&msg[0]))
	}
	return logits, act, nil
}
