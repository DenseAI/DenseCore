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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sync"
	"unsafe"
)

type native struct {
	mu              sync.Mutex
	handle          *C.DenseCoreDecisionHandle
	tokenizer       *Tokenizer
	cfg             Config
	tokenizerSHA256 string
}

func Open(path string, threads int) (Backend, error) {
	return OpenWithOptions(path, threads, OpenOptions{})
}
func OpenWithOptions(path string, threads int, options OpenOptions) (Backend, error) {
	precision, err := options.precision()
	if err != nil {
		return nil, err
	}
	if threads < 1 || threads > 256 {
		return nil, fmt.Errorf("threads must be between 1 and 256")
	}
	p := C.CString(path)
	defer C.free(unsafe.Pointer(p))
	var msg [4096]C.char
	h := C.DenseCoreDecisionLoadEx(p, C.int(threads), C.int(precision), &msg[0], C.size_t(len(msg)))
	if h == nil {
		return nil, fmt.Errorf("load decision model: %s", C.GoString(&msg[0]))
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
	architecture := "laya"
	if raw, exists := meta["general.architecture"]; exists {
		if err := json.Unmarshal(raw, &architecture); err != nil {
			return nil, err
		}
	}
	if architecture != "laya" && architecture != "julia" {
		return nil, fmt.Errorf("unsupported decision architecture %q", architecture)
	}
	prefix := architecture + "."
	maxLen, headLen := 512, 192
	names := []string{"[CLS]", "[SEP]", "[MASK]"}
	c := Config{Model: "laya", MaskText: "[MASK]", Temperature: [3]float64{1, 1, 1}}
	if architecture == "julia" {
		maxLen, headLen = 8192, 512
		c.Model = "julia-1"
		c.MaskText = "<mask>"
		c.StrictEncoding = true
		names = []string{"<bos>", "<eos>", "<mask>"}
	}
	c.MaxLen = integer(prefix+"max_len", maxLen)
	c.HeadMaxLen = integer(prefix+"head_max_len", headLen)
	c.ModelMaxLen = c.MaxLen
	nativeMax := integer("densecore.native_max_len", c.MaxLen)
	if nativeMax < 1 {
		return nil, fmt.Errorf("invalid native context limit")
	}
	c.MaxLen = min(c.MaxLen, nativeMax)
	if architecture == "laya" {
		for i := 0; i < 3; i++ {
			if raw, exists := meta[fmt.Sprintf("laya.temperature.%d", i)]; exists {
				if err := json.Unmarshal(raw, &c.Temperature[i]); err != nil {
					return nil, err
				}
			}
		}
		if raw, exists := meta["laya.temperature_by_options"]; exists {
			var encoded string
			if json.Unmarshal(raw, &encoded) == nil {
				raw = json.RawMessage(encoded)
			}
			if err := json.Unmarshal(raw, &c.Buckets); err != nil {
				return nil, err
			}
		}
	}
	var e error
	tokenizerRaw := []byte(C.GoString(C.DenseCoreDecisionTokenizerJSON(h)))
	sum := sha256.Sum256(tokenizerRaw)
	n.tokenizerSHA256 = hex.EncodeToString(sum[:])
	if architecture == "julia" {
		n.tokenizer, e = NewJuliaTokenizer(tokenizerRaw)
	} else {
		n.tokenizer, e = NewTokenizer(tokenizerRaw)
	}
	if e != nil {
		return nil, e
	}
	for i, out := range []*int32{&c.CLS, &c.SEP, &c.Mask} {
		id, exists := n.tokenizer.TokenID(names[i])
		if !exists {
			return nil, fmt.Errorf("tokenizer missing %s", names[i])
		}
		*out = id
	}
	if architecture == "julia" && (c.CLS != 2 || c.SEP != 1 || c.Mask != 4 || c.HeadMaxLen+4 >= c.MaxLen) {
		return nil, fmt.Errorf("unsupported Julia tokenizer/context configuration")
	}
	if c.MaxLen < 16 || c.MaxLen > 8192 || c.HeadMaxLen < 16 || c.HeadMaxLen > c.MaxLen {
		return nil, fmt.Errorf("unsupported decision context configuration")
	}
	n.cfg = c
	ok = true
	return n, nil
}
func (n *native) TokenizerSHA256() string            { return n.tokenizerSHA256 }
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
		return nil, nil, fmt.Errorf("decision inference: %s", C.GoString(&msg[0]))
	}
	return logits, act, nil
}

func (n *native) InferBatch(inputs []BatchInput) ([]BatchOutput, error) {
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.handle == nil {
		return nil, fmt.Errorf("decision model is closed")
	}
	if len(inputs) < 1 || len(inputs) > 64 {
		return nil, fmt.Errorf("native batch size must be 1..64")
	}
	// C-owned buffers keep nested pointer arrays valid under cgo pointer rules.
	var allocations []unsafe.Pointer
	defer func() {
		for _, p := range allocations {
			C.free(p)
		}
	}()
	alloc := func(bytes uintptr) unsafe.Pointer {
		p := C.malloc(C.size_t(bytes))
		allocations = append(allocations, p)
		return p
	}
	ip := alloc(uintptr(len(inputs)) * unsafe.Sizeof(C.DenseCoreDecisionInput{}))
	op := alloc(uintptr(len(inputs)) * unsafe.Sizeof(C.DenseCoreDecisionOutput{}))
	if ip == nil || op == nil {
		return nil, fmt.Errorf("allocate native batch")
	}
	ins := unsafe.Slice((*C.DenseCoreDecisionInput)(ip), len(inputs))
	outs := unsafe.Slice((*C.DenseCoreDecisionOutput)(op), len(inputs))
	for i, input := range inputs {
		if len(input.IDs) == 0 || len(input.Markers) == 0 {
			return nil, fmt.Errorf("empty decision input")
		}
		ids := alloc(uintptr(len(input.IDs)) * 4)
		markers := alloc(uintptr(len(input.Markers)) * 4)
		logits := alloc(uintptr(len(input.Markers)) * 4)
		action := alloc(8)
		if ids == nil || markers == nil || logits == nil || action == nil {
			return nil, fmt.Errorf("allocate native batch tensors")
		}
		copy(unsafe.Slice((*int32)(ids), len(input.IDs)), input.IDs)
		copy(unsafe.Slice((*int32)(markers), len(input.Markers)), input.Markers)
		ins[i] = C.DenseCoreDecisionInput{tokens: (*C.int32_t)(ids), token_count: C.size_t(len(input.IDs)), question_type: C.int(input.QType), markers: (*C.int32_t)(markers), marker_count: C.size_t(len(input.Markers))}
		outs[i] = C.DenseCoreDecisionOutput{logits: (*C.float)(logits), act_logits: (*C.float)(action)}
	}
	var msg [4096]C.char
	if C.DenseCoreDecisionPredictBatch(n.handle, (*C.DenseCoreDecisionInput)(ip), C.size_t(len(inputs)), (*C.DenseCoreDecisionOutput)(op), &msg[0], C.size_t(len(msg))) == 0 {
		return nil, fmt.Errorf("decision batch inference: %s", C.GoString(&msg[0]))
	}
	result := make([]BatchOutput, len(inputs))
	for i, input := range inputs {
		result[i].Logits = append([]float32(nil), unsafe.Slice((*float32)(unsafe.Pointer(outs[i].logits)), len(input.Markers))...)
		result[i].Action = append([]float32(nil), unsafe.Slice((*float32)(unsafe.Pointer(outs[i].act_logits)), 2)...)
	}
	return result, nil
}
