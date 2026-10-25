// Benchmark the serving encoder pipeline without HTTP, JSON parsing or model load.
package main

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"maps"
	"math"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"syscall"
	"time"

	"github.com/DenseAI/DenseCore/server/internal/decision"
)

type sequence struct {
	IDs     []int32   `json:"token_ids"`
	Keys    []*string `json:"token_keys"`
	Mask    []int     `json:"attention_mask"`
	Markers []int32   `json:"markers"`
	QType   int       `json:"question_type"`
}
type requestCase struct {
	ID        string           `json:"id"`
	Raw       json.RawMessage  `json:"request"`
	Truncated *bool            `json:"truncated"`
	Sequences []sequence       `json:"sequences"`
	Request   decision.Request `json:"-"`
	Useful    int              `json:"-"`
	Compute   int              `json:"-"`
}
type workload struct {
	Schema     string            `json:"schema"`
	Provenance map[string]string `json:"provenance"`
	Requests   []requestCase     `json:"requests"`
}

func loadWorkload(path string) (workload, error) {
	var w workload
	data, err := os.ReadFile(path)
	if err != nil {
		return w, err
	}
	if err = json.Unmarshal(data, &w); err != nil {
		return w, err
	}
	if w.Schema != "systemone-workload-v1" || len(w.Requests) == 0 {
		return w, fmt.Errorf("nonempty systemone-workload-v1 required")
	}
	seen := map[string]bool{}
	for i := range w.Requests {
		c := &w.Requests[i]
		if c.ID == "" || seen[c.ID] || c.Truncated == nil || *c.Truncated {
			return w, fmt.Errorf("invalid duplicate/truncated/unverified request")
		}
		seen[c.ID] = true
		c.Request, err = decision.Parse(c.Raw)
		if err != nil {
			return w, err
		}
		if len(c.Sequences) != len(c.Request.Questions) {
			return w, fmt.Errorf("question receipt count mismatch")
		}
		keys := map[string]int32{}
		for j, s := range c.Sequences {
			if len(s.IDs) == 0 || len(s.IDs) != len(s.Keys) || len(s.Markers) == 0 || s.QType != c.Request.Questions[j].QType {
				return w, fmt.Errorf("invalid sequence receipt")
			}
			if s.Mask != nil && (len(s.Mask) != len(s.IDs) || slices.ContainsFunc(s.Mask, func(v int) bool { return v != 1 })) {
				return w, fmt.Errorf("pipeline native ABI requires unpadded receipts")
			}
			local := map[string]bool{}
			for k, id := range s.IDs {
				if id < 0 {
					return w, fmt.Errorf("negative token")
				}
				if s.Keys[k] == nil {
					continue
				}
				key := *s.Keys[k]
				if key == "" || local[key] {
					return w, fmt.Errorf("duplicate/empty logical token key")
				}
				local[key] = true
				if old, ok := keys[key]; ok && old != id {
					return w, fmt.Errorf("shared logical key changes token")
				}
				keys[key] = id
			}
			for _, m := range s.Markers {
				if m < 0 || int(m) >= len(s.IDs) {
					return w, fmt.Errorf("marker out of range")
				}
			}
			c.Compute += len(s.IDs)
		}
		c.Useful = len(keys)
		if c.Useful == 0 {
			return w, fmt.Errorf("no useful tokens")
		}
	}
	return w, nil
}

// Instrumentation covers actual tokenizer/native calls. The remaining measured
// pipeline time includes construction, collation, calibration and map formatting.
type measuredBackend struct {
	decision.Backend
	TokenizeTime time.Duration
	InferTime    time.Duration
	Inputs       []decision.BatchInput
}

func (b *measuredBackend) Tokenize(s string) ([]int32, error) {
	t := time.Now()
	out, e := b.Backend.Tokenize(s)
	b.TokenizeTime += time.Since(t)
	return out, e
}
func (b *measuredBackend) Infer(ids, markers []int32, qtype int) ([]float32, []float32, error) {
	b.Inputs = append(b.Inputs, decision.BatchInput{IDs: ids, Markers: markers, QType: qtype})
	t := time.Now()
	v, a, e := b.Backend.Infer(ids, markers, qtype)
	b.InferTime += time.Since(t)
	return v, a, e
}
func (b *measuredBackend) InferBatch(in []decision.BatchInput) ([]decision.BatchOutput, error) {
	b.Inputs = append(b.Inputs, in...)
	t := time.Now()
	out, e := b.Backend.(decision.BatchBackend).InferBatch(in)
	b.InferTime += time.Since(t)
	return out, e
}
func (b *measuredBackend) reset() { b.TokenizeTime = 0; b.InferTime = 0; b.Inputs = nil }

type record struct {
	RunID      string             `json:"run_id"`
	Attempt    int                `json:"attempt_id"`
	BatchID    int                `json:"batch_id"`
	BatchSize  int                `json:"batch_size"`
	WorkloadID string             `json:"workload_id"`
	Start      float64            `json:"start_s"`
	End        float64            `json:"end_s"`
	Useful     int                `json:"useful_tokens"`
	Compute    int                `json:"compute_tokens"`
	Success    bool               `json:"success"`
	Error      *string            `json:"error"`
	Unresolved bool               `json:"unresolved_backend_work"`
	Response   map[string]any     `json:"response"`
	Stage      map[string]float64 `json:"batch_stage_s"`
}

func checkReceipts(rows []requestCase, inputs []decision.BatchInput, outputs []map[string]any) error {
	if len(rows) != len(outputs) {
		return fmt.Errorf("response count mismatch")
	}
	index := 0
	for i, c := range rows {
		for _, seq := range c.Sequences {
			if index >= len(inputs) {
				return fmt.Errorf("missing actual encoder input")
			}
			in := inputs[index]
			index++
			if !slices.Equal(in.IDs, seq.IDs) || !slices.Equal(in.Markers, seq.Markers) || in.QType != seq.QType {
				return fmt.Errorf("actual tokenizer/build input differs from fixed receipt")
			}
		}
		usage, ok := outputs[i]["usage"].(map[string]any)
		if !ok {
			return fmt.Errorf("missing usage")
		}
		if usage["input_tokens"] != c.Compute || usage["truncated"] != false || usage["state_tokens_dropped"] != 0 {
			return fmt.Errorf("actual usage mismatch or truncation")
		}
		collapsed, ok := usage["collapsed_options"].(map[string]any)
		if !ok || len(collapsed) != 0 {
			return fmt.Errorf("collapsed or unverified options")
		}
		answers, ok := outputs[i]["answers"].(map[string]any)
		if !ok || len(answers) != len(c.Request.Questions) {
			return fmt.Errorf("answer count mismatch")
		}
	}
	if index != len(inputs) {
		return fmt.Errorf("unexpected internal encoder repeats")
	}
	return nil
}
func execute(ctx context.Context, b *measuredBackend, rows []requestCase) ([]map[string]any, time.Duration, error) {
	requests := make([]decision.Request, len(rows))
	for i := range rows {
		requests[i] = rows[i].Request
	}
	b.reset()
	start := time.Now()
	var out []map[string]any
	var err error
	if len(rows) == 1 {
		var one map[string]any
		one, err = decision.Predict(ctx, b, requests[0])
		out = []map[string]any{one}
	} else {
		out, err = decision.PredictBatch(ctx, b, requests)
	}
	elapsed := time.Since(start)
	if err == nil {
		err = checkReceipts(rows, b.Inputs, out)
	} // Proof checks are outside pipeline timing.
	return out, elapsed, err
}
func measure(ctx context.Context, b *measuredBackend, cases []requestCase, batch int, duration time.Duration, runID string) ([]record, map[string]any) {
	origin := time.Now()
	deadline := origin.Add(duration)
	var records []record
	attempt := 0
	active := time.Duration(0)
	var tokenizerTotal, nativeTotal time.Duration
	for time.Now().Before(deadline) && ctx.Err() == nil {
		rows := make([]requestCase, batch)
		for j := range rows {
			rows[j] = cases[(attempt+j)%len(cases)]
		}
		start := time.Since(origin).Seconds()
		out, took, err := execute(ctx, b, rows)
		end := time.Since(origin).Seconds()
		active += took
		tokenizerTotal += b.TokenizeTime
		nativeTotal += b.InferTime
		stage := map[string]float64{"pipeline": took.Seconds(), "tokenize": b.TokenizeTime.Seconds(), "native": b.InferTime.Seconds(), "prepare_collate_decode_format": (took - b.TokenizeTime - b.InferTime).Seconds()}
		var message *string
		if err != nil {
			s := err.Error()
			message = &s
		}
		for j, c := range rows {
			var response map[string]any
			if j < len(out) {
				response = out[j]
			}
			records = append(records, record{runID, attempt + j, attempt, batch, c.ID, start, end, c.Useful, c.Compute, err == nil, message, false, response, stage})
		}
		attempt += batch
	}
	elapsed := time.Since(origin).Seconds()
	successful, useful, compute, decisions := 0, 0, 0, 0
	var latencies []float64
	for _, r := range records {
		if r.Success {
			successful++
			useful += r.Useful
			compute += r.Compute
			latencies = append(latencies, r.End-r.Start)
			decisions += len(r.Response["answers"].(map[string]any))
		}
	}
	slices.Sort(latencies)
	percentile := func(q float64) any {
		if len(latencies) == 0 {
			return nil
		}
		return latencies[min(len(latencies)-1, int(math.Ceil(q*float64(len(latencies))))-1)]
	}
	return records, map[string]any{"elapsed_s": elapsed, "submission_stop_s": duration.Seconds(), "pipeline_active_s": active.Seconds(), "record_count": len(records),
		"successful_requests": successful, "useful_tokens": useful, "successful_compute_tokens": compute, "decisions": decisions,
		"tokens_per_s": float64(useful) / elapsed, "decisions_per_s": float64(decisions) / elapsed,
		"error_rate": func() float64 {
			if len(records) == 0 {
				return 0
			}
			return 1 - float64(successful)/float64(len(records))
		}(),
		"latency_s":            map[string]any{"p50": percentile(.5), "p95": percentile(.95), "p99": percentile(.99)},
		"batch_stage_totals_s": map[string]float64{"pipeline": active.Seconds(), "tokenize": tokenizerTotal.Seconds(), "native": nativeTotal.Seconds(), "prepare_collate_decode_format": (active - tokenizerTotal - nativeTotal).Seconds()}}

}

// Canonical bytes: encoding/json.Marshal of an ASCII string-key map, sorted keys,
// no whitespace and no trailing newline; identical to Python sort_keys with compact separators.
func pipelineImplementation(executableHash, nativeHash string) string {
	raw, _ := json.Marshal(map[string]string{"executable_sha256": executableHash, "native_library_sha256": nativeHash})
	sum := sha256.Sum256(raw)
	return hex.EncodeToString(sum[:])
}

func hashFile(path string) (string, error) {
	f, e := os.Open(path)
	if e != nil {
		return "", e
	}
	defer f.Close()
	h := sha256.New()
	if _, e = io.Copy(h, f); e != nil {
		return "", e
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
func nativeIdentity() (map[string]string, error) {
	raw, e := os.ReadFile("/proc/self/maps")
	if e != nil {
		return nil, e
	}
	paths := map[string]bool{}
	for _, line := range strings.Split(string(raw), "\n") {
		fields := strings.Fields(line)
		if len(fields) >= 6 && strings.Contains(filepath.Base(fields[5]), "libdensecore_decision") {
			paths[strings.Join(fields[5:], " ")] = true
		}
	}
	if len(paths) != 1 {
		return nil, fmt.Errorf("cannot uniquely establish loaded native library identity")
	}
	for path := range paths {
		sum, e := hashFile(path)
		return map[string]string{"path": path, "sha256": sum}, e
	}
	panic("unreachable")
}
func nativeDependencies() (map[string]string, error) {
	raw, e := os.ReadFile("/proc/self/maps")
	if e != nil {
		return nil, e
	}
	dependencies := map[string]string{}
	for _, line := range strings.Split(string(raw), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 6 {
			continue
		}
		path := strings.Join(fields[5:], " ")
		name := filepath.Base(path)
		if !strings.HasPrefix(name, "libggml") || !strings.Contains(name, ".so") {
			continue
		}
		hash, e := hashFile(path)
		if e != nil {
			return nil, e
		}
		if old, ok := dependencies[name]; ok && old != hash {
			return nil, fmt.Errorf("conflicting loaded runtime dependency %s", name)
		}
		dependencies[name] = hash
	}
	for _, prefix := range []string{"libggml-cpu.so", "libggml-base.so"} {
		found := false
		for name := range dependencies {
			if name == prefix || strings.HasPrefix(name, prefix+".") {
				found = true
			}
		}
		if !found {
			return nil, fmt.Errorf("required loaded runtime dependency absent: %s", prefix)
		}
	}
	return dependencies, nil
}

func writeJSON(path string, value any) error {
	raw, e := json.MarshalIndent(value, "", "  ")
	if e != nil {
		return e
	}
	return os.WriteFile(path, append(raw, '\n'), 0600)
}
func mainRun() error {
	model := flag.String("model", "", "model GGUF")
	threads := flag.Int("threads", 1, "native CPU threads")
	precision := flag.String("precision", "fp32", "fp32 or q8_0")
	batch := flag.Int("batch-size", 1, "independent request batch 1..64 (native sequences chunked at 64)")
	duration := flag.Float64("duration", 120, "submission duration in seconds")
	warmup := flag.Int("warmup", 2, "warmup batch rounds")
	workloadPath := flag.String("workload", "", "fixed System One workload JSON")
	output := flag.String("output", "", "artifact prefix")
	flag.Parse()
	if *model == "" || *workloadPath == "" || *output == "" || *threads < 1 || *threads > 256 || *batch < 1 || *batch > 64 || *duration <= 0 || math.IsNaN(*duration) || math.IsInf(*duration, 0) || *duration > float64(math.MaxInt64)/float64(time.Second) || *warmup < 0 || (*precision != "fp32" && *precision != "q8_0") {
		return fmt.Errorf("invalid benchmark controls")
	}
	w, e := loadWorkload(*workloadPath)
	if e != nil {
		return e
	}
	modelHash, e := hashFile(*model)
	if e != nil {
		return e
	}
	if w.Provenance["model_sha256"] != modelHash {
		return fmt.Errorf("workload/model identity mismatch")
	}
	workloadHash, e := hashFile(*workloadPath)
	if e != nil {
		return e
	}
	exe, e := os.Executable()
	if e != nil {
		return e
	}
	exeHash, e := hashFile(exe)
	if e != nil {
		return e
	}
	native, e := nativeIdentity()
	if e != nil {
		return e
	}
	backend, e := decision.OpenWithOptions(*model, *threads, decision.OpenOptions{Precision: *precision})
	if e != nil {
		return e
	}
	defer backend.Close()
	identity, ok := backend.(interface{ TokenizerSHA256() string })
	if !ok {
		return fmt.Errorf("loaded backend cannot prove tokenizer identity")
	}
	tokenizerHash := identity.TokenizerSHA256()
	if tokenizerHash != w.Provenance["tokenizer_sha256"] {
		return fmt.Errorf("actual loaded tokenizer differs from workload provenance")
	}
	measured := &measuredBackend{Backend: backend}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	for i := 0; i < *warmup; i++ {
		rows := make([]requestCase, *batch)
		for j := range rows {
			rows[j] = w.Requests[(i**batch+j)%len(w.Requests)]
		}
		if _, _, e = execute(ctx, measured, rows); e != nil {
			return fmt.Errorf("warmup: %w", e)
		}
	}
	var runBytes [16]byte
	if _, e = rand.Read(runBytes[:]); e != nil {
		return e
	}
	runID := hex.EncodeToString(runBytes[:])
	var started time.Time
	dependencies, e := nativeDependencies()
	if e != nil {
		return e
	}
	implementationHash := pipelineImplementation(exeHash, native["sha256"])
	config := map[string]any{"runtime_dependencies_sha256": dependencies, "schema": "encoder-pipeline-config-v1", "boundary": "encoder_pipeline", "status": "diagnostic_only", "model_sha256": modelHash, "workload_sha256": workloadHash, "tokenizer_sha256": tokenizerHash, "implementation_sha256": implementationHash, "native_library_sha256": native["sha256"], "executable_sha256": exeHash, "precision": *precision, "batch_size": *batch, "threads": *threads, "result_cache": false}
	if e = os.MkdirAll(filepath.Dir(*output), 0700); e != nil {
		return e
	}
	if e = writeJSON(*output+".config.json", config); e != nil {
		return e
	}
	configHash, e := hashFile(*output + ".config.json")
	if e != nil {
		return e
	}
	started = time.Now().UTC()
	records, meta := measure(ctx, measured, w.Requests, *batch, time.Duration(*duration*float64(time.Second)), runID)
	meta["manifest_sha256"] = configHash // Diagnostic configuration, not a passed-quality economics manifest.
	afterDependencies, e := nativeDependencies()
	if e != nil {
		return e
	}
	if !maps.Equal(dependencies, afterDependencies) {
		return fmt.Errorf("runtime dependencies changed during measured run")
	}
	meta["runtime_dependencies_sha256"] = dependencies
	meta["runtime_dependencies_stable"] = true
	meta["runtime_dependencies_origin"] = "observed_process_maps"
	meta["implementation_sha256"] = implementationHash
	meta["native_library_sha256"] = native["sha256"]
	meta["schema"] = "systemone-run-v1"
	meta["boundary"] = "encoder_pipeline"
	meta["status"] = "diagnostic_only"
	meta["run_id"] = runID
	meta["started_at"] = started.Format(time.RFC3339Nano)
	meta["finished_at"] = time.Now().UTC().Format(time.RFC3339Nano)
	meta["threads"] = *threads
	meta["batch_size"] = *batch
	meta["concurrency"] = 1
	meta["warmup_rounds"] = *warmup
	meta["warmup_requests"] = *warmup * *batch
	meta["model_sha256"] = modelHash
	meta["workload_sha256"] = workloadHash
	meta["tokenizer_sha256"] = tokenizerHash
	meta["executable_sha256"] = exeHash
	meta["native_library"] = native
	meta["precision"] = *precision
	meta["resource_identity_status"] = "unavailable"
	meta["host_id"], _ = os.Hostname()
	meta["observed_host"] = map[string]any{"architecture": runtime.GOARCH, "os": runtime.GOOS, "logical_cpu_count": runtime.NumCPU()}
	meta["memory"] = map[string]any{"status": "unavailable", "reason": "native process RSS sampling not implemented in this Go runner"}
	meta["quality_verdict"] = "unverified"
	meta["economics_verdict"] = "unavailable"
	meta["stage_times"] = map[string]any{"scope": "per-batch; repeated on request records, do not sum per request", "includes": "prompt construction, tokenization, collation, forward, calibration and Go answer-map formatting", "excludes": "JSON parsing, JSON serialization, HTTP, model load, warmup, post-call receipt verification"}
	if ctx.Err() != nil {
		meta["submission_stop_s"] = math.Min(*duration, meta["elapsed_s"].(float64))
		meta["interrupted"] = true
	}
	if e = os.MkdirAll(filepath.Dir(*output), 0700); e != nil {
		return e
	}
	f, e := os.OpenFile(*output+".records.jsonl", os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0600)
	if e != nil {
		return e
	}
	encoder := json.NewEncoder(f)
	for _, record := range records {
		if e = encoder.Encode(record); e != nil {
			f.Close()
			return e
		}
	}
	if e = f.Close(); e != nil {
		return e
	}
	meta["records_sha256"], e = hashFile(*output + ".records.jsonl")
	if e != nil {
		return e
	}
	return writeJSON(*output+".metadata.json", meta)
}
func main() {
	if e := mainRun(); e != nil {
		fmt.Fprintln(os.Stderr, e)
		os.Exit(1)
	}
}
