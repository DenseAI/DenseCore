/**
 * @file bench_native.cpp
 * @brief Native C++ benchmark for DenseCore vs llama.cpp comparison
 *
 * Measures TTFT, generation throughput (tok/s), and inter-token latency.
 * Links directly against libdensecore.so - no Python overhead.
 *
 * Build:
 *   g++ -O2 -o bench_native bench_native.cpp -I../core/include -L../core/build -ldensecore -lpthread
 * Run:
 *   LD_LIBRARY_PATH=../core/build ./bench_native /path/to/model.gguf [max_tokens] [threads]
 */

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <algorithm>
#include <cstdlib>

#include "densecore.h"

using Clock = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;

struct BenchState {
    std::mutex mu;
    std::condition_variable cv;
    std::atomic<bool> finished{false};
    std::atomic<int> token_count{0};

    TimePoint start_time;
    TimePoint first_token_time;
    TimePoint end_time;
    std::vector<TimePoint> token_times;
    bool got_first{false};
    bool got_end{false};
    bool record_token_times{false};

    struct BatchSync* sync{nullptr};
};

struct BatchSync {
    std::mutex mu;
    std::condition_variable cv;
    int finished_count{0};
    int target{0};
};

void BenchCallback(const char* token, int is_final, void* user_data) {
    auto* s = static_cast<BenchState*>(user_data);
    if (!s) return;
    (void)token;

    auto now = Clock::now();

    {
        std::lock_guard<std::mutex> lock(s->mu);
        if (is_final) {
            s->finished.store(true, std::memory_order_release);
            s->end_time = now;
            s->got_end = true;
        } else {
            if (!s->got_first) {
                s->first_token_time = now;
                s->got_first = true;
            }
            if (s->record_token_times) {
                s->token_times.push_back(now);
            }
            s->token_count.fetch_add(1, std::memory_order_relaxed);
        }
    }

    if (is_final && s->sync) {
        {
            std::lock_guard<std::mutex> lock(s->sync->mu);
            s->sync->finished_count++;
        }
        s->sync->cv.notify_one();
    }
    s->cv.notify_one();
}

struct BenchResult {
    double ttft_ms;
    double tps;
    double tps_steady;
    int tokens;
    double itl_avg_ms;
    double itl_p50_ms;
    double itl_p90_ms;
    double itl_p99_ms;
    double total_time_s;
};

enum class PromptMode {
    kText,
    kTokenIds,
};

struct PromptPayload {
    PromptMode mode{PromptMode::kText};
    std::string text;
    std::vector<int> ids;
};

enum class ThroughputMode {
    kDecodeExcludingFirst,
    kAllGenerated,
};

static bool ParseBoolEnv(const char* name, bool default_value) {
    const char* v = std::getenv(name);
    if (!v || !*v) return default_value;
    if (std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 || std::strcmp(v, "TRUE") == 0 ||
        std::strcmp(v, "yes") == 0 || std::strcmp(v, "on") == 0) {
        return true;
    }
    if (std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 || std::strcmp(v, "FALSE") == 0 ||
        std::strcmp(v, "no") == 0 || std::strcmp(v, "off") == 0) {
        return false;
    }
    return default_value;
}

static double PercentileFromSorted(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    if (p <= 0.0) return sorted.front();
    if (p >= 1.0) return sorted.back();

    const double pos = p * static_cast<double>(sorted.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    if (lo == hi) return sorted[lo];

    const double w = pos - static_cast<double>(lo);
    return sorted[lo] * (1.0 - w) + sorted[hi] * w;
}

static ThroughputMode ParseThroughputMode() {
    const char* env = std::getenv("DENSECORE_BENCH_TPS_MODE");
    if (!env || !*env) {
        return ThroughputMode::kDecodeExcludingFirst;
    }
    if (std::strcmp(env, "all") == 0 || std::strcmp(env, "ALL") == 0 ||
        std::strcmp(env, "llama") == 0 || std::strcmp(env, "llama_compatible") == 0) {
        return ThroughputMode::kAllGenerated;
    }
    return ThroughputMode::kDecodeExcludingFirst;
}

static std::vector<int> BuildFairPromptIds(int prompt_tokens) {
    std::vector<int> ids(static_cast<size_t>(prompt_tokens));
    for (int i = 0; i < prompt_tokens; ++i) {
        // Deterministic, low-range non-special IDs for exact prompt-length benchmarking.
        ids[static_cast<size_t>(i)] = 10 + (i % 50);
    }
    return ids;
}

static int SubmitOneRequest(DenseCoreHandle engine, const PromptPayload& prompt_payload, int max_tokens, BenchState* s) {
    if (prompt_payload.mode == PromptMode::kTokenIds) {
        return SubmitRequestIdsWithSamplingEx(
            engine,
            prompt_payload.ids.data(),
            static_cast<int>(prompt_payload.ids.size()),
            max_tokens,
            nullptr,   // lora
            0.0f,      // temperature: greedy
            1.0f,      // top_p
            1,         // top_k
            1.0f,      // repetition_penalty
            nullptr,   // stop_sequences
            0,         // json_mode
            BenchCallback,
            s);
    }

    return SubmitRequestWithSamplingEx(
        engine,
        prompt_payload.text.c_str(),
        max_tokens,
        nullptr,   // lora
        0.0f,      // temperature: greedy
        1.0f,      // top_p
        1,         // top_k
        1.0f,      // repetition_penalty
        nullptr,   // stop_sequences
        0,         // json_mode
        BenchCallback,
        s);
}

static bool SubmitBatchSequential(DenseCoreHandle engine, const PromptPayload& prompt_payload, int max_tokens,
                                  std::vector<BenchState>* states, const TimePoint& start_time) {
    for (auto& s : *states) {
        s.start_time = start_time;
    }
    for (size_t i = 0; i < states->size(); ++i) {
        int req_id = SubmitOneRequest(engine, prompt_payload, max_tokens, &(*states)[i]);
        if (req_id < 0) {
            fprintf(stderr, "SubmitRequest failed (sequential, idx=%zu): %s\n", i, DenseCoreGetLastError());
            return false;
        }
    }
    return true;
}

static bool SubmitBatchParallel(DenseCoreHandle engine, const PromptPayload& prompt_payload, int max_tokens,
                                std::vector<BenchState>* states, const TimePoint& launch_time) {
    const size_t n = states->size();
    if (n == 0) return true;

    for (auto& s : *states) {
        s.start_time = launch_time;
    }

    struct Barrier {
        std::mutex mu;
        std::condition_variable cv;
        size_t ready{0};
        bool go{false};
    };

    Barrier barrier;
    std::vector<int> submit_results(n, -1);
    std::vector<std::string> submit_errors(n);
    std::vector<std::thread> workers;
    workers.reserve(n);

    try {
        for (size_t i = 0; i < n; ++i) {
            workers.emplace_back([&, i] {
                {
                    std::unique_lock<std::mutex> lock(barrier.mu);
                    barrier.ready++;
                    barrier.cv.notify_one();
                    barrier.cv.wait(lock, [&] { return barrier.go; });
                }
                const int req_id = SubmitOneRequest(engine, prompt_payload, max_tokens, &(*states)[i]);
                submit_results[i] = req_id;
                if (req_id < 0) {
                    const char* err = DenseCoreGetLastError();
                    submit_errors[i] = err ? err : "unknown submit error";
                }
            });
        }
    } catch (...) {
        fprintf(stderr, "Parallel submit setup failed; falling back to sequential submit.\n");
        for (auto& t : workers) {
            if (t.joinable()) t.join();
        }
        return SubmitBatchSequential(engine, prompt_payload, max_tokens, states, launch_time);
    }

    {
        std::unique_lock<std::mutex> lock(barrier.mu);
        barrier.cv.wait(lock, [&] { return barrier.ready == n; });
        barrier.go = true;
    }
    barrier.cv.notify_all();

    for (auto& t : workers) {
        if (t.joinable()) t.join();
    }

    for (size_t i = 0; i < n; ++i) {
        if (submit_results[i] < 0) {
            fprintf(stderr, "SubmitRequest failed (parallel, idx=%zu): %s\n", i, submit_errors[i].c_str());
            return false;
        }
    }
    return true;
}

BenchResult RunBenchmark(DenseCoreHandle engine, const PromptPayload& prompt_payload, int max_tokens, int batch_size,
                         bool parallel_submit, bool submit_thread_safe, bool record_token_times,
                         ThroughputMode throughput_mode) {
    if (batch_size < 1) batch_size = 1;

    std::vector<BenchState> states(static_cast<size_t>(batch_size));
    BatchSync sync;
    sync.target = batch_size;

    for (auto& s : states) {
        s.token_times.reserve(max_tokens + 16);
        s.sync = &sync;
        s.record_token_times = record_token_times;
    }

    const TimePoint start_time = Clock::now();
    const bool use_parallel = parallel_submit && submit_thread_safe && batch_size > 1;
    bool submit_ok = false;
    if (use_parallel) {
        submit_ok = SubmitBatchParallel(engine, prompt_payload, max_tokens, &states, start_time);
    } else {
        submit_ok = SubmitBatchSequential(engine, prompt_payload, max_tokens, &states, start_time);
    }
    if (!submit_ok) return {};

    {
        std::unique_lock<std::mutex> lock(sync.mu);
        sync.cv.wait_for(lock, std::chrono::seconds(240), [&] { return sync.finished_count >= sync.target; });
    }

    if (sync.finished_count < sync.target) {
        fprintf(stderr, "Timeout waiting for generation (batch=%d, done=%d)\n", batch_size, sync.finished_count);
        return {};
    }

    TimePoint end_time = start_time;
    TimePoint min_first_token_time{};
    TimePoint max_first_token_time{};
    bool has_first = false;
    bool all_have_first = true;

    // Calculate metrics
    BenchResult r{};
    r.tokens = 0;
    int total_decode_tokens = 0;
    int total_generated_tokens = 0;
    double sum_ttft_ms = 0.0;
    int ttft_count = 0;

    std::vector<double> itl_ms;
    itl_ms.reserve(static_cast<size_t>(batch_size) * static_cast<size_t>(max_tokens));

    for (auto& s : states) {
        r.tokens += s.token_count.load();
        if (s.got_end && s.end_time > end_time) {
            end_time = s.end_time;
        }
        if (s.got_first) {
            const double ttft_ms = std::chrono::duration<double, std::milli>(s.first_token_time - s.start_time).count();
            sum_ttft_ms += ttft_ms;
            ttft_count++;
            if (!has_first || s.first_token_time < min_first_token_time) {
                min_first_token_time = s.first_token_time;
            }
            if (!has_first || s.first_token_time > max_first_token_time) {
                max_first_token_time = s.first_token_time;
            }
            has_first = true;
        } else {
            all_have_first = false;
        }
        const int token_count = s.token_count.load();
        if (token_count > 0) {
            total_generated_tokens += token_count;
        }
        if (token_count > 1) {
            total_decode_tokens += (token_count - 1);
        }

        for (size_t i = 1; i < s.token_times.size(); ++i) {
            double dt = std::chrono::duration<double, std::milli>(
                s.token_times[i] - s.token_times[i - 1]).count();
            itl_ms.push_back(dt);
        }
    }

    r.total_time_s = std::chrono::duration<double>(end_time - start_time).count();
    if (ttft_count > 0) {
        r.ttft_ms = sum_ttft_ms / ttft_count;
    }
    // Generation throughput:
    // - decode mode: exclude per-sequence first token (legacy decode-only view)
    // - all mode: include all generated tokens after first token arrival
    const int throughput_tokens =
        (throughput_mode == ThroughputMode::kAllGenerated) ? total_generated_tokens : total_decode_tokens;
    if (has_first && throughput_tokens > 0) {
        const double decode_time_s = std::chrono::duration<double>(end_time - min_first_token_time).count();
        if (decode_time_s > 0.0) {
            r.tps = static_cast<double>(throughput_tokens) / decode_time_s;
        }
    }
    // Steady-state decode throughput:
    // t_start = max(first_token_time_i), count tokens at/after t_start only.
    if (all_have_first && has_first) {
        int tokens_after_t_start = 0;
        if (record_token_times) {
            for (const auto& s : states) {
                for (const auto& t : s.token_times) {
                    if (t >= max_first_token_time) tokens_after_t_start++;
                }
            }
        } else {
            // Approximation without per-token timestamps:
            // - If a sequence first token is at/after t_start, count all tokens.
            // - Otherwise exclude that sequence's first token.
            for (const auto& s : states) {
                const int tc = s.token_count.load();
                if (tc <= 0) continue;
                if (s.first_token_time >= max_first_token_time) {
                    tokens_after_t_start += tc;
                } else {
                    tokens_after_t_start += std::max(0, tc - 1);
                }
            }
        }
        if (tokens_after_t_start > 0) {
            const double steady_time_s = std::chrono::duration<double>(end_time - max_first_token_time).count();
            if (steady_time_s > 0.0) {
                r.tps_steady = static_cast<double>(tokens_after_t_start) / steady_time_s;
            }
        }
    }

    if (!itl_ms.empty()) {
        std::sort(itl_ms.begin(), itl_ms.end());
        double sum = 0;
        for (auto v : itl_ms) sum += v;
        const size_t n = itl_ms.size();
        r.itl_avg_ms = sum / n;
        r.itl_p50_ms = PercentileFromSorted(itl_ms, 0.50);
        r.itl_p90_ms = PercentileFromSorted(itl_ms, 0.90);
        r.itl_p99_ms = PercentileFromSorted(itl_ms, 0.99);
    }

    return r;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <model.gguf> [prompt_tokens=128] [gen_tokens=128] [threads=0] [runs=2] "
                "[--batch-size N] [--fair-mode] [--parallel-submit] [--csv]\n",
                argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    int prompt_tokens = argc > 2 ? std::max(1, atoi(argv[2])) : 128;
    int gen_tokens = argc > 3 ? std::max(1, atoi(argv[3])) : 128;
    int threads = argc > 4 ? atoi(argv[4]) : 0;
    int runs = argc > 5 ? std::max(1, atoi(argv[5])) : 2;
    int batch_size = 1;
    bool csv_only = false;
    bool fair_mode = false;
    bool parallel_submit = false;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--csv") == 0) {
            csv_only = true;
            continue;
        }
        if (std::strcmp(argv[i], "--fair-mode") == 0) {
            fair_mode = true;
            continue;
        }
        if (std::strcmp(argv[i], "--parallel-submit") == 0) {
            parallel_submit = true;
            continue;
        }
        if (std::strcmp(argv[i], "--batch-size") == 0 && (i + 1) < argc) {
            batch_size = std::max(1, atoi(argv[i + 1]));
            ++i;
        }
    }
    const bool submit_thread_safe = ParseBoolEnv("DENSECORE_BENCH_SUBMIT_THREAD_SAFE", true);
    const bool record_token_times = ParseBoolEnv("DENSECORE_BENCH_RECORD_TOKEN_TIMES", false);
    const ThroughputMode throughput_mode = ParseThroughputMode();

#if defined(_WIN32)
    _putenv_s("DENSECORE_BENCH_MODE", "1");
#else
    setenv("DENSECORE_BENCH_MODE", "1", 0);
#endif

    if (!csv_only) {
        printf("================================================================\n");
        printf("  DenseCore Native Benchmark\n");
        printf("================================================================\n");
        printf("  Model:         %s\n", model_path);
        printf("  Prompt tokens: %d\n", prompt_tokens);
        printf("  Gen tokens:    %d\n", gen_tokens);
        printf("  Batch size:    %d\n", batch_size);
        printf("  Threads:       %d (0=auto)\n", threads);
        printf("  Runs:          %d\n", runs);
        printf("  Fair mode:     %s\n", fair_mode ? "on (exact prompt token IDs)" : "off (text prompt)");
        printf("  Parallel submit: %s\n", parallel_submit ? "on" : "off");
        printf("  Submit thread-safe: %s (set DENSECORE_BENCH_SUBMIT_THREAD_SAFE=0 to force sequential)\n",
               submit_thread_safe ? "yes" : "no");
        printf("  Throughput mode: %s (set DENSECORE_BENCH_TPS_MODE=all|decode)\n",
               throughput_mode == ThroughputMode::kAllGenerated ? "all_generated(llama-compatible)" : "decode_only");
        printf("  Record token times: %s (set DENSECORE_BENCH_RECORD_TOKEN_TIMES=1 for ITL precision)\n",
               record_token_times ? "yes" : "no");
        printf("================================================================\n\n");
    }

    // Init engine
    if (!csv_only) printf("[1/4] Initializing engine...\n");
    auto t0 = Clock::now();
    DenseCoreHandle engine = InitEngine(model_path, nullptr, threads);
    auto t1 = Clock::now();

    if (!engine) {
        fprintf(stderr, "InitEngine failed: %s\n", DenseCoreGetLastError());
        return 1;
    }

    double load_time = std::chrono::duration<double>(t1 - t0).count();
    if (!csv_only) printf("  Model loaded in %.2f s\n\n", load_time);

    PromptPayload prompt_payload;
    if (fair_mode) {
        prompt_payload.mode = PromptMode::kTokenIds;
        prompt_payload.ids = BuildFairPromptIds(prompt_tokens);
    } else {
        prompt_payload.mode = PromptMode::kText;
        prompt_payload.text = "Write a comprehensive guide to ";
        const int repeat = std::max(1, prompt_tokens / 6);
        for (int i = 1; i < repeat; ++i) {
            prompt_payload.text += "Write a comprehensive guide to ";
        }
    }

    // Warmup (full generation run) - excluded from measured statistics.
    if (!csv_only) printf("Performing warmup run...\n");
    (void)RunBenchmark(engine, prompt_payload, gen_tokens, batch_size, parallel_submit, submit_thread_safe,
                       record_token_times, throughput_mode);
    if (!csv_only) printf("Warmup run complete.\n\n");

    double sum_ttft = 0.0;
    double sum_tps = 0.0;
    double sum_tps_steady = 0.0;
    double sum_itl_avg = 0.0;
    double sum_itl_p50 = 0.0;
    double sum_itl_p90 = 0.0;
    double sum_itl_p99 = 0.0;
    int last_tokens = 0;

    if (!csv_only) printf("[3/4] Benchmark runs (%d x %d tokens, batch=%d)...\n", runs, gen_tokens, batch_size);
    for (int i = 0; i < runs; ++i) {
        BenchResult r = RunBenchmark(engine, prompt_payload, gen_tokens, batch_size, parallel_submit, submit_thread_safe,
                                     record_token_times, throughput_mode);
        sum_ttft += r.ttft_ms;
        sum_tps += r.tps;
        sum_tps_steady += r.tps_steady;
        sum_itl_avg += r.itl_avg_ms;
        sum_itl_p50 += r.itl_p50_ms;
        sum_itl_p90 += r.itl_p90_ms;
        sum_itl_p99 += r.itl_p99_ms;
        last_tokens = r.tokens;
        if (!csv_only) {
            printf("  Run %d/%d: %d tokens, %.2f t/s, TTFT %.2f ms\n", i + 1, runs, r.tokens, r.tps, r.ttft_ms);
        }
    }

    if (!csv_only) printf("\n[4/4] Finalizing report...\n");

    const double inv_runs = 1.0 / runs;
    const double avg_ttft = sum_ttft * inv_runs;
    const double avg_tps = sum_tps * inv_runs;
    const double avg_tps_steady = sum_tps_steady * inv_runs;
    const double avg_itl_avg = sum_itl_avg * inv_runs;
    const double avg_itl_p50 = sum_itl_p50 * inv_runs;
    const double avg_itl_p90 = sum_itl_p90 * inv_runs;
    const double avg_itl_p99 = sum_itl_p99 * inv_runs;

    // Report
    if (!csv_only) {
        printf("================================================================\n");
        printf("  Results (avg of %d runs)\n", runs);
        printf("================================================================\n");
        printf("  +--------------------------+------------+\n");
        printf("  | Metric                   | Value      |\n");
        printf("  +--------------------------+------------+\n");
        printf("  | Model Load Time          | %7.2f s  |\n", load_time);
        printf("  | Time To First Token      | %7.2f ms |\n", avg_ttft);
        printf("  | Generation Speed         | %7.2f t/s|\n", avg_tps);
        printf("  | Generation Speed (steady)| %7.2f t/s|\n", avg_tps_steady);
        printf("  | Inter-Token Latency avg  | %7.2f ms |\n", avg_itl_avg);
        printf("  | Inter-Token Latency p50  | %7.2f ms |\n", avg_itl_p50);
        printf("  | Inter-Token Latency p90  | %7.2f ms |\n", avg_itl_p90);
        printf("  | Inter-Token Latency p99  | %7.2f ms |\n", avg_itl_p99);
        printf("  | Tokens Generated         | %7d    |\n", last_tokens);
        printf("  +--------------------------+------------+\n");

        // Version
        const char* ver = GetLibraryVersionString();
        if (ver) printf("\n  DenseCore version: %s\n", ver);
    }

    FreeEngine(engine);
    if (csv_only) {
        // CSV columns:
        // ttft_ms,tps,tps_steady,tokens_generated,itl_avg_ms,itl_p50_ms,itl_p90_ms,itl_p99_ms
        printf("%.2f,%.2f,%.2f,%d,%.2f,%.2f,%.2f,%.2f\n", avg_ttft, avg_tps, avg_tps_steady, last_tokens, avg_itl_avg,
               avg_itl_p50, avg_itl_p90, avg_itl_p99);
    } else {
        printf("\n  Engine freed. Benchmark complete.\n");
    }

    return 0;
}
