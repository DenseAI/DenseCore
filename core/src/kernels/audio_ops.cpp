/**
 * @file audio_ops.cpp
 * @brief Audio processing kernels (Whisper)
 */

#include "densecore/hal/op_registry.h"
#include "densecore/hal/transformer_ops.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <map>
#include <mutex>
#include <tuple>
#include <unordered_map>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace densecore {
namespace kernels {

using Complex = std::complex<float>;

// =============================================================================
// Optimized Iterative FFT
// =============================================================================
static void ReverseBits(std::vector<Complex>& x) {
    size_t n = x.size();
    size_t j = 0;
    for (size_t i = 1; i < n; ++i) {
        size_t bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            std::swap(x[i], x[j]);
        }
    }
}

static bool IsPowerOfTwo(size_t n) {
    return (n > 0) && ((n & (n - 1)) == 0);
}

// =============================================================================
// Helper Wrappers for AVX
// =============================================================================
#ifdef __AVX2__
#include <immintrin.h>

// Complex multiplication (a+bi)(c+di) = (ac-bd) + (ad+bc)i
// Layout: [R0, I0, R1, I1, R2, I2, R3, I3]
// We process 4 complex numbers at a time with AVX2 (8 floats)
static inline __m256 ComplexMul_AVX(__m256 a, __m256 b) {
    // a = [R0, I0, R1, I1, ...]
    // b = [C0, D0, C1, D1, ...]

    // Want: [R0C0-I0D0, R0D0+I0C0, ...]

    // 1. Prepare vectors for ac, bd, ad, bc
    // a_real = [R0, R0, R1, R1, ...] (duplicating Reals)
    // a_imag = [I0, I0, I1, I1, ...] (duplicating Imags)

    __m256 a_real = _mm256_moveldup_ps(a);
    __m256 a_imag = _mm256_movehdup_ps(a);

    // b_swap: [D0, C0, D1, C1 ...]
    __m256 b_swap = _mm256_permute_ps(b, 0xB1);  // Swap adjacent

    // Real Part of output: R*C - I*D
    // Imag Part of output: R*D + I*C

    // t1 = a_real * b = [R0C0, R0D0, R1C1, R1D1...]
    __m256 t1 = _mm256_mul_ps(a_real, b);

    // t2 = a_imag * b_swap = [I0D0, I0C0, I1D1, I1C1...]
    __m256 t2 = _mm256_mul_ps(a_imag, b_swap);

    // For even indices (Real out): t1 - t2 = R0C0 - I0D0
    // For odd indices (Imag out): t1 + t2 = R0D0 + I0C0
    // We need addsub but vertical?
    // _mm256_addsub_ps does: even-sub, odd-add.
    // We want R*C - I*D (Even indices). So Subtract.
    // We want R*D + I*C (Odd indices). So Add.
    // _mm256_addsub_ps(X, Y) = [X0-Y0, X1+Y1, ...]
    // So output = addsub(t1, t2).

    return _mm256_addsub_ps(t1, t2);
}
#endif

static void DFT_AVX(std::vector<Complex>& x) {
    size_t n = x.size();
    if (n <= 1) return;

    std::vector<Complex> out(n);
    const float two_pi = -2.0f * (float)M_PI;

#ifdef __AVX2__
    // Precompute W matrix logic could be cache-managed, but for now:
    // We compute rows on the fly or just small optimizations.
    // For maximizing N=400 performance, we'll just vectorize the dot product.

    // Vectorized implementation of O(N^2) DFT
    // out[k] = sum_t ( x[t] * W[k*t] )

    const float* x_ptr = reinterpret_cast<const float*>(x.data());

    // Since computing sin/cos inside the loop is slow, precompute input angles?
    // Better: Precompute the Twist Factors for N=400 once.
    // But this function is stateless. We can use a static cache for W table.

    // Thread-safe twiddle factor cache with double-checked locking
    static std::unordered_map<int, std::vector<Complex>> w_cache;
    static std::mutex w_cache_mutex;

    // Fast path: check without lock (safe for read on existing entry)
    std::vector<Complex> const* W_ptr = nullptr;
    {
        std::lock_guard<std::mutex> lock(w_cache_mutex);
        auto it = w_cache.find((int)n);
        if (it == w_cache.end()) {
            // Compute twiddle factors while holding lock
            std::vector<Complex> W(n * n);
            for (size_t k = 0; k < n; ++k) {
                for (size_t t = 0; t < n; ++t) {
                    float angle = two_pi * t * k / n;
                    W[k * n + t] = Complex(std::cos(angle), std::sin(angle));
                }
            }
            w_cache[(int)n] = std::move(W);
            it = w_cache.find((int)n);
        }
        W_ptr = &it->second;
    }
    const std::vector<Complex>& W = *W_ptr;
    const float* w_ptr = reinterpret_cast<const float*>(W.data());

    for (size_t k = 0; k < n; ++k) {
        __m256 sum_vec = _mm256_setzero_ps();
        const float* row_w = w_ptr + k * n * 2;  // 2 floats per complex

        size_t t = 0;
        // Process 4 complex numbers (8 floats) per iter
        for (; t + 4 <= n; t += 4) {
            __m256 vx = _mm256_loadu_ps(x_ptr + t * 2);
            __m256 vw = _mm256_loadu_ps(row_w + t * 2);
            __m256 prod = ComplexMul_AVX(vx, vw);
            sum_vec = _mm256_add_ps(sum_vec, prod);
        }

        // Horizontal sum of 4 complex numbers in sum_vec
        // sum_vec: [R0, I0, R1, I1, R2, I2, R3, I3]
        // Dest: sum(R) + i*sum(I)

        // Add pairs
        // permute 2x128 -> add
        __m128 lo = _mm256_castps256_ps128(sum_vec);
        __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
        __m128 combine = _mm_add_ps(lo, hi);  // [R0+R2, I0+I2, R1+R3, I1+I3]

        // Add shuffle
        __m128 swap = _mm_movehl_ps(combine, combine);  // [R1+R3, I1+I3, ...]
        __m128 final_sum = _mm_add_ps(combine, swap);   // [SumR, SumI, ...]

        float res[4];
        _mm_storeu_ps(res, final_sum);
        out[k] = Complex(res[0], res[1]);

        // Tail
        for (; t < n; ++t) {
            out[k] += x[t] * W[k * n + t];
        }
    }
#else
    // Check cache even for scalar to save trig muls
    // Thread-safe twiddle factor cache with double-checked locking
    static std::unordered_map<int, std::vector<Complex>> w_cache;
    static std::mutex w_cache_mutex;

    std::vector<Complex> const* W_ptr = nullptr;
    {
        std::lock_guard<std::mutex> lock(w_cache_mutex);
        auto it = w_cache.find((int)n);
        if (it == w_cache.end()) {
            std::vector<Complex> W(n * n);
            for (size_t k = 0; k < n; ++k) {
                for (size_t t = 0; t < n; ++t) {
                    float angle = two_pi * t * k / n;
                    W[k * n + t] = Complex(std::cos(angle), std::sin(angle));
                }
            }
            w_cache[(int)n] = std::move(W);
            it = w_cache.find((int)n);
        }
        W_ptr = &it->second;
    }
    const std::vector<Complex>& W = *W_ptr;

    for (size_t k = 0; k < n; ++k) {
        Complex sum(0.0f, 0.0f);
        for (size_t t = 0; t < n; ++t) {
            sum += x[t] * W[k * n + t];
        }
        out[k] = sum;
    }
#endif
    x = std::move(out);
}

static void IterativeFFT(std::vector<Complex>& x) {
    size_t n = x.size();
    if (n <= 1) return;

    if (IsPowerOfTwo(n)) {
        ReverseBits(x);

        for (size_t len = 2; len <= n; len <<= 1) {
            float ang = -2.0f * (float)M_PI / len;
            Complex wlen(std::cos(ang), std::sin(ang));

#ifdef __AVX2__
            // Vectorized butterfly for stages where len/2 >= 4 (can process 4 butterflies)
            if (len >= 8) {
                // Precompute twiddle factors for this stage
                std::vector<Complex> twiddles(len / 2);
                Complex w(1.0f, 0.0f);
                for (size_t j = 0; j < len / 2; ++j) {
                    twiddles[j] = w;
                    w *= wlen;
                }

                for (size_t i = 0; i < n; i += len) {
                    size_t j = 0;
                    // Process 4 butterflies at a time (8 complex numbers involved)
                    for (; j + 4 <= len / 2; j += 4) {
                        // Load 4 u values: x[i+j], x[i+j+1], x[i+j+2], x[i+j+3]
                        __m256 u_vec = _mm256_loadu_ps(reinterpret_cast<float*>(&x[i + j]));

                        // Load 4 v values to be multiplied: x[i+j+len/2], ...
                        __m256 v_raw = _mm256_loadu_ps(reinterpret_cast<float*>(&x[i + j + len / 2]));

                        // Load 4 twiddle factors
                        __m256 tw = _mm256_loadu_ps(reinterpret_cast<float*>(&twiddles[j]));

                        // Complex multiply: v_raw * tw using ComplexMul_AVX
                        __m256 v_vec = ComplexMul_AVX(v_raw, tw);

                        // Butterfly: u + v, u - v
                        __m256 add_result = _mm256_add_ps(u_vec, v_vec);
                        __m256 sub_result = _mm256_sub_ps(u_vec, v_vec);

                        // Store results
                        _mm256_storeu_ps(reinterpret_cast<float*>(&x[i + j]), add_result);
                        _mm256_storeu_ps(reinterpret_cast<float*>(&x[i + j + len / 2]), sub_result);
                    }

                    // Scalar tail for remaining butterflies
                    for (; j < len / 2; ++j) {
                        Complex u = x[i + j];
                        Complex v = x[i + j + len / 2] * twiddles[j];
                        x[i + j] = u + v;
                        x[i + j + len / 2] = u - v;
                    }
                }
            } else {
                // Small stages: use scalar (overhead of SIMD not worth it)
                for (size_t i = 0; i < n; i += len) {
                    Complex w(1.0f, 0.0f);
                    for (size_t j = 0; j < len / 2; ++j) {
                        Complex u = x[i + j];
                        Complex v = x[i + j + len / 2] * w;
                        x[i + j] = u + v;
                        x[i + j + len / 2] = u - v;
                        w *= wlen;
                    }
                }
            }
#else
            // Scalar fallback for non-AVX2 platforms
            for (size_t i = 0; i < n; i += len) {
                Complex w(1.0f, 0.0f);
                for (size_t j = 0; j < len / 2; ++j) {
                    Complex u = x[i + j];
                    Complex v = x[i + j + len / 2] * w;
                    x[i + j] = u + v;
                    x[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
#endif
        }
    } else {
        DFT_AVX(x);
    }
}

// =============================================================================
// Mel Filterbank Utilities
// =============================================================================
static float hz_to_mel(float freq) {
    return 2595.0f * std::log10(1.0f + freq / 700.0f);
}

static float mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

// Generate Mel Filterbank matrix [n_mels, n_fft/2 + 1]
// Standard triangular filters (Slaney-style typically, but this is HTK-style common in ML)
static std::vector<std::vector<float>> CreateMelFilters(int n_mels, int n_fft, int sample_rate) {
    std::vector<std::vector<float>> filters(n_mels, std::vector<float>(n_fft / 2 + 1, 0.0f));

    float mel_min = hz_to_mel(0.0f);
    float mel_max = hz_to_mel(sample_rate / 2.0f);

    // Create n_mels + 2 points
    std::vector<float> mel_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        mel_points[i] = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
    }

    std::vector<int> bin_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        float freq = mel_to_hz(mel_points[i]);
        bin_points[i] = std::floor((n_fft + 1) * freq / sample_rate);
    }

    // Construct filters
    for (int i = 0; i < n_mels; ++i) {
        int start = bin_points[i];
        int center = bin_points[i + 1];
        int end = bin_points[i + 2];

        // Rising edge
        for (int j = start; j < center; ++j) {
            filters[i][j] = (float)(j - start) / (center - start);
        }
        // Falling edge
        for (int j = center; j < end; ++j) {
            filters[i][j] = 1.0f - (float)(j - center) / (end - center);
        }
    }

    return filters;
}

class CpuAudioOps : public AudioOps {
public:
    bool Supports(DeviceType device) const override { return device == DeviceType::CPU; }

    OpCapabilities GetCapabilities() const override { return {.supports_fp16 = false, .priority = 1}; }

    void Execute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const void* params) override {
        if (inputs.empty() || outputs.empty() || !params) return;

        // Try to interpret generic execute as MelSpectrogram request if inputs/params match
        // In a strictly typed system, we would check an opcode or type ID.
        // Assuming known context from OpType::MelSpectrogram registration:
        const auto* mel_params = static_cast<const MelSpectrogramParams*>(params);
        if (mel_params) {
            MelSpectrogram(*inputs[0], mel_params->n_fft, mel_params->hop_length, mel_params->n_mels,
                           mel_params->sample_rate, outputs[0]);
        }
    }

    // =========================================================================
    // Mel Spectrogram Implementation
    // =========================================================================
    void MelSpectrogram(const Tensor& waveform, int n_fft, int hop_length, int n_mels, int sample_rate,
                        Tensor* output) override {
        // 1. Hann Window Pre-calculation
        std::vector<float> window(n_fft);
        for (int i = 0; i < n_fft; ++i) {
            window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / n_fft));
        }

        // 2. Setup
        const float* audio = static_cast<const float*>(waveform.data);
        int64_t num_samples = waveform.shape[0];
        int64_t num_frames = (num_samples - n_fft) / hop_length + 1;
        float* out_data = static_cast<float*>(output->data);

        // 3. Generate Mel Filters using provided sample_rate
        // Use static cache to avoid recomputing filters on every call
        static std::map<std::tuple<int, int, int>, std::vector<std::vector<float>>> filter_cache;
        static std::mutex filter_cache_mutex;

        std::vector<std::vector<float>> mel_filters_local;
        {
            std::lock_guard<std::mutex> lock(filter_cache_mutex);
            auto key = std::make_tuple(n_mels, n_fft, sample_rate);
            if (filter_cache.find(key) == filter_cache.end()) {
                filter_cache[key] = CreateMelFilters(n_mels, n_fft, sample_rate);
            }
            mel_filters_local = filter_cache[key];
        }
        const auto& mel_filters = mel_filters_local;

        std::vector<Complex> buffer(n_fft);

        // 4. Processing Loop
        for (int i = 0; i < num_frames; ++i) {
            // Windowing
            for (int j = 0; j < n_fft; ++j) {
                int64_t idx = i * hop_length + j;
                float val = (idx < num_samples) ? audio[idx] : 0.0f;
                buffer[j] = Complex(val * window[j], 0.0f);
            }

            // FFT
            IterativeFFT(buffer);

            // Compute Power Spectrum
            std::vector<float> power_spec(n_fft / 2 + 1);
            for (int j = 0; j <= n_fft / 2; ++j) {
                float abs = std::abs(buffer[j]);
                power_spec[j] = abs * abs;
            }

            // Apply Mel Filterbank & Log
            for (int m = 0; m < n_mels; ++m) {
                float val = 0.0f;
                for (int j = 0; j <= n_fft / 2; ++j) {
                    val += power_spec[j] * mel_filters[m][j];
                }

                // Log10(val + 1e-10) -> Log Mel Spectrogram
                val = std::log10(val + 1e-10f);

                // Write to output: [n_mels, num_frames] Row-major
                // Memory layout: m changes slowest, i changes fastest.
                // out[m, i] corresponds to out_data[m * num_frames + i]
                int64_t out_idx = m * num_frames + i;
                if (out_idx < output->shape[0] * output->shape[1]) {  // Boundary check
                    out_data[out_idx] = val;
                }
            }
        }
    }
};

DENSECORE_REGISTER_OP(CpuAudioOps, OpType::MelSpectrogram, DeviceType::CPU);

}  // namespace kernels
}  // namespace densecore
