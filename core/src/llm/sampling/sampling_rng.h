#pragma once

#include <random>
#include "densecore/runtime/inference.h"

// Internal request-owned RNG entry point. SamplingParams and the public
// SampleToken symbol retain their existing ABI.
int SampleTokenWithRng(GgmlTensorHandle* logits, int idx, const SamplingParams& params, std::mt19937& rng);

// Uses precisely the regular sampler's grammar, penalties, top-k/min-p/top-p
// order. Produces token-ID-indexed probabilities without consuming RNG state.
std::vector<double> BuildSamplingDistribution(GgmlTensorHandle* logits, int idx, const SamplingParams& params);
