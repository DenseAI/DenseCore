#pragma once

#include <random>
#include "densecore/runtime/inference.h"

// Internal request-owned RNG entry point. SamplingParams and the public
// SampleToken symbol retain their existing ABI.
int SampleTokenWithRng(GgmlTensorHandle* logits, int idx, const SamplingParams& params, std::mt19937& rng);
