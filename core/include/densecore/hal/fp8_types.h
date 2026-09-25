#ifndef DENSECORE_HAL_FP8_TYPES_H
#define DENSECORE_HAL_FP8_TYPES_H

#include <cstdint>

namespace densecore {

/**
 * @brief FP8 E5M2 format (IEEE 754 standard).
 * Structure: 1 sign bit, 5 exponent bits, 2 mantissa bits.
 * Range: ~10^-5 to 57344.
 * Small dynamic range, but standard IEEE behavior with Infinity and NaN.
 */
struct float8_e5m2 {
    uint8_t storage;

    float8_e5m2() = default;
    explicit float8_e5m2(uint8_t v) : storage(v) {}

    // Explicit conversion to uint8_t for storage access
    explicit operator uint8_t() const { return storage; }
};

/**
 * @brief FP8 E4M3FN format (NVIDIA H100 / OpenAI standard).
 * Structure: 1 sign bit, 4 exponent bits, 3 mantissa bits.
 * Range: ~10^-9 to 448.
 * No Infinity, extended range, used for weights/gradients.
 * Represents NaN only when sign=0/1, exponent=1111, mantissa=111 (0x7F, 0xFF).
 */
struct float8_e4m3fn {
    uint8_t storage;

    float8_e4m3fn() = default;
    explicit float8_e4m3fn(uint8_t v) : storage(v) {}

    // Explicit conversion to uint8_t for storage access
    explicit operator uint8_t() const { return storage; }
};

}  // namespace densecore

#endif  // DENSECORE_HAL_FP8_TYPES_H
