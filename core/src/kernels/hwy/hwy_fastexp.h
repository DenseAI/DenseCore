/**
 * @file hwy_fastexp.h
 * @brief BitCast-based fast exp() approximation for Highway SIMD
 *
 * Pure inline template — must be #included inside an already-open
 * HWY_NAMESPACE (after HWY_BEFORE_NAMESPACE() and namespace opening).
 *
 * NO include guard: Highway's foreach_target.h re-compiles the including
 * .cc file once per ISA target, each time with a different HWY_NAMESPACE.
 * A traditional #ifndef guard would prevent re-inclusion for the 2nd+ targets.
 */

template <class D, class V> HWY_INLINE V FastExpHwy(D d, V x) {
    namespace hn = hwy::HWY_NAMESPACE;

    const auto kLog2E = hn::Set(d, 1.4426950408889634f);
    const auto kMinExp = hn::Set(d, -50.0f);
    const auto kMaxExp = hn::Set(d, 50.0f);

    // Clamp
    x = hn::Max(hn::Min(x, kMaxExp), kMinExp);

    // y = x * log2(e)
    const auto y = hn::Mul(x, kLog2E);

    // i = floor(y), f = y - i
    const auto i_f = hn::Floor(y);
    const auto f = hn::Sub(y, i_f);

    // Polynomial: p = 1 + f*(c0 + f*(c1 + f*c2))
    const auto c0 = hn::Set(d, 0.6960656421638072f);
    const auto c1 = hn::Set(d, 0.224494337302845f);
    const auto c2 = hn::Set(d, 0.07944023841053369f);

    auto p = hn::MulAdd(f, c2, c1);
    p = hn::MulAdd(f, p, c0);
    p = hn::MulAdd(f, p, hn::Set(d, 1.0f));

    // 2^i via bit manipulation: reinterpret (i + 127) << 23 as float
    using DI = hn::RebindToSigned<D>;
    const DI di;
    const auto i_int = hn::ConvertTo(di, i_f);
    const auto exp_bits = hn::ShiftLeft<23>(hn::Add(i_int, hn::Set(di, 127)));
    const auto two_i = hn::BitCast(d, exp_bits);

    return hn::Mul(two_i, p);
}

// Scalar version for bit-exact consistency with vectorized path
HWY_INLINE float FastExpScalar(float x) {
    const float kLog2E = 1.4426950408889634f;
    const float kMinExp = -50.0f;
    const float kMaxExp = 50.0f;

    // Clamp
    x = std::max(std::min(x, kMaxExp), kMinExp);

    // y = x * log2(e)
    const float y = x * kLog2E;

    // i = floor(y), f = y - i
    const float i_f = std::floor(y);
    const float f = y - i_f;

    // Polynomial: p = 1 + f*(c0 + f*(c1 + f*c2))
    const float c0 = 0.6960656421638072f;
    const float c1 = 0.224494337302845f;
    const float c2 = 0.07944023841053369f;

    float p = c2 * f + c1;
    p = p * f + c0;
    p = p * f + 1.0f;

    // 2^i via bit manipulation
    int32_t i_int = static_cast<int32_t>(i_f);
    int32_t exp_bits = (i_int + 127) << 23;
    union {
        int32_t i;
        float f;
    } bits = {exp_bits};
    const float two_i = bits.f;

    return p * two_i;
}
