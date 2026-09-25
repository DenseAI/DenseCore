#ifndef DENSECORE_LLM_RUNTIME_SPIN_WAIT_H
#define DENSECORE_LLM_RUNTIME_SPIN_WAIT_H

#include <thread>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#endif
namespace densecore::llm::runtime {
inline void SpinPause(int spin_count) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    if ((spin_count & 0x3F) != 0) {
        _mm_pause();
        return;
    }
#elif defined(__aarch64__)
    if ((spin_count & 0x3F) != 0) {
        asm volatile("yield");
        return;
    }
#endif
    std::this_thread::yield();
}

}  // namespace densecore::llm::runtime
#endif
