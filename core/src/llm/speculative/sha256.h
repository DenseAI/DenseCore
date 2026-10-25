#pragma once
// Private streaming SHA-256, FIPS 180-4. No external crypto dependency.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <istream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace densecore::llm::speculative::qualification {
class Sha256 {
public:
    void Update(const void* data, size_t size) {
        if (!data && size) throw std::invalid_argument("SHA256 null input");
        constexpr uint64_t max_bytes = UINT64_MAX / 8;
        if (size > max_bytes - bytes_) throw std::length_error("SHA256 bit length overflow");
        bytes_ += size;
        const auto* input = static_cast<const uint8_t*>(data);
        while (size) {
            const size_t copy = std::min(size, buffer_.size() - used_);
            std::memcpy(buffer_.data() + used_, input, copy);
            input += copy; size -= copy; used_ += copy;
            if (used_ == buffer_.size()) { Transform(buffer_.data()); used_ = 0; }
        }
    }
    void Update(std::string_view data) { Update(data.data(), data.size()); }
    std::string HexDigest() const {
        Sha256 final = *this;
        final.buffer_[final.used_++] = 0x80;
        if (final.used_ > 56) {
            std::fill(final.buffer_.begin() + final.used_, final.buffer_.end(), 0);
            final.Transform(final.buffer_.data()); final.used_ = 0;
        }
        std::fill(final.buffer_.begin() + final.used_, final.buffer_.begin() + 56, 0);
        const uint64_t bits = bytes_ * 8;
        for (int i = 0; i < 8; ++i) final.buffer_[63 - i] = static_cast<uint8_t>(bits >> (8 * i));
        final.Transform(final.buffer_.data());
        constexpr char hex[] = "0123456789abcdef";
        std::string result(64, '0');
        size_t offset = 0;
        for (uint32_t word : final.state_) {
            for (int shift = 24; shift >= 0; shift -= 8) {
                const uint8_t byte = static_cast<uint8_t>(word >> shift);
                result[offset++] = hex[byte >> 4]; result[offset++] = hex[byte & 15];
            }
        }
        return result;
    }
private:
    static uint32_t Rotate(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }
    void Transform(const uint8_t* block) {
        static constexpr uint32_t constants[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t words[64];
        for (size_t i = 0; i < 16; ++i) {
            words[i] = (uint32_t(block[i * 4]) << 24) | (uint32_t(block[i * 4 + 1]) << 16) |
                       (uint32_t(block[i * 4 + 2]) << 8) | uint32_t(block[i * 4 + 3]);
        }
        for (size_t i = 16; i < 64; ++i) {
            const uint32_t s0 = Rotate(words[i - 15],7) ^ Rotate(words[i - 15],18) ^ (words[i - 15] >> 3);
            const uint32_t s1 = Rotate(words[i - 2],17) ^ Rotate(words[i - 2],19) ^ (words[i - 2] >> 10);
            words[i] = words[i - 16] + s0 + words[i - 7] + s1;
        }
        uint32_t a=state_[0],b=state_[1],c=state_[2],d=state_[3],e=state_[4],f=state_[5],g=state_[6],h=state_[7];
        for (size_t i = 0; i < 64; ++i) {
            const uint32_t sum1 = Rotate(e,6) ^ Rotate(e,11) ^ Rotate(e,25);
            const uint32_t choice = (e & f) ^ (~e & g);
            const uint32_t t1 = h + sum1 + choice + constants[i] + words[i];
            const uint32_t sum0 = Rotate(a,2) ^ Rotate(a,13) ^ Rotate(a,22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = sum0 + majority;
            h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        state_[0]+=a;state_[1]+=b;state_[2]+=c;state_[3]+=d;
        state_[4]+=e;state_[5]+=f;state_[6]+=g;state_[7]+=h;
    }
    std::array<uint32_t,8> state_{{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                                  0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19}};
    std::array<uint8_t,64> buffer_{};
    size_t used_ = 0;
    uint64_t bytes_ = 0;
};
inline std::string Sha256Hex(std::string_view input) { Sha256 hash; hash.Update(input); return hash.HexDigest(); }
inline std::string Sha256Stream(std::istream& input) {
    Sha256 hash;
    std::array<char,64 * 1024> buffer;
    while (input) {
        input.read(buffer.data(), buffer.size());
        hash.Update(buffer.data(), static_cast<size_t>(input.gcount()));
    }
    if (input.bad() || !input.eof()) throw std::ios_base::failure("SHA256 stream read failed");
    return hash.HexDigest();
}
} // namespace densecore::llm::speculative::qualification
