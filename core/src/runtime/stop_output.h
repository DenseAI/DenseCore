#pragma once

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace densecore::runtime {

inline size_t StopOutputCapacity(const std::vector<std::string>& stops) {
    size_t capacity = 0;
    for (const auto& stop : stops) capacity = std::max(capacity, stop.size());
    return capacity;
}

namespace stop_output_detail {
// View two existing strings; never put a whole token piece into the bounded
// request buffer just to search across the token boundary.
struct Text {
    std::string_view pending;
    std::string_view piece;
    size_t size() const { return pending.size() + piece.size(); }
    char at(size_t i) const { return i < pending.size() ? pending[i] : piece[i - pending.size()]; }
    bool boundary(size_t i) const {
        return i == size() || (static_cast<unsigned char>(at(i)) & 0xc0) != 0x80;
    }
    bool matches(size_t position, std::string_view value) const {
        for (size_t i = 0; i < value.size(); ++i) if (at(position + i) != value[i]) return false;
        return true;
    }
    std::string prefix(size_t count) const {
        std::string result;
        result.reserve(count);
        const size_t held = std::min(count, pending.size());
        result.append(pending.data(), held);
        if (count > held) result.append(piece.data(), count - held);
        return result;
    }
};
}

// Input is the existing UTF-8 collector's complete emitted piece (possibly
// empty). Partial token bytes stay in that collector, never discarded here.
// Return true on a full stop; piece becomes only safe visible output. Pending
// contains at most max_stop_bytes-1 bytes and only a possible stop prefix.
inline bool FilterStopOutput(std::string& pending, std::string& piece,
                             const std::vector<std::string>& stops) {
    const size_t maximum = StopOutputCapacity(stops);
    if (maximum == 0) return false; // Preserve the no-stop path without copying.
    if (pending.size() > std::numeric_limits<size_t>::max() - piece.size())
        throw std::length_error("stop output length overflow");
    const stop_output_detail::Text text{pending, piece};
    size_t first = std::string::npos;
    for (const auto& stop : stops) {
        if (stop.empty() || stop.size() > text.size()) continue;
        for (size_t position = 0; position <= text.size() - stop.size(); ++position) {
            if (position >= first) break;
            if (text.boundary(position) && text.boundary(position + stop.size()) && text.matches(position, stop)) {
                first = position;
                break;
            }
        }
    }
    if (first != std::string::npos) {
        if (pending.empty()) piece.resize(first);
        else {
            auto output = text.prefix(first);
            piece.swap(output);
        }
        pending.clear();
        return true;
    }
    size_t keep = 0;
    for (const auto& stop : stops) {
        if (stop.empty()) continue;
        for (size_t length = std::min(text.size(), stop.size() - 1); length > keep; --length) {
            const size_t start = text.size() - length;
            if (text.boundary(start) && text.matches(start, std::string_view(stop).substr(0, length))) {
                keep = length;
                break;
            }
        }
    }
    if (pending.empty() && keep == 0) return false;
    const size_t visible = text.size() - keep;
    auto output = text.prefix(visible); // Ordinary output storage, not another persistent request buffer.
    if (visible < pending.size()) {
        pending.erase(0, visible);
        pending.append(piece);
    } else {
        const size_t offset = visible - pending.size();
        pending.assign(piece.data() + offset, keep);
    }
    piece.swap(output);
    return false;
}

inline std::string FinishStopOutput(std::string& pending, bool discard) {
    std::string tail;
    if (discard) pending.clear();
    else tail.swap(pending);
    return tail;
}

} // namespace densecore::runtime
