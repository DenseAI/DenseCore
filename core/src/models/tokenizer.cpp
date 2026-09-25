#include "densecore/models/tokenizer.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <queue>
#include <regex>
#include <stdexcept>
#include <unordered_map>

#include "densecore/models/model_descriptor.h"
#include "runtime/runtime_env.h"

/**
 * BPE tokenizer implementation.
 *
 * Fast path (preferred): uses GGUF-provided tokenizer.ggml.merges ranks
 * with byte-level encoding (GPT-2/Qwen-style).
 *
 * Fallback path: legacy merge-by-vocab existence for models that do not
 * ship merge ranks.
 */

// ============================================================================
// UTF-8 Utilities
// ============================================================================

static int Utf8CharLen(unsigned char c) {
    if ((c & 0x80) == 0) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

std::vector<std::string> Tokenizer::SplitToChars(const std::string& text) {
    std::vector<std::string> chars;
    chars.reserve(text.size());

    size_t i = 0;
    while (i < text.size()) {
        int len = Utf8CharLen(static_cast<unsigned char>(text[i]));
        if (i + len > text.size()) {
            len = static_cast<int>(text.size() - i);
        }
        chars.push_back(text.substr(i, len));
        i += len;
    }

    return chars;
}

// ============================================================================
// BPE Core
// ============================================================================

bool Tokenizer::HasToken(const TransformerModel* model, const std::string& token) {
    return model->token_to_id.find(token) != model->token_to_id.end();
}

float Tokenizer::GetTokenScore(const TransformerModel* model, const std::string& token) {
    auto it = model->token_to_id.find(token);
    if (it == model->token_to_id.end()) {
        return std::numeric_limits<float>::max();
    }

    int id = it->second;
    if (!model->token_scores.empty() && id < static_cast<int>(model->token_scores.size())) {
        return model->token_scores[id];
    }

    // Fallback: lower id = higher merge priority.
    return static_cast<float>(id);
}

int Tokenizer::FindBestMerge(const TransformerModel* model, const std::vector<std::string>& tokens) {
    if (tokens.size() < 2) {
        return -1;
    }

    int best_idx = -1;
    float best_score = std::numeric_limits<float>::max();

    for (size_t i = 0; i + 1 < tokens.size(); i++) {
        std::string merged = tokens[i] + tokens[i + 1];
        if (HasToken(model, merged)) {
            float score = GetTokenScore(model, merged);
            if (score < best_score) {
                best_score = score;
                best_idx = static_cast<int>(i);
            }
        }
    }

    return best_idx;
}

namespace {

bool IsDebugTokenizerEnabled() {
    static const bool enabled = []() {
        const char* env = densecore::env::GetDiagnosticEnv("DENSECORE_DEBUG_TOKENIZER");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

constexpr char kMergeKeySep = '\x1f';

std::string EncodeUtf8(uint32_t cp) {
    std::string out;
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

std::vector<std::string> SplitUtf8Units(const std::string& text) {
    std::vector<std::string> units;
    units.reserve(text.size());

    size_t i = 0;
    while (i < text.size()) {
        int len = Utf8CharLen(static_cast<unsigned char>(text[i]));
        if (i + len > text.size()) {
            len = static_cast<int>(text.size() - i);
        }
        units.push_back(text.substr(i, len));
        i += len;
    }

    return units;
}

std::string NormalizeSentencePieceText(const std::string& text) {
    if (text.empty()) {
        return {};
    }

    std::string normalized;
    normalized.reserve(text.size() * 3 + 3);
    if (text.front() != ' ') {
        normalized.append("\xE2\x96\x81");
    }
    for (char ch : text) {
        if (ch == ' ') {
            normalized.append("\xE2\x96\x81");
        } else {
            normalized.push_back(ch);
        }
    }
    return normalized;
}

int FindSentencePieceUnkId(const TransformerModel* model) {
    if (!model) {
        return -1;
    }

    auto lit = model->token_to_id.find("<unk>");
    if (lit != model->token_to_id.end()) {
        return lit->second;
    }

    for (size_t i = 0; i < model->token_types.size(); ++i) {
        if (model->token_types[i] == 2) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

float GetSentencePieceScore(const TransformerModel* model, int token_id) {
    if (!model || token_id < 0 || token_id >= static_cast<int>(model->token_scores.size())) {
        return 0.0f;
    }
    return model->token_scores[static_cast<size_t>(token_id)];
}

bool IsLikelyControlTokenLiteral(const std::string& token) {
    if (token.size() >= 4 && token[0] == '<' && token[1] == '|' && token[token.size() - 2] == '|' &&
        token[token.size() - 1] == '>') {
        return true;
    }
    // Qwen3 thinking delimiters use <...> not <|...|>
    if (token == "<think>" || token == "</think>" || token == "<bos>" || token == "<eos>" || token == "<pad>" ||
        token == "<unk>" || token == "<mask>") {
        return true;
    }
    // Gemma chat-turn markers use <...> (not <|...|>) and must tokenize as a
    // single atomic special id (start_of_turn=105 / end_of_turn=106). Some GGUF
    // exports omit/garble tokenizer.ggml.token_type, in which case the
    // token_types-based check in IsAtomicSpecialTokenLiteral can't classify
    // them and they would otherwise decompose into literal text, corrupting the
    // turn structure. Recognize them by name so they survive intact (and are
    // hidden from detokenized output, like the other control tokens above).
    if (token == "<start_of_turn>" || token == "<end_of_turn>") {
        return true;
    }
    if (token == "[CLS]" || token == "[SEP]" || token == "[PAD]" || token == "[UNK]" || token == "[MASK]") {
        return true;
    }
    return false;
}

bool IsGemma4GenerationChannelToken(const TransformerModel* model, const std::string& token) {
    return model && densecore::models::DescribeModel(model).variant == ModelVariant::GEMMA4 &&
           (token == "<|channel>" || token == "<channel|>");
}

bool IsAtomicSpecialTokenLiteral(const TransformerModel* model, const std::string& token) {
    if (!model || token.empty()) {
        return false;
    }

    auto it = model->token_to_id.find(token);
    if (it == model->token_to_id.end()) {
        return false;
    }

    const int token_id = it->second;
    if (token_id >= 0 && token_id < static_cast<int>(model->token_types.size())) {
        const int32_t token_type = model->token_types[static_cast<size_t>(token_id)];
        if (token_type != 1 && token_type != 6) {
            return true;
        }
    }

    return IsLikelyControlTokenLiteral(token);
}

const std::array<std::string, 256>& ByteToUnicode() {
    static const std::array<std::string, 256> table = []() {
        std::array<std::string, 256> out;

        std::vector<int> bs;
        bs.reserve(256);

        for (int b = 33; b <= 126; ++b) bs.push_back(b);
        for (int b = 161; b <= 172; ++b) bs.push_back(b);
        for (int b = 174; b <= 255; ++b) bs.push_back(b);

        std::array<bool, 256> present{};
        for (int b : bs) present[b] = true;

        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (!present[b]) {
                bs.push_back(b);
                cs.push_back(256 + n);
                ++n;
            }
        }

        for (size_t i = 0; i < bs.size(); ++i) {
            out[bs[i]] = EncodeUtf8(static_cast<uint32_t>(cs[i]));
        }

        return out;
    }();

    return table;
}

const std::unordered_map<std::string, uint8_t>& UnicodeToByte() {
    static const std::unordered_map<std::string, uint8_t> inv = []() {
        std::unordered_map<std::string, uint8_t> m;
        m.reserve(256);
        const auto& b2u = ByteToUnicode();
        for (int b = 0; b < 256; ++b) {
            m.emplace(b2u[b], static_cast<uint8_t>(b));
        }
        return m;
    }();

    return inv;
}

std::string MakeMergeKey(const std::string& left, const std::string& right) {
    std::string key;
    key.reserve(left.size() + right.size() + 1);
    key.append(left);
    key.push_back(kMergeKeySep);
    key.append(right);
    return key;
}

std::string AsciiLowerCopy(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

bool UseQwenPretokenizer(const TransformerModel* model) {
    const auto family = densecore::models::ResolveTokenizerFamily(model);
    return family == densecore::models::TokenizerFamily::QWEN_BYTE_BPE ||
           family == densecore::models::TokenizerFamily::QWEN35_UNICODE_BPE;
}

bool UseQwen35Pretokenizer(const TransformerModel* model) {
    return densecore::models::ResolveTokenizerFamily(model) == densecore::models::TokenizerFamily::QWEN35_UNICODE_BPE;
}

bool UseLFM2Pretokenizer(const TransformerModel* model) {
    return densecore::models::ResolveTokenizerFamily(model) == densecore::models::TokenizerFamily::LFM2_BYTE_BPE;
}

bool UseGemmaPretokenizer(const TransformerModel* model) {
    return densecore::models::ResolveTokenizerFamily(model) == densecore::models::TokenizerFamily::GEMMA_SENTENCEPIECE;
}

bool UseSentencePieceUnigramTokenizer(const TransformerModel* model) {
    if (!model || model->token_scores.empty()) {
        return false;
    }
    switch (densecore::models::ResolveTokenizerFamily(model)) {
    case densecore::models::TokenizerFamily::GEMMA_SENTENCEPIECE:
    case densecore::models::TokenizerFamily::LLAMA_SENTENCEPIECE: return true;
    default: return false;
    }
}

bool UseWordPieceTokenizer(const TransformerModel* model) {
    return densecore::models::ResolveTokenizerFamily(model) == densecore::models::TokenizerFamily::BERT_WORDPIECE;
}

bool UseBertBpeTokenizer(const TransformerModel* model) {
    return densecore::models::ResolveTokenizerFamily(model) == densecore::models::TokenizerFamily::BERT_BPE;
}

bool IsByteLevelBpeTokenizer(const TransformerModel* model) {
    if (!model) return false;

    switch (densecore::models::ResolveTokenizerFamily(model)) {
    case densecore::models::TokenizerFamily::GEMMA_SENTENCEPIECE: return false;
    case densecore::models::TokenizerFamily::GPT2_BYTE_BPE:
    case densecore::models::TokenizerFamily::QWEN_BYTE_BPE:
    case densecore::models::TokenizerFamily::QWEN35_UNICODE_BPE:
    case densecore::models::TokenizerFamily::LFM2_BYTE_BPE:
    case densecore::models::TokenizerFamily::GLM_BYTE_BPE: return true;
    case densecore::models::TokenizerFamily::BERT_WORDPIECE: return false;
    case densecore::models::TokenizerFamily::BERT_BPE: return false;
    case densecore::models::TokenizerFamily::LLAMA_SENTENCEPIECE:
    case densecore::models::TokenizerFamily::UNKNOWN: break;
    }

    // Heuristic fallback for GGUFs that omit tokenizer_type but include byte-BPE merges.
    if (model->bpe_merge_ranks.empty()) {
        return false;
    }

    const auto& b2u = ByteToUnicode();
    int probe_hits = 0;
    const uint8_t probes[] = {0, 10, 32, 128, 173, 255};
    for (uint8_t b : probes) {
        if (model->token_to_id.find(b2u[b]) != model->token_to_id.end()) {
            ++probe_hits;
        }
    }
    return probe_hits >= 3;
}

bool UseByteUnicodeDetokenization(const TransformerModel* model) {
    if (!model) return false;
    return IsByteLevelBpeTokenizer(model);
}

int ResolveEndTokenId(const TransformerModel* model) {
    if (!model) {
        return -1;
    }
    if (densecore::models::ResolveTokenizerFamily(model) == densecore::models::TokenizerFamily::BERT_WORDPIECE &&
        model->sep_token_id >= 0) {
        return model->sep_token_id;
    }
    return model->eos_token_id;
}

std::string DetokenizeImpl(const TransformerModel* model, int token_id) {
    if (!model || token_id < 0 || token_id >= static_cast<int>(model->vocab_tokens.size())) {
        return "";
    }

    const std::string& token = model->vocab_tokens[static_cast<size_t>(token_id)];

    if (IsGemma4GenerationChannelToken(model, token)) {
        return token;
    }

    if (token_id < static_cast<int>(model->token_types.size())) {
        const int32_t token_type = model->token_types[static_cast<size_t>(token_id)];
        if (token_type != 1 && token_type != 6) {
            return "";
        }
    }
    if (IsLikelyControlTokenLiteral(token)) {
        return "";
    }

    if (token.size() == 6 && token[0] == '<' && token[1] == '0' && token[2] == 'x' && token[5] == '>') {
        char hex[3] = {token[3], token[4], 0};
        int byte_val = 0;
        if (std::sscanf(hex, "%x", &byte_val) == 1) {
            return std::string(1, static_cast<char>(byte_val));
        }
    }

    const auto units = SplitUtf8Units(token);
    std::string out;
    out.reserve(token.size());

    if (!UseByteUnicodeDetokenization(model)) {
        for (const std::string& unit : units) {
            if (unit == "▁") {
                out.push_back(' ');
            } else {
                out.append(unit);
            }
        }
        return out;
    }

    const auto& u2b = UnicodeToByte();
    for (const std::string& unit : units) {
        auto it = u2b.find(unit);
        if (it != u2b.end()) {
            out.push_back(static_cast<char>(it->second));
            continue;
        }
        if (unit == "▁") {
            out.push_back(' ');
            continue;
        }
        out.append(unit);
    }
    return out;
}

struct Utf8Codepoint {
    uint32_t cp = 0;
    size_t start = 0;
    size_t len = 0;
};

std::vector<Utf8Codepoint> DecodeUtf8Codepoints(const std::string& text) {
    std::vector<Utf8Codepoint> out;
    out.reserve(text.size());

    size_t i = 0;
    while (i < text.size()) {
        const size_t start = i;
        const unsigned char c0 = static_cast<unsigned char>(text[i]);
        uint32_t cp = c0;
        size_t len = 1;

        if ((c0 & 0x80) == 0) {
            cp = c0;
            len = 1;
        } else if ((c0 & 0xE0) == 0xC0 && i + 1 < text.size()) {
            cp = (static_cast<uint32_t>(c0 & 0x1F) << 6) |
                 static_cast<uint32_t>(static_cast<unsigned char>(text[i + 1]) & 0x3F);
            len = 2;
        } else if ((c0 & 0xF0) == 0xE0 && i + 2 < text.size()) {
            cp = (static_cast<uint32_t>(c0 & 0x0F) << 12) |
                 (static_cast<uint32_t>(static_cast<unsigned char>(text[i + 1]) & 0x3F) << 6) |
                 static_cast<uint32_t>(static_cast<unsigned char>(text[i + 2]) & 0x3F);
            len = 3;
        } else if ((c0 & 0xF8) == 0xF0 && i + 3 < text.size()) {
            cp = (static_cast<uint32_t>(c0 & 0x07) << 18) |
                 (static_cast<uint32_t>(static_cast<unsigned char>(text[i + 1]) & 0x3F) << 12) |
                 (static_cast<uint32_t>(static_cast<unsigned char>(text[i + 2]) & 0x3F) << 6) |
                 static_cast<uint32_t>(static_cast<unsigned char>(text[i + 3]) & 0x3F);
            len = 4;
        }

        out.push_back({cp, start, len});
        i += len;
    }

    return out;
}

bool IsUnicodeWhitespace(uint32_t cp) {
    switch (cp) {
    case 0x0009:
    case 0x000A:
    case 0x000B:
    case 0x000C:
    case 0x000D:
    case 0x0020:
    case 0x0085:
    case 0x00A0:
    case 0x1680:
    case 0x2028:
    case 0x2029:
    case 0x202F:
    case 0x205F:
    case 0x3000: return true;
    default: return (cp >= 0x2000 && cp <= 0x200A);
    }
}

bool IsUnicodeNumber(uint32_t cp) {
    if (cp >= '0' && cp <= '9') return true;
    if (cp >= 0x0660 && cp <= 0x0669) return true;
    if (cp >= 0x06F0 && cp <= 0x06F9) return true;
    if (cp >= 0x0966 && cp <= 0x096F) return true;
    if (cp >= 0x09E6 && cp <= 0x09EF) return true;
    if (cp >= 0x0A66 && cp <= 0x0A6F) return true;
    if (cp >= 0x0AE6 && cp <= 0x0AEF) return true;
    if (cp >= 0x0B66 && cp <= 0x0B6F) return true;
    if (cp >= 0x0BE6 && cp <= 0x0BEF) return true;
    if (cp >= 0x0C66 && cp <= 0x0C6F) return true;
    if (cp >= 0x0CE6 && cp <= 0x0CEF) return true;
    if (cp >= 0x0D66 && cp <= 0x0D6F) return true;
    if (cp >= 0x0E50 && cp <= 0x0E59) return true;
    if (cp >= 0x0ED0 && cp <= 0x0ED9) return true;
    if (cp >= 0x1040 && cp <= 0x1049) return true;
    if (cp >= 0x17E0 && cp <= 0x17E9) return true;
    if (cp >= 0xFF10 && cp <= 0xFF19) return true;
    return false;
}

bool IsUnicodeAccentMark(uint32_t cp) {
    return (cp >= 0x0300 && cp <= 0x036F) || (cp >= 0x1AB0 && cp <= 0x1AFF) || (cp >= 0x1DC0 && cp <= 0x1DFF) ||
           (cp >= 0x20D0 && cp <= 0x20FF) || (cp >= 0xFE20 && cp <= 0xFE2F);
}

bool IsUnicodePunctuationOrSymbol(uint32_t cp) {
    if (cp < 128) {
        return std::ispunct(static_cast<unsigned char>(cp)) != 0;
    }

    return (cp >= 0x2000 && cp <= 0x206F) || (cp >= 0x20A0 && cp <= 0x20CF) || (cp >= 0x2100 && cp <= 0x214F) ||
           (cp >= 0x2190 && cp <= 0x21FF) || (cp >= 0x2200 && cp <= 0x22FF) || (cp >= 0x2300 && cp <= 0x23FF) ||
           (cp >= 0x2460 && cp <= 0x24FF) || (cp >= 0x2500 && cp <= 0x257F) || (cp >= 0x25A0 && cp <= 0x25FF) ||
           (cp >= 0x2600 && cp <= 0x27BF) || (cp >= 0x2B00 && cp <= 0x2BFF) || (cp >= 0x3001 && cp <= 0x303F) ||
           (cp >= 0xFE10 && cp <= 0xFE19) || (cp >= 0xFE30 && cp <= 0xFE6F) || (cp >= 0xFF01 && cp <= 0xFF0F) ||
           (cp >= 0xFF1A && cp <= 0xFF20) || (cp >= 0xFF3B && cp <= 0xFF40) || (cp >= 0xFF5B && cp <= 0xFF65) ||
           (cp >= 0x1F000 && cp <= 0x1FAFF);
}

bool IsUnicodeLetter(uint32_t cp) {
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) {
        return true;
    }
    if (cp < 128) {
        return false;
    }
    if (IsUnicodeWhitespace(cp) || IsUnicodeNumber(cp) || IsUnicodeAccentMark(cp) || IsUnicodePunctuationOrSymbol(cp)) {
        return false;
    }
    return true;
}

bool IsUnicodeLetterOrMark(uint32_t cp, bool qwen35_mode) {
    return IsUnicodeLetter(cp) || (qwen35_mode && IsUnicodeAccentMark(cp));
}

bool IsCjkCodepoint(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x20000 && cp <= 0x2A6DF) ||
           (cp >= 0x2A700 && cp <= 0x2B73F) || (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}

std::string AsciiLowerSpan(const std::string& value) {
    std::string out = value;
    for (char& ch : out) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return out;
}

std::string SliceCodepointSpan(const std::string& text, const std::vector<Utf8Codepoint>& cps, size_t begin,
                               size_t end) {
    if (begin >= end || begin >= cps.size()) {
        return {};
    }
    const size_t byte_begin = cps[begin].start;
    const size_t byte_end = (end >= cps.size()) ? text.size() : cps[end].start;
    return text.substr(byte_begin, byte_end - byte_begin);
}

std::vector<std::string> PretokenizeQwenUnicode(const std::string& text, bool qwen35_mode) {
    const auto cps = DecodeUtf8Codepoints(text);
    std::vector<std::string> pieces;
    pieces.reserve(std::max<size_t>(1, cps.size() / 2));

    auto get_cp = [&](size_t pos) -> uint32_t { return pos < cps.size() ? cps[pos].cp : 0xFFFFFFFFu; };
    auto is_letterish = [&](size_t pos) -> bool {
        return pos < cps.size() && IsUnicodeLetterOrMark(cps[pos].cp, qwen35_mode);
    };
    auto is_number = [&](size_t pos) -> bool { return pos < cps.size() && IsUnicodeNumber(cps[pos].cp); };
    auto is_whitespace = [&](size_t pos) -> bool { return pos < cps.size() && IsUnicodeWhitespace(cps[pos].cp); };
    auto is_punctish = [&](size_t pos) -> bool {
        if (pos >= cps.size()) return false;
        const uint32_t cp = cps[pos].cp;
        return !IsUnicodeWhitespace(cp) && !IsUnicodeNumber(cp) && !IsUnicodeLetterOrMark(cp, qwen35_mode);
    };

    size_t prev_end = 0;
    auto add_piece = [&](size_t end) {
        if (end > prev_end) {
            pieces.push_back(SliceCodepointSpan(text, cps, prev_end, end));
        }
        prev_end = end;
    };

    size_t pos = 0;
    while (pos < cps.size()) {
        const uint32_t cp = get_cp(pos);

        if (cp == '\'' && pos + 1 < cps.size()) {
            const uint32_t c1 = static_cast<uint32_t>(std::tolower(static_cast<unsigned char>(get_cp(pos + 1))));
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') {
                pos += 2;
                add_piece(pos);
                continue;
            }
            if (pos + 2 < cps.size()) {
                const uint32_t c2 = static_cast<uint32_t>(std::tolower(static_cast<unsigned char>(get_cp(pos + 2))));
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) {
                    pos += 3;
                    add_piece(pos);
                    continue;
                }
            }
        }

        if (cp != '\r' && cp != '\n' && !IsUnicodeNumber(cp) && (is_letterish(pos) || is_letterish(pos + 1))) {
            ++pos;
            while (pos < cps.size() && is_letterish(pos)) {
                ++pos;
            }
            add_piece(pos);
            continue;
        }

        if (is_number(pos)) {
            if (qwen35_mode) {
                ++pos;
                add_piece(pos);
            } else {
                size_t count = 0;
                while (pos < cps.size() && is_number(pos) && count < 3) {
                    ++pos;
                    ++count;
                }
                add_piece(pos);
            }
            continue;
        }

        size_t pos2 = pos;
        if (cp == ' ' && is_punctish(pos + 1)) {
            ++pos2;
        }
        if (is_punctish(pos2)) {
            pos = pos2;
            while (pos < cps.size() && is_punctish(pos)) {
                ++pos;
            }
            while (pos < cps.size() && (get_cp(pos) == '\r' || get_cp(pos) == '\n')) {
                ++pos;
            }
            add_piece(pos);
            continue;
        }

        size_t ws = 0;
        size_t last_newline_end = 0;
        while (is_whitespace(pos + ws)) {
            const uint32_t cp_ws = get_cp(pos + ws);
            if (cp_ws == '\r' || cp_ws == '\n') {
                last_newline_end = pos + ws + 1;
            }
            ++ws;
        }
        if (last_newline_end > 0) {
            pos = last_newline_end;
            add_piece(pos);
            continue;
        }
        if (ws > 1 && get_cp(pos + ws) != 0xFFFFFFFFu) {
            pos += ws - 1;
            add_piece(pos);
            continue;
        }
        if (ws > 0) {
            pos += ws;
            add_piece(pos);
            continue;
        }

        ++pos;
        add_piece(pos);
    }

    if (pieces.empty()) {
        pieces.push_back(text);
    }
    return pieces;
}

bool WordPieceShouldLower(const TransformerModel* model) {
    if (!model) return false;
    const std::string lowered = AsciiLowerCopy(model->tokenizer_type);
    if (lowered.find("uncased") != std::string::npos) {
        return true;
    }
    if (lowered.find("cased") != std::string::npos) {
        return false;
    }
    if (model->token_to_id.find("▁the") != model->token_to_id.end() &&
        model->token_to_id.find("▁The") == model->token_to_id.end()) {
        return true;
    }
    for (const auto& item : model->token_to_id) {
        const std::string& token = item.first;
        if (token.rfind("▁", 0) != 0 || token.size() < 4) {
            continue;
        }
        const unsigned char first_ascii = static_cast<unsigned char>(token[3]);
        if (first_ascii < 'a' || first_ascii > 'z') {
            continue;
        }
        std::string upper = token;
        upper[3] = static_cast<char>(std::toupper(first_ascii));
        if (model->token_to_id.find(upper) == model->token_to_id.end()) {
            return true;
        }
    }
    return model->token_to_id.find("the") != model->token_to_id.end() &&
           model->token_to_id.find("The") == model->token_to_id.end();
}

bool WordPieceUsesSentencePieceSurface(const TransformerModel* model) {
    if (!model) return false;
    return model->token_to_id.find("▁the") != model->token_to_id.end() ||
           model->token_to_id.find("▁.") != model->token_to_id.end() ||
           model->token_to_id.find("▁,") != model->token_to_id.end();
}

std::vector<std::string> PretokenizeForWordPiece(const TransformerModel* model, const std::string& text) {
    const bool lowercase = WordPieceShouldLower(model);
    const auto cps = DecodeUtf8Codepoints(text);
    std::vector<std::string> pieces;
    pieces.reserve(std::max<size_t>(1, cps.size() / 2));

    std::string current;
    auto flush = [&]() {
        if (!current.empty()) {
            pieces.push_back(lowercase ? AsciiLowerSpan(current) : current);
            current.clear();
        }
    };

    for (size_t i = 0; i < cps.size(); ++i) {
        const uint32_t cp = cps[i].cp;
        const std::string unit = text.substr(cps[i].start, cps[i].len);
        if (IsUnicodeWhitespace(cp)) {
            flush();
            continue;
        }
        if (IsCjkCodepoint(cp) || IsUnicodePunctuationOrSymbol(cp)) {
            flush();
            pieces.push_back(lowercase ? AsciiLowerSpan(unit) : unit);
            continue;
        }
        current.append(unit);
    }
    flush();
    return pieces;
}

int FindWordPieceUnkId(const TransformerModel* model) {
    if (!model) {
        return -1;
    }
    if (model->unk_token_id >= 0) {
        return model->unk_token_id;
    }
    auto lit = model->token_to_id.find("[UNK]");
    if (lit != model->token_to_id.end()) {
        return lit->second;
    }
    lit = model->token_to_id.find("<unk>");
    if (lit != model->token_to_id.end()) {
        return lit->second;
    }
    return FindSentencePieceUnkId(model);
}

void AppendSentencePieceSurfaceWordPieceTokens(const TransformerModel* model, const std::string& span,
                                               std::vector<int>* out) {
    if (!model || !out || span.empty()) {
        return;
    }

    const int unk_id = FindWordPieceUnkId(model);
    for (const std::string& piece : PretokenizeForWordPiece(model, span)) {
        if (piece.empty()) {
            continue;
        }

        const auto cps = DecodeUtf8Codepoints(piece);
        std::vector<int> piece_ids;
        bool failed = cps.empty();
        size_t start = 0;
        while (!failed && start < cps.size()) {
            size_t end = cps.size();
            int found_id = -1;
            size_t found_end = start;

            while (end > start) {
                const size_t byte_begin = cps[start].start;
                const size_t byte_end = (end < cps.size()) ? cps[end].start : piece.size();
                std::string candidate = piece.substr(byte_begin, byte_end - byte_begin);
                if (start == 0) {
                    candidate.insert(0, "▁");
                }
                auto it = model->token_to_id.find(candidate);
                if (it != model->token_to_id.end()) {
                    found_id = it->second;
                    found_end = end;
                    break;
                }
                --end;
            }

            if (found_id < 0) {
                failed = true;
                break;
            }
            piece_ids.push_back(found_id);
            start = found_end;
        }

        if (failed) {
            if (unk_id >= 0) {
                out->push_back(unk_id);
            }
        } else {
            out->insert(out->end(), piece_ids.begin(), piece_ids.end());
        }
    }
}

void AppendWordPieceTokens(const TransformerModel* model, const std::string& span, std::vector<int>* out) {
    if (!model || !out || span.empty()) {
        return;
    }
    if (WordPieceUsesSentencePieceSurface(model)) {
        AppendSentencePieceSurfaceWordPieceTokens(model, span, out);
        return;
    }

    const int unk_id = FindWordPieceUnkId(model);
    for (const std::string& piece : PretokenizeForWordPiece(model, span)) {
        if (piece.empty()) {
            continue;
        }

        const auto cps = DecodeUtf8Codepoints(piece);
        std::vector<int> piece_ids;
        bool failed = cps.empty();
        size_t start = 0;
        while (!failed && start < cps.size()) {
            size_t end = cps.size();
            int found_id = -1;
            size_t found_end = start;

            while (end > start) {
                const size_t byte_begin = cps[start].start;
                const size_t byte_end = (end < cps.size()) ? cps[end].start : piece.size();
                std::string candidate = piece.substr(byte_begin, byte_end - byte_begin);
                if (start > 0) {
                    candidate.insert(0, "##");
                }
                auto it = model->token_to_id.find(candidate);
                if (it != model->token_to_id.end()) {
                    found_id = it->second;
                    found_end = end;
                    break;
                }
                --end;
            }

            if (found_id < 0) {
                failed = true;
                break;
            }
            piece_ids.push_back(found_id);
            start = found_end;
        }

        if (failed) {
            if (unk_id >= 0) {
                out->push_back(unk_id);
            }
            continue;
        }
        out->insert(out->end(), piece_ids.begin(), piece_ids.end());
    }
}

void AppendBertBpeTokens(const TransformerModel* model, const std::string& span, std::vector<int>* out) {
    if (!model || !out || span.empty()) {
        return;
    }

    const std::string normalized = NormalizeSentencePieceText(span);
    const auto cps = DecodeUtf8Codepoints(normalized);
    if (cps.empty()) {
        return;
    }

    const int unk_id = FindWordPieceUnkId(model);
    size_t max_virtual_bytes = 0;
    for (const std::string& token : model->vocab_tokens) {
        if (token.rfind("▁", 0) == 0 && token.size() > 3) {
            max_virtual_bytes = std::max(max_virtual_bytes, token.size() - 3);
        }
    }
    if (max_virtual_bytes == 0) {
        return;
    }

    const double kInf = std::numeric_limits<double>::infinity();
    std::vector<double> best(cps.size() + 1, kInf);
    std::vector<int> best_id(cps.size(), -1);
    std::vector<size_t> best_next(cps.size(), cps.size());
    best[cps.size()] = 0.0;

    for (int i_signed = static_cast<int>(cps.size()) - 1; i_signed >= 0; --i_signed) {
        const size_t i = static_cast<size_t>(i_signed);
        for (size_t end = i + 1; end <= cps.size(); ++end) {
            const size_t byte_begin = cps[i].start;
            const size_t byte_end = (end < cps.size()) ? cps[end].start : normalized.size();
            const size_t virtual_len = byte_end - byte_begin;
            if (virtual_len > max_virtual_bytes) {
                break;
            }

            std::string candidate = "▁";
            candidate.append(normalized, byte_begin, virtual_len);
            auto it = model->token_to_id.find(candidate);
            if (it == model->token_to_id.end() || !std::isfinite(best[end])) {
                continue;
            }

            const double candidate_cost = std::log1p(static_cast<double>(std::max(0, it->second))) + best[end];
            if (candidate_cost < best[i]) {
                best[i] = candidate_cost;
                best_id[i] = it->second;
                best_next[i] = end;
            }
        }
        if (best_id[i] < 0 && cps[i].len == 3 && normalized.compare(cps[i].start, 3, "▁") == 0 &&
            std::isfinite(best[i + 1])) {
            best[i] = best[i + 1];
            best_next[i] = i + 1;
        }
    }

    for (size_t i = 0; i < cps.size();) {
        const int id = best_id[i];
        if (id >= 0) {
            out->push_back(id);
            const size_t next = best_next[i];
            if (next <= i) {
                break;
            }
            i = next;
            continue;
        }
        if (best_next[i] == i + 1 && cps[i].len == 3 && normalized.compare(cps[i].start, 3, "▁") == 0) {
            ++i;
            continue;
        }
        if (unk_id >= 0) {
            out->push_back(unk_id);
        }
        ++i;
    }
}

std::vector<std::string> PretokenizeForByteBpe(const TransformerModel* model, const std::string& text) {
    if (UseGemmaPretokenizer(model)) {
        // Hugging Face GemmaTokenizer normalizes spaces to U+2581 before BPE
        // and relies on byte fallback inside the BPE model. Over-splitting the
        // text here prevents merges and explodes prompt length.
        std::string normalized;
        normalized.reserve(text.size() * 3);
        for (char ch : text) {
            if (ch == ' ') {
                normalized.append("\xE2\x96\x81");
            } else {
                normalized.push_back(ch);
            }
        }
        return {std::move(normalized)};
    }

    // GPT-2 default (ASCII-centric fallback).
    static const std::regex kPatternGpt2(
        "'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\\sA-Za-z0-9]+|\\s+(?!\\S)|\\s+",
        std::regex_constants::ECMAScript | std::regex_constants::icase);

    // Qwen2/Qwen3-style pre-tokenization approximation.
    // Matches llama.cpp/HF behavior more closely than GPT-2 fallback by:
    // - splitting numbers into 1..3 chunks
    // - preserving newline-aware punctuation chunks
    static const std::regex kPatternQwen("'s|'t|'re|'ve|'m|'ll|'d|[^\\r\\nA-Za-z0-9]?[A-Za-z]+|[0-9]{1,3}| "
                                         "?[^\\sA-Za-z0-9]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                                         std::regex_constants::ECMAScript | std::regex_constants::icase);

    std::vector<std::string> pieces;
    pieces.reserve(std::max<size_t>(1, text.size() / 4));

    if (UseQwen35Pretokenizer(model)) {
        return PretokenizeQwenUnicode(text, true);
    }
    if (UseLFM2Pretokenizer(model)) {
        // LFM2 / LFM2.5 use a cl100k/Llama-3-family pre-tokenizer:
        //   (?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
        // The Unicode-aware splitter matches this (multilingual \p{L}); pass
        // qwen35_mode=false to keep digits grouped in runs of 1..3 (\p{N}{1,3}).
        return PretokenizeQwenUnicode(text, /*qwen35_mode=*/false);
    }

    try {
        const std::regex& pattern = UseQwenPretokenizer(model) ? kPatternQwen : kPatternGpt2;
        for (std::sregex_iterator it(text.begin(), text.end(), pattern), end; it != end; ++it) {
            if (!it->str().empty()) {
                pieces.push_back(it->str());
            }
        }
    } catch (...) {
        // Fall back to whole-string piece if regex engine rejects pattern.
    }

    if (pieces.empty()) {
        pieces.push_back(text);
    }

    return pieces;
}

std::vector<std::string> EncodePieceSymbols(const TransformerModel* model, const std::string& piece) {
    if (UseGemmaPretokenizer(model)) {
        return SplitUtf8Units(piece);
    }

    const auto& b2u = ByteToUnicode();
    std::vector<std::string> symbols;
    symbols.reserve(piece.size());

    for (unsigned char b : piece) {
        symbols.push_back(b2u[b]);
    }

    return symbols;
}

std::vector<std::string> MergeWithBpeRanks(const TransformerModel* model, std::vector<std::string> symbols) {
    if (symbols.size() < 2 || model->bpe_merge_ranks.empty()) {
        return symbols;
    }

    while (symbols.size() > 1) {
        int best_rank = std::numeric_limits<int>::max();
        int best_pos = -1;

        for (int i = 0; i + 1 < static_cast<int>(symbols.size()); ++i) {
            auto it = model->bpe_merge_ranks.find(MakeMergeKey(symbols[i], symbols[i + 1]));
            if (it != model->bpe_merge_ranks.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos = i;
            }
        }

        if (best_pos < 0) {
            break;
        }

        symbols[best_pos] += symbols[best_pos + 1];
        symbols.erase(symbols.begin() + best_pos + 1);
    }

    return symbols;
}

void AppendByteFallbackTokens(const TransformerModel* model, const std::string& raw_bytes, std::vector<int>* out) {
    if (!out) return;

    const auto& b2u = ByteToUnicode();

    for (unsigned char b : raw_bytes) {
        const std::string& byte_symbol = b2u[b];
        auto byte_it = model->token_to_id.find(byte_symbol);
        if (byte_it != model->token_to_id.end()) {
            out->push_back(byte_it->second);
            continue;
        }

        char buf[8];
        std::snprintf(buf, sizeof(buf), "<0x%02X>", static_cast<unsigned int>(b));
        auto hex_it = model->token_to_id.find(buf);
        if (hex_it != model->token_to_id.end()) {
            out->push_back(hex_it->second);
            continue;
        }

        std::string raw_ch(1, static_cast<char>(b));
        auto raw_it = model->token_to_id.find(raw_ch);
        if (raw_it != model->token_to_id.end()) {
            out->push_back(raw_it->second);
        }
    }
}

void AppendSymbolsToIds(const TransformerModel* model, const std::vector<std::string>& symbols, std::vector<int>* out) {
    if (!out) return;

    if (!IsByteLevelBpeTokenizer(model)) {
        for (const std::string& sym : symbols) {
            auto it = model->token_to_id.find(sym);
            if (it != model->token_to_id.end()) {
                out->push_back(it->second);
                continue;
            }

            AppendByteFallbackTokens(model, sym, out);
        }
        return;
    }

    const auto& u2b = UnicodeToByte();

    for (const std::string& sym : symbols) {
        auto it = model->token_to_id.find(sym);
        if (it != model->token_to_id.end()) {
            out->push_back(it->second);
            continue;
        }

        // Decode byte-level unicode symbols back to raw bytes and fall back.
        std::string raw_bytes;
        const auto units = SplitUtf8Units(sym);
        for (const std::string& unit : units) {
            auto b_it = u2b.find(unit);
            if (b_it != u2b.end()) {
                raw_bytes.push_back(static_cast<char>(b_it->second));
            } else {
                raw_bytes.append(unit);
            }
        }
        AppendByteFallbackTokens(model, raw_bytes, out);
    }
}

struct MergePair {
    float score;
    size_t position;
    size_t generation;

    bool operator<(const MergePair& other) const { return score > other.score; }
};

std::vector<std::string> MergeWithPriorityQueue(const TransformerModel* model, std::vector<std::string> tokens) {
    if (tokens.size() < 2) {
        return tokens;
    }

    std::vector<size_t> generation(tokens.size(), 0);
    std::priority_queue<MergePair> pq;

    auto try_add_merge = [&](size_t pos) {
        if (pos >= tokens.size() - 1) return;

        std::string merged = tokens[pos] + tokens[pos + 1];
        if (Tokenizer::HasToken(model, merged)) {
            float score = Tokenizer::GetTokenScore(model, merged);
            pq.push({score, pos, generation[pos]});
        }
    };

    for (size_t i = 0; i + 1 < tokens.size(); ++i) {
        try_add_merge(i);
    }

    while (!pq.empty()) {
        MergePair top = pq.top();
        pq.pop();

        if (top.position >= tokens.size() - 1) {
            continue;
        }
        if (top.generation != generation[top.position]) {
            continue;
        }

        size_t pos = top.position;
        std::string merged = tokens[pos] + tokens[pos + 1];

        if (!Tokenizer::HasToken(model, merged)) {
            continue;
        }

        tokens[pos] = merged;
        tokens.erase(tokens.begin() + pos + 1);

        if (pos > 0) {
            generation[pos - 1]++;
        }
        generation[pos]++;

        if (generation.size() > tokens.size()) {
            generation.resize(tokens.size());
        }

        if (pos > 0) {
            try_add_merge(pos - 1);
        }
        try_add_merge(pos);
    }

    return tokens;
}

void AppendSentencePieceUnigram(const TransformerModel* model, const std::string& span, std::vector<int>* out) {
    if (!model || !out || span.empty()) {
        return;
    }

    const std::string normalized = NormalizeSentencePieceText(span);
    const auto cps = DecodeUtf8Codepoints(normalized);
    if (cps.empty()) {
        return;
    }

    size_t max_token_bytes = 0;
    for (const auto& token : model->vocab_tokens) {
        max_token_bytes = std::max(max_token_bytes, token.size());
    }
    if (max_token_bytes == 0) {
        return;
    }

    const int unk_id = FindSentencePieceUnkId(model);
    const float kNegInf = -std::numeric_limits<float>::infinity();
    std::vector<float> best(cps.size() + 1, kNegInf);
    std::vector<int> best_id(cps.size(), -1);
    std::vector<size_t> best_next(cps.size(), cps.size());
    best[cps.size()] = 0.0f;

    for (int i = static_cast<int>(cps.size()) - 1; i >= 0; --i) {
        const size_t byte_begin = cps[static_cast<size_t>(i)].start;
        for (size_t j = static_cast<size_t>(i) + 1; j <= cps.size(); ++j) {
            const size_t byte_end = (j < cps.size()) ? cps[j].start : normalized.size();
            if (byte_end - byte_begin > max_token_bytes) {
                break;
            }

            auto it = model->token_to_id.find(normalized.substr(byte_begin, byte_end - byte_begin));
            if (it == model->token_to_id.end() || !std::isfinite(best[j])) {
                continue;
            }

            const float candidate = GetSentencePieceScore(model, it->second) + best[j];
            if (candidate > best[static_cast<size_t>(i)]) {
                best[static_cast<size_t>(i)] = candidate;
                best_id[static_cast<size_t>(i)] = it->second;
                best_next[static_cast<size_t>(i)] = j;
            }
        }

        if (best_id[static_cast<size_t>(i)] < 0 && unk_id >= 0 && std::isfinite(best[static_cast<size_t>(i) + 1])) {
            best[static_cast<size_t>(i)] = GetSentencePieceScore(model, unk_id) + best[static_cast<size_t>(i) + 1];
            best_id[static_cast<size_t>(i)] = unk_id;
            best_next[static_cast<size_t>(i)] = static_cast<size_t>(i) + 1;
        }
    }

    if (best_id[0] < 0) {
        if (model->arch_flags.is_gemma4) {
            throw std::runtime_error("Gemma4 tokenizer could not segment prompt with GGUF sentencepiece metadata");
        }
        return;
    }

    for (size_t i = 0; i < cps.size();) {
        const int id = best_id[i];
        if (id < 0) {
            break;
        }
        out->push_back(id);
        const size_t next = best_next[i];
        if (next <= i) {
            break;
        }
        i = next;
    }
}

}  // namespace

// ============================================================================
// Tokenize
// ============================================================================

std::vector<int> Tokenizer::Tokenize(const TransformerModel* model, const std::string& text, bool add_bos,
                                     bool add_eos) {
    std::vector<int> result;
    if (!model) {
        return result;
    }

    const auto prompt_begins_with_bos_literal = [&]() {
        if (!model || model->bos_token_id < 0 || model->bos_token_id >= static_cast<int>(model->vocab_tokens.size())) {
            return false;
        }
        const std::string& bos_literal = model->vocab_tokens[static_cast<size_t>(model->bos_token_id)];
        return !bos_literal.empty() && text.rfind(bos_literal, 0) == 0;
    };

    if (add_bos && model->bos_token_id >= 0 && !prompt_begins_with_bos_literal()) {
        result.push_back(model->bos_token_id);
    }

    if (text.empty()) {
        const int end_token_id = ResolveEndTokenId(model);
        if (add_eos && end_token_id >= 0) {
            result.push_back(end_token_id);
        }
        return result;
    }

    const auto tokenize_plain_span = [&](const std::string& span) {
        if (span.empty()) return;
        if (UseSentencePieceUnigramTokenizer(model) && model->bpe_merge_ranks.empty()) {
            AppendSentencePieceUnigram(model, span, &result);
            return;
        }
        if (UseWordPieceTokenizer(model)) {
            AppendWordPieceTokens(model, span, &result);
            return;
        }
        if (UseBertBpeTokenizer(model)) {
            AppendBertBpeTokens(model, span, &result);
            return;
        }
        if (!model->bpe_merge_ranks.empty()) {
            const std::vector<std::string> pieces = PretokenizeForByteBpe(model, span);
            for (const std::string& piece : pieces) {
                std::vector<std::string> symbols = EncodePieceSymbols(model, piece);
                symbols = MergeWithBpeRanks(model, std::move(symbols));
                AppendSymbolsToIds(model, symbols, &result);
            }
            return;
        }

        // Legacy fallback path (kept for tokenizer formats without merges).
        std::vector<std::string> tokens = SplitToChars(span);
        tokens = MergeWithPriorityQueue(model, std::move(tokens));

        for (const std::string& tok : tokens) {
            auto it = model->token_to_id.find(tok);
            if (it != model->token_to_id.end()) {
                result.push_back(it->second);
            } else {
                for (size_t i = 0; i < tok.size(); i++) {
                    std::string single_char = tok.substr(i, 1);
                    auto c_it = model->token_to_id.find(single_char);
                    if (c_it != model->token_to_id.end()) {
                        result.push_back(c_it->second);
                    } else {
                        char buf[8];
                        std::snprintf(buf, sizeof(buf), "<0x%02X>", static_cast<unsigned char>(tok[i]));
                        auto b_it = model->token_to_id.find(buf);
                        if (b_it != model->token_to_id.end()) {
                            result.push_back(b_it->second);
                        }
                    }
                }
            }
        }
    };

    // Handle registered control tokens atomically on both the merges path and
    // the no-merges fallback path so Gemma4 turn markers survive intact.
    size_t cursor = 0;
    size_t span_start = 0;
    while (cursor < text.size()) {
        if (text[cursor] == '<' || text[cursor] == '[') {
            const char close_ch = text[cursor] == '<' ? '>' : ']';
            const size_t special_close = text.find(close_ch, cursor + 1);
            if (special_close != std::string::npos) {
                const size_t tok_end = special_close + 1;
                const std::string special = text.substr(cursor, tok_end - cursor);
                if (IsAtomicSpecialTokenLiteral(model, special)) {
                    auto it = model->token_to_id.find(special);
                    tokenize_plain_span(text.substr(span_start, cursor - span_start));
                    result.push_back(it->second);
                    cursor = tok_end;
                    span_start = cursor;
                    continue;
                }
            }
        }
        ++cursor;
    }
    tokenize_plain_span(text.substr(span_start));

    const int end_token_id = ResolveEndTokenId(model);
    if (add_eos && end_token_id >= 0) {
        result.push_back(end_token_id);
    }

    if (IsDebugTokenizerEnabled()) {
        std::fprintf(stderr, "[TokenizerDebug] arch=%d tokenizer_type=%s add_bos=%d add_eos=%d text=\"%s\"\n",
                     static_cast<int>(model->arch), model->tokenizer_type.c_str(), add_bos ? 1 : 0, add_eos ? 1 : 0,
                     text.c_str());
        std::fprintf(stderr, "[TokenizerDebug] token_count=%zu ids=", result.size());
        for (size_t i = 0; i < result.size(); ++i) {
            if (i != 0) std::fprintf(stderr, ",");
            std::fprintf(stderr, "%d", result[i]);
        }
        std::fprintf(stderr, "\n");
        const size_t preview = std::min<size_t>(result.size(), 16);
        for (size_t i = 0; i < preview; ++i) {
            const int id = result[i];
            const std::string raw =
                (id >= 0 && id < static_cast<int>(model->vocab_tokens.size())) ? model->vocab_tokens[id] : "";
            const std::string text_piece = DetokenizeImpl(model, id);
            std::fprintf(stderr, "[TokenizerDebug] tok[%zu] id=%d raw=\"%s\" text=\"%s\"\n", i, id, raw.c_str(),
                         text_piece.c_str());
        }
    }

    return result;
}

// ============================================================================
// Detokenize
// ============================================================================

void Tokenizer::BuildStreamTokenPieceCache(TransformerModel* model) {
    if (!model) return;
    model->stream_token_pieces.resize(model->vocab_tokens.size());
    for (size_t i = 0; i < model->vocab_tokens.size(); ++i) {
        model->stream_token_pieces[i] = DetokenizeImpl(model, static_cast<int>(i));
    }
}

std::string Tokenizer::Detokenize(const TransformerModel* model, int token_id) {
    if (!model) {
        return "";
    }
    auto log_debug = [&](const std::string& piece) {
        if (!IsDebugTokenizerEnabled()) {
            return;
        }
        const std::string raw = (token_id >= 0 && token_id < static_cast<int>(model->vocab_tokens.size()))
                                    ? model->vocab_tokens[token_id]
                                    : "";
        std::fprintf(stderr, "[TokenizerDebug] detokenize id=%d raw=\"%s\" text=\"%s\"\n", token_id, raw.c_str(),
                     piece.c_str());
    };
    if (token_id >= 0 && token_id < static_cast<int>(model->stream_token_pieces.size())) {
        const std::string& piece = model->stream_token_pieces[static_cast<size_t>(token_id)];
        log_debug(piece);
        return piece;
    }
    std::string piece = DetokenizeImpl(model, token_id);
    log_debug(piece);
    return piece;
}

std::string Tokenizer::DetokenizeStructured(const TransformerModel* model, int token_id) {
    if (!model || token_id < 0 || token_id >= static_cast<int>(model->vocab_tokens.size())) {
        return "";
    }
    static const char* kStructuredBoundaries[] = {
        "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
    };
    for (const char* boundary : kStructuredBoundaries) {
        const auto it = model->token_to_id.find(boundary);
        if (it != model->token_to_id.end() && it->second == token_id) {
            return boundary;
        }
    }
    return Detokenize(model, token_id);
}

std::string Tokenizer::DetokenizeMultiple(const TransformerModel* model, const std::vector<int>& token_ids) {
    std::string result;
    for (int id : token_ids) {
        if (id == model->bos_token_id || id == model->eos_token_id || id == model->sep_token_id ||
            id == model->pad_token_id) {
            continue;
        }
        result += Detokenize(model, id);
    }
    return result;
}
