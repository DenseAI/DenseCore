#ifndef DENSECORE_TOKENIZER_H
#define DENSECORE_TOKENIZER_H

#include <string>
#include <unordered_map>
#include <vector>

#include "densecore/models/model_types.h"

/**
 * BPE Tokenizer for GGUF models.
 *
 * Implements Byte-Pair Encoding (BPE) tokenization using the vocabulary
 * and merge scores loaded from the GGUF model file.
 */
class Tokenizer {
public:
    /**
     * Tokenize input text using BPE algorithm.
     *
     * @param model TransformerModel containing vocab and scores
     * @param text Input text to tokenize
     * @param add_bos Whether to prepend BOS token
     * @param add_eos Whether to append EOS token
     * @return Vector of token IDs
     */
    static std::vector<int> Tokenize(const TransformerModel* model, const std::string& text, bool add_bos = true,
                                     bool add_eos = false);

    /**
     * Convert a single token ID back to string.
     */
    static std::string Detokenize(const TransformerModel* model, int token_id);

    /**
     * Convert multiple token IDs back to string.
     */
    static std::string DetokenizeMultiple(const TransformerModel* model, const std::vector<int>& token_ids);

    /**
     * Precompute stream-safe detokenized pieces for hot decode/token streaming.
     */
    static void BuildStreamTokenPieceCache(TransformerModel* model);

    /**
     * Check if a token exists in the vocabulary.
     */
    static bool HasToken(const TransformerModel* model, const std::string& token);

    /**
     * Get the score for a token (lower = higher merge priority).
     */
    static float GetTokenScore(const TransformerModel* model, const std::string& token);

private:
    /**
     * Split UTF-8 string into individual bytes/characters.
     * Handles multi-byte UTF-8 sequences correctly.
     */
    static std::vector<std::string> SplitToChars(const std::string& text);

    /**
     * Find the best merge pair in the current token sequence.
     * Returns the index of the first token in the pair, or -1 if no merge found.
     */
    static int FindBestMerge(const TransformerModel* model, const std::vector<std::string>& tokens);
};

#endif  // DENSECORE_TOKENIZER_H
