// Standalone oracle for llama.cpp 74bca3950054eed1795f525cb9cf06bfe796492d.
// This tool deliberately links only llama.cpp, never DenseCore's ggml libraries.
#include "llama.h"
#include "llama-ext.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc != 7) throw std::runtime_error(
            "usage: llama_nextn_logits MODEL HIDDEN_F32 TOKEN_IDS_CSV OUTPUT_PREFIX CHUNK_SIZE THREADS");
        const int chunk = std::stoi(argv[5]), threads = std::stoi(argv[6]);
        if (chunk < 1 || chunk > 16 || threads < 1) throw std::runtime_error("invalid chunk/threads");
        std::vector<llama_token> tokens;
        std::stringstream csv(argv[3]);
        std::string item;
        while (std::getline(csv, item, ',')) {
            size_t consumed = 0;
            const long value = std::stol(item, &consumed);
            if (consumed != item.size() || value < 0 || value > INT32_MAX) throw std::runtime_error("invalid token ID");
            tokens.push_back(static_cast<llama_token>(value));
        }
        if (tokens.empty() || tokens.size() > 128) throw std::runtime_error("require 1..128 tokens");
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.load_mtp = true;
        mp.use_extra_bufts = false;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], mp), llama_model_free);
        if (!model || llama_model_n_layer_nextn(model.get()) != 1) throw std::runtime_error("require one NextN head");
        const int embedding = llama_model_n_embd(model.get());
        const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
        for (auto token : tokens) if (token >= vocab) throw std::runtime_error("token outside vocabulary");
        const size_t count = tokens.size() * static_cast<size_t>(embedding);
        std::vector<float> hidden(count);
        std::ifstream input(argv[2], std::ios::binary);
        if (!input.read(reinterpret_cast<char*>(hidden.data()), count * sizeof(float)) || input.peek() != EOF)
            throw std::runtime_error("hidden input must contain exactly N*embedding float32 values");
        for (float value : hidden) if (!std::isfinite(value)) throw std::runtime_error("nonfinite hidden input");
        auto cp = llama_context_default_params();
        cp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        cp.n_ctx = 128; cp.n_batch = 16; cp.n_ubatch = 16; cp.n_seq_max = 1;
        cp.n_threads = threads; cp.n_threads_batch = threads;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        if (!ctx) throw std::runtime_error("MTP context initialization failed");
        llama_set_embeddings_nextn(ctx.get(), true, false);
        std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(
            llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
        if (!batch) throw std::runtime_error("batch allocation failed");
        const std::string prefix(argv[4]);
        std::ofstream logits(prefix + ".logits.f32", std::ios::binary);
        std::ofstream feedback(prefix + ".hidden.f32", std::ios::binary);
        if (!logits || !feedback) throw std::runtime_error("cannot create output files");
        std::vector<int> argmax;
        for (size_t begin = 0; begin < tokens.size(); begin += chunk) {
            llama_batch_ext_clear(batch.get());
            const size_t length = std::min(static_cast<size_t>(chunk), tokens.size() - begin);
            for (size_t j = 0; j < length; ++j) {
                const int idx = llama_batch_ext_add_token(batch.get(), 0, tokens[begin+j]);
                const llama_pos position = static_cast<llama_pos>(begin+j);
                if (idx < 0 || !llama_batch_ext_set_pos(batch.get(), idx, &position) ||
                    !llama_batch_ext_set_embd_state(batch.get(), idx,
                        {hidden.data() + (begin+j)*embedding, 1, static_cast<size_t>(embedding)}) ||
                    !llama_batch_ext_set_output_logits(batch.get(), idx, true))
                    throw std::runtime_error("cannot bind MTP batch inputs");
            }
            if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0)
                throw std::runtime_error("NextN execution failed");
            for (size_t j = 0; j < length; ++j) {
                const float* row = llama_get_logits_ith(ctx.get(), static_cast<int32_t>(j));
                const float* state = llama_get_embeddings_nextn_ith(ctx.get(), static_cast<int32_t>(j));
                if (!row || !state) throw std::runtime_error("missing oracle outputs");
                for (int v = 0; v < vocab; ++v) if (!std::isfinite(row[v])) throw std::runtime_error("nonfinite oracle logits");
                argmax.push_back(static_cast<int>(std::max_element(row, row+vocab)-row));
                logits.write(reinterpret_cast<const char*>(row), vocab*sizeof(float));
                feedback.write(reinterpret_cast<const char*>(state), embedding*sizeof(float));
            }
        }
        logits.close(); feedback.close();
        if (!logits || !feedback) throw std::runtime_error("output write failed");
        std::ofstream meta(prefix + ".json");
        meta << "{\"oracle_revision\":\"74bca3950054eed1795f525cb9cf06bfe796492d\",\"rows\":"
             << tokens.size() << ",\"embedding\":" << embedding << ",\"vocab\":" << vocab
             << ",\"chunk_size\":" << chunk << ",\"threads\":" << threads << ",\"argmax\":[";
        for (size_t i = 0; i < argmax.size(); ++i) { if (i) meta << ','; meta << argmax[i]; }
        meta << "]}\n";
        if (!meta) throw std::runtime_error("metadata write failed");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "NextN oracle: " << e.what() << '\n';
        return 1;
    }
}
