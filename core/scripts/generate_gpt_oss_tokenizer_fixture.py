#!/usr/bin/env python3
"""Generate the small real-vocabulary GPT-OSS tokenizer oracle fixture.

uv run --with tiktoken==0.14.0 python core/scripts/generate_gpt_oss_tokenizer_fixture.py MODEL.gguf
The GGUF can be a header-only range download (first 16MiB of the pinned file).
No tensor data or complete model is needed. Runtime has no Python dependency.
"""
import argparse
import json
import pathlib
import struct

import tiktoken

REVISION = "15dcfc7e7c608c6161507211ed43c5b590801b4b"
MODEL_SHA256 = "27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901"
SAMPLES = ["Hello, world!", "HTTPServer XMLParser camelCase", "I'm we'RE aren't it'LL",
           "a\u0301 cafe\u0301 한국어 中文 日本語", "123456789 Ⅷ ١٢٣٤", " A\n\nB?!/  \r\n  Z"]


def metadata(path):
    with open(path, "rb") as stream:
        magic, version, _, count = struct.unpack("<4sIQQ", stream.read(24))
        if magic != b"GGUF" or version != 3:
            raise ValueError("Expected GGUF v3")

        def read(kind):
            if kind == 8:
                length = struct.unpack("<Q", stream.read(8))[0]
                return stream.read(length).decode()
            if kind == 9:
                element, length = struct.unpack("<IQ", stream.read(12))
                return [read(element) for _ in range(length)]
            fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
                   10: "Q", 11: "q", 12: "d"}[kind]
            return struct.unpack("<" + fmt, stream.read(struct.calcsize(fmt)))[0]

        result = {}
        for _ in range(count):
            key = read(8)
            result[key] = read(struct.unpack("<I", stream.read(4))[0])
        return result


def generate(data):
    # GPT byte encoder preserves visible Latin-1 and maps remaining byte values
    # to U+0100 onward. Each encoded string can be searched before pretokenizing:
    # retain all candidate merges, including those across piece boundaries.
    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    codepoints = byte_values[:]
    next_codepoint = 256
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            codepoints.append(next_codepoint)
            next_codepoint += 1
    encoder = dict(zip(byte_values, map(chr, codepoints)))
    encoded = ["".join(encoder[value] for value in text.encode()) for text in SAMPLES]
    vocab = [(token, index) for index, token in enumerate(data["tokenizer.ggml.tokens"])
             if token and any(token in text for text in encoded)]
    merges = []
    for rank, pair in enumerate(data["tokenizer.ggml.merges"]):
        left, right = pair.split(" ", 1)
        if any(left + right in text for text in encoded):
            merges.append((left + "\x1f" + right, rank))
    quote = lambda value: json.dumps(value, ensure_ascii=False).replace("\\u001f", "\\037")
    output = (f"// ggml-org/gpt-oss-20b-GGUF@{REVISION}\n"
              f"// GGUF SHA256 {MODEL_SHA256}; oracle tiktoken 0.14.0 o200k_base.\n")
    for name, entries in [("Vocab", vocab), ("Merges", merges)]:
        output += f"const std::pair<const char*, int> kGptOssFixture{name}[] = {{\n"
        output += "".join(f"    {{{quote(token)}, {index}}},\n" for token, index in entries) + "};\n"
    output += "const std::pair<const char*, std::vector<int>> kGptOssTokenizerCases[] = {\n"
    oracle = tiktoken.get_encoding("o200k_base")
    for text in SAMPLES:
        ids = ", ".join(map(str, oracle.encode(text, disallowed_special=())))
        output += f"    {{{quote(text)}, {{{ids}}}}},\n"
    return output + "};\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=pathlib.Path)
    args = parser.parse_args()
    target = pathlib.Path(__file__).resolve().parents[1] / "tests/models/fixtures/gpt_oss_tokenizer.inl"
    target.write_text(generate(metadata(args.model)))
