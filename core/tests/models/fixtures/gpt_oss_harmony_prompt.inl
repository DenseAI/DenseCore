// Oracle: pinned ggml-org/gpt-oss-20b-GGUF@15dcfc7e7c608c6161507211ed43c5b590801b4b Jinja template.
// tools=[], builtin_tools=[], add_generation_prompt=true, strftime_now=2026-10-25.
const char* const kGptOssCanonicalPromptOracle = R"GPTOSS(<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2026-10-25

Reasoning: medium

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|><|start|>developer<|message|># Instructions

Be concise

<|end|><|start|>user<|message|>Hi<|end|><|start|>assistant<|channel|>final<|message|>Hello<|end|><|start|>user<|message|>Next<|end|><|start|>assistant)GPTOSS";
