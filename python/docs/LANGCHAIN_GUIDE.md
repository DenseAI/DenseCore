# LangChain and LangGraph

DenseCore provides optional LangChain wrappers for local GGUF inference. The
integration is useful when an application already uses LangChain abstractions; it
does not add a hosted service, tool sandbox, or output-quality guarantee.

## Install

```bash
pip install "densecore[langchain]"
```

Supply exactly one model source to each wrapper:

- `model_path` for a local GGUF file.
- `hf_repo_id` for a supported Hugging Face model repository.

## Text Completion

```python
from densecore.integrations import DenseCoreLLM

llm = DenseCoreLLM(
    hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    temperature=0.7,
    max_tokens=256,
)

print(llm.invoke("Explain CPU inference in one sentence."))
```

Use the wrapper in a normal LangChain runnable pipeline:

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("Write one sentence about {topic}.")
chain = prompt | llm | StrOutputParser()
print(chain.invoke({"topic": "NUMA locality"}))
```

## Chat

```python
from densecore.integrations import DenseCoreChatModel
from langchain_core.messages import HumanMessage, SystemMessage

chat = DenseCoreChatModel(
    hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    temperature=0.2,
    max_tokens=128,
)

response = chat.invoke(
    [
        SystemMessage(content="Answer concisely."),
        HumanMessage(content="What is a GGUF model file?"),
    ]
)
print(response.content)
```

## Tools

`bind_tools()` asks the model to emit a tool-call payload and exposes parsed calls
on the returned `AIMessage`. It does not execute tools. Treat every model-proposed
tool call as untrusted input: validate its name and arguments, enforce
authorization, and run it in an application-controlled boundary.

```python
from densecore.integrations import DenseCoreChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool


@tool
def add_numbers(left: int, right: int) -> str:
    """Add two integer values."""
    return str(left + right)


chat = DenseCoreChatModel(hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct")
response = chat.bind_tools([add_numbers]).invoke(
    [HumanMessage(content="Use the add_numbers tool for 25 plus 4.")]
)

for call in response.tool_calls:
    if call["name"] == "add_numbers":
        print(add_numbers.invoke(call["args"]))
```

Do not pass model-controlled expressions to `eval`, shell commands, database
queries, file paths, or network clients without a deliberate validation and policy
layer.

## Structured Output

`with_structured_output()` adds a JSON schema to the prompt, enables JSON mode,
and parses the response. This is not constrained decoding: schema validation can
fail, especially for small or unqualified models. Handle parser errors and validate
the returned value before using it in a workflow.

```python
from pydantic import BaseModel


class Answer(BaseModel):
    summary: str
    confidence: float


structured_chat = chat.with_structured_output(Answer)
result = structured_chat.invoke("Summarize: DenseCore is CPU-first.")
print(result.additional_kwargs.get("parsed"))
```

## LangGraph Node

For a one-node graph, use `create_densecore_node()`. The state must contain a
`messages` sequence; the node appends an `AIMessage` to that sequence.

```python
from typing import TypedDict

from densecore.integrations import create_densecore_node
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph


class ConversationState(TypedDict):
    messages: list[BaseMessage]


workflow = StateGraph(ConversationState)
workflow.add_node(
    "llm",
    create_densecore_node(
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        max_tokens=128,
    ),
)
workflow.set_entry_point("llm")
workflow.add_edge("llm", END)

app = workflow.compile()
result = app.invoke({"messages": [HumanMessage(content="Hello")]})
print(result["messages"][-1].content)
```

## Production Boundary

- Run the exact model, prompt format, and decoding configuration through a
  representative output-quality check before deployment.
- Apply timeouts, concurrency limits, request-size limits, and observability in
  the hosting application or DenseCore server deployment.
- Keep tools capability-scoped and fail closed on validation or policy errors.
- For server deployment, use [the deployment guide](../../docs/DEPLOYMENT.md).
