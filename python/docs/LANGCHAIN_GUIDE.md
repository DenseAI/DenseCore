# LangChain & LangGraph Integration Guide

Complete guide to using DenseCore with LangChain and LangGraph for building advanced LLM applications.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [LangChain Integration](#langchain-integration)
4. [LangGraph Integration](#langgraph-integration)
5. [Production Examples](#production-examples)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

## Installation

Install DenseCore with LangChain/LangGraph support:

```bash
pip install densecore[langchain]
```

This installs:
- `langchain-core` - Core LangChain abstractions
- `langchain-community` - Community integrations
- `langchain` - Full LangChain framework
- `langgraph` - State machine and workflow support

## Quick Start

### Basic LangChain Usage

```python
from densecore.integrations import DenseCoreLLM

# Initialize LLM
llm = DenseCoreLLM(
    hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    temperature=0.7,
    max_tokens=256
)

# Simple generation
response = llm("Explain quantum computing in one sentence.")
print(response)
```

### Basic LangGraph Usage

```python
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from densecore.integrations import create_densecore_node
from typing import TypedDict, Sequence

# Define state
class State(TypedDict):
    messages: Sequence

# Create workflow
workflow = StateGraph(State)

# Add DenseCore node
node = create_densecore_node(
    hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    temperature=0.7
)
workflow.add_node("llm", node)
workflow.set_entry_point("llm")
workflow.add_edge("llm", END)

# Compile and run
app = workflow.compile()
result = app.invoke({"messages": [HumanMessage(content="Hello!")]})
print(result["messages"][-1].content)
```

## LangChain Integration

### DenseCoreLLM

Standard LLM wrapper for text completion tasks.

#### Features

- ✅ Synchronous generation
- ✅ Streaming support
- ✅ Async operations
- ✅ LangChain callback integration
- ✅ Full parameter control

#### Basic Usage

```python
from densecore.integrations import DenseCoreLLM

llm = DenseCoreLLM(
    hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    model_path=None,  # Or use local path
    temperature=0.7,
    max_tokens=256,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
    stop=["</s>", "\n\n"]
)

# Generate
response = llm("What is machine learning?")
```

#### Streaming

```python
for token in llm.stream("Write a short story:"):
    print(token, end="", flush=True)
```

#### Async

```python
import asyncio

async def generate():
    response = await llm.agenerate(["Question 1", "Question 2"])
    for generation in response.generations:
        print(generation[0].text)

asyncio.run(generate())
```

### DenseCoreChatModel

Chat-oriented interface for conversational AI.

#### Basic Usage

```python
from densecore.integrations import DenseCoreChatModel
from langchain_core.messages import HumanMessage, SystemMessage

chat = DenseCoreChatModel(
    hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    temperature=0.7,
    max_tokens=200
)

messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="What is Python?")
]

response = chat(messages)
print(response.content)
```

#### Multi-turn Conversation

messages = [
    SystemMessage(content="You are a helpful coding assistant."),
    HumanMessage(content="How do I read a file in Python?"),
]

# First response
response1 = chat(messages)
messages.append(response1)

# Follow-up
messages.append(HumanMessage(content="What about writing to a file?"))
response2 = chat(messages)
print(response2.content)
```

#### Tool Calling with `bind_tools()`

DenseCore supports tool calling for building agentic applications:

```python
from densecore.integrations import DenseCoreChatModel
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# Define tools
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

@tool
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    return f"Weather in {location}: Sunny, 72°F"

# Bind tools to model
chat = DenseCoreChatModel(hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct")
chat_with_tools = chat.bind_tools([calculator, get_weather])

# Invoke - model may decide to call tools
response = chat_with_tools.invoke([
    HumanMessage(content="What is 25 * 4?")
])

# Check for tool calls
if response.tool_calls:
    print(f"Tool calls: {response.tool_calls}")
else:
    print(f"Response: {response.content}")
```

#### Structured Output with `with_structured_output()`

Force the model to output structured JSON:

```python
from pydantic import BaseModel
from densecore.integrations import DenseCoreChatModel

class Person(BaseModel):
    name: str
    age: int
    occupation: str

chat = DenseCoreChatModel(hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct")
structured_chat = chat.with_structured_output(Person)

# Model will output JSON matching the schema
result = structured_chat.invoke("Tell me about Alice, a 30-year-old engineer")
```

### LangChain Chains

#### Simple Chain

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

llm = DenseCoreLLM(hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct")

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a haiku about {topic}:"
)
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"topic": "artificial intelligence"})
print(result)
```

#### Sequential Chain

```python
first_prompt = PromptTemplate(
    input_variables=["subject"],
    template="Suggest a creative project idea about {subject}:"
)
chain_one = first_prompt | llm | StrOutputParser()

second_prompt = PromptTemplate(
    input_variables=["idea"],
    template="Expand on this idea with implementation details:\n{idea}"
)
chain_two = second_prompt | llm | StrOutputParser()

overall_chain = chain_one | chain_two
result = overall_chain.invoke({"subject": "AI art"})
```

#### Conversation with Memory

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

chat = DenseCoreChatModel(hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)
chain = prompt | chat
store = {}

def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

conversation = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)

conversation.invoke(
    {"input": "My name is Alice"},
    config={"configurable": {"session_id": "demo"}},
)
conversation.invoke(
    {"input": "What's my name?"},
    config={"configurable": {"session_id": "demo"}},
)
```

## LangGraph Integration

### Node Creation

#### Simple Node

```python
from densecore.integrations import create_densecore_node

# Create a node
node = create_densecore_node(
    hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    node_name="assistant",
    system_prompt="You are a helpful assistant.",
    temperature=0.7,
    max_tokens=256
)

# Use in graph
workflow.add_node("assistant", node)
```

#### Multi-Agent Workflow

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Sequence
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]

# Create specialized agents
researcher = create_densecore_node(
    hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    node_name="researcher",
    system_prompt="You are a researcher. Provide factual information.",
    temperature=0.3
)

writer = create_densecore_node(
    hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    node_name="writer",
    system_prompt="You are a writer. Make information engaging.",
    temperature=0.8
)

# Build workflow
workflow = StateGraph(AgentState)
workflow.add_node("research", researcher)
workflow.add_node("write", writer)

workflow.set_entry_point("research")
workflow.add_edge("research", "write")
workflow.add_edge("write", END)

app = workflow.compile()
```

### ReAct Agents with Tool Calling

DenseCore provides full support for building ReAct-style agents that can use tools.

#### Quick Start with `create_react_agent()`

The easiest way to create a tool-using agent:

```python
from densecore.integrations import DenseCoreChatModel, create_react_agent
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# Define tools
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for '{query}': Example search results here"

# Create model and agent
llm = DenseCoreChatModel(hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct")
agent = create_react_agent(
    llm,
    tools=[calculator, search],
    system_prompt="You are a helpful assistant with access to tools."
)

# Run the agent
result = agent.invoke({
    "messages": [HumanMessage(content="What is 15 * 7?")]
})
print(result["messages"][-1].content)
```

#### Custom Agent with Tool Node

For more control, build your own agent workflow:

```python
from densecore.integrations import (
    DenseCoreChatModel,
    create_tool_node,
    should_continue,
    AgentState
)
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# Define tools
@tool
def get_weather(location: str) -> str:
    """Get weather for a location."""
    return f"Weather in {location}: Sunny, 72°F"

# Create model with tools
llm = DenseCoreChatModel(hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct")
llm_with_tools = llm.bind_tools([get_weather])

# Agent node
def agent_node(state):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": messages + [response]}

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", create_tool_node([get_weather]))

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "end": END}
)
workflow.add_edge("tools", "agent")

app = workflow.compile()

# Run
result = app.invoke({
    "messages": [HumanMessage(content="What's the weather in Paris?")]
})
```

### Tool Execution

```python
from langchain_core.tools import tool
from densecore.integrations import DenseCoreToolExecutor

# Define tools
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))

@tool
def word_count(text: str) -> str:
    """Count words in text."""
    return f"Words: {len(text.split())}"

# Create executor
tools = [calculator, word_count]
executor = DenseCoreToolExecutor(
    tools=tools,
    handle_tool_errors=True,
    max_retries=3
)

# Execute tool
result = executor.invoke(
    {"tool": "calculator", "tool_input": {"expression": "25 * 4"}}
)
print(result)  # "100"
```

### Conditional Routing

```python
def route_based_on_query(state):
    """Route to different nodes based on query content."""
    query = state["messages"][-1].content.lower()

    if "technical" in query or "code" in query:
        return "technical_expert"
    elif "creative" in query or "story" in query:
        return "creative_expert"
    else:
        return "general_assistant"

# Add conditional edges
workflow.add_conditional_edges(
    "router",
    route_based_on_query,
    {
        "technical_expert": "technical_node",
        "creative_expert": "creative_node",
        "general_assistant": "general_node"
    }
)
```

### Checkpointing

```python
from densecore.integrations import GraphCheckpoint

# Create checkpoint manager
checkpoint = GraphCheckpoint(checkpoint_dir="./.checkpoints")

# Save state
state = {
    "messages": [...],
    "current_step": "processing",
    "data": {...}
}
checkpoint.save("workflow_123", state)

# Later... resume from checkpoint
restored_state = checkpoint.load("workflow_123")
if restored_state:
    # Continue workflow from saved state
    result = app.invoke(restored_state)
```

## Production Examples

### RAG (Retrieval-Augmented Generation)

See [production_rag.py](../examples/production_rag.py) for a complete production-ready RAG system featuring:

- Document ingestion and chunking
- Vector store integration (Chroma)
- Semantic search
- Context-aware generation
- Error handling and retries
- Performance monitoring

```python
from examples.production_rag import ProductionRAGSystem, RAGConfig

# Initialize
config = RAGConfig(
    hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
    temperature=0.3,
    top_k_docs=3
)

rag = ProductionRAGSystem(config)

# Ingest documents
rag.ingest_documents(my_documents)

# Query
result = rag.query("What is the main topic?", return_sources=True)
print(result["answer"])
```

## Best Practices

### Performance Optimization

1. **Batch Requests**: Use async operations for multiple queries
```python
async def batch_generate(prompts):
    tasks = [llm.agenerate([p]) for p in prompts]
    return await asyncio.gather(*tasks)
```

2. **Streaming for Long Outputs**: Use streaming for better UX
```python
for token in llm.stream(long_prompt):
    display(token)  # Show incrementally
```

3. **Parameter Tuning**:
   - Lower `temperature` (0.1-0.3) for factual tasks
   - Higher `temperature` (0.7-0.9) for creative tasks
   - Use `top_p` instead of `top_k` for better quality

### Error Handling

```python
try:
    response = llm.invoke("Your prompt")
    print(response)
except Exception as e:
    logger.error(f"Generation failed: {e}")
    # Implement fallback logic
```

### Memory Management

```python
# Keep history bounded per session in your history store
store = {"demo": InMemoryChatMessageHistory()}
store["demo"].add_user_message("hello")
store["demo"].add_ai_message("hi")
store["demo"].messages = store["demo"].messages[-10:]
```

### Monitoring

```python
import logging

# Enable verbose logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("densecore")

# Log generation times
import time
start = time.time()
response = llm("Test")
print(f"Generation took {time.time() - start:.2f}s")
```

## Troubleshooting

### Common Issues

**1. Import Errors**

```
ImportError: LangChain integration requires langchain-core
```

**Solution**: Install optional dependencies
```bash
pip install densecore[langchain]
```

**2. Model Loading Fails**

```python
# Use explicit model path
llm = DenseCoreLLM(
    model_path="/path/to/model.gguf",
    hf_repo_id="owner/repo"  # For tokenizer
)
```

**3. Slow Generation**

- Reduce `max_tokens`
- Use quantized models (Q4, Q5)
- Increase `threads` parameter
- Enable streaming for perceived speed

**4. Quality Issues**

- Adjust `temperature` (lower = more focused)
- Use `repetition_penalty` to avoid loops
- Add stop sequences to prevent over-generation
- Improve prompts with examples (few-shot)

### Getting Help

- 📚 [Full Documentation](https://github.com/DenseCore/DenseCore)
- 💬 [GitHub Issues](https://github.com/DenseCore/DenseCore/issues)
- 📧 Email: support@densecore.dev

## Advanced Topics

### Custom Callbacks

```python
from langchain.callbacks.base import BaseCallbackHandler

class CustomCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM started with {len(prompts)} prompts")

    def on_llm_end(self, response, **kwargs):
        print(f"LLM finished: {response}")

llm = DenseCoreLLM(hf_repo_id="...", callbacks=[CustomCallback()])
```

### Integration with LangSmith

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"

# All LangChain operations now traced!
llm = DenseCoreLLM(hf_repo_id="...")
```

### Custom Prompt Templates

```python
from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert in {domain}."),
        ("human", "{question}"),
    ]
)

messages = chat_prompt.invoke(
    {"domain": "Python programming", "question": "What are decorators?"}
).to_messages()

chat = DenseCoreChatModel(hf_repo_id="...")
response = chat(messages)
```

## Examples Summary

- **[langchain_example.py](../examples/langchain_example.py)**: Complete LangChain examples
- **[langgraph_example.py](../examples/langgraph_example.py)**: LangGraph workflows and agents
- **[production_rag.py](../examples/production_rag.py)**: Production-ready RAG system

Run examples:
```bash
cd python/examples
python langchain_example.py
python langgraph_example.py
python production_rag.py
```

---

**Built with ❤️ by the DenseCore Team**
