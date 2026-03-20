"""
LangChain Integration Examples for DenseCore

This script demonstrates how to use DenseCore with LangChain for various tasks:
1. Basic LLM usage
2. Chat model with conversation
3. Building chains
4. Streaming generation
5. Async operations

Requirements:
    pip install densecore[langchain]
"""

import asyncio

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory

# DenseCore imports
from densecore.integrations import DenseCoreChatModel, DenseCoreLLM


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def example_1_basic_llm():
    """Example 1: Basic LLM usage"""
    _print_header("Example 1: Basic LLM Usage")

    # Initialize DenseCore LLM
    llm = DenseCoreLLM(
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",  # Or use model_path
        temperature=0.7,
        max_tokens=128,
    )

    # Simple generation
    prompt = "Explain quantum computing in one sentence."
    response = llm(prompt)
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")


def example_2_chat_model():
    """Example 2: Chat model with conversation"""
    _print_header("Example 2: Chat Model with Conversation")

    # Initialize chat model
    chat = DenseCoreChatModel(
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        temperature=0.7,
        max_tokens=200,
    )

    # Create conversation
    messages = [
        SystemMessage(content="You are a helpful AI assistant specialized in Python programming."),
        HumanMessage(content="What is a Python decorator?"),
    ]

    response = chat(messages)
    print(f"\nUser: {messages[1].content}")
    print(f"Assistant: {response.content}")

    # Continue conversation
    messages.append(response)
    messages.append(HumanMessage(content="Can you show me a simple example?"))

    response2 = chat(messages)
    print(f"\nUser: {messages[-1].content}")
    print(f"Assistant: {response2.content}")


def example_3_simple_chain():
    """Example 3: Building a simple runnable chain"""
    _print_header("Example 3: Runnable Chain")

    llm = DenseCoreLLM(
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        temperature=0.8,
        max_tokens=150,
    )

    # Create prompt template
    prompt = PromptTemplate(input_variables=["topic"], template="Write a haiku about {topic}:")

    # Build an LCEL runnable
    chain = prompt | llm | StrOutputParser()

    # Run chain
    topic = "artificial intelligence"
    result = chain.invoke({"topic": topic})
    print(f"\nTopic: {topic}")
    print(f"Haiku:\n{result}")


def example_4_sequential_chain():
    """Example 4: Sequential runnable composition"""
    _print_header("Example 4: Sequential Runnable")

    llm = DenseCoreLLM(
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        temperature=0.7,
        max_tokens=100,
    )

    # First chain: Generate a topic
    first_prompt = PromptTemplate(
        input_variables=["subject"],
        template="Suggest one specific topic about {subject}. Just the topic name:",
    )
    chain_one = first_prompt | llm | StrOutputParser()

    # Second chain: Write about the topic
    second_prompt = PromptTemplate(
        input_variables=["topic"], template="Write one interesting fact about {topic}:"
    )
    chain_two = second_prompt | llm | StrOutputParser()
    overall_chain = chain_one | chain_two

    # Run
    result = overall_chain.invoke({"subject": "machine learning"})
    print(f"\nFinal result: {result}")


def example_5_streaming():
    """Example 5: Streaming generation"""
    _print_header("Example 5: Streaming Generation")

    llm = DenseCoreLLM(
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        temperature=0.7,
        max_tokens=200,
    )

    prompt = "Write a short story about a robot:"

    print(f"\nPrompt: {prompt}")
    print("Response: ", end="", flush=True)

    for chunk in llm.stream(prompt):
        print(chunk, end="", flush=True)

    print("\n")


async def example_6_async():
    """Example 6: Async operations"""
    _print_header("Example 6: Async Operations")

    llm = DenseCoreLLM(
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        temperature=0.7,
        max_tokens=100,
    )

    # Multiple async requests
    prompts = [
        "What is Python?",
        "What is JavaScript?",
        "What is Rust?",
    ]

    print("\nGenerating responses asynchronously for 3 prompts...")

    # Create tasks
    tasks = [llm.agenerate([prompt]) for prompt in prompts]

    # Wait for all
    results = await asyncio.gather(*tasks)

    # Print results
    for prompt, result in zip(prompts, results):
        print(f"\nQ: {prompt}")
        print(f"A: {result.generations[0][0].text[:100]}...")


async def example_7_async_streaming():
    """Example 7: Async streaming"""
    _print_header("Example 7: Async Streaming")

    chat = DenseCoreChatModel(
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        temperature=0.7,
        max_tokens=150,
    )

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Explain async/await in Python briefly."),
    ]

    print("\nStreaming response:")
    print("Assistant: ", end="", flush=True)

    async for chunk in chat.astream(messages):
        print(chunk.content, end="", flush=True)

    print("\n")


def example_8_memory():
    """Example 8: Chat history with RunnableWithMessageHistory"""
    _print_header("Example 8: Message History")

    chat = DenseCoreChatModel(
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        temperature=0.7,
        max_tokens=100,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | chat
    store: dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    print("\nConversation 1:")
    response1 = with_history.invoke(
        {"input": "My name is Alice."},
        config={"configurable": {"session_id": "demo-session"}},
    )
    print(f"Response: {response1.content}")

    print("\nConversation 2:")
    response2 = with_history.invoke(
        {"input": "What is my name?"},
        config={"configurable": {"session_id": "demo-session"}},
    )
    print(f"Response: {response2.content}")


def main():
    """Run all examples"""
    _print_header("DenseCore + LangChain Examples")

    # Run synchronous examples
    example_1_basic_llm()
    example_2_chat_model()
    example_3_simple_chain()
    example_4_sequential_chain()
    example_5_streaming()
    example_8_memory()

    # Run async examples
    _print_header("Running Async Examples...")
    asyncio.run(example_6_async())
    asyncio.run(example_7_async_streaming())

    _print_header("All examples completed")


if __name__ == "__main__":
    main()
