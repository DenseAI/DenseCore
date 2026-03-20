"""
LangGraph Integration Examples for DenseCore

This script demonstrates how to use DenseCore with LangGraph for building
stateful, multi-step workflows and agents.

Examples:
1. Simple state machine
2. Multi-step workflow
3. Conditional routing
4. Tool-using agent
5. Checkpointing and persistence

Requirements:
    pip install densecore[langchain]
"""

import operator
from collections.abc import Sequence
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool

# LangGraph imports
from langgraph.graph import END, StateGraph

# DenseCore imports
from densecore.integrations import (
    DenseCoreToolExecutor,
    GraphCheckpoint,
    create_densecore_node,
)


# Define state schemas
class SimpleState(TypedDict):
    """Simple state with just messages"""

    messages: Sequence[BaseMessage]


class WorkflowState(TypedDict):
    """Workflow state with additional fields"""

    messages: Sequence[BaseMessage]
    current_step: str
    iteration_count: int


def example_1_simple_graph():
    """Example 1: Simple single-node graph"""
    print("\n" + "=" * 80)
    print("Example 1: Simple State Machine")
    print("=" * 80)

    # Create graph
    workflow = StateGraph(SimpleState)

    # Add DenseCore node
    llm_node = create_densecore_node(
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        node_name="assistant",
        system_prompt="You are a helpful AI assistant.",
        temperature=0.7,
        max_tokens=150,
    )

    workflow.add_node("assistant", llm_node)

    # Set entry point and end
    workflow.set_entry_point("assistant")
    workflow.add_edge("assistant", END)

    # Compile graph
    app = workflow.compile()

    # Run the graph
    initial_state = {"messages": [HumanMessage(content="What is the capital of France?")]}

    print("\nRunning graph...")
    result = app.invoke(initial_state)

    print(f"\nUser: {result['messages'][0].content}")
    print(f"Assistant: {result['messages'][-1].content}")


def example_2_multi_step_workflow():
    """Example 2: Multi-step workflow"""
    print("\n" + "=" * 80)
    print("Example 2: Multi-Step Workflow")
    print("=" * 80)

    # Create graph with multiple steps
    workflow = StateGraph(WorkflowState)

    # Step 1: Generate initial idea
    idea_generator = create_densecore_node(
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        node_name="idea_generator",
        system_prompt="You generate creative ideas. Be brief.",
        temperature=0.9,
        max_tokens=100,
    )

    # Step 2: Refine the idea
    refiner = create_densecore_node(
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        node_name="refiner",
        system_prompt="You refine and improve ideas. Be concise.",
        temperature=0.7,
        max_tokens=150,
    )

    # Add nodes
    workflow.add_node("generate", idea_generator)
    workflow.add_node("refine", refiner)

    # Define edges
    workflow.set_entry_point("generate")
    workflow.add_edge("generate", "refine")
    workflow.add_edge("refine", END)

    # Compile
    app = workflow.compile()

    # Run
    initial_state = {
        "messages": [HumanMessage(content="Suggest a name for a café")],
        "current_step": "start",
        "iteration_count": 0,
    }

    print("\nRunning multi-step workflow...")
    result = app.invoke(initial_state)

    print(f"\nInitial request: {result['messages'][0].content}")
    print(f"Generated idea: {result['messages'][1].content}")
    print(f"Refined idea: {result['messages'][-1].content}")


def example_3_conditional_routing():
    """Example 3: Conditional routing based on state"""
    print("\n" + "=" * 80)
    print("Example 3: Conditional Routing")
    print("=" * 80)

    class RouterState(TypedDict):
        messages: Sequence[BaseMessage]
        route: str

    # Create nodes for different routes
    technical_expert = create_densecore_node(
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        node_name="technical",
        system_prompt="You are a technical expert. Answer with precision.",
        temperature=0.3,
        max_tokens=150,
    )

    creative_expert = create_densecore_node(
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        node_name="creative",
        system_prompt="You are a creative writer. Be imaginative.",
        temperature=0.9,
        max_tokens=150,
    )

    # Router function
    def route_query(state: RouterState) -> str:
        """Determine which expert to route to"""
        query = state["messages"][-1].content.lower()

        # Simple keyword-based routing
        if any(word in query for word in ["code", "program", "technical", "how to"]):
            print("  → Routing to technical expert")
            return "technical"
        else:
            print("  → Routing to creative expert")
            return "creative"

    # Build graph
    workflow = StateGraph(RouterState)
    workflow.add_node("technical", technical_expert)
    workflow.add_node("creative", creative_expert)

    # Conditional edges
    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        route_query,
        {
            "technical": "technical",
            "creative": "creative",
        },
    )

    # Router node (passthrough)
    def router_node(state: RouterState) -> RouterState:
        return state

    workflow.add_node("router", router_node)
    workflow.add_edge("technical", END)
    workflow.add_edge("creative", END)

    # Compile
    app = workflow.compile()

    # Test with different queries
    queries = [
        "How do I write a Python function?",
        "Write a poem about the ocean",
    ]

    for query in queries:
        print(f"\n\nQuery: {query}")
        result = app.invoke(
            {
                "messages": [HumanMessage(content=query)],
                "route": "",
            }
        )
        print(f"Response: {result['messages'][-1].content}")


def example_4_tool_execution():
    """Example 4: Agent with tool execution"""
    print("\n" + "=" * 80)
    print("Example 4: Tool Execution")
    print("=" * 80)

    # Define tools
    @tool
    def calculator(expression: str) -> str:
        """Evaluate a mathematical expression"""
        try:
            result = eval(expression)
            return f"The result is: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

    @tool
    def word_count(text: str) -> str:
        """Count words in a text"""
        count = len(text.split())
        return f"Word count: {count}"

    tools = [calculator, word_count]

    # Create tool executor
    tool_executor = DenseCoreToolExecutor(tools)

    # Test tool invocation
    print("\nTesting calculator tool:")
    result1 = tool_executor.invoke({"tool": "calculator", "tool_input": {"expression": "15 * 7 + 3"}})
    print(f"  {result1}")

    print("\nTesting word_count tool:")
    result2 = tool_executor.invoke(
        {"tool": "word_count", "tool_input": {"text": "Hello world from DenseCore"}}
    )
    print(f"  {result2}")


def example_5_checkpointing():
    """Example 5: Checkpointing for long-running workflows"""
    print("\n" + "=" * 80)
    print("Example 5: Checkpointing")
    print("=" * 80)

    # Create checkpoint manager
    checkpoint = GraphCheckpoint(checkpoint_dir="./.checkpoints")

    # Simulate a multi-step workflow
    workflow_state = {
        "messages": [
            HumanMessage(content="Start workflow"),
            AIMessage(content="Step 1 complete"),
        ],
        "current_step": "step_1",
        "iteration_count": 1,
    }

    # Save checkpoint
    checkpoint_id = "example_workflow_001"
    checkpoint.save(checkpoint_id, workflow_state)
    print(f"\n✓ Saved checkpoint '{checkpoint_id}'")

    # Simulate workflow interruption...
    print("  (Simulating workflow interruption...)")

    # Resume from checkpoint
    restored_state = checkpoint.load(checkpoint_id)
    print(f"\n✓ Restored checkpoint '{checkpoint_id}'")
    print(f"  Current step: {restored_state['current_step']}")
    print(f"  Messages: {len(restored_state['messages'])}")
    print(f"  Iteration: {restored_state['iteration_count']}")


def example_6_advanced_agent():
    """Example 6: Advanced multi-agent workflow"""
    print("\n" + "=" * 80)
    print("Example 6: Advanced Multi-Agent Workflow")
    print("=" * 80)

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        next_agent: str

    # Create specialized agents
    researcher = create_densecore_node(
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        node_name="researcher",
        system_prompt="You are a researcher. Provide factual information.",
        temperature=0.3,
        max_tokens=150,
    )

    writer = create_densecore_node(
        hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        node_name="writer",
        system_prompt="You are a writer. Make information engaging.",
        temperature=0.7,
        max_tokens=150,
    )

    # Build collaborative workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("researcher", researcher)
    workflow.add_node("writer", writer)

    # Define workflow
    workflow.set_entry_point("researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", END)

    # Compile
    app = workflow.compile()

    # Run collaborative task
    initial_state = {
        "messages": [HumanMessage(content="Tell me about quantum computing")],
        "next_agent": "researcher",
    }

    print("\nRunning collaborative workflow...")
    print("  1. Researcher gathers facts")
    print("  2. Writer makes it engaging")

    result = app.invoke(initial_state)

    print(f"\n\nFinal output ({len(result['messages'])} messages):")
    print(f"Last message: {result['messages'][-1].content}")


def main():
    """Run all examples"""
    print("\n🔄 DenseCore + LangGraph Examples")
    print("=" * 80)

    example_1_simple_graph()
    example_2_multi_step_workflow()
    example_3_conditional_routing()
    example_4_tool_execution()
    example_5_checkpointing()
    example_6_advanced_agent()

    print("\n" + "=" * 80)
    print("✅ All LangGraph examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
