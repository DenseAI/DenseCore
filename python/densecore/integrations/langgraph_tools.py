"""
LangGraph integration for DenseCore.

This module provides utilities for building LangGraph workflows with DenseCore,
including:
- Node creation for LLM calls
- Tool execution with retry logic
- ReAct-style agent creation (create_react_agent)
- ToolNode for processing tool calls
- State management and checkpointing
"""

import logging
from inspect import isawaitable
from collections.abc import Sequence
from typing import (
    Any,
    Callable,
    Optional,
    TypedDict,
    TypeVar,
)

from langchain_core.messages import SystemMessage

try:
    from langgraph.graph import END, StateGraph
except ImportError as e:
    raise ImportError(
        "LangGraph integration requires langgraph. Install with: pip install densecore[langchain]"
    ) from e

try:
    # langgraph>=1.0
    from langgraph.graph.state import CompiledStateGraph as CompiledGraph
except ImportError:
    try:
        # older langgraph versions
        from langgraph.graph.graph import CompiledGraph  # type: ignore[no-redef]
    except ImportError:
        # Runtime type only; graph compilation still works without this symbol.
        CompiledGraph = Any  # type: ignore[misc,assignment]

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool

from .langchain import DenseCoreChatModel

logger = logging.getLogger(__name__)

StateT = TypeVar("StateT", bound=dict[str, Any])


# Type for agent state
class AgentState(TypedDict, total=False):
    """Standard state for ReAct agents."""

    messages: Sequence[BaseMessage]
    next: str


class DenseCoreToolExecutor:
    """
    Lightweight tool executor compatible with LangGraph tool invocation patterns.

    Accepts either LangGraph ToolInvocation objects or dict-like payloads with:
      - tool: tool name
      - tool_input: tool args
    """

    def __init__(
        self,
        tools: list[BaseTool],
        handle_tool_errors: bool = True,
        max_retries: int = 3,
    ) -> None:
        self.tools: dict[str, BaseTool] = {tool.name: tool for tool in tools}
        self.handle_tool_errors = handle_tool_errors
        self.max_retries = max(0, max_retries)

    def _parse_invocation(self, invocation: Any) -> tuple[Optional[str], Any]:
        if isinstance(invocation, dict):
            return invocation.get("tool"), invocation.get("tool_input", {})
        return getattr(invocation, "tool", None), getattr(invocation, "tool_input", {})

    def _error(self, message: str) -> dict[str, Any]:
        return {"error": message}

    def invoke(self, invocation: Any) -> Any:
        tool_name, tool_input = self._parse_invocation(invocation)
        if not tool_name:
            return self._error("Missing tool name in invocation")

        tool = self.tools.get(tool_name)
        if tool is None:
            return self._error(f"Tool '{tool_name}' not found")

        last_error: Optional[Exception] = None
        for _ in range(self.max_retries + 1):
            try:
                return tool.invoke(tool_input)
            except Exception as e:  # pragma: no cover - exercised in integration tests
                last_error = e
                continue

        if self.handle_tool_errors:
            return self._error(str(last_error) if last_error else f"Tool '{tool_name}' failed")

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Tool '{tool_name}' failed with unknown error")

    async def ainvoke(self, invocation: Any) -> Any:
        tool_name, tool_input = self._parse_invocation(invocation)
        if not tool_name:
            return self._error("Missing tool name in invocation")

        tool = self.tools.get(tool_name)
        if tool is None:
            return self._error(f"Tool '{tool_name}' not found")

        last_error: Optional[Exception] = None
        for _ in range(self.max_retries + 1):
            try:
                if hasattr(tool, "ainvoke"):
                    return await tool.ainvoke(tool_input)

                result = tool.invoke(tool_input)
                if isawaitable(result):
                    return await result
                return result
            except Exception as e:  # pragma: no cover - exercised in integration tests
                last_error = e
                continue

        if self.handle_tool_errors:
            return self._error(str(last_error) if last_error else f"Tool '{tool_name}' failed")

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Tool '{tool_name}' failed with unknown error")


def create_densecore_node(
    model_path: Optional[str] = None,
    hf_repo_id: Optional[str] = None,
    node_name: str = "llm",
    system_prompt: Optional[str] = None,
    **generation_kwargs: Any,
) -> Callable[[StateT], StateT]:
    """
    Create a LangGraph node that uses DenseCore for text generation.

    This factory function creates a stateful node that can be added to a LangGraph
    workflow. The node reads from the state's "messages" key and appends the
    generated response.

    Args:
        model_path: Path to GGUF model file
        hf_repo_id: HuggingFace repo ID (alternative to model_path)
        node_name: Name for the node (for debugging/logging)
        system_prompt: Optional system prompt to prepend
        **generation_kwargs: Additional generation parameters (temperature, max_tokens, etc.)

    Returns:
        A callable that can be used as a LangGraph node

    Example:
        >>> from langgraph.graph import StateGraph
        >>> from densecore.integrations import create_densecore_node
        >>>
        >>> # Define state schema
        >>> class AgentState(TypedDict):
        ...     messages: List[BaseMessage]
        ...
        >>> workflow = StateGraph(AgentState)
        >>>
        >>> # Add DenseCore node
        >>> llm_node = create_densecore_node(
        ...     hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        ...     temperature=0.7,
        ...     max_tokens=256
        ... )
        >>> workflow.add_node("llm", llm_node)
        >>>
        >>> # Build and compile graph
        >>> workflow.set_entry_point("llm")
        >>> workflow.add_edge("llm", END)
        >>> app = workflow.compile()
    """
    # Initialize the chat model
    chat_model = DenseCoreChatModel(
        model_path=model_path,
        hf_repo_id=hf_repo_id,
        **generation_kwargs,
    )

    def node_function(state: StateT) -> StateT:
        """Node function that processes messages."""
        messages = list(state.get("messages", []))

        # Optionally prepend system prompt
        if system_prompt:
            # Check if first message is system prompt
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=system_prompt)] + messages

        try:
            # Generate response
            logger.debug(f"Node '{node_name}' generating response for {len(messages)} messages")
            response = chat_model.invoke(messages)

            # Update state with new message
            # logic: append the new response to original state messages
            # Note: we used local 'messages' (with prepended system) for generation
            # but we usually only append the NEW response to the graph state history
            # unless we want to persist the system prompt modification.
            # LangGraph usually expects additive updates.
            base_messages = list(state.get("messages", []))
            return {**state, "messages": base_messages + [response]}

        except Exception as e:
            logger.error(f"Error in node '{node_name}': {e}")
            # Add error message to state
            error_msg = AIMessage(content=f"Error: {str(e)}")
            base_messages = list(state.get("messages", []))
            return {**state, "messages": base_messages + [error_msg]}

    return node_function


def create_agent_node(
    llm: DenseCoreChatModel,
    tools: list[BaseTool],
    node_name: str = "agent",
) -> Callable[[StateT], StateT]:
    """
    Create an agent node that can use tools.
    """
    # Simply use the model with tools bound
    # DenseCoreChatModel handles tool call parsing automatically if bind_tools was used.
    # However, create_react_agent binds tools *before* passing to this,
    # so we assume `llm` already has tools bound OR we bind them here.

    # If LLM doesn't have tools bound, bind them.
    if not getattr(llm, "_bound_tools", None):
        llm = llm.bind_tools(tools)

    def node_function(state: StateT) -> StateT:
        """Agent node function."""
        messages = list(state.get("messages", []))

        try:
            logger.debug(f"Agent node '{node_name}' processing {len(messages)} messages")
            response = llm.invoke(messages)

            # Response will be AIMessage, possibly with tool_calls set
            return {**state, "messages": messages + [response]}

        except Exception as e:
            logger.error(f"Error in agent node '{node_name}': {e}")
            error_msg = AIMessage(content=f"Error: {str(e)}")
            return {**state, "messages": messages + [error_msg]}

    return node_function


class GraphCheckpoint:
    """
    Simple checkpointing utility for LangGraph workflows.

    Provides basic save/load functionality for graph states to enable
    resumable workflows.

    Example:
        >>> from densecore.integrations import GraphCheckpoint
        >>>
        >>> checkpoint = GraphCheckpoint()
        >>>
        >>> # Save state
        >>> state = {"messages": [...], "step": 5}
        >>> checkpoint.save("my_workflow", state)
        >>>
        >>> # Load state
        >>> restored_state = checkpoint.load("my_workflow")
    """

    def __init__(self, checkpoint_dir: str = "./.checkpoints"):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        import os

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, checkpoint_id: str, state: dict[str, Any]) -> None:
        """
        Save a checkpoint.

        Args:
            checkpoint_id: Unique identifier for the checkpoint
            state: State dictionary to save
        """
        import json
        import os

        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")

        # Convert state to JSON-serializable format
        serializable_state = self._make_serializable(state)

        with open(checkpoint_path, "w") as f:
            json.dump(serializable_state, f, indent=2)

        logger.info(f"Saved checkpoint '{checkpoint_id}' to {checkpoint_path}")

    def load(self, checkpoint_id: str) -> Optional[dict[str, Any]]:
        """
        Load a checkpoint.

        Args:
            checkpoint_id: Unique identifier for the checkpoint

        Returns:
            Loaded state dictionary, or None if not found
        """
        import json
        import os

        checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.json")

        if not os.path.exists(checkpoint_path):
            logger.warning("Checkpoint '%s' not found", checkpoint_id)
            return None

        with open(checkpoint_path, encoding="utf-8") as f:
            state = json.load(f)

        logger.info("Loaded checkpoint '%s' from %s", checkpoint_id, checkpoint_path)
        return state

    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, BaseMessage):
            # Convert LangChain messages to dicts
            return {
                "type": obj.__class__.__name__,
                "content": obj.content,
            }
        else:
            # Return as-is for primitive types
            return obj


def create_tool_node(
    tools: list[BaseTool],
    handle_errors: bool = True,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """
    Create a tool execution node for LangGraph.

    This node processes tool calls from the previous agent response and
    returns ToolMessages with the results.

    Args:
        tools: List of available tools
        handle_errors: Whether to catch and return tool errors as messages

    Returns:
        A callable node function for LangGraph

    Example:
        >>> from langchain_core.tools import tool
        >>> from densecore.integrations import create_tool_node
        >>>
        >>> @tool
        ... def add_numbers(left: int, right: int) -> str:
        ...     '''Add two numbers.'''
        ...     return str(left + right)
        >>>
        >>> tool_node = create_tool_node([add_numbers])
        >>> workflow.add_node("tools", tool_node)
    """
    tool_map = {tool.name: tool for tool in tools}

    def tool_node(state: dict[str, Any]) -> dict[str, Any]:
        """Execute tools based on the last AI message's tool calls."""
        messages = state.get("messages", [])
        if not messages:
            return state

        last_message = messages[-1]

        # Check if last message has tool calls
        tool_calls = getattr(last_message, "tool_calls", None)
        if not tool_calls:
            return state

        # Execute each tool call
        tool_messages = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id", f"call_{len(tool_messages)}")

            if tool_name not in tool_map:
                error_msg = f"Tool '{tool_name}' not found"
                if handle_errors:
                    tool_messages.append(
                        ToolMessage(content=error_msg, tool_call_id=tool_id, name=tool_name)
                    )
                continue

            tool = tool_map[tool_name]
            try:
                result = tool.invoke(tool_args)
                tool_messages.append(
                    ToolMessage(content=str(result), tool_call_id=tool_id, name=tool_name)
                )
            except Exception as e:
                if handle_errors:
                    tool_messages.append(
                        ToolMessage(content=f"Error: {e}", tool_call_id=tool_id, name=tool_name)
                    )
                else:
                    raise

        # Update state with tool messages while preserving other state keys
        return {**state, "messages": messages + tool_messages}

    return tool_node


def should_continue(state: dict[str, Any]) -> str:
    """
    Routing function for ReAct agent.

    Checks if the last message has tool calls. If so, route to tools node.
    Otherwise, end the conversation.

    Args:
        state: Current graph state with messages

    Returns:
        "tools" if there are tool calls, "end" otherwise

    Example:
        >>> workflow.add_conditional_edges(
        ...     "agent",
        ...     should_continue,
        ...     {"tools": "tools", "end": END}
        ... )
    """
    messages = state.get("messages", [])
    if not messages:
        return "end"

    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None)

    if tool_calls:
        return "tools"
    return "end"


def create_react_agent(
    model: DenseCoreChatModel,
    tools: list[BaseTool],
    *,
    system_prompt: Optional[str] = None,
    max_iterations: int = 10,
) -> CompiledGraph:
    """
    Create a ReAct-style agent using DenseCore and LangGraph.

    This function creates a complete agent graph with:
    - An agent node that uses the LLM with bound tools
    - A tool execution node
    - Conditional routing based on tool calls

    Args:
        model: DenseCoreChatModel instance
        tools: List of tools available to the agent
        system_prompt: Optional system prompt for the agent
        max_iterations: Maximum number of agent iterations (safety limit)

    Returns:
        Compiled LangGraph that can be invoked

    Example:
        >>> from densecore.integrations import DenseCoreChatModel, create_react_agent
        >>> from langchain_core.tools import tool
        >>> from langchain_core.messages import HumanMessage
        >>>
        >>> @tool
        ... def add_numbers(left: int, right: int) -> str:
        ...     '''Add two numbers.'''
        ...     return str(left + right)
        >>>
        >>> llm = DenseCoreChatModel(hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct")
        >>> agent = create_react_agent(llm, [add_numbers])
        >>>
        >>> result = agent.invoke({
        ...     "messages": [HumanMessage(content="Use add_numbers for 25 and 4.")]
        ... })
        >>> print(result["messages"][-1].content)
    """
    # Bind tools to the model
    model_with_tools = model.bind_tools(tools)

    # Create agent node
    def agent_node(state: dict[str, Any]) -> dict[str, Any]:
        """Agent node that calls the LLM with tools."""
        messages = list(state.get("messages", []))
        iteration = int(state.get("_densecore_iteration", 0))

        # Add system prompt if provided and not already present
        if system_prompt:
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(content=system_prompt)] + messages

        # Call the model
        response = model_with_tools.invoke(messages)

        # Return the updated list of messages to ensure history is preserved
        return {**state, "messages": messages + [response], "_densecore_iteration": iteration + 1}

    # Create tool node
    tool_node = create_tool_node(tools)

    # Build the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional edge from agent
    def should_continue_with_limit(state: dict[str, Any]) -> str:
        if int(state.get("_densecore_iteration", 0)) >= max_iterations:
            return "end"
        return should_continue(state)

    workflow.add_conditional_edges(
        "agent",
        should_continue_with_limit,
        {"tools": "tools", "end": END},
    )

    # Add edge from tools back to agent
    workflow.add_edge("tools", "agent")

    # Compile and return
    return workflow.compile()
