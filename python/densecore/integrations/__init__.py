"""
DenseCore integrations with popular LLM frameworks.

This module provides integrations for:
- LangChain: LLM and ChatModel wrappers with tool calling support
- LangGraph: Node creation, tool execution, and ReAct agents
"""

from typing import TYPE_CHECKING

# Lazy imports to avoid requiring optional dependencies
if TYPE_CHECKING:
    from .langchain import DenseCoreChatModel, DenseCoreLLM
    from .langgraph_tools import (
        AgentState,
        DenseCoreToolExecutor,
        GraphCheckpoint,
        create_densecore_node,
        create_react_agent,
        create_tool_node,
        should_continue,
    )

__all__ = [
    # LangChain
    "DenseCoreLLM",
    "DenseCoreChatModel",
    # LangGraph
    "create_densecore_node",
    "DenseCoreToolExecutor",
    "create_react_agent",
    "create_tool_node",
    "should_continue",
    "AgentState",
    "GraphCheckpoint",
]


def __getattr__(name: str):
    """Lazy loading of integration modules."""
    if name in ("DenseCoreLLM", "DenseCoreChatModel"):
        try:
            from .langchain import DenseCoreChatModel, DenseCoreLLM

            return DenseCoreLLM if name == "DenseCoreLLM" else DenseCoreChatModel
        except ImportError as e:
            raise ImportError(
                "LangChain integration requires optional dependencies. "
                "Install with: pip install densecore[langchain]"
            ) from e

    if name in (
        "DenseCoreToolExecutor",
        "create_densecore_node",
        "create_react_agent",
        "create_tool_node",
        "should_continue",
        "AgentState",
        "GraphCheckpoint",
    ):
        try:
            from .langgraph_tools import (
                AgentState,
                DenseCoreToolExecutor,
                GraphCheckpoint,
                create_densecore_node,
                create_react_agent,
                create_tool_node,
                should_continue,
            )

            mapping = {
                "DenseCoreToolExecutor": DenseCoreToolExecutor,
                "create_densecore_node": create_densecore_node,
                "create_react_agent": create_react_agent,
                "create_tool_node": create_tool_node,
                "should_continue": should_continue,
                "AgentState": AgentState,
                "GraphCheckpoint": GraphCheckpoint,
            }
            return mapping[name]
        except ImportError as e:
            raise ImportError(
                "LangGraph integration requires optional dependencies. "
                "Install with: pip install densecore[langchain]"
            ) from e

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
