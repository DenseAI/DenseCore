"""
Tests for LangGraph integration.

This module tests the DenseCore LangGraph utilities including:
- Node creation
- Tool execution
- State management
- Checkpointing
"""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest

# Try to import LangGraph dependencies
try:
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import tool
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# Import after checking availability
if LANGGRAPH_AVAILABLE:
    from densecore.integrations import (
        DenseCoreToolExecutor,
        GraphCheckpoint,
        create_densecore_node,
    )


pytestmark = pytest.mark.skipif(
    not LANGGRAPH_AVAILABLE, reason="LangGraph dependencies not installed"
)


def _tool_invocation(tool_name: str, tool_input: dict):
    """Create a tool invocation payload compatible across LangGraph versions."""
    return {"tool": tool_name, "tool_input": tool_input}


class TestNodeCreation:
    """Tests for create_densecore_node"""

    @patch("densecore.integrations.langgraph_tools.DenseCoreChatModel")
    def test_node_creation(self, mock_chat_class):
        """Test basic node creation"""
        node_func = create_densecore_node(
            hf_repo_id="test/model", node_name="test_node", temperature=0.7
        )

        assert callable(node_func)
        mock_chat_class.assert_called_once()

    @patch("densecore.integrations.langgraph_tools.DenseCoreChatModel")
    def test_node_execution(self, mock_chat_class):
        """Test node execution with state"""
        # Setup mock
        mock_chat = Mock()
        mock_response = AIMessage(content="Test response")
        mock_chat.invoke.return_value = mock_response
        mock_chat_class.return_value = mock_chat

        # Create node
        node_func = create_densecore_node(hf_repo_id="test/model", node_name="test")

        # Execute with state
        state = {"messages": [HumanMessage(content="Hello")]}

        result = node_func(state)

        assert "messages" in result
        assert len(result["messages"]) == 2
        assert result["messages"][-1].content == "Test response"

    @patch("densecore.integrations.langgraph_tools.DenseCoreChatModel")
    def test_node_with_system_prompt(self, mock_chat_class):
        """Test node with system prompt"""
        mock_chat = Mock()
        mock_chat.invoke.return_value = AIMessage(content="Response")
        mock_chat_class.return_value = mock_chat

        node_func = create_densecore_node(hf_repo_id="test/model", system_prompt="You are helpful")

        state = {"messages": [HumanMessage(content="Hi")]}
        result = node_func(state)

        # Node should handle system prompt internally
        assert "messages" in result


class TestToolExecutor:
    """Tests for DenseCoreToolExecutor"""

    def test_tool_executor_initialization(self):
        """Test tool executor initialization"""

        @tool
        def test_tool(input: str) -> str:
            """A test tool"""
            return f"Result: {input}"

        executor = DenseCoreToolExecutor([test_tool])

        assert "test_tool" in executor.tools
        assert executor.handle_tool_errors is True
        assert executor.max_retries == 3

    def test_tool_execution(self):
        """Test successful tool execution"""

        @tool
        def add_numbers(left: int, right: int) -> str:
            """Add two integers."""
            return str(left + right)

        executor = DenseCoreToolExecutor([add_numbers])

        invocation = _tool_invocation("add_numbers", {"left": 2, "right": 2})

        result = executor.invoke(invocation)
        assert result == "4"

    def test_tool_error_handling(self):
        """Test tool error handling"""

        @tool
        def failing_tool(input: str) -> str:
            """A tool that always fails"""
            raise ValueError("Tool error")

        executor = DenseCoreToolExecutor([failing_tool], max_retries=2)

        invocation = _tool_invocation("failing_tool", {"input": "test"})

        result = executor.invoke(invocation)

        # Should return error dict
        assert isinstance(result, dict)
        assert "error" in result

    def test_unknown_tool(self):
        """Test handling of unknown tool"""
        executor = DenseCoreToolExecutor([])

        invocation = _tool_invocation("unknown_tool", {})

        result = executor.invoke(invocation)

        assert isinstance(result, dict)
        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_async_tool_execution(self):
        """Test async tool execution"""

        @tool
        async def async_tool(input: str) -> str:
            """An async tool"""
            return f"Async: {input}"

        executor = DenseCoreToolExecutor([async_tool])

        invocation = _tool_invocation("async_tool", {"input": "test"})

        result = await executor.ainvoke(invocation)
        assert result == "Async: test"


class TestGraphCheckpoint:
    """Tests for GraphCheckpoint"""

    def test_checkpoint_initialization(self):
        """Test checkpoint manager initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = GraphCheckpoint(checkpoint_dir=tmpdir)
            assert os.path.exists(tmpdir)

    def test_save_and_load_checkpoint(self):
        """Test saving and loading checkpoints"""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = GraphCheckpoint(checkpoint_dir=tmpdir)

            # Create test state
            state = {
                "messages": [HumanMessage(content="Test")],
                "step": 5,
                "data": {"key": "value"},
            }

            # Save checkpoint
            checkpoint.save("test_checkpoint", state)

            # Load checkpoint
            loaded_state = checkpoint.load("test_checkpoint")

            assert loaded_state is not None
            assert loaded_state["step"] == 5
            assert loaded_state["data"] == {"key": "value"}

    def test_load_nonexistent_checkpoint(self):
        """Test loading a checkpoint that doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = GraphCheckpoint(checkpoint_dir=tmpdir)

            result = checkpoint.load("nonexistent")
            assert result is None

    def test_message_serialization(self):
        """Test that messages are properly serialized"""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = GraphCheckpoint(checkpoint_dir=tmpdir)

            state = {"messages": [HumanMessage(content="Hello"), AIMessage(content="Hi there")]}

            checkpoint.save("msg_test", state)
            loaded = checkpoint.load("msg_test")

            assert loaded is not None
            assert len(loaded["messages"]) == 2
            assert loaded["messages"][0]["type"] == "HumanMessage"
            assert loaded["messages"][0]["content"] == "Hello"
            assert loaded["messages"][1]["type"] == "AIMessage"
            assert loaded["messages"][1]["content"] == "Hi there"


class TestGraphIntegration:
    """Integration tests for LangGraph"""

    @patch("densecore.integrations.langgraph_tools.DenseCoreChatModel")
    def test_simple_graph_execution(self, mock_chat_class):
        """Test simple graph with DenseCore node"""
        # Setup mock
        mock_chat = Mock()
        mock_chat.invoke.return_value = AIMessage(content="Graph response")
        mock_chat_class.return_value = mock_chat

        # Create graph
        from collections.abc import Sequence
        from typing import TypedDict

        from langchain_core.messages import BaseMessage

        class SimpleState(TypedDict):
            messages: Sequence[BaseMessage]

        workflow = StateGraph(SimpleState)

        # Add DenseCore node
        node = create_densecore_node(hf_repo_id="test/model")
        workflow.add_node("llm", node)

        # Setup workflow
        workflow.set_entry_point("llm")
        workflow.add_edge("llm", END)

        # Compile
        app = workflow.compile()

        # Test execution
        result = app.invoke({"messages": [HumanMessage(content="Test")]})

        assert "messages" in result
        assert len(result["messages"]) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
