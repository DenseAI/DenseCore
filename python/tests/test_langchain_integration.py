"""
Tests for LangChain integration.

This module tests the DenseCore LangChain wrappers including:
- DenseCoreLLM basic functionality
- DenseCoreChatModel functionality
- Streaming support
- Async operations
- LangChain chain compatibility
"""

import asyncio
from unittest.mock import Mock, patch

import pytest

# Try to import LangChain dependencies
try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.prompts import PromptTemplate

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Import after checking availability
if LANGCHAIN_AVAILABLE:
    from densecore.engine import GenerationOutput
    from densecore.integrations import DenseCoreChatModel, DenseCoreLLM


pytestmark = pytest.mark.skipif(
    not LANGCHAIN_AVAILABLE, reason="LangChain dependencies not installed"
)


class TestDenseCoreLLM:
    """Tests for DenseCoreLLM wrapper"""

    def test_initialization_with_repo_id(self):
        """Test LLM initialization with HF repo ID"""
        llm = DenseCoreLLM(hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct", temperature=0.7, max_tokens=100)

        assert llm.hf_repo_id == "Qwen/Qwen2.5-0.5B-Instruct"
        assert llm.temperature == 0.7
        assert llm.max_tokens == 100
        assert llm._llm_type == "densecore"

    def test_initialization_with_model_path(self):
        """Test LLM initialization with model path"""
        llm = DenseCoreLLM(model_path="/path/to/model.gguf", temperature=0.5)

        assert llm.model_path == "/path/to/model.gguf"
        assert llm.temperature == 0.5

    def test_initialization_fails_without_model(self):
        """Test that initialization fails without model path or repo ID"""
        with pytest.raises(ValueError):
            DenseCoreLLM(temperature=0.7)

    @patch("densecore.integrations.langchain.DenseCore")
    def test_call_method(self, mock_engine_class):
        """Test basic call method"""
        # Setup mock
        mock_engine = Mock()
        mock_engine.generate.return_value = GenerationOutput(
            text="Test response", tokens=10, finish_reason="stop", generation_time=0.5
        )
        mock_engine_class.return_value = mock_engine

        # Create LLM and call
        llm = DenseCoreLLM(hf_repo_id="test/model", temperature=0.7)
        response = llm("test prompt")

        assert response == "Test response"
        mock_engine.generate.assert_called_once()

    @patch("densecore.integrations.langchain.DenseCore")
    def test_stream_method(self, mock_engine_class):
        """Test streaming method"""
        # Setup mock
        mock_engine = Mock()
        mock_engine.stream.return_value = iter(["Hello", " ", "world"])
        mock_engine_class.return_value = mock_engine

        # Create LLM and stream
        llm = DenseCoreLLM(hf_repo_id="test/model")
        tokens = list(llm.stream("test prompt"))

        assert tokens == ["Hello", " ", "world"]
        mock_engine.stream.assert_called_once()

    @patch("densecore.integrations.langchain.DenseCore")
    def test_async_call(self, mock_engine_class):
        """Test async call method"""
        # Setup mock
        mock_engine = Mock()

        async def mock_generate_async(*args, **kwargs):
            return GenerationOutput(
                text="Async response", tokens=5, finish_reason="stop", generation_time=0.3
            )

        mock_engine.generate_async = mock_generate_async
        mock_engine_class.return_value = mock_engine

        # Create LLM and call async
        llm = DenseCoreLLM(hf_repo_id="test/model")

        async def run_test():
            response = await llm.agenerate(["test prompt"])
            return response.generations[0][0].text

        result = asyncio.run(run_test())
        assert result == "Async response"

    @patch("densecore.integrations.langchain.DenseCore")
    def test_chain_compatibility(self, mock_engine_class):
        """Test compatibility with LangChain runnables"""
        # Setup mock
        mock_engine = Mock()
        mock_engine.generate.return_value = GenerationOutput(
            text="Chain output", tokens=8, finish_reason="stop"
        )
        mock_engine_class.return_value = mock_engine

        # Create LLM
        llm = DenseCoreLLM(hf_repo_id="test/model", temperature=0.7)

        # Create simple runnable chain
        prompt = PromptTemplate(input_variables=["topic"], template="Write about {topic}")
        chain = prompt | llm

        # Run chain
        result = chain.invoke({"topic": "AI"})
        assert result == "Chain output"


class TestDenseCoreChatModel:
    """Tests for DenseCoreChatModel wrapper"""

    def test_initialization(self):
        """Test chat model initialization"""
        chat = DenseCoreChatModel(
            hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct", temperature=0.8, max_tokens=150
        )

        assert chat.hf_repo_id == "Qwen/Qwen2.5-0.5B-Instruct"
        assert chat.temperature == 0.8
        assert chat.max_tokens == 150
        assert chat._llm_type == "densecore-chat"

    def test_message_formatting(self):
        """Test message format conversion"""
        chat = DenseCoreChatModel(hf_repo_id="test/model")

        messages = [
            SystemMessage(content="You are helpful"),
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there"),
        ]

        formatted = chat._format_messages(messages)

        assert "System: You are helpful" in formatted
        assert "User: Hello" in formatted
        assert "Assistant: Hi there" in formatted
        assert "Assistant: " in formatted  # Prompt for next response

    @patch("densecore.integrations.langchain.DenseCore")
    def test_generate_method(self, mock_engine_class):
        """Test chat generation"""
        # Setup mock
        mock_engine = Mock()
        mock_engine.generate.return_value = GenerationOutput(
            text="Chat response", tokens=10, finish_reason="stop"
        )
        mock_engine_class.return_value = mock_engine

        # Create chat model
        chat = DenseCoreChatModel(hf_repo_id="test/model")

        messages = [HumanMessage(content="Hello")]
        result = chat(messages)

        assert isinstance(result, AIMessage)
        assert result.content == "Chat response"

    @patch("densecore.integrations.langchain.DenseCore")
    def test_stream_chat(self, mock_engine_class):
        """Test chat streaming"""
        # Setup mock
        mock_engine = Mock()
        mock_engine.stream.return_value = iter(["Hello", " ", "there"])
        mock_engine_class.return_value = mock_engine

        # Create chat model
        chat = DenseCoreChatModel(hf_repo_id="test/model")

        messages = [HumanMessage(content="Hi")]
        chunks = list(chat.stream(messages))

        assert len(chunks) >= 3
        assert [chunk.content for chunk in chunks[:3]] == ["Hello", " ", "there"]
        assert all(hasattr(chunk, "content") for chunk in chunks)


class TestParameterPropagation:
    """Test that parameters are correctly propagated"""

    @patch("densecore.integrations.langchain.DenseCore")
    def test_generation_config_creation(self, mock_engine_class):
        """Test that generation config is correctly created"""
        mock_engine = Mock()
        mock_engine.generate.return_value = GenerationOutput(
            text="test", tokens=5, finish_reason="stop"
        )
        mock_engine_class.return_value = mock_engine

        llm = DenseCoreLLM(
            hf_repo_id="test/model", temperature=0.5, top_p=0.9, top_k=50, max_tokens=200
        )

        # Call with different parameters
        llm("test", temperature=0.7, max_tokens=100)

        # Verify engine was called
        assert mock_engine.generate.called
        call_kwargs = mock_engine.generate.call_args[1]

        # Check config was passed
        assert "config" in call_kwargs
        config = call_kwargs["config"]

        # Verify override worked
        assert config.temperature == 0.7
        assert config.max_tokens == 100


@pytest.mark.asyncio
class TestAsyncOperations:
    """Test async functionality"""

    @patch("densecore.integrations.langchain.DenseCore")
    async def test_async_generation(self, mock_engine_class):
        """Test async generation"""
        mock_engine = Mock()

        async def mock_gen(*args, **kwargs):
            return GenerationOutput(text="Async test", tokens=5, finish_reason="stop")

        mock_engine.generate_async = mock_gen
        mock_engine_class.return_value = mock_engine

        llm = DenseCoreLLM(hf_repo_id="test/model")
        result = await llm.agenerate(["test prompt"])

        assert result.generations[0][0].text == "Async test"

    @patch("densecore.integrations.langchain.DenseCore")
    async def test_async_streaming(self, mock_engine_class):
        """Test async streaming"""
        mock_engine = Mock()

        async def mock_stream(*args, **kwargs):
            for token in ["a", "b", "c"]:
                yield token

        mock_engine.stream_async = mock_stream
        mock_engine_class.return_value = mock_engine

        llm = DenseCoreLLM(hf_repo_id="test/model")
        tokens = []

        async for token in llm.astream("test"):
            tokens.append(token)

        assert tokens == ["a", "b", "c"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
