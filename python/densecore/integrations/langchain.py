"""
LangChain integration for DenseCore.

This module provides LangChain-compatible wrappers for DenseCore,
enabling seamless integration with LangChain chains, agents, and workflows.

Features:
- DenseCoreLLM: Standard LLM wrapper for text completion
- DenseCoreChatModel: Chat model with tool calling support
- bind_tools(): Bind tools to chat model for agent workflows
- with_structured_output(): Force structured JSON output
"""

import json
import logging
import re
from collections.abc import AsyncIterator, Iterator
from typing import Any, Optional

try:
    from langchain_core.callbacks import (
        AsyncCallbackManagerForLLMRun,
        CallbackManagerForLLMRun,
    )
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import LLM, BaseChatModel
    from langchain_core.messages import (
        AIMessage,
        AIMessageChunk,
        BaseMessage,
        ChatMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )
    from langchain_core.outputs import (
        ChatGeneration,
        ChatGenerationChunk,
        ChatResult,
        Generation,
        GenerationChunk,
        LLMResult,
    )
    from langchain_core.runnables import Runnable, RunnableConfig
    from langchain_core.tools import BaseTool
    from langchain_core.utils.function_calling import convert_to_openai_tool
except ImportError as e:
    raise ImportError(
        "LangChain integration requires langchain-core. "
        "Install with: pip install densecore[langchain]"
    ) from e

try:
    from langchain_core.pydantic_v1 import Field, root_validator
except ImportError:
    from pydantic import Field, root_validator

try:
    from pydantic import BaseModel as PydanticBaseModel
except ImportError:
    PydanticBaseModel = None  # type: ignore

from ..chat_template import format_chat_prompt
from ..config import GenerationConfig, ModelConfig
from ..engine import DenseCore, GenerationOutput

logger = logging.getLogger(__name__)

_MODEL_FAMILY_HINTS = {
    "qwen": "qwen",
    "llama": "llama",
    "mistral": "mistral",
    "gemma": "gemma",
    "glm": "glm",
    "minimax": "minimax",
}


def _infer_model_family(model_path: Optional[str], hf_repo_id: Optional[str]) -> str:
    """Infer a coarse model family name from model_path or hf_repo_id."""
    source = (hf_repo_id or model_path or "").lower()
    for key, family in _MODEL_FAMILY_HINTS.items():
        if key in source:
            return family
    return "generic"


def _tool_call_instruction_for_family(family: str) -> str:
    """
    Provide a robust, model-agnostic tool call instruction.
    We keep it consistent across families to maximize parseability.
    """
    return (
        "You can call tools by responding with a JSON object of this form:\n"
        '{"tool_calls":[{"id":"call_0","type":"function","function":{"name":"<tool_name>","arguments":<json>}}]}\n'
        "If you are not calling a tool, respond with normal text only."
    )


# ==============================================================================
# Robust Tool Call Parsing Utilities
# ==============================================================================


def _extract_json_object(text: str, start: int = 0) -> tuple[Optional[str], int]:
    """
    Extract a complete JSON object from text, handling nested braces.

    Args:
        text: Text to extract from
        start: Starting position

    Returns:
        Tuple of (json_string, end_position) or (None, -1) if not found
    """
    brace_start = text.find("{", start)
    if brace_start == -1:
        return None, -1

    depth = 0
    in_string = False
    escape_next = False

    for i in range(brace_start, len(text)):
        char = text[i]

        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start : i + 1], i + 1

    # Unterminated JSON object
    return None, -1


def _repair_json(json_str: str) -> str:
    """
    Attempt to repair common JSON errors.

    Handles:
    - Trailing commas
    - Single quotes instead of double quotes
    - Unquoted keys
    """
    # Remove trailing commas before } or ]
    import re

    repaired = re.sub(r",\s*([}\]])", r"\1", json_str)

    # Replace single quotes with double quotes (simple cases only)
    # This is risky so we only do it if the string doesn't parse
    try:
        json.loads(repaired)
        return repaired
    except json.JSONDecodeError:
        # Try replacing single quotes
        repaired = re.sub(r"(?<!\\)'([^']*?)(?<!\\)'", r'"\1"', repaired)
        return repaired


def _safe_json_parse(text: str) -> Optional[dict]:
    """Attempt to parse JSON with repair fallback."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            repaired = _repair_json(text)
            return json.loads(repaired)
        except json.JSONDecodeError:
            return None


def _parse_structured_payload(text: str, schema_class: Optional[Any] = None) -> Optional[Any]:
    """Parse text into JSON payload, optionally validating with a schema class."""
    parsed = _safe_json_parse(text)
    if parsed is None:
        json_str, _ = _extract_json_object(text, 0)
        if json_str:
            parsed = _safe_json_parse(json_str)

    if parsed is None:
        return None

    if schema_class is None:
        return parsed

    try:
        if hasattr(schema_class, "model_validate"):
            return schema_class.model_validate(parsed)
        if hasattr(schema_class, "parse_obj"):
            return schema_class.parse_obj(parsed)
    except Exception as e:
        logger.warning(
            f"Failed to validate structured output against schema {getattr(schema_class, '__name__', 'unknown')}: {e}"
        )
        return parsed

    return parsed


class DenseCoreLLM(LLM):
    """
    LangChain LLM wrapper for DenseCore.

    This class provides a LangChain-compatible interface for DenseCore's
    text completion capabilities, supporting streaming, async operations,
    and LangChain's callback system.

    Example:
        >>> from densecore.integrations import DenseCoreLLM
        >>> llm = DenseCoreLLM(
        ...     model_path="./model.gguf",
        ...     hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        ...     temperature=0.7,
        ...     max_tokens=256
        ... )
        >>> response = llm("What is the capital of France?")
        >>> print(response)

        >>> # With langchain_core composition
        >>> from langchain_core.prompts import PromptTemplate
        >>>
        >>> prompt = PromptTemplate.from_template("Write a short poem about {topic}")
        >>> chain = prompt | llm
        >>> result = chain.invoke({"topic": "AI"})
    """

    # DenseCore specific parameters
    model_path: Optional[str] = Field(default=None, description="Path to GGUF model file")
    hf_repo_id: Optional[str] = Field(default=None, description="HuggingFace repo ID")
    threads: int = Field(default=0, description="Number of threads (0=auto)")

    # Generation parameters
    temperature: float = Field(default=1.0, ge=0.0, description="Sampling temperature")
    max_tokens: int = Field(default=256, gt=0, description="Maximum tokens to generate")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling probability")
    top_k: int = Field(default=0, ge=0, description="Top-k sampling")
    repetition_penalty: float = Field(default=1.0, ge=0.0, description="Repetition penalty")
    stop: Optional[list[str]] = Field(default=None, description="Stop sequences")

    # Internal state
    _engine: Optional[DenseCore] = None

    class Config:
        """Pydantic config."""

        extra = "forbid"
        arbitrary_types_allowed = True

    @root_validator(pre=False, skip_on_failure=True)
    def validate_environment(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate that model path or repo ID is provided."""
        if not values.get("model_path") and not values.get("hf_repo_id"):
            raise ValueError("Either model_path or hf_repo_id must be provided")
        return values

    @property
    def engine(self) -> DenseCore:
        """Lazy initialization of DenseCore engine."""
        if self._engine is None:
            config = ModelConfig(
                model_path=self.model_path or "",
                threads=self.threads,
            )
            self._engine = DenseCore(
                model_path=self.model_path,
                threads=self.threads,
                hf_repo_id=self.hf_repo_id,
                config=config,
                verbose=False,
            )
        return self._engine

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "densecore"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Get identifying parameters."""
        return {
            "model_path": self.model_path,
            "hf_repo_id": self.hf_repo_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        """Backward-compatible convenience wrapper."""
        return self.invoke(prompt, **kwargs)

    def _get_generation_config(self, **kwargs: Any) -> GenerationConfig:
        """Create GenerationConfig from parameters."""
        # Merge instance params with call-time kwargs
        params = {
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "top_k": kwargs.get("top_k", self.top_k),
            "repetition_penalty": kwargs.get("repetition_penalty", self.repetition_penalty),
            "stop_sequences": kwargs.get("stop", self.stop) or [],
        }
        return GenerationConfig(**params)

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Call the DenseCore engine.

        Args:
            prompt: Input prompt
            stop: Stop sequences
            run_manager: Callback manager
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if stop:
            kwargs["stop"] = stop

        config = self._get_generation_config(**kwargs)

        try:
            want_metadata = run_manager is not None
            json_mode = bool(kwargs.get("json_mode", False))
            result = self.engine.generate(
                prompt=prompt,
                max_tokens=config.max_tokens,
                config=config,
                json_mode=json_mode,
                return_dict_in_generate=want_metadata,
            )

            # Notify callback if provided
            if run_manager:
                if isinstance(result, str):
                    result_text = result
                    llm_output = {}
                else:
                    result_text = result.text
                    usage = getattr(result, "usage", {}) or {}
                    llm_output = {
                        "finish_reason": getattr(result, "finish_reason", "stop"),
                        "usage": usage,
                        "tokens": usage.get("completion_tokens", 0),
                    }

                run_manager.on_llm_end(
                    LLMResult(
                        generations=[[Generation(text=result_text)]],
                        llm_output=llm_output,
                    )
                )
                return result_text

            # If no run_manager, engine returns plain string for speed
            return result

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            if run_manager:
                run_manager.on_llm_error(e)
            raise

    def _stream(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """
        Stream the output from DenseCore.

        Args:
            prompt: Input prompt
            stop: Stop sequences
            run_manager: Callback manager
            **kwargs: Additional generation parameters

        Yields:
            Generated text chunks
        """
        if stop:
            kwargs["stop"] = stop

        config = self._get_generation_config(**kwargs)

        try:
            for token in self.engine.stream(
                prompt=prompt,
                max_tokens=config.max_tokens,
                config=config,
            ):
                chunk = GenerationChunk(text=token)
                if run_manager:
                    run_manager.on_llm_new_token(token)
                yield chunk

        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            if run_manager:
                run_manager.on_llm_error(e)
            raise

    async def _acall(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Async call to DenseCore.

        Args:
            prompt: Input prompt
            stop: Stop sequences
            run_manager: Async callback manager
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if stop:
            kwargs["stop"] = stop

        config = self._get_generation_config(**kwargs)

        try:
            json_mode = bool(kwargs.get("json_mode", False))
            result = await self.engine.generate_async(
                prompt=prompt,
                max_tokens=config.max_tokens,
                config=config,
                json_mode=json_mode,
            )
            result_text = result if isinstance(result, str) else result.text

            if run_manager:
                # We don't have detailed stats (tokens, finish_reason) from async str result yet
                # We'll construct a basic LLMResult
                await run_manager.on_llm_end(
                    LLMResult(
                        generations=[[Generation(text=result_text)]],
                        llm_output={
                            "model_name": self.model_path,
                        },
                    )
                )

            return result_text

        except Exception as e:
            logger.error(f"Error during async generation: {e}")
            if run_manager:
                await run_manager.on_llm_error(e)
            raise

    async def _astream(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationChunk]:
        """
        Async stream from DenseCore.

        Args:
            prompt: Input prompt
            stop: Stop sequences
            run_manager: Async callback manager
            **kwargs: Additional generation parameters

        Yields:
            Generated text chunks
        """
        if stop:
            kwargs["stop"] = stop

        config = self._get_generation_config(**kwargs)

        try:
            async for token in self.engine.stream_async(
                prompt=prompt,
                max_tokens=config.max_tokens,
                config=config,
            ):
                chunk = GenerationChunk(text=token)
                if run_manager:
                    await run_manager.on_llm_new_token(token)
                yield chunk

        except Exception as e:
            logger.error(f"Error during async streaming: {e}")
            if run_manager:
                await run_manager.on_llm_error(e)
            raise


class DenseCoreChatModel(BaseChatModel):
    """
    LangChain ChatModel wrapper for DenseCore.

    This class provides a chat-oriented interface compatible with LangChain's
    chat models, supporting conversation history and system messages.

    Example:
        >>> from densecore.integrations import DenseCoreChatModel
        >>> from langchain_core.messages import HumanMessage, SystemMessage
        >>>
        >>> chat = DenseCoreChatModel(
        ...     model_path="./model.gguf",
        ...     hf_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        ...     temperature=0.7
        ... )
        >>>
        >>> messages = [
        ...     SystemMessage(content="You are a helpful assistant."),
        ...     HumanMessage(content="What is quantum computing?")
        ... ]
        >>> response = chat(messages)
        >>> print(response.content)
    """

    # DenseCore specific parameters
    model_path: Optional[str] = Field(default=None, description="Path to GGUF model file")
    hf_repo_id: Optional[str] = Field(default=None, description="HuggingFace repo ID")
    threads: int = Field(default=0, description="Number of threads (0=auto)")

    # Generation parameters
    temperature: float = Field(default=1.0, ge=0.0, description="Sampling temperature")
    max_tokens: int = Field(default=256, gt=0, description="Maximum tokens to generate")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling")
    top_k: int = Field(default=0, ge=0, description="Top-k sampling")
    repetition_penalty: float = Field(default=1.0, ge=0.0, description="Repetition penalty")

    # Internal state
    _engine: Optional[DenseCore] = None

    class Config:
        """Pydantic config."""

        extra = "forbid"
        arbitrary_types_allowed = True

    @root_validator(pre=False, skip_on_failure=True)
    def validate_environment(cls, values: dict) -> dict:
        """Validate that model path or repo ID is provided."""
        if not values.get("model_path") and not values.get("hf_repo_id"):
            raise ValueError("Either model_path or hf_repo_id must be provided")
        return values

    @property
    def engine(self) -> DenseCore:
        """Lazy initialization of DenseCore engine."""
        if self._engine is None:
            config = ModelConfig(
                model_path=self.model_path or "",
                threads=self.threads,
            )
            self._engine = DenseCore(
                model_path=self.model_path,
                threads=self.threads,
                hf_repo_id=self.hf_repo_id,
                config=config,
                verbose=False,
            )
        return self._engine

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "densecore-chat"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Get identifying parameters for caching."""
        return {
            "model_path": self.model_path,
            "hf_repo_id": self.hf_repo_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools": [t.get("function", {}).get("name") for t in getattr(self, "_bound_tools", [])],
        }

    def __call__(self, messages: list[BaseMessage], **kwargs: Any) -> AIMessage:
        """Backward-compatible convenience wrapper."""
        return self.invoke(messages, **kwargs)

    def bind_tools(
        self,
        tools: list[Any],
        *,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> "DenseCoreChatModel":
        """
        Bind tools to the chat model for agent workflows.

        This returns a new instance of the model with tools bound. When invoked,
        the model will generate tool calls in its response if appropriate.

        Args:
            tools: List of tools to bind. Can be:
                - LangChain BaseTool instances
                - Pydantic models
                - Python functions with type hints
                - Dicts in OpenAI tool format
            tool_choice: Optional tool choice strategy ('auto', 'required', or tool name)
            **kwargs: Additional arguments

        Returns:
            New DenseCoreChatModel instance with tools bound

        Example:
            >>> from langchain_core.tools import tool
            >>>
            >>> @tool
            ... def add_numbers(left: int, right: int) -> str:
            ...     '''Add two numbers.'''
            ...     return str(left + right)
            >>>
            >>> llm = DenseCoreChatModel(hf_repo_id="...")
            >>> llm_with_tools = llm.bind_tools([add_numbers])
            >>> response = llm_with_tools.invoke([HumanMessage("What is 2+2?")])
        """
        # Convert tools to OpenAI-compatible format
        formatted_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                # Already in dict format
                formatted_tools.append(tool)
            elif hasattr(tool, "name") and hasattr(tool, "description"):
                # LangChain BaseTool or similar
                try:
                    formatted_tools.append(convert_to_openai_tool(tool))
                except Exception:
                    # Fallback for custom tools
                    formatted_tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": getattr(tool, "name", "unknown"),
                                "description": getattr(tool, "description", ""),
                                "parameters": getattr(tool, "args_schema", {}).schema()
                                if hasattr(tool, "args_schema")
                                else {},
                            },
                        }
                    )
            elif callable(tool):
                # Python function - convert to tool format
                try:
                    formatted_tools.append(convert_to_openai_tool(tool))
                except Exception:
                    func_name = getattr(tool, "__name__", "function")
                    formatted_tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": func_name,
                                "description": getattr(tool, "__doc__", "") or "",
                                "parameters": {"type": "object", "properties": {}},
                            },
                        }
                    )

        # Create a copy with tools bound
        new_model = self.__class__(
            model_path=self.model_path,
            hf_repo_id=self.hf_repo_id,
            threads=self.threads,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
        )
        # Store bound tools as private attribute
        object.__setattr__(new_model, "_bound_tools", formatted_tools)
        object.__setattr__(new_model, "_tool_choice", tool_choice)
        object.__setattr__(new_model, "_original_tools", tools)

        return new_model

    def with_structured_output(
        self,
        schema: Any,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> "DenseCoreChatModel":
        """
        Force the model to output structured JSON matching a schema.

        Args:
            schema: Schema to enforce. Can be:
                - Pydantic model class
                - Dict with JSON schema
                - TypedDict class
            include_raw: If True, include raw response alongside structured output
            **kwargs: Additional arguments

        Returns:
            New DenseCoreChatModel that outputs structured data

        Example:
            >>> from pydantic import BaseModel
            >>>
            >>> class Person(BaseModel):
            ...     name: str
            ...     age: int
            >>>
            >>> llm = DenseCoreChatModel(hf_repo_id="...")
            >>> structured_llm = llm.with_structured_output(Person)
            >>> result = structured_llm.invoke("Tell me about Alice, age 30")
        """
        # Convert schema to JSON schema format
        if hasattr(schema, "schema"):
            # Pydantic model
            json_schema = schema.schema()
        elif isinstance(schema, dict):
            json_schema = schema
        else:
            # Try to infer schema
            json_schema = {"type": "object"}

        # Create a copy with structured output enabled
        new_model = self.__class__(
            model_path=self.model_path,
            hf_repo_id=self.hf_repo_id,
            threads=self.threads,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
        )
        object.__setattr__(new_model, "_output_schema", json_schema)
        object.__setattr__(
            new_model, "_output_schema_class", schema if hasattr(schema, "schema") else None
        )
        object.__setattr__(new_model, "_include_raw", include_raw)

        return new_model

    def _parse_tool_calls(self, text: str) -> list[dict[str, Any]]:
        """
        Parse tool calls from generated text with robust error handling.

        Supports multiple formats:
        - {"tool_calls":[{"function":{"name":"...","arguments":{...}}}]}
        - <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        - {"name": "...", "arguments": {...}}
        - ReAct style: "Action: <name>" + "Action Input: <json>"
        - Partial/malformed JSON with repair attempts
        """
        tool_calls: list[dict[str, Any]] = []

        # Strategy 0: OpenAI-style tool_calls JSON
        try:
            parsed = _safe_json_parse(text)
            if parsed and isinstance(parsed, dict) and "tool_calls" in parsed:
                tool_calls_raw = parsed.get("tool_calls") or []
                for i, tc in enumerate(tool_calls_raw):
                    func = tc.get("function", {}) if isinstance(tc, dict) else {}
                    name = func.get("name")
                    args = func.get("arguments", {})
                    if name:
                        tool_calls.append(
                            {
                                "id": tc.get("id", f"call_{i}"),
                                "name": name,
                                "args": args if isinstance(args, dict) else {},
                            }
                        )
                if tool_calls:
                    return tool_calls
        except Exception:
            pass

        # Strategy 1: Look for XML-style tool calls
        xml_pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
        for match in xml_pattern.finditer(text):
            inner = match.group(1).strip()
            parsed = _safe_json_parse(inner)
            if parsed and isinstance(parsed, dict):
                name = parsed.get("name")
                args = parsed.get("arguments", {})
                if name:
                    tool_calls.append(
                        {
                            "id": f"call_{len(tool_calls)}",
                            "name": name,
                            "args": args if isinstance(args, dict) else {},
                        }
                    )
            else:
                logger.debug("Failed to parse tool call XML content: %s", inner[:100])

        if tool_calls:
            return tool_calls

        # Strategy 2: Look for incomplete XML-style (missing closing tag)
        partial_pattern = re.compile(r"<tool_call>\s*(\{.+)", re.DOTALL)
        partial_match = partial_pattern.search(text)
        if partial_match:
            json_str, _ = _extract_json_object(partial_match.group(1))
            if json_str:
                parsed = _safe_json_parse(json_str)
                if parsed and isinstance(parsed, dict):
                    name = parsed.get("name")
                    args = parsed.get("arguments", {})
                    if name:
                        tool_calls.append(
                            {
                                "id": "call_0",
                                "name": name,
                                "args": args if isinstance(args, dict) else {},
                            }
                        )
                        logger.warning(
                            "Parsed incomplete tool call (missing </tool_call>): %s", name
                        )

        if tool_calls:
            return tool_calls

        # Strategy 3: Look for raw JSON function calls
        # Pattern: {"name": "...", "arguments": {...}}
        pos = 0
        while pos < len(text):
            json_str, end_pos = _extract_json_object(text, pos)
            if json_str is None:
                break

            parsed = _safe_json_parse(json_str)
            if parsed and isinstance(parsed, dict):
                # Check if it looks like a function call
                if "name" in parsed and ("arguments" in parsed or "args" in parsed):
                    name = parsed.get("name")
                    args = parsed.get("arguments") or parsed.get("args", {})
                    if name and isinstance(name, str):
                        tool_calls.append(
                            {
                                "id": f"call_{len(tool_calls)}",
                                "name": name,
                                "args": args if isinstance(args, dict) else {},
                            }
                        )

            pos = end_pos

        if tool_calls:
            return tool_calls

        # Strategy 4: ReAct format (Action / Action Input)
        action_match = re.search(r"^Action\s*:\s*(.+)$", text, flags=re.MULTILINE)
        input_match = re.search(r"^Action Input\s*:\s*(\{[\s\S]*\})\s*$", text, flags=re.MULTILINE)
        if action_match and input_match:
            name = action_match.group(1).strip()
            args = _safe_json_parse(input_match.group(1).strip()) or {}
            if name:
                tool_calls.append(
                    {
                        "id": "call_0",
                        "name": name,
                        "args": args if isinstance(args, dict) else {},
                    }
                )

        return tool_calls

    def _format_messages(self, messages: list[BaseMessage]) -> str:
        """
        Format messages into a prompt string.

        Handles all message types including ToolMessage for multi-turn tool use.
        """
        structured_messages: list[dict[str, Any]] = []
        extra_system_messages: list[str] = []

        # Check if we have tools bound
        bound_tools = getattr(self, "_bound_tools", [])
        tool_choice = getattr(self, "_tool_choice", None)
        if tool_choice == "none":
            bound_tools = []

        # Add tool definitions if present
        if bound_tools:
            tools_desc = "Available tools:\n"
            for tool in bound_tools:
                func = tool.get("function", {})
                tools_desc += f"- {func.get('name', 'unknown')}: {func.get('description', '')}\n"
            model_family = _infer_model_family(self.model_path, self.hf_repo_id)
            tools_desc += "\n" + _tool_call_instruction_for_family(model_family) + "\n"
            if tool_choice in ("required", "any", True):
                tools_desc += "You must emit at least one tool call in the required JSON format.\n"
            elif isinstance(tool_choice, str) and tool_choice not in ("auto", "none", "required"):
                tools_desc += f'You must prefer tool "{tool_choice}" when producing a tool call.\n'
            extra_system_messages.append(tools_desc)

        output_schema = getattr(self, "_output_schema", None)
        if output_schema:
            extra_system_messages.append(
                "Return a valid JSON object matching this schema exactly:\n"
                f"{json.dumps(output_schema, ensure_ascii=False)}"
            )

        for message in messages:
            if isinstance(message, SystemMessage):
                structured_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                structured_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": message.content,
                }
                if hasattr(message, "tool_calls") and message.tool_calls:
                    assistant_message["tool_calls"] = [
                        {
                            "id": tc.get("id", f"call_{index}"),
                            "type": "function",
                            "function": {
                                "name": tc.get("name", "unknown"),
                                "arguments": json.dumps(tc.get("args", {}), ensure_ascii=False),
                            },
                        }
                        for index, tc in enumerate(message.tool_calls)
                    ]
                structured_messages.append(assistant_message)
            elif isinstance(message, ToolMessage):
                tool_name = getattr(message, "name", "tool")
                structured_messages.append(
                    {"role": "tool", "name": tool_name, "content": message.content}
                )
            elif isinstance(message, ChatMessage):
                role = message.role or "User"
                structured_messages.append({"role": role.lower(), "content": message.content})
            else:
                structured_messages.append({"role": "user", "content": message.content})

        return format_chat_prompt(
            self.hf_repo_id or self.model_path,
            structured_messages,
            extra_system_messages=extra_system_messages,
        )

    def _get_generation_config(self, **kwargs: Any) -> GenerationConfig:
        """Create GenerationConfig from parameters."""
        params = {
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "top_k": kwargs.get("top_k", self.top_k),
            "repetition_penalty": kwargs.get("repetition_penalty", self.repetition_penalty),
            "stop_sequences": kwargs.get("stop", []),
        }
        return GenerationConfig(**params)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate chat completion with tool calling support.

        Args:
            messages: List of chat messages
            stop: Stop sequences
            run_manager: Callback manager
            **kwargs: Additional generation parameters

        Returns:
            ChatResult with generated message (including tool_calls if detected)
        """
        prompt = self._format_messages(messages)

        if stop:
            kwargs["stop"] = stop

        if getattr(self, "_output_schema", None):
            kwargs["json_mode"] = True

        config = self._get_generation_config(**kwargs)

        try:
            json_mode = bool(
                kwargs.get("json_mode", False) or getattr(self, "_output_schema", None)
            )
            result = self.engine.generate(
                prompt=prompt,
                max_tokens=config.max_tokens,
                config=config,
                json_mode=json_mode,
            )

            # Handle string or GenerationOutput result
            if isinstance(result, str):
                text = result
                tokens = 0
                finish_reason = "stop"
                generation_time = 0.0
            else:
                text = result.text
                tokens = getattr(result, "tokens", 0)
                finish_reason = getattr(result, "finish_reason", "stop")
                generation_time = getattr(result, "generation_time", 0.0)

            # Parse tool calls if tools are bound
            tool_calls = []
            if (
                getattr(self, "_bound_tools", None)
                and getattr(self, "_tool_choice", None) != "none"
            ):
                tool_calls = self._parse_tool_calls(text)

            structured_payload = None
            raw_text = text
            if getattr(self, "_output_schema", None):
                structured_payload = _parse_structured_payload(
                    text,
                    schema_class=getattr(self, "_output_schema_class", None),
                )
                if structured_payload is not None and not getattr(self, "_include_raw", False):
                    if isinstance(structured_payload, (dict, list)):
                        text = json.dumps(structured_payload, ensure_ascii=False)
                    else:
                        text = str(structured_payload)

            additional_kwargs = {}
            if structured_payload is not None:
                additional_kwargs["parsed"] = structured_payload
                if getattr(self, "_include_raw", False):
                    additional_kwargs["raw"] = raw_text

            # Create AIMessage with tool_calls if present
            if tool_calls:
                message = AIMessage(
                    content=text,
                    tool_calls=tool_calls,
                    additional_kwargs=additional_kwargs,
                )
            else:
                message = AIMessage(content=text, additional_kwargs=additional_kwargs)

            generation = ChatGeneration(message=message)

            return ChatResult(
                generations=[generation],
                llm_output={
                    "tokens": tokens,
                    "finish_reason": finish_reason,
                    "generation_time": generation_time,
                    "tool_calls": tool_calls,
                },
            )

        except Exception as e:
            logger.error("Error during chat generation: %s", e)
            if run_manager:
                run_manager.on_llm_error(e)
            raise

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Async generate chat completion with tool calling support.

        Args:
            messages: List of chat messages
            stop: Stop sequences
            run_manager: Async callback manager
            **kwargs: Additional generation parameters

        Returns:
            ChatResult with generated message (including tool_calls if detected)
        """
        prompt = self._format_messages(messages)

        if stop:
            kwargs["stop"] = stop

        if getattr(self, "_output_schema", None):
            kwargs["json_mode"] = True

        config = self._get_generation_config(**kwargs)

        try:
            json_mode = bool(
                kwargs.get("json_mode", False) or getattr(self, "_output_schema", None)
            )
            result = await self.engine.generate_async(
                prompt=prompt,
                max_tokens=config.max_tokens,
                config=config,
                json_mode=json_mode,
            )

            # Handle string or GenerationOutput result
            if isinstance(result, str):
                text = result
                tokens = 0
                finish_reason = "stop"
                generation_time = 0.0
            else:
                text = result.text
                tokens = getattr(result, "tokens", 0)
                finish_reason = getattr(result, "finish_reason", "stop")
                generation_time = getattr(result, "generation_time", 0.0)

            # Parse tool calls if tools are bound
            tool_calls = []
            if (
                getattr(self, "_bound_tools", None)
                and getattr(self, "_tool_choice", None) != "none"
            ):
                tool_calls = self._parse_tool_calls(text)

            structured_payload = None
            raw_text = text
            if getattr(self, "_output_schema", None):
                structured_payload = _parse_structured_payload(
                    text,
                    schema_class=getattr(self, "_output_schema_class", None),
                )
                if structured_payload is not None and not getattr(self, "_include_raw", False):
                    if isinstance(structured_payload, (dict, list)):
                        text = json.dumps(structured_payload, ensure_ascii=False)
                    else:
                        text = str(structured_payload)

            additional_kwargs = {}
            if structured_payload is not None:
                additional_kwargs["parsed"] = structured_payload
                if getattr(self, "_include_raw", False):
                    additional_kwargs["raw"] = raw_text

            # Create AIMessage with tool_calls if present
            if tool_calls:
                message = AIMessage(
                    content=text,
                    tool_calls=tool_calls,
                    additional_kwargs=additional_kwargs,
                )
            else:
                message = AIMessage(content=text, additional_kwargs=additional_kwargs)

            generation = ChatGeneration(message=message)

            return ChatResult(
                generations=[generation],
                llm_output={
                    "tokens": tokens,
                    "finish_reason": finish_reason,
                    "generation_time": generation_time,
                    "tool_calls": tool_calls,
                },
            )

        except Exception as e:
            logger.error("Error during async chat generation: %s", e)
            if run_manager:
                await run_manager.on_llm_error(e)
            raise

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        Stream chat completion.

        Args:
            messages: List of chat messages
            stop: Stop sequences
            run_manager: Callback manager
            **kwargs: Additional generation parameters

        Yields:
            ChatGenerationChunk tokens
        """
        prompt = self._format_messages(messages)

        if stop:
            kwargs["stop"] = stop

        config = self._get_generation_config(**kwargs)

        try:
            for token in self.engine.stream(
                prompt=prompt,
                max_tokens=config.max_tokens,
                config=config,
            ):
                message = AIMessageChunk(content=token)
                generation = ChatGenerationChunk(message=message)

                if run_manager:
                    run_manager.on_llm_new_token(token)

                yield generation

        except Exception as e:
            logger.error(f"Error during chat streaming: {e}")
            if run_manager:
                run_manager.on_llm_error(e)
            raise

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """
        Async stream chat completion.

        Args:
            messages: List of chat messages
            stop: Stop sequences
            run_manager: Async callback manager
            **kwargs: Additional generation parameters

        Yields:
            ChatGenerationChunk tokens
        """
        prompt = self._format_messages(messages)

        if stop:
            kwargs["stop"] = stop

        config = self._get_generation_config(**kwargs)

        try:
            async for token in self.engine.stream_async(
                prompt=prompt,
                max_tokens=config.max_tokens,
                config=config,
            ):
                message = AIMessageChunk(content=token)
                generation = ChatGenerationChunk(message=message)

                if run_manager:
                    await run_manager.on_llm_new_token(token)

                yield generation

        except Exception as e:
            logger.error(f"Error during async chat streaming: {e}")
            if run_manager:
                await run_manager.on_llm_error(e)
            raise


# ==============================================================================
# Embeddings Wrapper for VectorStore Integration
# ==============================================================================


class DenseCoreEmbeddings(Embeddings):
    """
    LangChain Embeddings wrapper for DenseCore.

    Enables seamless integration with LangChain VectorStores like FAISS, Chroma, etc.
    Uses DenseCore's CPU-optimized embedding engine under the hood.

    Args:
        model_path: Path to GGUF embedding model (e.g., bge-small.gguf)
        pooling_strategy: Pooling method ("mean", "cls", "last", "max")
        normalize: Whether to L2 normalize embeddings for cosine similarity
        threads: CPU threads for inference (0 = auto-detect)

    Example:
        >>> from densecore.integrations.langchain import DenseCoreEmbeddings
        >>> from langchain_community.vectorstores import FAISS
        >>>
        >>> embeddings = DenseCoreEmbeddings(
        ...     model_path="./bge-small.gguf",
        ...     pooling_strategy="mean",
        ...     normalize=True,
        ... )
        >>>
        >>> # Create vector store
        >>> vectorstore = FAISS.from_texts(
        ...     ["Document 1", "Document 2", "Document 3"],
        ...     embeddings
        ... )
        >>> results = vectorstore.similarity_search("query")
    """

    def __init__(
        self,
        model_path: str,
        pooling_strategy: str = "mean",
        normalize: bool = True,
        threads: int = 0,
    ) -> None:
        """
        Initialize DenseCoreEmbeddings.

        Args:
            model_path: Path to GGUF embedding model file
            pooling_strategy: Pooling strategy ("mean", "cls", "last", "max")
            normalize: L2 normalize embeddings (recommended for cosine similarity)
            threads: Number of CPU threads (0 = auto-detect optimal count)
        """
        from ..embedding import EmbeddingConfig, EmbeddingModel

        self._normalize = normalize

        config = EmbeddingConfig(
            pooling=pooling_strategy,
            normalize=normalize,
        )

        self._model = EmbeddingModel(
            model_path=model_path,
            config=config,
            threads=threads,
            verbose=False,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of documents.

        Uses batch embedding API when available for optimal performance.

        Args:
            texts: List of document texts to embed

        Returns:
            List of embedding vectors (each as list of floats)
        """
        if not texts:
            return []

        # Use batch API if available, otherwise fall back to sequential
        if hasattr(self._model, "_has_batch_embedding") and self._model._has_batch_embedding:
            embeddings = self._model.embed_batch(texts, show_progress=False)
        else:
            embeddings = self._model.embed(texts)

        # Convert numpy arrays to list format for LangChain compatibility
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector as list of floats
        """
        embedding = self._model.embed(text)
        return embedding.tolist()

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        return self._model.dimension

    def __repr__(self) -> str:
        return f"DenseCoreEmbeddings(dim={self._model.dimension}, normalize={self._normalize})"
