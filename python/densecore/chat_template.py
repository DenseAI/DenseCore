from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any, Optional

DEFAULT_CHAT_SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass(frozen=True)
class PromptProfile:
    family: str
    kind: str
    system_role: str
    user_role: str
    assistant_role: str
    open_tag: str
    close_tag: str
    default_system_prompt: str
    image_token: str
    video_token: str
    audio_token: str


def _default_profiles() -> list[PromptProfile]:
    return [
        PromptProfile(
            family="qwen",
            kind="chatml",
            system_role="system",
            user_role="user",
            assistant_role="assistant",
            open_tag="<|im_start|>",
            close_tag="<|im_end|>\n",
            default_system_prompt=DEFAULT_CHAT_SYSTEM_PROMPT,
            image_token="<|vision_start|><|image_pad|><|vision_end|>",
            video_token="<|vision_start|><|video_pad|><|vision_end|>",
            audio_token="",
        ),
        PromptProfile(
            family="gemma",
            kind="turn_tags",
            system_role="system",
            user_role="user",
            assistant_role="model",
            open_tag="<|turn>",
            close_tag="<turn|>\n",
            default_system_prompt=DEFAULT_CHAT_SYSTEM_PROMPT,
            image_token="\n\n<|image|>\n\n",
            video_token="\n\n<|video|>\n\n",
            audio_token="<|audio|>",
        ),
        PromptProfile(
            family="generic",
            kind="generic_transcript",
            system_role="System",
            user_role="User",
            assistant_role="Assistant",
            open_tag="",
            close_tag="",
            default_system_prompt=DEFAULT_CHAT_SYSTEM_PROMPT,
            image_token="",
            video_token="",
            audio_token="",
        ),
    ]


@lru_cache(maxsize=1)
def _profile_specs() -> list[dict[str, Any]]:
    try:
        text = (
            resources.files("densecore")
            .joinpath("prompt_profiles.json")
            .read_text(encoding="utf-8")
        )
        data = json.loads(text)
        profiles = data.get("profiles", [])
        if isinstance(profiles, list):
            return profiles
    except Exception:
        return []
    return []


def resolve_prompt_profile(model_hint: Optional[str]) -> PromptProfile:
    source = (model_hint or "").strip().lower()
    specs = _profile_specs()
    for spec in specs:
        matches = spec.get("match_substrings", [])
        if any(token in source for token in matches):
            roles = spec.get("roles", {})
            tags = spec.get("tags", {})
            multimodal = spec.get("multimodal", {})
            return PromptProfile(
                family=spec.get("family", "generic"),
                kind=spec.get("kind", "generic_transcript"),
                system_role=roles.get("system", "system"),
                user_role=roles.get("user", "user"),
                assistant_role=roles.get("assistant", "assistant"),
                open_tag=tags.get("open", ""),
                close_tag=tags.get("close", ""),
                default_system_prompt=spec.get("default_system_prompt", DEFAULT_CHAT_SYSTEM_PROMPT),
                image_token=multimodal.get("image", ""),
                video_token=multimodal.get("video", ""),
                audio_token=multimodal.get("audio", ""),
            )
    for profile in _default_profiles():
        if profile.family != "generic" and profile.family in source:
            return profile
    return _default_profiles()[-1]


def qwen_thinking_enabled(model_hint: Optional[str], enable_thinking: Optional[bool]) -> bool:
    if enable_thinking is not None:
        return enable_thinking
    lower = (model_hint or "").strip().lower()
    if any(token in lower for token in ("qwen3.5", "qwen3_5", "qwen3-5", "qwen35")):
        return os.getenv("DENSECORE_QWEN35_ENABLE_THINKING", "false").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    if "qwen3" in lower:
        return os.getenv("DENSECORE_QWEN3_ENABLE_THINKING", "true").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    return True


def gemma_thinking_enabled(enable_thinking: Optional[bool]) -> bool:
    return bool(enable_thinking)


def _normalize_content(content: str) -> str:
    return content.strip()


def _has_structured_content(message: dict[str, Any]) -> bool:
    return isinstance(message.get("content"), list)


def _render_structured_text(parts: list[dict[str, Any]], profile: PromptProfile) -> str:
    rendered: list[str] = []
    for part in parts:
        if part.get("text"):
            rendered.append(str(part["text"]))
        elif (
            part.get("type") == "image"
            or part.get("image")
            or part.get("image_url")
            or part.get("url")
        ):
            rendered.append(profile.image_token)
        elif part.get("type") == "video" or part.get("video"):
            rendered.append(profile.video_token)
        elif part.get("type") == "audio" or part.get("audio"):
            rendered.append(profile.audio_token)
    return "".join(rendered)


def _render_qwen_content(message: dict[str, Any], profile: PromptProfile) -> str:
    if not _has_structured_content(message):
        return _normalize_content(str(message.get("content", "")))
    return _normalize_content(_render_structured_text(message.get("content", []), profile))


def _render_qwen_assistant_message(message: dict[str, Any], profile: PromptProfile) -> str:
    content = _render_qwen_content(message, profile)
    reasoning = _normalize_content(str(message.get("reasoning_content", "")))
    if reasoning:
        return _normalize_content(f"<think>\n{reasoning}\n</think>\n\n{content}")

    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return content

    parts: list[str] = []
    if content:
        parts.append(content)
    for tool_call in tool_calls:
        function = tool_call.get("function", {})
        name = function.get("name", "unknown")
        args = function.get("arguments", "")
        parsed_args: dict[str, Any] = {}
        if isinstance(args, str) and args:
            try:
                parsed_args = json.loads(args)
            except json.JSONDecodeError:
                parsed_args = {"arguments": args}
        elif isinstance(args, dict):
            parsed_args = args
        block = [f"<tool_call>\n<function={name}>"]
        for key, value in parsed_args.items():
            if isinstance(value, str):
                rendered = value
            else:
                rendered = json.dumps(value, ensure_ascii=False)
            block.append(f"<parameter={key}>\n{rendered}\n</parameter>")
        block.append("</function>\n</tool_call>")
        parts.append("\n".join(block))
    return _normalize_content("\n\n".join(parts))


def _render_qwen_tool_response(message: dict[str, Any], profile: PromptProfile) -> str:
    content = _render_qwen_content(message, profile)
    if not content:
        content = _normalize_content(str(message.get("content", "")))
    return _normalize_content(f"<tool_response>\n{content}\n</tool_response>")


def _strip_gemma_thinking_channels(content: str) -> str:
    content = _normalize_content(content)
    if "<channel|>" not in content:
        return content
    rendered: list[str] = []
    for part in content.split("<channel|>"):
        marker = part.find("<|channel>")
        rendered.append(part[:marker] if marker >= 0 else part)
    return _normalize_content("".join(rendered))


def _render_gemma_content(message: dict[str, Any], profile: PromptProfile, role: str) -> str:
    if not _has_structured_content(message):
        content = _normalize_content(str(message.get("content", "")))
        return _strip_gemma_thinking_channels(content) if role == "model" else content
    content = _normalize_content(_render_structured_text(message.get("content", []), profile))
    return _strip_gemma_thinking_channels(content) if role == "model" else content


def _render_gemma_assistant_message(message: dict[str, Any], profile: PromptProfile) -> str:
    content = _render_gemma_content(message, profile, "model")
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return content
    parts = [content] if content else []
    for tool_call in tool_calls:
        function = tool_call.get("function", {})
        name = function.get("name", "unknown")
        args = function.get("arguments", "")
        if isinstance(args, dict):
            args = json.dumps(args, ensure_ascii=False)
        parts.append(f"<|tool_call>call:{name}{{{args}}}<tool_call|>")
    return _normalize_content("".join(parts))


def _render_gemma_tool_response(message: dict[str, Any], profile: PromptProfile) -> str:
    tool_responses = message.get("tool_responses") or []
    if tool_responses:
        parts: list[str] = []
        for response in tool_responses:
            name = response.get("name") or "unknown"
            value = response.get("response")
            if isinstance(value, str):
                rendered = f'value:<|"|>{value}<|"|>'
            else:
                rendered = "value:" + json.dumps(value, ensure_ascii=False)
            parts.append(f"<|tool_response>response:{name}{{{rendered}}}<tool_response|>")
        return _normalize_content("".join(parts))

    content = _render_gemma_content(message, profile, "tool")
    if not content:
        return ""
    name = message.get("name", "")
    return _normalize_content(
        f'<|tool_response>response:{name}{{value:<|"|>{content}<|"|>}}<tool_response|>'
    )


def _render_generic_assistant_message(message: dict[str, Any]) -> str:
    content = _normalize_content(str(message.get("content", "")))
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return content
    parts = [content] if content else []
    for tool_call in tool_calls:
        function = tool_call.get("function", {})
        name = function.get("name", "unknown")
        args = function.get("arguments", {})
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        parts.append(f'<tool_call>{{"name":"{name}","arguments":{args}}}</tool_call>')
    return _normalize_content("\n".join(part for part in parts if part))


def format_chat_prompt(
    model_hint: Optional[str],
    messages: list[dict[str, Any]],
    *,
    enable_thinking: Optional[bool] = None,
    extra_system_messages: Optional[list[str]] = None,
) -> str:
    if not messages and not extra_system_messages:
        return ""

    profile = resolve_prompt_profile(model_hint)
    normalized_messages = list(messages)
    if extra_system_messages:
        for content in reversed(extra_system_messages):
            if _normalize_content(content):
                normalized_messages.insert(0, {"role": "system", "content": content})

    if profile.family == "qwen":
        thinking_enabled = qwen_thinking_enabled(model_hint, enable_thinking)
        parts: list[str] = []
        for index, message in enumerate(normalized_messages):
            role = str(message.get("role", "user")).strip().lower()
            if role == "system":
                if index == 0:
                    content = _render_qwen_content(message, profile)
                    if content:
                        parts.append(
                            f"{profile.open_tag}{profile.system_role}\n{content}{profile.close_tag}"
                        )
            elif role == "user":
                content = _render_qwen_content(message, profile)
                if content:
                    parts.append(f"{profile.open_tag}user\n{content}{profile.close_tag}")
            elif role == "assistant":
                content = _render_qwen_assistant_message(message, profile)
                if content:
                    parts.append(
                        f"{profile.open_tag}{profile.assistant_role}\n{content}{profile.close_tag}"
                    )
            elif role == "tool":
                content = _render_qwen_tool_response(message, profile)
                if content:
                    parts.append(f"{profile.open_tag}user\n{content}{profile.close_tag}")
        assistant_prefix = "<think>\n" if thinking_enabled else "<think>\n\n</think>\n\n"
        parts.append(f"{profile.open_tag}{profile.assistant_role}\n{assistant_prefix}")
        return "".join(parts)

    if profile.family == "gemma":
        thinking_enabled = gemma_thinking_enabled(enable_thinking)
        system_parts: list[str] = []
        filtered: list[dict[str, Any]] = []
        for message in normalized_messages:
            role = str(message.get("role", "user")).strip().lower()
            if role in ("system", "developer"):
                content = _render_gemma_content(message, profile, profile.system_role)
                if content:
                    system_parts.append(content)
            else:
                filtered.append(message)

        parts: list[str] = ["<bos>"]
        if thinking_enabled or system_parts:
            system_content = ""
            if thinking_enabled:
                system_content += "<|think|>"
            if system_parts:
                system_content += "\n\n".join(system_parts)
            parts.append(
                f"{profile.open_tag}{profile.system_role}\n{system_content}{profile.close_tag}"
            )

        for message in filtered:
            role = str(message.get("role", "user")).strip().lower()
            if role == "user":
                content = _render_gemma_content(message, profile, "user")
                if content:
                    parts.append(f"{profile.open_tag}user\n{content}{profile.close_tag}")
            elif role == "assistant":
                content = _render_gemma_assistant_message(message, profile)
                if content:
                    parts.append(
                        f"{profile.open_tag}{profile.assistant_role}\n{content}{profile.close_tag}"
                    )
            elif role == "tool":
                content = _render_gemma_tool_response(message, profile)
                if content:
                    parts.append(f"{profile.open_tag}user\n{content}{profile.close_tag}")
        parts.append(f"{profile.open_tag}{profile.assistant_role}\n")
        return "".join(parts)

    system_messages = [
        message
        for message in normalized_messages
        if str(message.get("role", "")).strip().lower() == "system"
    ]
    conversation_messages = [
        message
        for message in normalized_messages
        if str(message.get("role", "")).strip().lower() in ("user", "assistant", "tool")
    ]
    if not system_messages:
        system_messages = [{"role": "system", "content": profile.default_system_prompt}]

    lines: list[str] = []
    for message in system_messages:
        content = _normalize_content(str(message.get("content", "")))
        if content:
            lines.append(f"System: {content}")
    for message in conversation_messages:
        role = str(message.get("role", "")).strip().lower()
        if role == "user":
            content = _normalize_content(str(message.get("content", "")))
            if content:
                lines.append(f"User: {content}")
        elif role == "assistant":
            content = _render_generic_assistant_message(message)
            if content:
                lines.append(f"Assistant: {content}")
        elif role == "tool":
            name = message.get("name", "tool")
            content = _normalize_content(str(message.get("content", "")))
            if content:
                lines.append(f"Tool ({name}): {content}")
    lines.append("Assistant: ")
    return "\n".join(lines)
