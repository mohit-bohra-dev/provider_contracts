"""AWS Bedrock LLM adapter using the Converse API."""
import json
import logging

_log = logging.getLogger(__name__)
from collections.abc import AsyncIterator
from typing import Any

try:
    import aioboto3
except ImportError:
    aioboto3 = None

from ._base import (
    AbstractLLMProvider,
    LLMMessage,
    LLMResponse,
    ToolCallRequest,
    ToolDefinition,
)

_DEFAULT_MODEL = "us.amazon.nova-pro-v1:0"
_DEFAULT_REGION = "us-east-1"
# Bedrock API-key endpoint — must be bedrock-RUNTIME for inference (Converse API)
_BEDROCK_API_KEY_ENDPOINT = "https://bedrock-runtime.{region}.amazonaws.com"


class BedrockProvider(AbstractLLMProvider):
    """Adapter for AWS Bedrock API using the unified Converse API.

    Supports two authentication modes:

    1. **Bedrock API Key** (bearer token, recommended for quick starts):
       Pass ``api_key=<your-key>``.  The key is sent as an
       ``Authorization: Bearer`` header; no AWS IAM credentials are needed.

    2. **IAM credentials** (recommended for production):
       Pass ``access_key_id`` + ``secret_access_key`` (and optionally
       ``session_token`` for temporary credentials).  Leave all three as
       ``None`` to fall back on the standard boto3 credential chain
       (environment variables, ``~/.aws/credentials``, instance metadata,
       etc.).
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        region: str = _DEFAULT_REGION,
        # Bedrock API key (bearer token) — simplest auth for quick starts
        api_key: str | None = None,
        # IAM credentials — leave None to use the boto3 default chain
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        session_token: str | None = None,
    ) -> None:
        if aioboto3 is None:
            raise ImportError(
                "aioboto3 is not installed. Please install provider-contracts[bedrock]."
            )

        self._model_id = model_id
        self._region = region
        self._session = aioboto3.Session()

        if api_key:
            # Bearer-token mode: set AWS_BEARER_TOKEN_BEDROCK — the official
            # botocore env var that triggers Authorization: Bearer injection.
            # This is exactly what LiteLLM does (start_litellm.ps1 L41).
            import os
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key
            _log.debug("bedrock: API key auth — set AWS_BEARER_TOKEN_BEDROCK")

        # IAM / default credential chain (or bearer token via env var above)
        self._client_kwargs: dict[str, Any] = {
            "region_name": region,
            "aws_access_key_id": access_key_id,
            "aws_secret_access_key": secret_access_key,
            "aws_session_token": session_token,
        }

    # Bedrock Converse only accepts "user" | "assistant" for non-system turns.
    _ROLE_MAP: dict[str, str] = {
        "user": "user",
        "assistant": "assistant",
        "tool": "user",       # tool results re-injected as user turn
        "function": "user",   # OpenAI compat alias
    }

    def _build_messages(
        self, messages: list[LLMMessage]
    ) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]]]:
        system: list[dict[str, Any]] | None = None
        chat_msgs: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "system":
                if system is None:
                    system = []
                system.append({"text": msg.content})
            else:
                bedrock_role = self._ROLE_MAP.get(msg.role, "user")
                chat_msgs.append({
                    "role": bedrock_role,
                    "content": [{"text": msg.content}]
                })
        if not chat_msgs:
            raise ValueError(
                "BedrockProvider.chat() requires at least one non-system message; "
                f"got roles: {[m.role for m in messages]}"
            )
        return system, chat_msgs

    def _to_bedrock_tools(self, tools: list[ToolDefinition]) -> dict[str, Any]:
        return {
            "tools": [
                {
                    "toolSpec": {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": {"json": tool.parameters},
                    }
                }
                for tool in tools
            ]
        }

    def _extract_text(self, content_blocks: list[dict[str, Any]]) -> str:
        text_chunks: list[str] = []
        for block in content_blocks:
            if "text" in block:
                text_chunks.append(block["text"])
        return "\n".join(text_chunks)

    def _parse_tool_calls(self, content_blocks: list[dict[str, Any]]) -> list[ToolCallRequest]:
        tool_calls: list[ToolCallRequest] = []
        for block in content_blocks:
            if "toolUse" not in block:
                continue
            tool_use = block["toolUse"]
            name = tool_use.get("name", "")
            if not name:
                continue
            input_data = tool_use.get("input", {})
            tool_calls.append(ToolCallRequest(name=name, arguments=input_data))
        return tool_calls

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse:
        if json_mode:
            instruction = "You must return a valid JSON object. Do not wrap the response in markdown blocks."
            system = f"{system}\n\n{instruction}" if system else instruction

        kwargs: dict[str, Any] = dict(
            modelId=self._model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
        )
        if system:
            kwargs["system"] = [{"text": system}]

        async with self._session.client("bedrock-runtime", **self._client_kwargs) as client:
            response = await client.converse(**kwargs)

        output_message = response.get("output", {}).get("message", {})
        content_blocks = output_message.get("content", [])
        usage = response.get("usage", {})
        
        return LLMResponse(
            content=self._extract_text(content_blocks),
            model=self._model_id,
            usage={
                "prompt_tokens": usage.get("inputTokens", 0),
                "completion_tokens": usage.get("outputTokens", 0),
            },
            tool_calls=self._parse_tool_calls(content_blocks),
        )

    async def chat(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False,
        tools: list[ToolDefinition] | None = None,
    ) -> LLMResponse:
        system, chat_msgs = self._build_messages(messages)
        if json_mode:
            instruction = {"text": "You must return a valid JSON object. Do not wrap the response in markdown blocks."}
            if system is None:
                system = [instruction]
            else:
                system.append(instruction)

        kwargs: dict[str, Any] = dict(
            modelId=self._model_id,
            messages=chat_msgs,
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
        )
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["toolConfig"] = self._to_bedrock_tools(tools)

        _log.debug("bedrock.chat kwargs: %s", {k: v for k, v in kwargs.items() if k != "system"})
        _log.debug("bedrock.chat messages count: %d", len(chat_msgs))

        async with self._session.client("bedrock-runtime", **self._client_kwargs) as client:
            response = await client.converse(**kwargs)

        _log.warning("bedrock.chat RAW response: %s", json.dumps(response, default=str))

        stop_reason = response.get("stopReason", "<missing>")
        output_message = response.get("output", {}).get("message", {})
        content_blocks = output_message.get("content", [])
        usage = response.get("usage", {})
        extracted = self._extract_text(content_blocks)

        _log.debug(
            "bedrock.chat stop_reason=%s content_blocks=%d extracted_len=%d",
            stop_reason, len(content_blocks), len(extracted),
        )
        if not extracted:
            _log.warning(
                "bedrock.chat got BLANK response. stop_reason=%s content_blocks=%s",
                stop_reason, content_blocks,
            )

        return LLMResponse(
            content=extracted,
            model=self._model_id,
            usage={
                "prompt_tokens": usage.get("inputTokens", 0),
                "completion_tokens": usage.get("outputTokens", 0),
            },
            tool_calls=self._parse_tool_calls(content_blocks),
        )

    def stream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncIterator[str]:
        return self._stream_impl(messages, temperature=temperature, max_tokens=max_tokens)

    async def _stream_impl(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        system, chat_msgs = self._build_messages(messages)
        kwargs: dict[str, Any] = dict(
            modelId=self._model_id,
            messages=chat_msgs,
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
        )
        if system:
            kwargs["system"] = system

        async with self._session.client("bedrock-runtime", **self._client_kwargs) as client:
            response = await client.converse_stream(**kwargs)
            stream = response.get("stream")
            if stream:
                async for event in stream:
                    if "contentBlockDelta" in event:
                        delta = event["contentBlockDelta"].get("delta", {})
                        if "text" in delta:
                            yield delta["text"]

    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError("Use a dedicated BedrockEmbeddingProvider.")
