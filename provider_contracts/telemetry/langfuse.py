"""Langfuse telemetry provider implementation."""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from ._base import (
    AbstractTelemetryProvider,
    SpanContext,
    SpanKind,
    current_span_context,
    trace_span,
)


class LangfuseTelemetryProvider(AbstractTelemetryProvider):
    """Telemetry provider that exports spans + metrics to Langfuse."""

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        host: str,
        *,
        environment: str = "local",
    ) -> None:
        self._public_key = public_key
        self._secret_key = secret_key
        self._host = host
        self._environment = environment
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            from langfuse import Langfuse  # type: ignore[import-not-found]
            self._client = Langfuse(
                public_key=self._public_key,
                secret_key=self._secret_key,
                host=self._host,
            )
        return self._client

    def _open_span(
        self,
        client: Any,
        name: str,
        attrs: dict[str, Any],
        *,
        as_type: str = "span",
    ) -> tuple[Any, Any]:
        """Open a current observation so nested spans share one Langfuse trace."""
        starter = getattr(client, "start_as_current_observation", None)
        if starter is not None:
            try:
                cm = starter(
                    name=name,
                    as_type=as_type,
                    metadata=attrs,
                    environment=self._environment,
                )
            except TypeError:
                cm = starter(name=name, as_type=as_type, metadata=attrs)
            span_obj = cm.__enter__()

            def _close() -> None:
                cm.__exit__(None, None, None)

            return span_obj, _close

        try:
            span_obj = client.start_observation(
                name=name,
                as_type=as_type,
                metadata=attrs,
                environment=self._environment,
            )
        except TypeError:
            span_obj = client.start_observation(name=name, as_type=as_type, metadata=attrs)
        return span_obj, span_obj.end

    @asynccontextmanager
    async def span(
        self,
        name: str,
        *,
        attributes: dict[str, Any] | None = None,
        kind: SpanKind | None = None,
    ) -> AsyncIterator[SpanContext]:
        client = self._get_client()
        attrs = dict(attributes or {})
        attrs.setdefault("environment", self._environment)
        # Log view shows the observation name, not metadata.tool.
        tool = attrs.get("tool")
        obs_name = tool.strip() if isinstance(tool, str) and tool.strip() else name
        resolved_kind = kind or (
            "llm" if name.startswith("llm.") else None
        )
        as_type = "generation" if resolved_kind == "llm" or attrs.get("span.kind") == "llm" else "span"
        if resolved_kind == "llm":
            attrs.setdefault("span.kind", "llm")

        span_obj, close = self._open_span(client, obs_name, attrs, as_type=as_type)

        async with trace_span(
            name,
            attributes=attrs,
            kind=resolved_kind,
            trace_id=str(span_obj.trace_id),
            span_id=str(span_obj.id),
        ) as ctx:
            # Prefer Langfuse ids / display name for the buffered snapshot.
            ctx.name = obs_name
            try:
                yield ctx
            finally:
                try:
                    update: dict[str, Any] = {}
                    if ctx.input is not None:
                        update["input"] = ctx.input
                    if ctx.output is not None:
                        update["output"] = ctx.output
                    if ctx.model:
                        update["model"] = ctx.model
                    if ctx.usage:
                        update["usage_details"] = {
                            "input": int(ctx.usage.get("prompt_tokens") or ctx.usage.get("prompt") or 0),
                            "output": int(
                                ctx.usage.get("completion_tokens") or ctx.usage.get("completion") or 0
                            ),
                        }
                    if update:
                        span_obj.update(**update)
                finally:
                    close()

    async def record_metric(
        self,
        name: str,
        value: float,
        *,
        unit: str = "1",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        client = self._get_client()
        attrs = attributes or {}
        active = current_span_context()
        trace_id = (
            (active.trace_id if active is not None else None)
            or attrs.get("agent.trace_id")
            or attrs.get("trace_id")
        )
        kwargs: dict[str, Any] = {
            "name": name,
            "value": value,
            "comment": f"Unit: {unit}",
            "metadata": attrs,
            "environment": self._environment,
        }
        if isinstance(trace_id, str) and trace_id:
            kwargs["trace_id"] = trace_id
        else:
            # Last resort: open a short-lived observation only when no active trace.
            span_obj = client.start_observation(name="metric_record", as_type="span")
            kwargs["trace_id"] = span_obj.trace_id
            try:
                try:
                    client.create_score(**kwargs)
                except TypeError:
                    kwargs.pop("environment", None)
                    kwargs.pop("metadata", None)
                    client.create_score(**kwargs)
            finally:
                span_obj.end()
            return

        try:
            client.create_score(**kwargs)
        except TypeError:
            kwargs.pop("environment", None)
            kwargs.pop("metadata", None)
            client.create_score(**kwargs)

    async def flush(self) -> None:
        """Flush buffered events to Langfuse."""
        if self._client:
            self._client.flush()
