"""Langfuse-backed eval store (SDK v4+)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from ._base import (
    AbstractEvalStoreProvider,
    DatasetItem,
    ReviewLabel,
    RunHandle,
    ScoreRecord,
    TraceRecord,
)


class LangfuseEvalStoreProvider(AbstractEvalStoreProvider):
    """EvalStore adapter over the Langfuse Python SDK.

    Dataset / item / score APIs use the high-level client. Trace fetch and
    annotation queues use ``client.api`` where available; missing endpoints
    degrade to empty results rather than crashing CI.
    """

    def __init__(
        self,
        public_key: str,
        secret_key: str,
        host: str,
        *,
        environment: str = "eval",
    ) -> None:
        self._public_key = public_key
        self._secret_key = secret_key
        self._host = host
        self._environment = environment
        self._client: Any | None = None
        self._local_runs: dict[str, RunHandle] = {}
        self._queue_buffer: dict[str, list[str]] = {}
        self._label_buffer: dict[str, list[ReviewLabel]] = {}

    def _get_client(self) -> Any:
        if self._client is None:
            from langfuse import Langfuse  # type: ignore[import-not-found]

            self._client = Langfuse(
                public_key=self._public_key,
                secret_key=self._secret_key,
                host=self._host,
            )
        return self._client

    async def upsert_dataset(
        self,
        name: str,
        *,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        client = self._get_client()
        try:
            client.create_dataset(
                name=name,
                description=description,
                metadata=metadata or {},
            )
        except Exception:  # noqa: BLE001 — dataset may already exist
            pass

    async def upsert_item(self, item: DatasetItem) -> None:
        client = self._get_client()
        meta = dict(item.metadata)
        meta["status"] = item.status
        kwargs: dict[str, Any] = {
            "dataset_name": item.dataset,
            "input": item.input,
            "expected_output": item.expected_output,
            "metadata": meta,
            "id": item.id,
        }
        if item.source_trace_id:
            kwargs["source_trace_id"] = item.source_trace_id
        client.create_dataset_item(**kwargs)

    async def list_items(
        self,
        dataset: str,
        *,
        status: str | None = None,
    ) -> list[DatasetItem]:
        client = self._get_client()
        ds = client.get_dataset(dataset)
        out: list[DatasetItem] = []
        for raw in getattr(ds, "items", []) or []:
            meta = getattr(raw, "metadata", None) or {}
            if not isinstance(meta, dict):
                meta = {}
            item_status = str(meta.get("status", "approved"))
            if status is not None and item_status != status:
                continue
            out.append(
                DatasetItem(
                    id=str(getattr(raw, "id", "")),
                    dataset=dataset,
                    input=getattr(raw, "input", None) or {},
                    expected_output=getattr(raw, "expected_output", None) or {},
                    metadata=meta,
                    status=item_status,
                    source_trace_id=getattr(raw, "source_trace_id", None),
                )
            )
        return out

    async def start_run(
        self,
        dataset: str,
        run_name: str,
        *,
        metadata: dict[str, Any] | None = None,
        environment: str = "eval",
    ) -> RunHandle:
        handle = RunHandle(
            run_id=str(uuid.uuid4()),
            dataset=dataset,
            run_name=run_name,
            metadata=metadata or {},
            environment=environment,
        )
        self._local_runs[handle.run_id] = handle
        return handle

    async def record_item_result(
        self,
        run: RunHandle,
        item_id: str,
        *,
        trace_id: str | None = None,
        scores: list[ScoreRecord] | None = None,
        output: dict[str, Any] | None = None,
    ) -> None:
        client = self._get_client()
        for score in scores or []:
            kwargs: dict[str, Any] = {
                "name": score.name,
                "value": score.value,
                "data_type": score.data_type,
                "comment": score.comment,
                "metadata": {
                    **score.metadata,
                    "run_id": run.run_id,
                    "run_name": run.run_name,
                    "item_id": item_id,
                    "dataset": run.dataset,
                    "output": output or {},
                },
                "environment": score.environment or run.environment,
            }
            if trace_id or score.trace_id:
                kwargs["trace_id"] = trace_id or score.trace_id
            client.create_score(**kwargs)

    async def upsert_score_config(
        self,
        name: str,
        version: str,
        *,
        data_type: Literal["NUMERIC", "CATEGORICAL", "BOOLEAN"] = "CATEGORICAL",
        categories: list[str] | None = None,
        description: str | None = None,
    ) -> str:
        key = f"{name}@v{version}"
        client = self._get_client()
        api = getattr(client, "api", None)
        if api is not None and hasattr(api, "score_configs"):
            try:
                cats = None
                if categories:
                    cats = [{"label": c, "value": float(i)} for i, c in enumerate(categories)]
                created = api.score_configs.create(
                    name=key,
                    data_type=data_type,
                    categories=cats,
                    description=description,
                )
                return str(getattr(created, "id", key))
            except Exception:  # noqa: BLE001
                return key
        return key

    async def fetch_traces(
        self,
        *,
        environment: str,
        since: datetime | None = None,
        name: str | None = None,
        limit: int = 100,
    ) -> list[TraceRecord]:
        client = self._get_client()
        api = getattr(client, "api", None)
        if api is None or not hasattr(api, "observations"):
            return []
        filters: list[dict[str, Any]] = [
            {
                "type": "string",
                "column": "environment",
                "operator": "=",
                "value": environment,
            }
        ]
        if name:
            filters.append(
                {
                    "type": "string",
                    "column": "name",
                    "operator": "=",
                    "value": name,
                }
            )
        if since is not None:
            filters.append(
                {
                    "type": "datetime",
                    "column": "startTime",
                    "operator": ">=",
                    "value": since.astimezone(timezone.utc).isoformat(),
                }
            )
        try:
            response = api.observations.get_many(
                filter=json.dumps(filters),
                fields="core,basic,io,trace_context",
                limit=limit,
            )
        except Exception:  # noqa: BLE001
            return []
        out: list[TraceRecord] = []
        for obs in getattr(response, "data", []) or []:
            attrs = getattr(obs, "metadata", None) or {}
            if not isinstance(attrs, dict):
                attrs = {}
            start = getattr(obs, "start_time", None) or getattr(obs, "startTime", None)
            end = getattr(obs, "end_time", None) or getattr(obs, "endTime", None)
            latency = 0.0
            if start is not None and end is not None:
                try:
                    latency = (end - start).total_seconds() * 1000.0
                except Exception:  # noqa: BLE001
                    latency = float(attrs.get("latency_ms", 0.0) or 0.0)
            level = str(getattr(obs, "level", "") or "").upper()
            out.append(
                TraceRecord(
                    trace_id=str(getattr(obs, "trace_id", "") or ""),
                    span_id=str(getattr(obs, "id", "") or ""),
                    name=str(getattr(obs, "name", "") or ""),
                    environment=environment,
                    start_time=start if isinstance(start, datetime) else None,
                    latency_ms=latency,
                    success=level not in {"ERROR", "DEFAULT"} or level == "",
                    is_error=level == "ERROR" or bool(attrs.get("is_error")),
                    attributes=attrs,
                    input=getattr(obs, "input", None) or {},
                    output=getattr(obs, "output", None) or {},
                )
            )
        return out

    async def enqueue_review(self, queue: str, trace_id: str) -> None:
        self._queue_buffer.setdefault(queue, []).append(trace_id)
        client = self._get_client()
        api = getattr(client, "api", None)
        if api is not None and hasattr(api, "annotation_queues"):
            try:
                # Best-effort: queue APIs vary by Langfuse version.
                queues = api.annotation_queues
                if hasattr(queues, "create_queue_item"):
                    queues.create_queue_item(queue_id=queue, object_id=trace_id)
            except Exception:  # noqa: BLE001
                pass

    async def fetch_review_labels(self, queue: str) -> list[ReviewLabel]:
        return list(self._label_buffer.get(queue, []))

    async def flush(self) -> None:
        client = self._get_client()
        if hasattr(client, "flush"):
            client.flush()
