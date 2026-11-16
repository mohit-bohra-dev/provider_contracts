"""JSONL-backed eval store — zero-credential local default."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ._base import (
    AbstractEvalStoreProvider,
    DatasetItem,
    ReviewLabel,
    RunHandle,
    ScoreRecord,
    TraceRecord,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


class JsonlEvalStoreProvider(AbstractEvalStoreProvider):
    """Persists datasets, runs, scores, traces, and reviews under a directory."""

    def __init__(self, directory: str = "./out/eval_store") -> None:
        self._root = Path(directory)
        self._root.mkdir(parents=True, exist_ok=True)
        (self._root / "datasets").mkdir(exist_ok=True)
        (self._root / "runs").mkdir(exist_ok=True)
        (self._root / "scores").mkdir(exist_ok=True)
        (self._root / "traces").mkdir(exist_ok=True)
        (self._root / "reviews").mkdir(exist_ok=True)
        (self._root / "score_configs").mkdir(exist_ok=True)

    def _dataset_path(self, name: str) -> Path:
        safe = name.replace("/", "_")
        return self._root / "datasets" / f"{safe}.jsonl"

    def _append(self, path: Path, obj: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, default=str) + "\n")

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    async def upsert_dataset(
        self,
        name: str,
        *,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        meta_path = self._root / "datasets" / "_meta.jsonl"
        rows = [r for r in self._read_jsonl(meta_path) if r.get("name") != name]
        rows.append(
            {
                "name": name,
                "description": description,
                "metadata": metadata or {},
                "updated_at": _now().isoformat(),
            }
        )
        meta_path.write_text(
            "\n".join(json.dumps(r, default=str) for r in rows) + "\n",
            encoding="utf-8",
        )

    async def upsert_item(self, item: DatasetItem) -> None:
        path = self._dataset_path(item.dataset)
        rows = [r for r in self._read_jsonl(path) if r.get("id") != item.id]
        rows.append(item.model_dump())
        path.write_text(
            "\n".join(json.dumps(r, default=str) for r in rows) + "\n",
            encoding="utf-8",
        )

    async def list_items(
        self,
        dataset: str,
        *,
        status: str | None = None,
    ) -> list[DatasetItem]:
        rows = self._read_jsonl(self._dataset_path(dataset))
        items = [DatasetItem(**r) for r in rows]
        if status is not None:
            items = [i for i in items if i.status == status]
        return items

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
        path = self._root / "runs" / f"{handle.run_id}.json"
        path.write_text(handle.model_dump_json(indent=2), encoding="utf-8")
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
        rec = {
            "run_id": run.run_id,
            "dataset": run.dataset,
            "item_id": item_id,
            "trace_id": trace_id,
            "output": output or {},
            "scores": [s.model_dump() for s in (scores or [])],
            "recorded_at": _now().isoformat(),
            "environment": run.environment,
        }
        self._append(self._root / "runs" / f"{run.run_id}.results.jsonl", rec)
        for score in scores or []:
            score_row = score.model_dump()
            score_row["run_id"] = run.run_id
            score_row["item_id"] = item_id
            if trace_id and not score_row.get("trace_id"):
                score_row["trace_id"] = trace_id
            self._append(self._root / "scores" / "scores.jsonl", score_row)

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
        path = self._root / "score_configs" / f"{key}.json"
        path.write_text(
            json.dumps(
                {
                    "id": key,
                    "name": name,
                    "version": version,
                    "data_type": data_type,
                    "categories": categories or [],
                    "description": description,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return key

    async def fetch_traces(
        self,
        *,
        environment: str,
        since: datetime | None = None,
        name: str | None = None,
        limit: int = 100,
    ) -> list[TraceRecord]:
        path = self._root / "traces" / f"{environment}.jsonl"
        out: list[TraceRecord] = []
        for row in self._read_jsonl(path):
            tr = TraceRecord(**row)
            if name is not None and tr.name != name:
                continue
            if since is not None and tr.start_time is not None and tr.start_time < since:
                continue
            out.append(tr)
            if len(out) >= limit:
                break
        return out

    async def seed_traces(self, traces: list[TraceRecord]) -> None:
        """Test helper: write synthetic traces for L8 fixture runs."""
        by_env: dict[str, list[TraceRecord]] = {}
        for tr in traces:
            by_env.setdefault(tr.environment, []).append(tr)
        for env, items in by_env.items():
            path = self._root / "traces" / f"{env}.jsonl"
            with path.open("a", encoding="utf-8") as fh:
                for tr in items:
                    fh.write(tr.model_dump_json() + "\n")

    async def enqueue_review(self, queue: str, trace_id: str) -> None:
        path = self._root / "reviews" / f"{queue}.queue.jsonl"
        self._append(
            path,
            {"trace_id": trace_id, "enqueued_at": _now().isoformat(), "queue": queue},
        )

    async def fetch_review_labels(self, queue: str) -> list[ReviewLabel]:
        path = self._root / "reviews" / f"{queue}.labels.jsonl"
        return [ReviewLabel(**r) for r in self._read_jsonl(path)]

    async def seed_review_labels(self, labels: list[ReviewLabel]) -> None:
        """Test helper: write synthetic review labels."""
        by_q: dict[str, list[ReviewLabel]] = {}
        for lab in labels:
            by_q.setdefault(lab.queue, []).append(lab)
        for queue, items in by_q.items():
            path = self._root / "reviews" / f"{queue}.labels.jsonl"
            with path.open("a", encoding="utf-8") as fh:
                for lab in items:
                    fh.write(lab.model_dump_json() + "\n")
