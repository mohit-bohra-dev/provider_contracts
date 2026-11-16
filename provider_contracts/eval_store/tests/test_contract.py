"""Contract tests for EvalStore providers (jsonl + mock)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from provider_contracts.eval_store._base import (
    DatasetItem,
    ReviewLabel,
    ScoreRecord,
    TraceRecord,
)
from provider_contracts.eval_store.jsonl import JsonlEvalStoreProvider
from provider_contracts.eval_store.mock import MockEvalStoreProvider


@pytest.fixture
def jsonl_store(tmp_path: Path) -> JsonlEvalStoreProvider:
    return JsonlEvalStoreProvider(directory=str(tmp_path / "eval_store"))


@pytest.fixture
def mock_store() -> MockEvalStoreProvider:
    return MockEvalStoreProvider()


@pytest.mark.asyncio
@pytest.mark.parametrize("store_fixture", ["jsonl_store", "mock_store"])
async def test_dataset_item_roundtrip(
    store_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    store = request.getfixturevalue(store_fixture)
    await store.upsert_dataset("loanops-l7", description="robustness")
    item = DatasetItem(
        id="g001_para_0",
        dataset="loanops-l7",
        input={"prompt": "escrow jumped"},
        expected_output={"refusal": False},
        metadata={"case_id": "g001"},
        status="approved",
    )
    await store.upsert_item(item)
    items = await store.list_items("loanops-l7", status="approved")
    assert len(items) == 1
    assert items[0].id == "g001_para_0"


@pytest.mark.asyncio
@pytest.mark.parametrize("store_fixture", ["jsonl_store", "mock_store"])
async def test_run_and_scores(
    store_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    store = request.getfixturevalue(store_fixture)
    await store.upsert_dataset("loanops-l7")
    run = await store.start_run("loanops-l7", "repeat-1", environment="eval")
    await store.record_item_result(
        run,
        "g001_para_0",
        scores=[ScoreRecord(name="pass", value=1.0, environment="eval")],
        output={"passed": True},
    )
    assert run.run_id
    key = await store.upsert_score_config(
        "outcome",
        "1",
        categories=["pass", "fail"],
    )
    assert "outcome@v1" in key


@pytest.mark.asyncio
async def test_jsonl_traces_and_reviews(jsonl_store: JsonlEvalStoreProvider) -> None:
    now = datetime.now(timezone.utc)
    await jsonl_store.seed_traces(
        [
            TraceRecord(
                trace_id="t1",
                name="mcp.tool.call",
                environment="prod",
                start_time=now,
                latency_ms=120.0,
                attributes={"tool": "search_docs", "mcp.client_name": "cursor"},
            )
        ]
    )
    traces = await jsonl_store.fetch_traces(environment="prod", name="mcp.tool.call")
    assert len(traces) == 1
    assert traces[0].attributes["tool"] == "search_docs"

    await jsonl_store.enqueue_review("loanops-ops-review", "t1")
    await jsonl_store.seed_review_labels(
        [
            ReviewLabel(
                queue="loanops-ops-review",
                trace_id="t1",
                reviewer="alice",
                label="pass",
                rubric="outcome",
                rubric_version="1",
            )
        ]
    )
    labels = await jsonl_store.fetch_review_labels("loanops-ops-review")
    assert len(labels) == 1
    assert labels[0].reviewer == "alice"


@pytest.mark.asyncio
async def test_mock_traces_filter(mock_store: MockEvalStoreProvider) -> None:
    mock_store.traces.append(
        TraceRecord(
            trace_id="t2",
            name="mcp.tool.call",
            environment="prod",
            latency_ms=50.0,
        )
    )
    mock_store.traces.append(
        TraceRecord(
            trace_id="t3",
            name="other",
            environment="prod",
            latency_ms=50.0,
        )
    )
    got = await mock_store.fetch_traces(environment="prod", name="mcp.tool.call")
    assert [t.trace_id for t in got] == ["t2"]
