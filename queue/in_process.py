"""In-process queue provider — executes tasks immediately in the same event loop, for dev/tests."""
import asyncio
import uuid
from collections.abc import Callable, Coroutine
from typing import Any

from ._base import AbstractQueueProvider, EnqueuedTask


class InProcessQueueProvider(AbstractQueueProvider):
    """
    Runs enqueued tasks immediately with asyncio.create_task.

    This provider is intentionally simple — it exists purely so the application
    can run in dev / test mode without a Redis broker.  Tasks execute in the
    current event loop with no retry or dead-letter semantics.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[Any]] = {}

    async def enqueue(
        self,
        func: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
        queue: str = "default",
        countdown: int = 0,
        **kwargs: Any,
    ) -> EnqueuedTask:
        task_id = str(uuid.uuid4())

        async def _run() -> None:
            if countdown > 0:
                await asyncio.sleep(countdown)
            await func(*args, **kwargs)

        loop_task = asyncio.create_task(_run(), name=task_id)
        self._tasks[task_id] = loop_task
        loop_task.add_done_callback(lambda _: self._tasks.pop(task_id, None))

        return EnqueuedTask(task_id=task_id, queue_name=queue)

    async def revoke(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task is None or task.done():
            return False
        task.cancel()
        return True
