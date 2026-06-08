"""Generic Celery + Redis queue adapter.

Designed to be project-agnostic: callers supply the Celery app instance and
the name of the registered task that wraps async callables. This removes any
hard dependency on a specific project's ``celery_app`` module.

Usage::

    from celery import Celery
    from provider_contracts.queue.celery_redis import CeleryRedisQueueProvider

    celery_app = Celery("myproject", broker="redis://localhost:6379/0")

    # Register a generic runner task on your Celery app:
    @celery_app.task(name="myproject.run_async")
    def run_async_task(module: str, qualname: str, args: list, kwargs: dict) -> None:
        import asyncio, importlib
        mod = importlib.import_module(module)
        func = mod
        for part in qualname.split("."):
            func = getattr(func, part)
        asyncio.run(func(*args, **kwargs))

    provider = CeleryRedisQueueProvider(
        celery_app=celery_app,
        task_name="myproject.run_async",
    )
"""
import uuid
from collections.abc import Callable, Coroutine
from typing import Any

from celery import Celery

from ._base import AbstractQueueProvider, EnqueuedTask


class CeleryRedisQueueProvider(AbstractQueueProvider):
    """
    Enqueue tasks via Celery using Redis as the broker.

    Due to Celery's design, async callables are serialised by module + qualname
    and executed inside ``asyncio.run`` on the worker via a registered runner task.

    Args:
        celery_app: A configured :class:`celery.Celery` application instance.
        task_name: Name of the registered Celery task that accepts
            ``(module, qualname, args, kwargs)`` and runs the async callable.
            Defaults to ``"provider_contracts.run_async"``.
    """

    def __init__(
        self,
        celery_app: Celery,
        *,
        task_name: str = "provider_contracts.run_async",
    ) -> None:
        self._app = celery_app
        self._task_name = task_name

    async def enqueue(
        self,
        func: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
        queue: str = "default",
        countdown: int = 0,
        **kwargs: Any,
    ) -> EnqueuedTask:
        task_id = str(uuid.uuid4())
        self._app.send_task(
            self._task_name,
            args=[func.__module__, func.__qualname__, list(args), kwargs],
            task_id=task_id,
            queue=queue,
            countdown=countdown,
        )
        return EnqueuedTask(task_id=task_id, queue_name=queue)

    async def revoke(self, task_id: str) -> bool:
        self._app.control.revoke(task_id, terminate=True)
        return True
