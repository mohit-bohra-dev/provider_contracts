"""Abstract base class for task/background-job queue providers."""
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from typing import Any

from pydantic import BaseModel


class EnqueuedTask(BaseModel):
    """Handle returned after a task is enqueued."""

    task_id: str
    queue_name: str
    status: str = "queued"


class AbstractQueueProvider(ABC):
    """
    Contract that every queue adapter must satisfy.

    Adapters must support enqueuing async callables and (optionally) immediate
    in-process execution for testing / dev environments.
    """

    @abstractmethod
    async def enqueue(
        self,
        func: Callable[..., Coroutine[Any, Any, Any]],
        *args: Any,
        queue: str = "default",
        countdown: int = 0,
        **kwargs: Any,
    ) -> EnqueuedTask:
        """
        Enqueue an async callable for background execution.

        Args:
            func: Async callable to execute.
            *args: Positional arguments forwarded to func.
            queue: Target queue name.
            countdown: Delay in seconds before the task is eligible to run.
            **kwargs: Keyword arguments forwarded to func.
        """
        ...

    @abstractmethod
    async def revoke(self, task_id: str) -> bool:
        """
        Cancel a pending / running task.

        Returns:
            True if the task was found and cancelled; False otherwise.
        """
        ...
