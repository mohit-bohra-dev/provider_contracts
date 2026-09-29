"""In-memory session store provider."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from provider_contracts.session_store._base import (
    AbstractSessionStoreProvider,
    ConversationSession,
    ConversationTurn,
)


class ProviderError(RuntimeError):
    """Raised when a session operation fails."""


class InMemorySessionStoreProvider(AbstractSessionStoreProvider):
    """In-memory session store. Test double / single-process only."""

    def __init__(self, ttl_minutes: int = 60, max_turns: int = 50) -> None:
        self.ttl_minutes = ttl_minutes
        self.max_turns = max_turns
        self._store: dict[str, ConversationSession] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None
        self._start_cleanup_task()

    def _start_cleanup_task(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            self._cleanup_task = loop.create_task(self._purge_loop())
        except RuntimeError:
            pass

    async def _purge_loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            now = datetime.now(UTC)
            async with self._lock:
                expired = [k for k, v in self._store.items() if v.expires_at < now]
                for k in expired:
                    del self._store[k]

    async def create_session(
        self,
        session_id: str,
        loan_id: str | None = None,
        rep_id: str | None = None,
    ) -> ConversationSession:
        now = datetime.now(UTC)
        session = ConversationSession(
            session_id=session_id,
            created_at=now,
            last_accessed_at=now,
            expires_at=now + timedelta(minutes=self.ttl_minutes),
            loan_id=loan_id,
            rep_id=rep_id,
        )
        async with self._lock:
            if self._cleanup_task is None:
                self._start_cleanup_task()
            self._store[session_id] = session
        return session

    async def get_session(self, session_id: str) -> ConversationSession | None:
        now = datetime.now(UTC)
        async with self._lock:
            session = self._store.get(session_id)
            if session is None:
                return None
            if session.expires_at < now:
                del self._store[session_id]
                return None
            session.last_accessed_at = now
            session.expires_at = now + timedelta(minutes=self.ttl_minutes)
            return session

    async def append_turn(self, session_id: str, turn: ConversationTurn) -> None:
        async with self._lock:
            session = self._store.get(session_id)
            if session is None:
                raise ProviderError(f"Session {session_id} not found or expired")
            session.turns.append(turn)
            if len(session.turns) > self.max_turns:
                session.turns = session.turns[-self.max_turns :]
            now = datetime.now(UTC)
            session.last_accessed_at = now
            session.expires_at = now + timedelta(minutes=self.ttl_minutes)

    async def delete_session(self, session_id: str) -> None:
        async with self._lock:
            self._store.pop(session_id, None)

    async def close(self) -> None:
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
