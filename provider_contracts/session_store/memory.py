"""In-memory session store provider."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from packages.common.providers.session_store import ConversationSession, ConversationTurn

from packages.common.providers.base import ProviderError
from packages.common.providers.session_store import ConversationSession


class InMemorySessionStoreProvider:
    """In-memory implementation of the SessionStoreProvider.

    Stores sessions in a dictionary and relies on a background asyncio task
    to purge expired sessions. Not meant for multi-process deployments.
    """

    def __init__(self, ttl_minutes: int = 60, max_turns: int = 50) -> None:
        self.ttl_minutes = ttl_minutes
        self.max_turns = max_turns
        self._store: dict[str, ConversationSession] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[None] | None = None
        self._start_cleanup_task()

    def _start_cleanup_task(self) -> None:
        """Start the background task to purge expired sessions."""
        try:
            loop = asyncio.get_running_loop()
            self._cleanup_task = loop.create_task(self._purge_loop())
        except RuntimeError:
            pass  # No running event loop yet (e.g. during sync initialization)

    async def _purge_loop(self) -> None:
        """Periodically purge expired sessions."""
        while True:
            await asyncio.sleep(60)  # Check every minute
            now = datetime.now(UTC)
            async with self._lock:
                expired_keys = [
                    k for k, v in self._store.items() if v.expires_at < now
                ]
                for k in expired_keys:
                    del self._store[k]

    async def create_session(
        self,
        session_id: str,
        loan_id: str | None = None,
        rep_id: str | None = None,
    ) -> ConversationSession:
        """Create a new conversation session."""
        now = datetime.now(UTC)
        expires_at = now + timedelta(minutes=self.ttl_minutes)

        session = ConversationSession(
            session_id=session_id,
            created_at=now,
            last_accessed_at=now,
            expires_at=expires_at,
            loan_id=loan_id,
            rep_id=rep_id,
        )

        async with self._lock:
            if self._cleanup_task is None:
                self._start_cleanup_task()
            self._store[session_id] = session

        return session

    async def get_session(self, session_id: str) -> ConversationSession | None:
        """Retrieve a session by ID, returning None if expired or not found."""
        now = datetime.now(UTC)
        
        async with self._lock:
            session = self._store.get(session_id)
            if session is None:
                return None
            
            if session.expires_at < now:
                del self._store[session_id]
                return None

            # Update last accessed and extend expiration
            session.last_accessed_at = now
            session.expires_at = now + timedelta(minutes=self.ttl_minutes)
            return session

    async def append_turn(self, session_id: str, turn: ConversationTurn) -> None:
        """Append a new turn to an existing session."""
        async with self._lock:
            session = self._store.get(session_id)
            if session is None:
                raise ProviderError(f"Session {session_id} not found or expired")
            
            session.turns.append(turn)
            if len(session.turns) > self.max_turns:
                session.turns = session.turns[-self.max_turns:]
                
            now = datetime.now(UTC)
            session.last_accessed_at = now
            session.expires_at = now + timedelta(minutes=self.ttl_minutes)

    async def delete_session(self, session_id: str) -> None:
        """Delete a session entirely."""
        async with self._lock:
            self._store.pop(session_id, None)

    async def close(self) -> None:
        """Clean up the background task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
