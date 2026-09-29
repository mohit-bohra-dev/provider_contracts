"""Postgres-backed session store (durable conversation history)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

try:
    import asyncpg
except ImportError:
    asyncpg = None  # type: ignore[assignment]

from provider_contracts.session_store._base import (
    AbstractSessionStoreProvider,
    ConversationSession,
    ConversationTurn,
)


class PostgresSessionStoreProvider(AbstractSessionStoreProvider):
    """Durable sessions in PostgreSQL.

    Requires ``asyncpg`` (``provider-contracts[pgvector]`` pulls it in).
    """

    def __init__(
        self,
        dsn: str,
        *,
        ttl_minutes: int = 60,
        max_turns: int = 50,
    ) -> None:
        if asyncpg is None:
            raise ImportError(
                "asyncpg is not installed. Install provider-contracts[pgvector]."
            )
        self._dsn = dsn
        self.ttl_minutes = ttl_minutes
        self.max_turns = max_turns
        self._pool: Any | None = None

    async def _get_pool(self) -> Any:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=4)
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_sessions (
                        session_id TEXT PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL,
                        last_accessed_at TIMESTAMPTZ NOT NULL,
                        expires_at TIMESTAMPTZ NOT NULL,
                        loan_id TEXT,
                        rep_id TEXT,
                        turns JSONB NOT NULL DEFAULT '[]'::jsonb
                    )
                    """
                )
        return self._pool

    def _row_to_session(self, row: Any) -> ConversationSession:
        turns_raw = row["turns"]
        if isinstance(turns_raw, str):
            turns_raw = json.loads(turns_raw)
        turns = [ConversationTurn.model_validate(t) for t in turns_raw]
        return ConversationSession(
            session_id=row["session_id"],
            created_at=row["created_at"],
            last_accessed_at=row["last_accessed_at"],
            expires_at=row["expires_at"],
            loan_id=row["loan_id"],
            rep_id=row["rep_id"],
            turns=turns,
        )

    async def create_session(
        self,
        session_id: str,
        loan_id: str | None = None,
        rep_id: str | None = None,
    ) -> ConversationSession:
        now = datetime.now(UTC)
        expires = now + timedelta(minutes=self.ttl_minutes)
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO conversation_sessions
                    (session_id, created_at, last_accessed_at, expires_at, loan_id, rep_id, turns)
                VALUES ($1, $2, $3, $4, $5, $6, '[]'::jsonb)
                ON CONFLICT (session_id) DO NOTHING
                """,
                session_id,
                now,
                now,
                expires,
                loan_id,
                rep_id,
            )
        return ConversationSession(
            session_id=session_id,
            created_at=now,
            last_accessed_at=now,
            expires_at=expires,
            loan_id=loan_id,
            rep_id=rep_id,
            turns=[],
        )

    async def get_session(self, session_id: str) -> ConversationSession | None:
        now = datetime.now(UTC)
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM conversation_sessions WHERE session_id = $1",
                session_id,
            )
            if row is None:
                return None
            if row["expires_at"] < now:
                await conn.execute(
                    "DELETE FROM conversation_sessions WHERE session_id = $1",
                    session_id,
                )
                return None
            expires = now + timedelta(minutes=self.ttl_minutes)
            await conn.execute(
                """
                UPDATE conversation_sessions
                SET last_accessed_at = $2, expires_at = $3
                WHERE session_id = $1
                """,
                session_id,
                now,
                expires,
            )
            session = self._row_to_session(row)
            session.last_accessed_at = now
            session.expires_at = expires
            return session

    async def append_turn(self, session_id: str, turn: ConversationTurn) -> None:
        session = await self.get_session(session_id)
        if session is None:
            raise RuntimeError(f"Session {session_id} not found or expired")
        session.turns.append(turn)
        if len(session.turns) > self.max_turns:
            session.turns = session.turns[-self.max_turns :]
        now = datetime.now(UTC)
        expires = now + timedelta(minutes=self.ttl_minutes)
        payload = json.dumps([t.model_dump(mode="json") for t in session.turns])
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE conversation_sessions
                SET turns = $2::jsonb, last_accessed_at = $3, expires_at = $4
                WHERE session_id = $1
                """,
                session_id,
                payload,
                now,
                expires,
            )

    async def delete_session(self, session_id: str) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM conversation_sessions WHERE session_id = $1",
                session_id,
            )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
