from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Optional

log = logging.getLogger("qwen2api.chat_pool")


class _Entry:
    __slots__ = ("chat_id", "created_at")

    def __init__(self, chat_id: str):
        self.chat_id = chat_id
        self.created_at = time.time()


class ChatIdPool:
    def __init__(
        self,
        client,
        *,
        target_per_account: int = 5,
        ttl_seconds: float = 600,
        default_model: str = "qwen3.6-plus",
    ):
        self._client = client
        self._target = target_per_account
        self._ttl = ttl_seconds
        self._default_model = default_model
        self._queues: dict[str, deque[_Entry]] = {}
        self._lock = asyncio.Lock()
        self._refill_task: Optional[asyncio.Task] = None
        self._shutdown = False

    async def start(self) -> None:
        self._refill_task = asyncio.create_task(self._refill_loop())
        log.info("[ChatIdPool] started (target=%s, ttl=%ss)", self._target, self._ttl)

    async def stop(self) -> None:
        self._shutdown = True
        if self._refill_task:
            self._refill_task.cancel()
            try:
                await self._refill_task
            except (asyncio.CancelledError, Exception):
                pass

    async def acquire(self, email: str, model: str | None = None) -> str | None:
        del model
        if not email:
            return None
        async with self._lock:
            queue = self._queues.get(email)
            if not queue:
                return None
            now = time.time()
            while queue:
                entry = queue.popleft()
                if now - entry.created_at < self._ttl:
                    return entry.chat_id
            return None

    async def _prewarm_one(self, account, model: str) -> None:
        try:
            if not getattr(account, "token", ""):
                return
            chat_id = await self._client.executor.create_chat(account.token, model)
            async with self._lock:
                queue = self._queues.setdefault(account.email, deque())
                queue.append(_Entry(chat_id))
        except Exception as exc:
            err = str(exc) or type(exc).__name__
            log.warning("[ChatIdPool] prewarm failed email=%s: %s", getattr(account, "email", "?"), err)

    async def _refill_loop(self) -> None:
        await asyncio.sleep(1.0)
        while not self._shutdown:
            try:
                await self._refill_once()
            except Exception as exc:
                log.warning("[ChatIdPool] refill error: %s", exc)
            await asyncio.sleep(30.0)

    async def _refill_once(self) -> None:
        pool = getattr(self._client, "account_pool", None)
        if pool is None:
            return
        valid_accounts = [
            account for account in (getattr(pool, "accounts", []) or [])
            if getattr(account, "token", "") and getattr(account, "status_code", "valid") == "valid"
        ]
        for account in valid_accounts:
            async with self._lock:
                queue_size = len(self._queues.get(account.email, []))
            if self._target - queue_size > 0:
                await self._prewarm_one(account, self._default_model)

    async def invalidate(self, email: str, chat_id: str) -> None:
        if not email or not chat_id:
            return
        async with self._lock:
            queue = self._queues.get(email)
            if not queue:
                return
            self._queues[email] = deque(entry for entry in queue if entry.chat_id != chat_id)

    async def flush_account(self, email: str) -> int:
        if not email:
            return 0
        async with self._lock:
            queue = self._queues.get(email)
            if not queue:
                return 0
            count = len(queue)
            self._queues[email] = deque()
            return count
