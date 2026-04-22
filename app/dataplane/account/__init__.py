"""AccountDirectory — high-concurrency hot-path account store."""

import asyncio
from typing import TYPE_CHECKING

from app.platform.config.snapshot import get_config
from app.platform.logging.logger import logger
from app.platform.runtime.clock import now_s
from app.control.account.repository import AccountRepository
from app.control.account.enums import FeedbackKind

from .table import AccountRuntimeTable
from .lease import AccountLease, new_lease
from .selector import current_strategy, select, select_any
from .sync import bootstrap as _bootstrap, apply_changes
from . import feedback as fb
from ..shared.enums import POOL_ID_TO_STR, StatusId

if TYPE_CHECKING:
    pass


class AccountDirectory:
    """High-concurrency, lock-minimal account store for the hot path."""

    def __init__(self, repository: AccountRepository) -> None:
        self._repo = repository
        self._table: AccountRuntimeTable | None = None
        self._lock = asyncio.Lock()
        self._sync_lock = asyncio.Lock()

    async def bootstrap(self) -> None:
        table = await _bootstrap(self._repo)
        async with self._lock:
            self._table = table
        logger.info("account directory ready: size={}", table.size)

    async def sync_if_changed(self) -> bool:
        if self._table is None:
            return False

        async with self._sync_lock:
            async with self._lock:
                table = self._table
            changed = await apply_changes(table, self._repo)
            if changed:
                logger.debug("account directory synced: revision={} size={}", table.revision, table.size)
            return changed

    async def reserve(
        self,
        pool_candidates: tuple[int, ...] | int,
        mode_id: int,
        *,
        exclude_tokens: list[str] | None = None,
        prefer_tags: list[str] | None = None,
        now_s_override: int | None = None,
    ) -> AccountLease | None:
        table = self._table
        if table is None:
            return None

        pools: tuple[int, ...] = (pool_candidates,) if isinstance(pool_candidates, int) else pool_candidates
        ts = now_s_override if now_s_override is not None else now_s()

        exclude_idxs: frozenset[int] | None = None
        if exclude_tokens:
            idxs = [table.idx_by_token[t] for t in exclude_tokens if t in table.idx_by_token]
            if idxs:
                exclude_idxs = frozenset(idxs)

        prefer_tag_idxs: set[int] | None = None
        if prefer_tags:
            sets = [table.tag_idx.get(tag) for tag in prefer_tags if tag in table.tag_idx]
            if sets:
                prefer_tag_idxs = set().union(*sets)

        async with self._lock:
            idx: int | None = None
            for pool_id in pools:
                idx = select(
                    table,
                    pool_id,
                    mode_id,
                    exclude_idxs=exclude_idxs,
                    prefer_tag_idxs=prefer_tag_idxs,
                    now_s=ts,
                )
                if idx is not None:
                    break

            if idx is None:
                return None

            fb.increment_inflight(table, idx)
            fb.update_last_use(table, idx, ts)
            token = table.get_token(idx)
            actual_pool = table.get_pool_id(idx)

        return new_lease(idx=idx, token=token, pool_id=actual_pool, mode_id=mode_id, selected_at=ts)

    async def reserve_any(
        self,
        pool_candidates: tuple[int, ...] | int,
        *,
        exclude_tokens: list[str] | None = None,
        prefer_tags: list[str] | None = None,
        now_s_override: int | None = None,
    ) -> AccountLease | None:
        table = self._table
        if table is None:
            return None

        pools: tuple[int, ...] = (pool_candidates,) if isinstance(pool_candidates, int) else pool_candidates
        ts = now_s_override if now_s_override is not None else now_s()

        exclude_idxs: frozenset[int] | None = None
        if exclude_tokens:
            idxs = [table.idx_by_token[t] for t in exclude_tokens if t in table.idx_by_token]
            if idxs:
                exclude_idxs = frozenset(idxs)

        prefer_tag_idxs: set[int] | None = None
        if prefer_tags:
            sets = [table.tag_idx.get(tag) for tag in prefer_tags if tag in table.tag_idx]
            if sets:
                prefer_tag_idxs = set().union(*sets)

        async with self._lock:
            idx: int | None = None
            for pool_id in pools:
                idx = select_any(
                    table,
                    pool_id,
                    exclude_idxs=exclude_idxs,
                    prefer_tag_idxs=prefer_tag_idxs,
                    now_s=ts,
                )
                if idx is not None:
                    break

            if idx is None:
                return None

            fb.increment_inflight(table, idx)
            fb.update_last_use(table, idx, ts)
            token = table.get_token(idx)
            actual_pool = table.get_pool_id(idx)

        return new_lease(idx=idx, token=token, pool_id=actual_pool, mode_id=-1, selected_at=ts)

    async def release(self, lease: AccountLease) -> None:
        table = self._table
        if table is None:
            return
        async with self._lock:
            fb.decrement_inflight(table, lease.idx)

    async def feedback(
        self,
        token: str,
        kind: FeedbackKind,
        mode_id: int,
        *,
        remaining: int | None = None,
        reset_at_ms: int | None = None,
        now_s_val: int | None = None,
    ) -> None:
        table = self._table
        if table is None:
            return

        idx = table.idx_by_token.get(token)
        if idx is None:
            return

        ts = now_s_val if now_s_val is not None else now_s()
        strategy = current_strategy()

        async with self._lock:
            if kind == FeedbackKind.SUCCESS:
                if strategy == "random":
                    fb.apply_success_random(table, idx)
                else:
                    fb.apply_success_quota(table, idx, mode_id)

            elif kind == FeedbackKind.RATE_LIMITED:
                if strategy == "random":
                    fb.apply_rate_limited_random(table, idx, cooling_sec=_pool_cooling_sec(int(table.pool_by_idx[idx])))
                else:
                    fb.apply_rate_limited_quota(table, idx, mode_id)
                fb.update_last_fail(table, idx, ts)

            elif kind == FeedbackKind.UNAUTHORIZED:
                fb.apply_auth_failure(table, idx)
                fb.update_last_fail(table, idx, ts)
                fb.apply_status_change(table, idx, int(StatusId.EXPIRED))

            elif kind == FeedbackKind.FORBIDDEN:
                fb.apply_forbidden(table, idx)
                fb.update_last_fail(table, idx, ts)

            elif kind == FeedbackKind.SERVER_ERROR:
                fb.apply_server_error(table, idx)
                fb.update_last_fail(table, idx, ts)

            if strategy == "quota" and remaining is not None and reset_at_ms is not None:
                fb.apply_quota_update(table, idx, mode_id, remaining, int(reset_at_ms // 1000))

    @property
    def size(self) -> int:
        return self._table.size if self._table else 0

    @property
    def revision(self) -> int:
        return self._table.revision if self._table else 0


_directory: AccountDirectory | None = None

_POOL_INTERVAL_CONFIG: dict[str, tuple[str, int]] = {
    "basic": ("account.refresh.basic_interval_sec", 36_000),
    "super": ("account.refresh.super_interval_sec", 7_200),
    "heavy": ("account.refresh.heavy_interval_sec", 7_200),
}


def _pool_cooling_sec(pool_id: int) -> int:
    pool_str = POOL_ID_TO_STR.get(pool_id, "basic")
    interval_key, default_interval = _POOL_INTERVAL_CONFIG.get(pool_str, _POOL_INTERVAL_CONFIG["basic"])
    return max(0, int(get_config(interval_key, default_interval)))


async def get_account_directory(repository: AccountRepository | None = None) -> AccountDirectory:
    global _directory
    if _directory is None:
        if repository is None:
            raise RuntimeError("AccountDirectory not bootstrapped — repository required on first call")
        _directory = AccountDirectory(repository)
        await _directory.bootstrap()
    return _directory


__all__ = ["AccountDirectory", "get_account_directory"]
