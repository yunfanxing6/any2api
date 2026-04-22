"""Hot-path account selector — pluggable quota/random strategies."""

import random
from typing import Literal

from app.platform.config.snapshot import get_config

from ..shared.enums import PoolId
from .table import AccountRuntimeTable

_W_HEALTH = 100.0
_W_QUOTA = 25.0
_W_RECENT = 15.0
_W_INFLIGHT = 20.0
_W_FAIL = 4.0
_RECENT_WINDOW_S = 15

_StrategyName = Literal["quota", "random"]
_STRATEGY_NAME: _StrategyName = "quota"


def set_strategy(name: _StrategyName) -> None:
    if name not in ("quota", "random"):
        raise ValueError(f"unknown selection strategy: {name!r}")
    global _STRATEGY_NAME
    _STRATEGY_NAME = name


def current_strategy() -> _StrategyName:
    return _STRATEGY_NAME


def select(
    table: AccountRuntimeTable,
    pool_id: int,
    mode_id: int,
    *,
    exclude_idxs: frozenset[int] | None = None,
    prefer_tag_idxs: set[int] | None = None,
    now_s: int,
) -> int | None:
    if _STRATEGY_NAME == "random":
        return _random_select(
            table,
            pool_id,
            exclude_idxs=exclude_idxs,
            prefer_tag_idxs=prefer_tag_idxs,
            now_s=now_s,
        )
    return _quota_select(
        table,
        pool_id,
        mode_id,
        exclude_idxs=exclude_idxs,
        prefer_tag_idxs=prefer_tag_idxs,
        now_s=now_s,
    )


def select_any(
    table: AccountRuntimeTable,
    pool_id: int,
    *,
    exclude_idxs: frozenset[int] | None = None,
    prefer_tag_idxs: set[int] | None = None,
    now_s: int,
) -> int | None:
    if _STRATEGY_NAME == "random":
        return _random_select(
            table,
            pool_id,
            exclude_idxs=exclude_idxs,
            prefer_tag_idxs=prefer_tag_idxs,
            now_s=now_s,
        )
    return _quota_select_any(
        table,
        pool_id,
        exclude_idxs=exclude_idxs,
        prefer_tag_idxs=prefer_tag_idxs,
        now_s=now_s,
    )


def _quota_select(
    table: AccountRuntimeTable,
    pool_id: int,
    mode_id: int,
    *,
    exclude_idxs: frozenset[int] | None = None,
    prefer_tag_idxs: set[int] | None = None,
    now_s: int,
) -> int | None:
    candidates = table.mode_available.get((pool_id, mode_id))
    if not candidates:
        return None

    reset_col = table._reset_col(mode_id)
    quota_col = table._quota_col(mode_id)
    _maybe_reset_windows(table, candidates, mode_id, reset_col, quota_col, pool_id, now_s)

    working = candidates.copy()
    if exclude_idxs:
        working -= exclude_idxs
    working = {idx for idx in working if int(quota_col[idx]) > 0}
    if not working:
        return None

    if prefer_tag_idxs:
        preferred = working & prefer_tag_idxs
        working = preferred if preferred else working

    return _best(table, working, quota_col, now_s)


def _quota_select_any(
    table: AccountRuntimeTable,
    pool_id: int,
    *,
    exclude_idxs: frozenset[int] | None = None,
    prefer_tag_idxs: set[int] | None = None,
    now_s: int,
) -> int | None:
    candidates = _pool_union(table, pool_id)
    if not candidates:
        return None

    working = candidates.copy()
    if exclude_idxs:
        working -= exclude_idxs
    if not working:
        return None

    if prefer_tag_idxs:
        preferred = working & prefer_tag_idxs
        working = preferred if preferred else working

    return _best_no_quota(table, working, now_s)


def _maybe_reset_windows(
    table: AccountRuntimeTable,
    candidates: set[int],
    mode_id: int,
    reset_col,
    quota_col,
    pool_id: int,
    now_s: int,
) -> None:
    if pool_id != int(PoolId.BASIC):
        return

    from app.control.account.quota_defaults import default_quota_window

    defaults = default_quota_window("basic", mode_id)
    if defaults is None:
        return

    add_back: list[int] = []
    for idx in list(candidates):
        reset_at = reset_col[idx]
        if reset_at == 0 or now_s < reset_at:
            continue
        if int(table.pool_by_idx[idx]) != pool_id:
            continue
        quota_col[idx] = defaults.total
        reset_col[idx] = now_s + defaults.window_seconds
        add_back.append(idx)

    bucket = table.mode_available.get((pool_id, mode_id))
    if bucket is not None:
        bucket.update(add_back)


def _best(table: AccountRuntimeTable, working: set[int], quota_col, now_s: int) -> int | None:
    best_idx = -1
    best_score = -1e18

    health_col = table.health_by_idx
    inflight_col = table.inflight_by_idx
    fail_col = table.fail_count_by_idx
    last_use_col = table.last_use_at_by_idx

    for idx in working:
        quota = int(quota_col[idx])
        if quota <= 0:
            continue
        health = float(health_col[idx])
        inflight = int(inflight_col[idx])
        fails = min(int(fail_col[idx]), 10)
        last_use = int(last_use_col[idx])

        score = health * _W_HEALTH + quota * _W_QUOTA - inflight * _W_INFLIGHT - fails * _W_FAIL
        if last_use > 0:
            age_s = now_s - last_use
            if age_s < _RECENT_WINDOW_S:
                score -= (1.0 - age_s / _RECENT_WINDOW_S) * _W_RECENT

        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx if best_idx >= 0 else None


def _best_no_quota(table: AccountRuntimeTable, working: set[int], now_s: int) -> int | None:
    best_idx = -1
    best_score = -1e18

    health_col = table.health_by_idx
    inflight_col = table.inflight_by_idx
    fail_col = table.fail_count_by_idx
    last_use_col = table.last_use_at_by_idx

    for idx in working:
        health = float(health_col[idx])
        inflight = int(inflight_col[idx])
        fails = min(int(fail_col[idx]), 10)
        last_use = int(last_use_col[idx])

        score = health * _W_HEALTH - inflight * _W_INFLIGHT - fails * _W_FAIL
        if last_use > 0:
            age_s = now_s - last_use
            if age_s < _RECENT_WINDOW_S:
                score -= (1.0 - age_s / _RECENT_WINDOW_S) * _W_RECENT

        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx if best_idx >= 0 else None


def _random_select(
    table: AccountRuntimeTable,
    pool_id: int,
    *,
    exclude_idxs: frozenset[int] | None = None,
    prefer_tag_idxs: set[int] | None = None,
    now_s: int,
) -> int | None:
    candidates = _pool_union(table, pool_id)
    if not candidates:
        return None

    max_inflight = int(get_config("account.selection.max_inflight", 8))
    cooling_col = table.cooling_until_s_by_idx
    inflight_col = table.inflight_by_idx

    working = candidates.copy()
    if exclude_idxs:
        working -= exclude_idxs
    working = {
        idx
        for idx in working
        if int(cooling_col[idx]) <= now_s and int(inflight_col[idx]) < max_inflight
    }
    if not working:
        return None

    if prefer_tag_idxs:
        preferred = working & prefer_tag_idxs
        working = preferred if preferred else working

    return random.choice(tuple(working))


def _pool_union(table: AccountRuntimeTable, pool_id: int) -> set[int]:
    out: set[int] = set()
    for (pid, _mode_id), accounts in table.mode_available.items():
        if pid == pool_id:
            out |= accounts
    return out


__all__ = ["select", "select_any", "set_strategy", "current_strategy"]
