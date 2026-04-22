"""File content cache for repeated-read hint repair."""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from threading import Lock

_MAX_ENTRIES = 200
_TTL_SECONDS = 900

_lock = Lock()
_store: "OrderedDict[tuple[str, str], tuple[str, float]]" = OrderedDict()

_CACHE_HINT_PATTERNS = (
    re.compile(r"File\s+unchanged\s+since\s+last\s+read", re.IGNORECASE),
    re.compile(r"unchanged\s+since\s+last\s+read", re.IGNORECASE),
    re.compile(r"refer\s+to\s+that\s+instead\s+of\s+re-?reading", re.IGNORECASE),
    re.compile(r"still\s+current\s+[—-]\s+refer\s+to", re.IGNORECASE),
)


def is_cache_hint(text: str) -> bool:
    if not text or len(text) > 500:
        return False
    return any(pattern.search(text) for pattern in _CACHE_HINT_PATTERNS)


def _normalize_path(path: str) -> str:
    if not isinstance(path, str):
        return ""
    return path.strip().replace("\\", "/").lower()


def _prune_expired(now: float) -> None:
    stale = [key for key, (_, ts) in _store.items() if now - ts > _TTL_SECONDS]
    for key in stale:
        _store.pop(key, None)


def put(api_key: str, file_path: str, content: str) -> None:
    if not file_path or not isinstance(content, str) or is_cache_hint(content):
        return
    key = (api_key or "", _normalize_path(file_path))
    now = time.time()
    with _lock:
        _prune_expired(now)
        _store[key] = (content, now)
        _store.move_to_end(key)
        while len(_store) > _MAX_ENTRIES:
            _store.popitem(last=False)


def get(api_key: str, file_path: str) -> str | None:
    if not file_path:
        return None
    key = (api_key or "", _normalize_path(file_path))
    now = time.time()
    with _lock:
        _prune_expired(now)
        entry = _store.get(key)
        if not entry:
            return None
        content, ts = entry
        if now - ts > _TTL_SECONDS:
            _store.pop(key, None)
            return None
        _store.move_to_end(key)
        return content
