from __future__ import annotations

import re

_URL_RE = re.compile(r"https?://[^\s)'\"<>]+", re.IGNORECASE)
_WIN_PATH_RE = re.compile(r"[A-Z]:[\\/](?:[^\s<>'\"|:?*]+[\\/])*[^\s<>'\"|:?*\\/]+", re.IGNORECASE)
_NIX_PATH_RE = re.compile(r"/(?:[\w.-]+/)+[\w.-]+")
_CAMEL_RE = re.compile(r"[a-z][a-z0-9]*(?:[A-Z][a-z0-9]+)+")

_STOPWORDS = {
    "http", "https", "the", "and", "for", "with", "from", "into", "this", "that",
    "给我", "这个", "那个", "然后", "请", "帮我", "需要", "操作",
}


def _extract_entities(text: str) -> set[str]:
    if not text:
        return set()
    entities: set[str] = set()
    for match in _URL_RE.findall(text):
        entities.add(match.rstrip(".,;"))
    for match in _WIN_PATH_RE.findall(text):
        norm = match.replace("\\", "/")
        entities.add(norm)
        entities.add(norm.rsplit("/", 1)[-1])
    for match in _NIX_PATH_RE.findall(text):
        entities.add(match)
        entities.add(match.rsplit("/", 1)[-1])
    for match in _CAMEL_RE.findall(text):
        if match.lower() not in _STOPWORDS:
            entities.add(match)
    for match in re.findall(r"\b[\w-]+\.[a-zA-Z0-9]{1,5}\b", text):
        if match.lower() not in _STOPWORDS:
            entities.add(match)
    return {entity for entity in entities if len(entity) >= 4}


def detect_topic_change(first_user_text: str, last_user_text: str, *, jaccard_threshold: float = 0.1) -> bool:
    if not first_user_text or not last_user_text or first_user_text.strip() == last_user_text.strip():
        return False
    last_entities = _extract_entities(last_user_text)
    first_entities = _extract_entities(first_user_text)
    if not last_entities or not first_entities:
        return False
    union = last_entities | first_entities
    if not union:
        return False
    return len(last_entities & first_entities) / len(union) < jaccard_threshold
