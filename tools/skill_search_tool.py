#!/usr/bin/env python3
"""Semantic skill search backed by the local Nemotron embedding server.

``skill_search(query, limit)`` embeds the query and the skill catalog
(``name | description``) through ``embedd.service`` (127.0.0.1:18901) and returns
the top-``limit`` skills by cosine similarity. Vectors are cached in-process and
invalidated with the same directory signature the skills catalog uses, so a
skill install/edit is picked up without a restart.

Falls back to a lexical match over ``find_all_skills()`` when the embedding
server is unreachable (keeps the tool useful instead of erroring).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.error
import urllib.request
from typing import Any, Optional
import itertools

logger = logging.getLogger(__name__)

EMBED_URL = os.environ.get("HERMES_EMBED_URL", "http://127.0.0.1:18901")
EMBED_TIMEOUT = float(os.environ.get("HERMES_EMBED_TIMEOUT", "30"))

# Cache: signature -> (list[(name, description, category)], matrix as list[list[float]])
_CACHE_LOCK = threading.Lock()
_CACHE: dict = {}
_CACHE_MAX = 4


def _post_json(path: str, payload: dict, timeout: float = EMBED_TIMEOUT) -> dict:
    req = urllib.request.Request(
        EMBED_URL + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _embed(texts: list[str], role: str) -> list[list[float]]:
    out: list[list[float]] = []
    # chunk to keep requests reasonable (server also chunks, but be polite)
    for i in range(0, len(texts), 64):
        chunk = texts[i : i + 64]
        resp = _post_json("/embed", {"texts": chunk, "role": role})
        out.extend(resp["vectors"])
    return out


def _cosine_topk(query_vec: list[float], matrix: list[list[float]], k: int) -> list[tuple[int, float]]:
    """Top-k indices by dot product (vectors are L2-normalized server-side)."""
    q = query_vec
    scored = []
    for i, v in enumerate(matrix):
        s = 0.0
        for a, b in zip(q, v):
            s += a * b
        scored.append((s, i))
    scored.sort(reverse=True)
    return [(i, s) for s, i in scored[:k]]


def _lexical_fallback(query: str, skills: list[dict], k: int) -> list[dict]:
    """Very small keyword scorer used only when the embedding server is down.
    Handles CJK via character bigrams (a pure [a-z0-9] tokenizer yields nothing for Chinese)."""
    import re

    tokens = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 1}
    cjk = re.findall(r"[\u4e00-\u9fff]", query)
    bigrams = {"".join(p) for p in itertools.pairwise(cjk)} if len(cjk) >= 2 else set()
    scored = []
    for s in skills:
        hay = f"{s['name']} {s.get('description', '')}".lower()
        score = sum(1 for t in tokens if t in hay)
        if bigrams:
            score += sum(1 for b in bigrams if b in hay)
        if score:
            scored.append((score, s))
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:k]]


def _catalog_signature() -> tuple:
    """Reuse the skills catalog's directory signature (mtime + child names)."""
    from tools.skills_tool import _get_disabled_skill_names, _skills_scan_signature, skill_search_dirs

    roots, _ = skill_search_dirs()
    disabled = _get_disabled_skill_names()
    return _skills_scan_signature([d for _t, d in roots], disabled)


def _build_index(signature: tuple) -> tuple[list[dict], list[list[float]]]:
    """Build (skills, vectors). Vectors may be empty when the embedding server is down —
    the caller then falls back to lexical ranking over the same skill list."""
    from tools.skills_tool import find_all_skills_with_plugins

    skills = find_all_skills_with_plugins()
    docs = [f"{s['name']} | {s.get('description', '')}" for s in skills]
    if not docs:
        return [], []
    try:
        vectors = _embed(docs, role="document")
    except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
        logger.warning("skill_search: embedding index unavailable (%s); lexical mode", e)
        return skills, []
    return skills, vectors


def skill_search(query: str, limit: int = 10, task_id: Optional[str] = None) -> str:
    """Semantic top-``limit`` skills for ``query`` (JSON: results with name/desc/category/score)."""
    from tools.registry import tool_error

    if not query or not query.strip():
        return tool_error("query must be a non-empty string", success=False)
    limit = max(1, min(int(limit), 50))
    try:
        signature = _catalog_signature()
        with _CACHE_LOCK:
            entry = _CACHE.get(signature)
        if entry is None:
            skills, matrix = _build_index(signature)
            with _CACHE_LOCK:
                if len(_CACHE) >= _CACHE_MAX:
                    _CACHE.clear()
                _CACHE[signature] = (skills, matrix)
            entry = (skills, matrix)
        skills, matrix = entry
        if not skills:
            return json.dumps(
                {"success": True, "query": query, "results": [], "count": 0,
                 "message": "No skills found."},
                ensure_ascii=False,
            )
        try:
            if not matrix:
                raise RuntimeError("no embedding index")
            qvec = _embed([query], role="query")[0]
            ranked = _cosine_topk(qvec, matrix, limit)
            results = [
                {"name": skills[i]["name"],
                 "description": skills[i].get("description", ""),
                 "category": skills[i].get("category", ""),
                 "score": round(float(s), 4)}
                for i, s in ranked
            ]
            source = "embedding"
        except (urllib.error.URLError, OSError, ValueError, KeyError, RuntimeError) as e:
            logger.warning("skill_search embedding unavailable (%s); lexical fallback", e)
            results = [
                {"name": s["name"], "description": s.get("description", ""),
                 "category": s.get("category", ""), "score": None}
                for s in _lexical_fallback(query, skills, limit)
            ]
            source = "lexical_fallback"
        hint = "Use skill_view(name) to load full content."
        if source == "lexical_fallback" and not results:
            hint = ("Embedding server unavailable and no lexical match (Chinese queries need "
                    "the embedding server). Retry shortly, or call skills_list to browse all skills.")
        return json.dumps(
            {"success": True, "query": query, "results": results,
             "count": len(results), "source": source,
             "total_skills": len(skills),
             "hint": hint},
            ensure_ascii=False,
        )
    except Exception as e:
        return tool_error(str(e), success=False)


SKILL_SEARCH_SCHEMA = {
    "name": "skill_search",
    "description": (
        "Semantic search over available skills by meaning (works for Chinese and English "
        "queries). Returns the top matches with name, description, category and score. "
        "Prefer this over reading the full skills list when you know roughly what you need."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What you are trying to do, in natural language (zh or en).",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default 10, capped at 50).",
            },
        },
        "required": ["query"],
    },
}


def _register() -> None:
    from tools.registry import registry

    registry.register(
        name="skill_search",
        toolset="skills",
        schema=SKILL_SEARCH_SCHEMA,
        handler=lambda args, **kw: skill_search(
            query=args.get("query", ""), limit=args.get("limit", 10), task_id=kw.get("task_id")
        ),
        emoji="🔎",
    )


_register()
