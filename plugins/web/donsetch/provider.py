"""DonSeTch local search + fetch provider (keyless + BYOK, single Rust process).

Wires DonSeTch's CLI into Hermes' native web backend registry so it can be set
as ``web.search_backend`` / ``web.extract_backend: donsetch``.

- ``search``: ``donsetch search --json`` → normalizes ``meta.results[]``
  (title/url/snippet) to the Hermes ``search_ok`` contract. The BYOK chain
  (exa → tavily → local keyless) and all fallback logic live inside DonSeTch.
- ``extract``: ``donsetch fetch --json`` → clean markdown, batched (≤12/CLI call).

Env: ``DONSETCH_BIN`` (path to the donsetch executable; defaults to ``donsetch``
on PATH). No API key needed — DonSeTch holds its own key store.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import Any

from plugins.web._common import (
    BaseWebSearchProvider,
    document,
    provider_env,
    run_extract,
    run_search,
    search_fail,
    search_ok,
    setup_schema,
    titled_rows,
)

logger = logging.getLogger(__name__)

_DEFAULT_FETCH_CHARS = 16000
_MAX_FETCH_BATCH = 12


def _donsetch_bin() -> str:
    return provider_env("DONSETCH_BIN") or shutil.which("donsetch") or ""


def _proxy_env() -> dict[str, str]:
    """Pass the process proxy through so DonSeTch's egress works behind a TUN/proxy."""
    env = os.environ.copy()
    # DonSeTch reads donsetch.toml; also honour ambient proxy vars if set.
    return env


def _run(args: list[str], timeout: int = 60) -> tuple[bool, str, str]:
    """Run the donsetch CLI; returns (ok, stdout, stderr)."""
    bin_path = _donsetch_bin()
    if not bin_path:
        return False, "", "donsetch binary not found (set DONSETCH_BIN or install donsetch)"
    try:
        p = subprocess.run(
            [bin_path, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_proxy_env(),
        )
    except subprocess.TimeoutExpired:
        return False, "", f"donsetch timed out after {timeout}s"
    except Exception as exc:
        return False, "", f"donsetch launch failed: {exc}"
    return p.returncode == 0, p.stdout, p.stderr


class DonSeTchWebSearchProvider(BaseWebSearchProvider):
    """Local DonSeTch: keyless multi-engine search + BYOK (exa/tavily) with auto-fallback."""

    NAME = "donsetch"
    DISPLAY_NAME = "DonSeTch"
    KEY_ENV = "DONSETCH_BIN"
    EXTRACT = True
    KEYLESS = True

    def is_available(self) -> bool:
        """DonSeTch needs no API key; availability = the binary exists."""
        return bool(_donsetch_bin())

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        def _body() -> dict[str, Any]:
            ok, out, err = _run(
                ["search", query, "--json", "--max-results", str(max(1, min(limit, 12)))],
                timeout=60,
            )
            if not ok and not out:
                return search_fail(f"DonSeTch search failed: {err.strip()[:300]}")
            try:
                data = json.loads(out)
            except json.JSONDecodeError:
                return search_fail(f"DonSeTch returned non-JSON output: {(out or err)[:200]}")
            if not data.get("ok"):
                e = data.get("error", {}) or {}
                return search_fail(
                    f"DonSeTch search error [{e.get('code', 'error')}]: {str(e.get('message', ''))[:250]}"
                )
            meta = data.get("meta", {})
            raw = meta.get("results", [])[:limit]
            # DonSeTch uses "snippet"; titled_rows expects "description".
            rows = [dict(r, description=r.get("snippet", "")) for r in raw]
            if not rows:
                return search_fail("DonSeTch returned 0 results")
            logger.info(
                "DonSeTch search '%s': %d results (provider=%s, %sms)",
                query, len(rows), meta.get("provider"), meta.get("elapsed_ms"),
            )
            return search_ok(titled_rows(rows, "description"))

        return run_search("DonSeTch", logger, _body)

    def extract(self, urls: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        def _fetch_one(url: str) -> dict[str, Any]:
            ok, out, err = _run(
                ["fetch", url, "--json", "--max-chars", str(_DEFAULT_FETCH_CHARS)],
                timeout=60,
            )
            if not ok and not out:
                return {"url": url, "title": "", "content": "", "error": err.strip()[:300]}
            try:
                data = json.loads(out)
            except json.JSONDecodeError:
                return {"url": url, "title": "", "content": "", "error": "non-JSON output"}
            if not data.get("ok"):
                e = data.get("error", {}) or {}
                return {
                    "url": url,
                    "title": "",
                    "content": "",
                    "error": f"[{e.get('code', 'error')}] {str(e.get('message', ''))[:250]}",
                }
            content = data.get("content", "") or ""
            # First markdown H1 (or "Title:" line) as the page title.
            title = ""
            for line in content.splitlines():
                s = line.strip()
                if s.startswith("# "):
                    title = s[2:].strip()
                    break
                if s.startswith("Title:"):
                    title = s[6:].strip()
                    break
            return document(url, title, content)

        results: list[dict[str, Any]] = []
        # Batch up to _MAX_FETCH_BATCH URLs per CLI call (DonSeTch reads them in parallel).
        batch = list(urls)
        while batch:
            chunk = batch[:_MAX_FETCH_BATCH]
            batch = batch[_MAX_FETCH_BATCH:]
            if len(chunk) == 1:
                results.append(_fetch_one(chunk[0]))
                continue
            ok, out, err = _run(["fetch", *chunk, "--json"], timeout=90)
            if not ok and not out:
                results.extend(
                    {"url": u, "title": "", "content": "", "error": err.strip()[:300]}
                    for u in chunk
                )
                continue
            try:
                data = json.loads(out)
            except json.JSONDecodeError:
                results.extend(
                    {"url": u, "title": "", "content": "", "error": "non-JSON output"}
                    for u in chunk
                )
                continue
            # Batch fetch returns one envelope; map content per URL when present.
            items = data.get("results") or data.get("pages") or []
            if items:
                for it in items:
                    u = it.get("url", "")
                    results.append(
                        document(u, it.get("title", ""), it.get("content", "") or "")
                    )
            else:
                # Single-content envelope: attribute to the first URL in the chunk.
                c = data.get("content", "")
                results.append(document(chunk[0], "", c) if data.get("ok")
                                else {"url": chunk[0], "title": "", "content": "",
                                      "error": str((data.get("error") or {}).get("message", ""))[:250]})
                results.extend(
                    {"url": u, "title": "", "content": "", "error": "not returned in batch"}
                    for u in chunk[1:]
                )
        return results

    def get_setup_schema(self) -> dict[str, Any]:
        return setup_schema(
            "DonSeTch",
            "free · local",
            "Keyless multi-engine search + BYOK (exa/tavily) with automatic fallback. Local Rust process, no account.",
            "DONSETCH_BIN",
            "Path to the donsetch executable (defaults to `donsetch` on PATH)",
            "https://github.com/dondai44423/donsetch",
        )
