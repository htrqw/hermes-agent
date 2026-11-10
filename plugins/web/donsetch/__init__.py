"""DonSeTch local search/fetch plugin — bundled, auto-loaded (keyless, no API key)."""
from __future__ import annotations

from plugins.web.donsetch.provider import DonSeTchWebSearchProvider


def register(ctx) -> None:
    ctx.register_web_search_provider(DonSeTchWebSearchProvider())
