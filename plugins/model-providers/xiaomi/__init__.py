"""Xiaomi MiMo provider profile.

MiMo Chat Completions defaults ``thinking`` to ON when the field is absent
(``thinking.type`` default ``enabled``), so a profile with no reasoning hook
silently ignores the user's thinking toggle: nothing reaches the wire and the
server re-enables thinking on every request. Emit ``extra_body.thinking``
explicitly on every request (the deepseek profile defends the same way).

The wire has no effort levels — only ``thinking.type`` ``enabled``/``disabled``
— so a requested level only decides on or off. All current MiMo models are
thinking-capable, so no per-model gate.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

# UI/session surfaces can report Off from the effort field even when ``enabled``
# is missing or still True (same shape the DeepSeek profile defends against).
_OFF_EFFORTS = frozenset({"none", "false", "disabled"})


class XiaomiProfile(ProviderProfile):
    """Xiaomi MiMo — explicit ``extra_body.thinking`` on every request."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        rc = reasoning_config if isinstance(reasoning_config, dict) else None
        off = bool(rc) and (
            rc.get("enabled") is False
            or str(rc.get("effort") or "").strip().lower() in _OFF_EFFORTS
        )
        # Always emit the toggle: an omitted field defaults to enabled server-side,
        # which is exactly what made the user's "thinking off" silently revert.
        return {"thinking": {"type": "disabled" if off else "enabled"}}, {}


xiaomi = XiaomiProfile(
    name="xiaomi", aliases=("mimo", "xiaomi-mimo"), env_vars=("XIAOMI_API_KEY",),
    base_url="https://api.xiaomimimo.com/v1",
    supports_health_check=False,  # /v1/models returns 401 even with valid key
    supports_vision=True,  # mimo-v2-omni is vision-capable
    supports_vision_tool_messages=False,  # rejects list-type tool content (400 "text is not set")
)

register_provider(xiaomi)
