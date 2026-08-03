"""Persistent settings: configure once, reuse every session.

* Settings file at the platform-standard location via ``platformdirs``:
  ``~/.config/my-agent-framework/config.json`` on Linux,
  ``~/Library/Application Support/my-agent-framework/config.json`` on macOS,
  ``%APPDATA%\\my-agent-framework\\config.json`` on Windows.
* API keys are **never** written to the JSON file. They go through the OS
  credential manager via ``keyring`` (Windows Credential Manager, macOS
  Keychain, Secret Service on Linux). If keyring is unavailable, the framework
  refuses to store the secret and tells the user to use an environment
  variable instead -- it will not fall back to plain text.
* Every setting can be overridden by an environment variable:
  ``MY_AGENT_MODEL``, ``MY_AGENT_HYBRID_MODE``, ``MY_AGENT_FALLBACK_MODEL``, ...
  (``MY_AGENT_`` + upper-cased key with ``-``/``.`` -> ``_``).

Nothing in this module performs network I/O.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("my_agent_framework.config")

ENV_PREFIX = "MY_AGENT_"
KEYRING_SERVICE = "my-agent-framework"

DEFAULTS: dict[str, Any] = {
    "model": "ollama:llama3.1:8b",
    "fallback-model": None,          # e.g. "anthropic:claude-haiku-4-5"
    "hybrid-mode": False,            # local-first with cloud fallback
    "size-class": None,              # auto-detect when null
    "context-window": None,          # profile default when null
    "max-tools": None,
    "max-iterations": None,
    "tool-timeout-s": 30,
    "session-timeout-s": 300,
    "audit-enabled": True,
    "audit-root": None,              # platform data dir when null
    "confirm-at-or-above": "write",  # "read_only" | "write" | "destructive" | null
    "dry-run": False,
    "user": None,
    "role": None,
}

_SECRET_KEYS = {"openai-api-key", "anthropic-api-key", "custom-api-key"}


def config_dir() -> Path:
    from platformdirs import user_config_dir  # lazy, offline

    return Path(user_config_dir("my-agent-framework", appauthor=False))


def config_path() -> Path:
    return config_dir() / "config.json"


def _env_name(key: str) -> str:
    return ENV_PREFIX + key.upper().replace("-", "_").replace(".", "_")


class Settings:
    """Load/save/query framework settings with env-var overrides."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or config_path()
        self._data: dict[str, Any] = {}
        self.load()

    # -- file I/O ----------------------------------------------------------

    def load(self) -> None:
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.exception("Could not read %s; using defaults.", self.path)
                self._data = {}
        else:
            self._data = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        try:  # best-effort: restrict to owner on POSIX
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    # -- access ------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        env = os.environ.get(_env_name(key))
        if env is not None:
            return _coerce(env)
        if key in self._data:
            return self._data[key]
        if key in DEFAULTS:
            return DEFAULTS[key]
        return default

    def set(self, key: str, value: Any) -> None:
        if key in _SECRET_KEYS:
            self.set_secret(key, str(value))
            return
        self._data[key] = _coerce(value) if isinstance(value, str) else value
        self.save()
        logger.info("Set %s = %r in %s", key, self._data[key], self.path)

    def unset(self, key: str) -> None:
        self._data.pop(key, None)
        self.save()

    def reset(self) -> None:
        self._data = {}
        if self.path.exists():
            self.path.unlink()
        logger.info("Settings reset; %s removed.", self.path)

    def as_dict(self, include_defaults: bool = True) -> dict[str, Any]:
        merged = dict(DEFAULTS) if include_defaults else {}
        merged.update(self._data)
        for key in list(merged):
            env = os.environ.get(_env_name(key))
            if env is not None:
                merged[key] = _coerce(env)
        return merged

    # -- secrets (OS credential manager only) ------------------------------

    def set_secret(self, key: str, value: str) -> None:
        try:
            import keyring  # lazy import; optional dependency
        except ImportError as exc:
            raise RuntimeError(
                "Storing API keys requires the 'keyring' package "
                "(pip install my-agent-framework[keyring]). This framework will "
                "not write secrets to plain-text config. Alternatively set the "
                f"environment variable {_env_name(key)}."
            ) from exc
        keyring.set_password(KEYRING_SERVICE, key, value)
        logger.info("Stored secret %r in the OS credential manager.", key)

    def get_secret(self, key: str) -> Optional[str]:
        env = os.environ.get(_env_name(key))
        if env:
            return env
        try:
            import keyring
        except ImportError:
            return None
        try:
            return keyring.get_password(KEYRING_SERVICE, key)
        except Exception:
            logger.exception("keyring lookup failed for %r", key)
            return None

    def delete_secret(self, key: str) -> None:
        try:
            import keyring

            keyring.delete_password(KEYRING_SERVICE, key)
        except Exception:
            pass


def _coerce(value: str) -> Any:
    """Coerce env-var / CLI strings to bool/int/float/null where obvious."""
    lowered = value.strip().lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("null", "none", ""):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


# ---------------------------------------------------------------------------
# Agent construction from settings
# ---------------------------------------------------------------------------

def agent_from_settings(
    settings: Optional[Settings] = None,
    tools: Any = None,
    **overrides: Any,
):
    """Build a :class:`~my_agent_framework.agent.SmallModelAgent` from stored
    settings (plus keyword overrides). Used by the CLI and handy in scripts."""
    from .agent import SmallModelAgent  # noqa: PLC0415 (avoid cycle)
    from .models import HybridChatModel
    from .safety import PermissionPolicy, Severity

    s = settings or Settings()

    model_spec: str = overrides.pop("model", None) or s.get("model")
    provider = model_spec.split(":", 1)[0].lower()
    model_kwargs: dict[str, Any] = {}
    secret = s.get_secret(f"{provider}-api-key")
    if secret:
        model_kwargs["api_key"] = secret

    llm: Any = model_spec
    if s.get("hybrid-mode") and s.get("fallback-model"):
        fb_spec = s.get("fallback-model")
        fb_provider = fb_spec.split(":", 1)[0].lower()
        fb_kwargs: dict[str, Any] = {}
        fb_secret = s.get_secret(f"{fb_provider}-api-key")
        if fb_secret:
            fb_kwargs["api_key"] = fb_secret
        llm = HybridChatModel(
            primary_spec=model_spec, fallback_spec=fb_spec,
            primary_kwargs=model_kwargs, fallback_kwargs=fb_kwargs,
        )
        model_kwargs = {}

    confirm = s.get("confirm-at-or-above")
    policy = overrides.pop("policy", None) or PermissionPolicy(
        confirm_at_or_above=Severity(confirm) if confirm else None,
        dry_run=bool(s.get("dry-run")),
    )

    kwargs: dict[str, Any] = dict(
        size_class=s.get("size-class"),
        context_window=s.get("context-window"),
        max_tools=s.get("max-tools"),
        max_iterations=s.get("max-iterations"),
        tool_timeout_s=float(s.get("tool-timeout-s")),
        session_timeout_s=float(s.get("session-timeout-s")),
        audit_enabled=bool(s.get("audit-enabled")),
        user=s.get("user") or os.environ.get("USER")
        or os.environ.get("USERNAME") or "unknown",
        role=s.get("role") or "unknown",
        model_kwargs=model_kwargs,
    )
    audit_root = s.get("audit-root")
    if audit_root:
        kwargs["audit_root"] = Path(audit_root)
    kwargs.update(overrides)
    return SmallModelAgent(llm, tools, policy=policy, **kwargs)
