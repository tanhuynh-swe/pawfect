"""Configuration loading and shared paths."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = ROOT / "workspace"
ASSETS = ROOT / "assets"
TOPICS = ROOT / "topics"


def _load_dotenv() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def load_config(path: Path | None = None) -> dict[str, Any]:
    _load_dotenv()
    cfg_path = path or (ROOT / "config.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    WORKSPACE.mkdir(exist_ok=True)
    return cfg


def env(name: str, default: str = "") -> str:
    _load_dotenv()
    return os.environ.get(name, default) or default


def slot_dir(slug: str) -> Path:
    """Per-video working directory."""
    d = WORKSPACE / slug
    d.mkdir(parents=True, exist_ok=True)
    return d
