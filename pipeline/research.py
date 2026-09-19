"""Topic research.

Picks the next video topic by scoring seed topics against real YouTube search
demand (Google's public suggest endpoint, no key needed) and grounds it with
Wikipedia facts so the script is not invented from nothing.

Nothing here scrapes or copies another creator's video. That matters: YouTube's
inauthentic-content policy targets channels that re-narrate existing videos.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import requests
import yaml

from .config import TOPICS, WORKSPACE

SUGGEST_URL = "https://suggestqueries.google.com/complete/search"
WIKI_SUMMARY = "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"
WIKI_EXTRACT = "https://{lang}.wikipedia.org/w/api.php"
HISTORY = WORKSPACE / "used_topics.json"
TIMEOUT = 20


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text[:60] or "video"


def _history() -> list[str]:
    if HISTORY.exists():
        return json.loads(HISTORY.read_text(encoding="utf-8"))
    return []


def mark_used(slug: str) -> None:
    used = _history()
    if slug not in used:
        used.append(slug)
        HISTORY.write_text(json.dumps(used, indent=2), encoding="utf-8")


def youtube_suggestions(query: str) -> list[str]:
    """What people actually type into YouTube. Free, no API key, no quota."""
    try:
        r = requests.get(
            SUGGEST_URL,
            params={"client": "firefox", "ds": "yt", "q": query},
            timeout=TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        r.raise_for_status()
        data = r.json()
        return [s for s in data[1] if isinstance(s, str)]
    except Exception:
        return []


def demand_score(title: str) -> tuple[int, list[str]]:
    """More autocomplete completions == more people searching that phrasing.

    Crude but free and directionally right. A topic with 8 completions is a
    topic people ask about; one with 0 usually is not.
    """
    stem = " ".join(title.lower().replace("?", "").split()[:5])
    sug = youtube_suggestions(stem)
    return len(sug), sug[:8]


def wikipedia_facts(title: str, lang: str = "en", sentences: int = 40) -> str:
    """Grounding material. Used as reference for the writer, never as script text."""
    if not title:
        return ""
    try:
        r = requests.get(
            WIKI_EXTRACT.format(lang=lang),
            params={
                "action": "query",
                "prop": "extracts",
                "explaintext": 1,
                "format": "json",
                "redirects": 1,
                "titles": title,
            },
            timeout=TIMEOUT,
            headers={"User-Agent": "PawfectLoveAnimals/1.0 (research)"},
        )
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract", "")
            if extract:
                parts = re.split(r"(?<=[.!?])\s+", extract)
                return " ".join(parts[:sentences])
    except Exception:
        pass
    return ""


def load_seeds() -> list[dict[str, Any]]:
    path = TOPICS / "seed_topics.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or []


def pick_topic(cfg: dict[str, Any], force_title: str | None = None) -> dict[str, Any]:
    seeds = load_seeds()
    used = set(_history())

    if force_title:
        match = next((s for s in seeds if s["title"] == force_title), None)
        chosen = match or {"title": force_title, "wiki": "", "angle": ""}
    else:
        pool = [s for s in seeds if slugify(s["title"]) not in used]
        if not pool:
            raise SystemExit(
                "Every seed topic has been used. Add more lines to "
                "topics/seed_topics.yaml — ideally ones your comments asked for."
            )
        limit = int(cfg["research"].get("candidates_per_run", 12))
        scored = []
        for seed in pool[:limit]:
            score, suggestions = demand_score(seed["title"])
            scored.append((score, seed, suggestions))
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, chosen, suggestions = scored[0]
        chosen = dict(chosen)
        chosen["demand_score"] = best_score
        chosen["related_searches"] = suggestions

    chosen["slug"] = slugify(chosen["title"])
    chosen["facts"] = wikipedia_facts(
        chosen.get("wiki", ""), cfg["research"].get("wikipedia_lang", "en")
    )
    if not chosen.get("related_searches"):
        _, sug = demand_score(chosen["title"])
        chosen["related_searches"] = sug
    return chosen


def save_brief(topic: dict[str, Any], out_dir: Path) -> Path:
    path = out_dir / "brief.json"
    path.write_text(json.dumps(topic, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
