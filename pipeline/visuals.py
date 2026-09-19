"""Stock footage.

Pexels and Pixabay both license their video and photo libraries for commercial
use with no attribution required, and both give free API keys with no card.
That is the entire visual budget: zero.

If a query returns nothing (or you are offline), a neutral generated card is
used so the render never dies halfway through a batch.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import requests

from .config import env

TIMEOUT = 30
UA = {"User-Agent": "PawfectLoveAnimals/1.0"}


def _cache_dir(out_dir: Path) -> Path:
    d = out_dir / "media"
    d.mkdir(exist_ok=True)
    return d


def _download(url: str, dest: Path) -> bool:
    try:
        with requests.get(url, stream=True, timeout=TIMEOUT, headers=UA) as r:
            r.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in r.iter_content(1 << 16):
                    fh.write(chunk)
        return dest.stat().st_size > 10_000
    except Exception as exc:
        print(f"    download failed: {exc}")
        dest.unlink(missing_ok=True)
        return False


def _rotate(items: list, variant: int) -> list:
    """Different shots of the same scene should not be the same clip."""
    if not items:
        return items
    k = variant % len(items)
    return items[k:] + items[:k]


def _pexels(query: str, cfg: dict[str, Any], variant: int = 0) -> str | None:
    key = env("PEXELS_API_KEY")
    if not key:
        return None
    try:
        r = requests.get(
            "https://api.pexels.com/videos/search",
            params={"query": query, "per_page": 12,
                    "orientation": cfg["visuals"]["orientation"], "size": "medium"},
            headers={"Authorization": key, **UA},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        for video in _rotate(r.json().get("videos", []), variant):
            # Measure the long edge, not the width: a 1080x1920 portrait clip
            # is plenty big but would fail a width>=1280 test.
            files = sorted(
                (f for f in video.get("video_files", [])
                 if f.get("width") and f.get("height")
                 and max(f["width"], f["height"]) >= 1280
                 and f.get("file_type") == "video/mp4"),
                key=lambda f: max(f["width"], f["height"]),
            )
            if files:
                return files[0]["link"]
    except Exception as exc:
        print(f"    pexels: {exc}")
    return None


def _pixabay(query: str, cfg: dict[str, Any], variant: int = 0) -> str | None:
    key = env("PIXABAY_API_KEY")
    if not key:
        return None
    try:
        r = requests.get(
            "https://pixabay.com/api/videos/",
            params={"key": key, "q": query, "per_page": 12, "safesearch": "true"},
            timeout=TIMEOUT, headers=UA,
        )
        r.raise_for_status()
        for hit in _rotate(r.json().get("hits", []), variant):
            videos = hit.get("videos", {})
            for size in ("large", "medium", "small"):
                if videos.get(size, {}).get("url"):
                    return videos[size]["url"]
    except Exception as exc:
        print(f"    pixabay: {exc}")
    return None


def _wikimedia(query: str, cfg: dict[str, Any], variant: int = 0) -> str | None:
    """Wikimedia Commons — no API key, no account, no rate limit worth worrying about.

    Everything returned is freely licensed. Quality is below Pexels and the hit
    rate on abstract queries is worse, but it means the pipeline produces a real
    video the first time you run it, with nothing to sign up for.
    """
    try:
        r = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": f"filetype:bitmap {query}",
                "gsrnamespace": 6,
                "gsrlimit": 20,
                "prop": "imageinfo",
                "iiprop": "url|size|mime",
                "iiurlwidth": 1920,
                "format": "json",
            },
            timeout=TIMEOUT,
            headers={"User-Agent": "PawfectLoveAnimals/1.0 (pet care videos)"},
        )
        r.raise_for_status()
        pages = list(r.json().get("query", {}).get("pages", {}).values())
        usable = []
        for page in pages:
            info = (page.get("imageinfo") or [{}])[0]
            mime = info.get("mime", "")
            width = info.get("width") or 0
            if mime in ("image/jpeg", "image/png") and width >= 1280:
                usable.append(info.get("thumburl") or info.get("url"))
        usable = [u for u in usable if u]
        if usable:
            return _rotate(usable, variant)[0]
    except Exception as exc:
        print(f"    wikimedia: {exc}")
    return None


PROVIDERS = {"pexels": _pexels, "pixabay": _pixabay, "wikimedia": _wikimedia}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _fallback_card(query: str, dest: Path, cfg: dict[str, Any], seconds: float) -> Path:
    """Neutral moving gradient. Never the goal, but better than a failed batch."""
    h = int(hashlib.md5(query.encode()).hexdigest()[:6], 16)
    c1 = f"0x{h:06x}"
    c2 = f"0x{(h ^ 0x334455) & 0xFFFFFF:06x}"
    w, hgt, fps = cfg["video"]["width"], cfg["video"]["height"], cfg["video"]["fps"]
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", f"gradients=s={w}x{hgt}:c0={c1}:c1={c2}:speed=0.02:d={seconds + 1}:r={fps}",
         "-t", str(seconds + 1), "-pix_fmt", "yuv420p", str(dest)],
        check=True,
    )
    return dest


def _suffix_for(url: str) -> str:
    lowered = url.lower().split("?")[0]
    for suffix in (".mp4", ".jpg", ".jpeg", ".png"):
        if lowered.endswith(suffix):
            return suffix
    return ".mp4"


STOPWORDS = {
    "a", "an", "the", "of", "on", "in", "at", "with", "and", "to", "for",
    "his", "her", "its", "their", "some", "very", "close", "up", "shot",
}


def _query_ladder(query: str, cfg: dict[str, Any]) -> list[str]:
    """Progressively simpler searches.

    Wikimedia Commons matches titles and categories, not scene descriptions, so
    "dog chewing object floor" finds nothing while "dog" finds thousands. Try
    the specific phrasing first — it gives the better picture when it hits —
    then fall back toward the plain noun rather than straight to a blank card.
    """
    words = [w for w in query.lower().split() if w not in STOPWORDS]
    ladder = []
    if query.strip():
        ladder.append(query.strip())
    if len(words) > 2:
        ladder.append(" ".join(words[:2]))
    if len(words) > 1:
        # The subject is usually the noun the rest describes.
        for animal in ("dog", "puppy", "cat", "kitten", "wolf", "vet"):
            if animal in words:
                ladder.append(animal)
                break
        else:
            ladder.append(words[0])
    ladder.append(cfg["visuals"]["fallback_query"])
    seen, out = set(), []
    for q in ladder:
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


def fetch_clip(query: str, index: int, seconds: float, cfg: dict[str, Any],
               out_dir: Path) -> Path:
    """Returns a video OR a still image; render.py handles both."""
    media = _cache_dir(out_dir)
    # Real media is cached; generated placeholder cards are not, so a rerun
    # retries only the scenes that failed and keeps everything that worked.
    if not cfg.get("_refresh"):
        # Newest wins. A --refresh can leave an older file beside the new one
        # (a Wikimedia .jpg next to a Pexels .mp4), and alphabetical order
        # would silently hand back the stale one on the next ordinary run.
        cached = [p for p in media.glob(f"{index:03d}.*")
                  if not p.name.endswith(".card.mp4")]
        if cached:
            return max(cached, key=lambda p: p.stat().st_mtime)

    for q in _query_ladder(query, cfg):
        for name in cfg["visuals"]["providers"]:
            provider = PROVIDERS.get(name)
            if not provider:
                continue
            url = provider(q, cfg, index)
            if not url:
                continue
            dest = media / f"{index:03d}{_suffix_for(url)}"
            if _download(url, dest):
                print(f"    scene {index}: {name} — '{q}'")
                return dest

    print(f"    scene {index}: no match for '{query}', using generated card")
    return _fallback_card(query or "pet", media / f"{index:03d}.card.mp4", cfg, seconds)
