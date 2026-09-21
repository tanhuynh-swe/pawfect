"""Stock footage.

Pexels and Pixabay both license their video and photo libraries for commercial
use with no attribution required, and both give free API keys with no card.
That is the entire visual budget: zero.

Two things matter more than the provider list:

  Relevance. The script hands each scene several queries, one per shot, in the
  order the words are spoken (see `script.py`). This module searches the query
  belonging to the shot it is filling, so shot three shows what sentence three
  is talking about instead of another angle on the scene's opening line.

  Tone. The channel is about animals caught mid-absurdity, so when a specific
  query finds nothing the ladder falls back through a playful phrasing of the
  subject before it falls back to the bare noun. A generic filler shot should
  still be a cat doing something ridiculous, not a stock-photo cat sitting.

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


def _used(cfg: dict[str, Any]) -> set[str]:
    """Source ids already spent in this build.

    Rotating the result list stops two shots of one scene colliding, but a
    video ten scenes long kept coming back to the same handful of popular
    clips, which is what makes a stock-footage video look assembled rather
    than shot. Nothing is used twice now, across the whole video.
    """
    return cfg.setdefault("_used_media", set())


def _best_video_file(files: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Pick the sharpest file that is not pointlessly huge.

    Measure the long edge, not the width: a 1080x1920 portrait clip is plenty
    big but would fail a width>=1280 test. Anything under 1280 is refused
    outright — upscaled to 1080p it reads as a screen recording from 2011.
    """
    usable = [
        f for f in files
        if f.get("file_type") == "video/mp4" and f.get("width") and f.get("height")
    ]
    if not usable:
        return None
    vis = cfg["visuals"]
    floor = int(vis.get("min_width", 1920))
    ceiling = int(vis.get("max_width", 2560))
    edge = lambda f: max(f["width"], f["height"])  # noqa: E731

    within = [f for f in usable if floor <= edge(f) <= ceiling]
    if within:
        return max(within, key=edge)
    above = [f for f in usable if edge(f) > ceiling]
    if above:
        # A 4K download is slow but still better than dropping the clip.
        return min(above, key=edge)
    best = max(usable, key=edge)
    return best if edge(best) >= 1280 else None


def _pexels(query: str, cfg: dict[str, Any], variant: int = 0,
            seconds: float = 6.0) -> str | None:
    key = env("PEXELS_API_KEY")
    if not key:
        return None
    seen = _used(cfg)
    try:
        r = requests.get(
            "https://api.pexels.com/videos/search",
            params={
                "query": query,
                "per_page": 24,
                "orientation": cfg["visuals"]["orientation"],
                # Asking for footage at least as long as the shot keeps the
                # loop seam out of the cut.
                "min_duration": max(3, int(seconds)),
            },
            headers={"Authorization": key, **UA},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        for video in _rotate(r.json().get("videos", []), variant):
            tag = f"pexels:{video.get('id')}"
            if tag in seen:
                continue
            chosen = _best_video_file(video.get("video_files", []), cfg)
            if chosen:
                seen.add(tag)
                return chosen["link"]
    except Exception as exc:
        print(f"    pexels: {exc}")
    return None


def _pixabay(query: str, cfg: dict[str, Any], variant: int = 0,
             seconds: float = 6.0) -> str | None:
    key = env("PIXABAY_API_KEY")
    if not key:
        return None
    seen = _used(cfg)
    params = {"key": key, "q": query, "per_page": 24, "safesearch": "true"}
    if cfg["visuals"].get("prefer_recent", True):
        # Pixabay's popular ordering is weighted by lifetime downloads, which
        # keeps surfacing footage uploaded a decade ago. Recent uploads look
        # like the phones and homes the audience owns now.
        params["order"] = "latest"
    try:
        r = requests.get(
            "https://pixabay.com/api/videos/", params=params,
            timeout=TIMEOUT, headers=UA,
        )
        r.raise_for_status()
        hits = _rotate(r.json().get("hits", []), variant)
        # A clip shorter than the shot has to loop, so prefer the ones that
        # don't, without refusing the short ones outright.
        hits.sort(key=lambda h: (h.get("duration") or 0) < seconds)
        for hit in hits:
            tag = f"pixabay:{hit.get('id')}"
            if tag in seen:
                continue
            videos = hit.get("videos", {})
            candidates = [
                v for v in videos.values()
                if isinstance(v, dict) and v.get("url")
                and max(v.get("width") or 0, v.get("height") or 0) >= 1280
            ]
            if not candidates:
                continue
            ceiling = int(cfg["visuals"].get("max_width", 2560))
            edge = lambda v: max(v.get("width") or 0, v.get("height") or 0)  # noqa: E731
            within = [v for v in candidates if edge(v) <= ceiling]
            chosen = max(within, key=edge) if within else min(candidates, key=edge)
            seen.add(tag)
            return chosen["url"]
    except Exception as exc:
        print(f"    pixabay: {exc}")
    return None


def _wikimedia(query: str, cfg: dict[str, Any], variant: int = 0,
               seconds: float = 6.0) -> str | None:
    """Wikimedia Commons — no API key, no account, no rate limit worth worrying about.

    Everything returned is freely licensed. Quality is below Pexels and the hit
    rate on abstract queries is worse, but it means the pipeline produces a real
    video the first time you run it, with nothing to sign up for.
    """
    seen = _used(cfg)
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
        for url in _rotate([u for u in usable if u], variant):
            if url not in seen:
                seen.add(url)
                return url
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

SUBJECTS = ("puppy", "kitten", "dog", "cat", "wolf", "vet", "owner")

# Beats where a joke would land badly. A query about a seizure or a poisoning
# must not be answered with a clip of a cat falling off a shelf.
SOBER = {
    "vet", "veterinarian", "clinic", "emergency", "sick", "illness", "pain",
    "poison", "toxic", "seizure", "vomit", "vomiting", "diarrhea", "blood",
    "injury", "injured", "wound", "surgery", "medication", "symptom",
    "symptoms", "dying", "euthanasia", "hurt", "limping", "distress",
}


def _query_ladder(query: str, cfg: dict[str, Any]) -> list[str]:
    """Progressively simpler searches.

    Wikimedia Commons matches titles and categories, not scene descriptions, so
    "dog chewing object floor" finds nothing while "dog" finds thousands. Try
    the specific phrasing first — it gives the better picture when it hits —
    then fall back toward the plain noun rather than straight to a blank card.

    Between the two sits a playful phrasing of the subject, because the rungs
    below the specific query are where a video picks up its filler shots, and
    filler on this channel should still be funny. That rung is skipped when the
    narration is on a medical or safety beat.
    """
    words = [w for w in query.lower().split() if w not in STOPWORDS]
    subject = next((s for s in SUBJECTS if s in words), words[0] if words else "")
    sober = bool(SOBER & set(words))

    ladder = []
    if query.strip():
        ladder.append(query.strip())
    if len(words) > 3:
        ladder.append(" ".join(words[:3]))
    if len(words) > 2:
        ladder.append(" ".join(words[:2]))
    if subject:
        if not sober:
            for term in cfg["visuals"].get("tone_terms", []):
                ladder.append(f"{term} {subject}")
        ladder.append(subject)
    # The last rung is the one a scene lands on when nothing else matched, so
    # a medical beat must not land on the comedy filler.
    if sober:
        ladder.append(cfg["visuals"].get("sober_fallback_query", "pet dog cat"))
    else:
        ladder.append(cfg["visuals"]["fallback_query"])

    seen, out = set(), []
    for q in ladder:
        q = q.strip()
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
            url = provider(q, cfg, index, seconds)
            if not url:
                continue
            dest = media / f"{index:03d}{_suffix_for(url)}"
            if _download(url, dest):
                print(f"    shot {index}: {name} — '{q}'")
                return dest

    print(f"    shot {index}: no match for '{query}', using generated card")
    return _fallback_card(query or "pet", media / f"{index:03d}.card.mp4", cfg, seconds)
