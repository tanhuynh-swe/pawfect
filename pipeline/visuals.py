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
import json
import re
import subprocess
import time
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


def _download(url: str, dest: Path, budget: float = 90.0) -> bool:
    """Fetch a clip, giving up if it is merely trickling.

    requests' timeout is per read, not per download, so a CDN that keeps
    sending a few bytes at a time never trips it. One Pixabay clip took nine
    minutes to deliver two megabytes and held up the whole build behind it.
    The budget caps the whole transfer, and a clip that cannot beat it is
    abandoned for the next candidate.
    """
    start = time.monotonic()
    try:
        with requests.get(url, stream=True, timeout=TIMEOUT, headers=UA) as r:
            r.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in r.iter_content(1 << 16):
                    fh.write(chunk)
                    if time.monotonic() - start > budget:
                        raise TimeoutError(
                            f"still downloading after {budget:.0f}s"
                        )
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


def _tokens(text: str) -> list[str]:
    """Descriptive words in a provider's own label for a clip."""
    return [t for t in re.split(r"[^a-z]+", text.lower()) if len(t) > 2]


def _mentions(tokens: list[str], word: str) -> bool:
    """Loose word match, so 'flicking' finds 'flick' and 'cats' finds 'cat'."""
    for token in tokens:
        if token == word:
            return True
        short, long = sorted((token, word), key=len)
        if len(short) >= 4 and long.startswith(short):
            return True
    return False


def _relevant(candidates: list[tuple[str, Any, str]], query: str, variant: int,
              seen: set[str], cfg_blocked: tuple[str, ...] = ()) -> list[tuple[str, Any]]:
    """Choose the candidate whose own description matches the query.

    Every provider searches loosely. Pexels ORs the words together, so
    "cat riding robot vacuum" comes back led by a person walking beside a
    robot vacuum and a man leaving a desk — no cat anywhere in it. The search
    reports a hit, the ladder never falls back, and the narration then plays
    over footage of the wrong thing entirely.

    Each provider labels its own clips: Pexels in the URL slug, Pixabay in
    tags, Wikimedia in the file title. Matching the query against that label
    is what separates a clip of the thing from a clip that shares a word with
    it. The subject is required outright — a scene about a cat may not be
    filled with a vacuum cleaner — and a query specific enough to name three
    things has to match one of the other two as well, or the ladder drops to a
    simpler search rather than settling for something unrelated.
    """
    words = [w for w in query.lower().split() if w not in STOPWORDS]
    if not words:
        return []
    subject = next((s for s in SUBJECTS if s in words), None)
    others = [w for w in words if w != subject]
    need_extra = 1 if len(words) >= 3 else 0
    wanted_species = [sp for sp in SPECIES if any(w in sp for w in words)]

    blocked = set(cfg_blocked)
    ranked: list[tuple[int, Any]] = []
    for tag, item, label in candidates:
        if tag in seen or tag in blocked:
            continue
        tokens = _tokens(label)
        if subject and not _mentions(tokens, subject):
            continue
        # The clip has to be of the animal asked for, not merely share a verb
        # or a landscape with one.
        if wanted_species and not any(
            any(t in sp for t in tokens) for sp in wanted_species
        ):
            continue
        extra = sum(1 for w in others if _mentions(tokens, w))
        if extra < need_extra:
            continue
        ranked.append((extra, (tag, item)))
    if not ranked:
        return []

    # Best match first; rotate between equally good ones so two shots of the
    # same scene do not both land on whatever the provider listed first.
    out: list[tuple[str, Any]] = []
    for score in sorted({r[0] for r in ranked}, reverse=True):
        tier = [r[1] for r in ranked if r[0] == score]
        out.extend(_rotate(tier, variant))
    return out


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
        candidates = [
            (f"pexels:{v.get('id')}", v, v.get("url") or "")
            for v in r.json().get("videos", [])
        ]
        for tag, video in _relevant(candidates, query, variant, seen,
                                tuple(cfg['visuals'].get('blocked_sources', []))):
            chosen = _best_video_file(video.get("video_files", []), cfg)
            if chosen:
                seen.add(tag)
                cfg["_last_source"] = tag
                return chosen["link"]
    except Exception as exc:
        print(f"    pexels: {exc}")
    return None


def _coverr(query: str, cfg: dict[str, Any], variant: int = 0,
            seconds: float = 6.0) -> str | None:
    """Coverr — curated free footage, smaller and better shot than the big libraries.

    Every clip carries a written title and description rather than a filename,
    which is the best relevance signal of any provider here.

    Two caveats that are not this code's to solve: a key is issued by emailing
    team@coverr.co rather than self-serve, and Coverr's terms ask for
    attribution with their logo, which neither Pexels nor Pixabay require.
    """
    key = env("COVERR_API_KEY")
    if not key:
        return None
    seen = _used(cfg)
    portrait = cfg["visuals"]["orientation"] == "portrait"
    try:
        r = requests.get(
            "https://api.coverr.co/videos",
            params={"query": query, "page_size": 24, "urls": "true"},
            headers={"Authorization": f"Bearer {key}", **UA},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        payload = r.json()
        # The envelope is not pinned down in the public docs, so accept the
        # shapes it is documented to return rather than guessing just one.
        hits = payload if isinstance(payload, list) else (
            payload.get("hits") or payload.get("videos") or payload.get("data") or []
        )
        candidates = []
        for v in hits:
            w = v.get("max_width") or 0
            h = v.get("max_height") or 0
            if max(w, h) < 1280:
                continue
            if portrait and w > h:
                continue
            if not portrait and h > w:
                continue
            label = f"{v.get('title') or ''} {v.get('description') or ''}"
            candidates.append((f"coverr:{v.get('id')}", v, label))
        for tag, video in _relevant(candidates, query, variant, seen,
                                tuple(cfg['visuals'].get('blocked_sources', []))):
            urls = video.get("urls") or {}
            link = urls.get("mp4_download") or urls.get("mp4")
            if link:
                seen.add(tag)
                cfg["_last_source"] = tag
                return link
    except Exception as exc:
        print(f"    coverr: {exc}")
    return None


def _pexels_photos(query: str, cfg: dict[str, Any], variant: int = 0,
                   seconds: float = 6.0) -> str | None:
    """Pexels stills, on the same key as the video search.

    Some shots simply are not filmed. There is no stock video of a cat being
    examined at a vet, but there are good photographs of exactly that, and
    render.py gives a still a slow push and drift so it plays as footage.

    Photographs also carry a written description rather than a URL slug, so
    the relevance check has a real sentence to match against: "Veterinarian
    using stethoscope to examine cat in a clinic setting" either mentions a
    cat or it does not.
    """
    key = env("PEXELS_API_KEY")
    if not key:
        return None
    seen = _used(cfg)
    try:
        r = requests.get(
            "https://api.pexels.com/v1/search",
            params={"query": query, "per_page": 24,
                    "orientation": cfg["visuals"]["orientation"]},
            headers={"Authorization": key, **UA},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        candidates = [
            (f"pexels_photo:{p.get('id')}", p, p.get("alt") or "")
            for p in r.json().get("photos", [])
        ]
        for tag, photo in _relevant(candidates, query, variant, seen,
                                tuple(cfg['visuals'].get('blocked_sources', []))):
            src = photo.get("src", {})
            link = src.get("large2x") or src.get("original") or src.get("large")
            if link and (photo.get("width") or 0) >= 1280:
                seen.add(tag)
                cfg["_last_source"] = tag
                return link
    except Exception as exc:
        print(f"    pexels photos: {exc}")
    return None


def _pixabay(query: str, cfg: dict[str, Any], variant: int = 0,
             seconds: float = 6.0) -> str | None:
    key = env("PIXABAY_API_KEY")
    if not key:
        return None
    seen = _used(cfg)
    params = {"key": key, "q": query, "per_page": 24, "safesearch": "true",
              "min_width": int(cfg["visuals"].get("min_width", 1920))}
    if cfg["visuals"].get("prefer_recent", False):
        # Off by default, and it should stay off. Pixabay's "latest" ordering
        # barely weighs the query: asked for a cat in a cardboard box it
        # returns kebab cooking, grazing horses and a glacier, because those
        # were uploaded most recently. "popular" is the only ordering that
        # actually ranks on the search term.
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

        # Pixabay's video search has no orientation parameter and its library
        # is overwhelmingly landscape. Cropping a 1920x1080 clip to 1080x1920
        # throws away two thirds of the frame, usually including the animal,
        # so portrait clips are used first when the build is vertical — but
        # landscape is still allowed rather than starving the search.
        if cfg["visuals"]["orientation"] == "portrait":
            def _tall(h: dict[str, Any]) -> bool:
                v = (h.get("videos") or {}).get("large") or {}
                return (v.get("height") or 0) > (v.get("width") or 0)
            hits.sort(key=lambda h: not _tall(h))

        blocked = [t.lower() for t in cfg["visuals"].get("exclude_terms", [])]
        if blocked:
            # Pixabay carries 3D character renders and cartoon animals beside
            # the real footage. One closed the last build on a grinning
            # cartoon bulldog, which reads as clipart next to real pets.
            hits = [h for h in hits
                    if not any(b in (h.get("tags") or "").lower() for b in blocked)]

        if cfg["visuals"].get("exclude_ai", True):
            # Pixabay carries AI-generated footage, tagged as such. Synthetic
            # animals land in the uncanny valley on a channel whose whole
            # promise is what real pets actually do, and using them would drag
            # in a synthetic-media disclosure this channel does not otherwise
            # need.
            hits = [h for h in hits if "ai generated" not in (h.get("tags") or "").lower()]

        candidates = [
            (f"pixabay:{h.get('id')}", h, h.get("tags") or "") for h in hits
        ]
        for tag, hit in _relevant(candidates, query, variant, seen,
                                tuple(cfg['visuals'].get('blocked_sources', []))):
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
            cfg["_last_source"] = tag
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
        candidates = []
        for page in pages:
            info = (page.get("imageinfo") or [{}])[0]
            mime = info.get("mime", "")
            width = info.get("width") or 0
            url = info.get("thumburl") or info.get("url")
            if url and mime in ("image/jpeg", "image/png") and width >= 1280:
                candidates.append((url, url, page.get("title") or url))
        for tag, url in _relevant(candidates, query, variant, seen,
                                tuple(cfg['visuals'].get('blocked_sources', []))):
            seen.add(tag)
            cfg["_last_source"] = tag
            return url
    except Exception as exc:
        print(f"    wikimedia: {exc}")
    return None


PROVIDERS = {"coverr": _coverr, "pexels": _pexels, "pixabay": _pixabay,
             "pexels_photos": _pexels_photos, "wikimedia": _wikimedia}
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


def _record_source(media: Path, index: int, provider: str, query: str,
                   url: str, cfg: dict[str, Any]) -> None:
    """Write down which clip filled which shot.

    Without this a shot that looks wrong cannot be traced back to the clip
    that produced it, so it cannot be blocked — the only way to find the id
    was to re-run searches and guess. It doubles as the attribution trail
    Coverr's licence asks for.
    """
    path = media / "sources.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        data = {}
    data[f"{index:03d}"] = {
        "provider": provider,
        "source": cfg.get("_last_source"),
        "query": query,
        "url": url,
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


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

# Words that mean an animal is the subject, including the breeds people search
# by. A query naming one of these must be answered by a clip whose own label
# also names an animal — without this, "beagle running" is satisfied by a
# couple jogging on a beach and "german shepherd running grass" by a flock of
# geese, because only "running" and "grass" had to match.
DOG_WORDS = {
    "dog", "dogs", "puppy", "puppies", "pup", "doggy", "canine",
    "labrador", "retriever", "beagle", "poodle", "husky", "corgi", "chihuahua",
    "pug", "shiba", "terrier", "collie", "dachshund", "rottweiler", "boxer",
    "bulldog", "spaniel", "shepherd", "dalmatian", "greyhound", "mastiff",
    "pointer", "setter", "samoyed", "akita", "malamute", "pomeranian",
}
CAT_WORDS = {"cat", "cats", "kitten", "kittens", "feline", "kitty"}

# Species, not "an animal". A gate that accepts any animal word lets a flock
# of geese tagged "animals" answer "german shepherd running grass", which is
# exactly what it did.
SPECIES = (DOG_WORDS, CAT_WORDS)

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

    # Drawn animation short-circuits the search entirely: there is nothing to
    # look for, because the scene is drawn to the query.
    if "toon" in cfg["visuals"]["providers"]:
        from . import toon
        # Skia is the better renderer (real beziers, gradients, native
        # anti-aliasing) but Pillow stays as a fallback so the pipeline still
        # draws on a machine where skia-python will not install.
        try:
            from . import toon_skia as backend
            engine = "skia"
        except ImportError:
            backend = toon
            engine = "pillow"
        species = toon.species_for(
            query, str(cfg["visuals"].get("toon_dog", "greydog")))
        if species != "dog" and engine == "pillow":
            # Only the Skia backend has the cat and the channel's own dog.
            # Pillow would quietly draw the tan puppy instead, which for a
            # cat script is the whole mismatch this renderer exists to
            # remove, so say so rather than let it pass.
            print(f"    shot {index}: WARNING '{query}' wants the {species} "
                  f"character, but the Pillow backend only draws the tan "
                  f"dog — install skia-python")
        # Consecutive shots of the same scene used to restart the animation
        # clock and the camera, so a run of three sleeping shots cut like a
        # glitch. Carrying the elapsed time and camera phase across the run
        # makes them read as one continuous take that happens to be cut.
        # Keyed on the species too: a cut from a sleeping dog to a sleeping
        # cat is a different take, not a continuation of the same one.
        name = f"{species}:{toon.scene_for(query)}"
        prev_name, prev_time, prev_cam = cfg.get("_toon_run", (None, 0.0, 0.0))
        if prev_name == name:
            cfg["_toon_offset"], cfg["_toon_cam"] = prev_time, prev_cam
        else:
            cfg["_toon_offset"], cfg["_toon_cam"] = 0.0, 0.0
        cfg["_toon_run"] = (name, cfg["_toon_offset"] + seconds,
                            cfg["_toon_cam"] + 0.5)

        dest = media / f"{index:03d}.mp4"
        backend.render_clip(query, seconds, cfg, dest)
        print(f"    shot {index}: drawn/{engine} — '{query}' -> {name}")
        return dest

    for q in _query_ladder(query, cfg):
        for name in cfg["visuals"]["providers"]:
            provider = PROVIDERS.get(name)
            if not provider:
                continue
            url = provider(q, cfg, index, seconds)
            if not url:
                continue
            dest = media / f"{index:03d}{_suffix_for(url)}"
            if _download(url, dest, float(cfg["visuals"].get("download_timeout", 90))):
                print(f"    shot {index}: {name} — '{q}'")
                _record_source(media, index, name, q, url, cfg)
                return dest

    print(f"    shot {index}: no match for '{query}', using generated card")
    return _fallback_card(query or "pet", media / f"{index:03d}.card.mp4", cfg, seconds)
