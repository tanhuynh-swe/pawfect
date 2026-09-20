"""Video assembly with ffmpeg.

One shot per `max_shot_seconds` of narration, so a 40-second scene gets six
different visuals instead of one long static clip. That single detail does more
for retention than anything else in this file.

Captions are burned in as ASS. Roughly 70% of YouTube pet content is watched
with sound off at some point; unburned captions lose those viewers.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .visuals import fetch_clip



def _run(args: list[str]) -> None:
    subprocess.run(args, check=True)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _shot_from_image(src: Path, dest: Path, seconds: float, cfg: dict[str, Any],
                     index: int) -> None:
    """Ken Burns: a still with a slow push and drift, so it reads as footage.

    Direction alternates by shot so consecutive stills don't feel identical.
    """
    w, h, fps = cfg["video"]["width"], cfg["video"]["height"], cfg["video"]["fps"]
    frames = max(2, int(round(seconds * fps)))
    zoom_in = index % 2 == 0
    z = ("min(1.0001+0.00035*on,1.14)" if zoom_in
         else "max(1.14-0.00035*on,1.0001)")
    drift = "iw/2-(iw/zoom/2)" if index % 4 < 2 else "iw/2-(iw/zoom/2)+(on/16)"
    vf = (
        f"scale={w * 2}:{h * 2}:force_original_aspect_ratio=increase,"
        f"crop={w * 2}:{h * 2},"
        f"zoompan=z='{z}':d={frames}:x='{drift}':y='ih/2-(ih/zoom/2)'"
        f":s={w}x{h}:fps={fps},"
        f"setsar=1,eq=saturation=1.06:contrast=1.03"
    )
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-loop", "1", "-i", str(src), "-t", f"{seconds:.3f}",
        "-vf", vf, "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", str(dest),
    ])


def _shot(src: Path, dest: Path, seconds: float, cfg: dict[str, Any],
          index: int = 0) -> None:
    if src.suffix.lower() in IMAGE_SUFFIXES:
        _shot_from_image(src, dest, seconds, cfg, index)
        return
    w, h, fps = cfg["video"]["width"], cfg["video"]["height"], cfg["video"]["fps"]
    vf = (
        f"scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},fps={fps},setsar=1,"
        f"eq=saturation=1.06:contrast=1.03"
    )
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-stream_loop", "-1", "-i", str(src),
        "-t", f"{seconds:.3f}", "-vf", vf, "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", str(dest),
    ])


def _shot_is_current(dest: Path, seconds: float, cfg: dict[str, Any]) -> bool:
    """True when `dest` already holds a shot of the right length.

    Shot lengths are derived from the narration, so refreshing the audio
    changes them. Reusing a shot cut for the previous audio desynchronises the
    visuals from the voice and leaves the video shorter than the narration,
    which -shortest then silently truncates.
    """
    if not dest.exists():
        return False
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(dest)],
        capture_output=True, text=True,
    )
    try:
        have = float(probe.stdout.strip())
    except ValueError:
        return False
    return abs(have - seconds) <= 1.5 / float(cfg["video"]["fps"])


def build_shots(scenes: list[dict[str, Any]], durations: list[float],
                cfg: dict[str, Any], out_dir: Path) -> list[Path]:
    shots_dir = out_dir / "shots"
    shots_dir.mkdir(exist_ok=True)
    max_shot = float(cfg["video"]["max_shot_seconds"])
    paths: list[Path] = []
    counter = 0

    for scene, dur in zip(scenes, durations):
        n = max(1, math.ceil(dur / max_shot))
        per = dur / n
        for k in range(n):
            dest = shots_dir / f"{counter:04d}.mp4"
            if cfg.get("_refresh") or not _shot_is_current(dest, per, cfg):
                clip = fetch_clip(
                    scene.get("visual_query", ""), counter, per, cfg, out_dir
                )
                _shot(clip, dest, per, cfg, counter)
            paths.append(dest)
            counter += 1

    # Shots left over from a build that needed more of them would otherwise sit
    # here and be picked up as though they belonged to this one.
    for stale in sorted(shots_dir.glob("*.mp4"))[counter:]:
        stale.unlink()
    return paths


def concat_shots(shots: list[Path], out_dir: Path) -> Path:
    listfile = out_dir / "shots.txt"
    listfile.write_text(
        "\n".join(f"file '{p.resolve()}'" for p in shots), encoding="utf-8"
    )
    silent = out_dir / "video_silent.mp4"
    _run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", str(listfile), "-c", "copy", str(silent),
    ])
    return silent



def _caption_chunks(scenes: list[dict[str, Any]], durations: list[float]):
    """Yield (start, end, text) for every caption line, timed by word count."""
    t = 0.0
    for scene, dur in zip(scenes, durations):
        words = scene["narration"].split()
        if not words:
            t += dur
            continue
        chunks, current = [], []
        for word in words:
            current.append(word)
            if len(current) >= 7 or word.endswith((".", "!", "?")):
                chunks.append(current)
                current = []
        if current:
            chunks.append(current)
        total_words = sum(len(c) for c in chunks)
        cursor = t
        for chunk in chunks:
            span = dur * (len(chunk) / total_words)
            yield cursor, cursor + span, " ".join(chunk).replace("\n", " ")
            cursor += span
        t += dur


def _srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def build_srt(scenes: list[dict[str, Any]], durations: list[float],
              out_dir: Path) -> Path:
    """A subtitle sidecar, uploaded to YouTube as a real caption track.

    This is the safety net for burned-in captions: some ffmpeg builds ship
    without libass, and YouTube captions also help search and accessibility,
    so it is worth writing either way.
    """
    lines = []
    for i, (start, end, text) in enumerate(_caption_chunks(scenes, durations), 1):
        lines.append(f"{i}\n{_srt_time(start)} --> {_srt_time(end)}\n{text}\n")
    path = out_dir / "captions.srt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path



CAPTION_FONTS = [
    "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
# Poppins carries no Vietnamese diacritics, so a vi script needs a font that
# does or every tone mark renders as a blank box.
CAPTION_FONTS_VI = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _caption_font(size: int, vietnamese: bool) -> ImageFont.FreeTypeFont:
    for path in (CAPTION_FONTS_VI if vietnamese else CAPTION_FONTS):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def _wrap(draw: ImageDraw.ImageDraw, text: str,
          font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines: list[str] = []
    line: list[str] = []
    for word in text.split():
        line.append(word)
        if draw.textlength(" ".join(line), font=font) > width and len(line) > 1:
            line.pop()
            lines.append(" ".join(line))
            line = [word]
    if line:
        lines.append(" ".join(line))
    return lines


def _draw_block(draw: ImageDraw.ImageDraw, lines: list[str],
                font: ImageFont.FreeTypeFont, centre_x: int, bottom_y: int,
                fill: tuple[int, int, int, int],
                outline: tuple[int, int, int, int]) -> None:
    """Draw centred lines sitting on `bottom_y`, outlined so they read on any shot."""
    step = int(font.size * 1.25)
    y = bottom_y - step * len(lines) + step // 2
    for line in lines:
        draw.text((centre_x, y), line, font=font, fill=fill, anchor="mm",
                  stroke_width=max(3, font.size // 12), stroke_fill=outline)
        y += step


def build_caption_overlay(scenes: list[dict[str, Any]], durations: list[float],
                          cfg: dict[str, Any], out_dir: Path) -> Path | None:
    """Draw the captions with Pillow for ffmpeg to composite.

    This ffmpeg is built without libass and without freetype, so neither the
    `ass` filter nor `drawtext` exists to burn text in. Pillow is already a
    dependency for thumbnails, typesets this perfectly well, and can be given
    a font that actually covers Vietnamese. Each distinct caption is drawn
    once to a transparent frame, and the concat list holds each frame for
    exactly as long as its line is on screen, so ffmpeg reads the result as an
    ordinary video track that a single overlay composites.
    """
    if not cfg["video"].get("burn_captions", True):
        return None

    v = cfg["video"]
    w, h = v["width"], v["height"]
    side = v.get("side_margin", 160)
    margin_v = v.get("caption_margin_v", 90)
    vietnamese = cfg.get("_language") == "vi"
    cap_font = _caption_font(v.get("caption_size", 58), vietnamese)
    ban_font = _caption_font(v.get("banner_size", 92), vietnamese)

    captions = list(_caption_chunks(scenes, durations))
    banners: list[tuple[float, float, str]] = []
    t = 0.0
    for scene, dur in zip(scenes, durations):
        text = (scene.get("on_screen_text") or "").strip()
        if text:
            banners.append((t + 0.2, min(t + 3.2, t + dur), text.upper()))
        t += dur
    if t <= 0:
        return None

    marks = {0.0, t}
    for start, end, _ in [*captions, *banners]:
        marks.update((round(start, 3), round(end, 3)))
    ordered = sorted(m for m in marks if 0.0 <= m <= t)

    frames = out_dir / "captions"
    shutil.rmtree(frames, ignore_errors=True)
    frames.mkdir(parents=True)

    def showing(cues: list[tuple[float, float, str]], at: float) -> str:
        for start, end, text in cues:
            if start <= at < end:
                return text
        return ""

    drawn: dict[tuple[str, str], Path] = {}
    entries: list[str] = []
    last: Path | None = None
    for start, end in zip(ordered, ordered[1:]):
        if end - start < 0.04:          # shorter than a frame at any sane fps
            continue
        middle = (start + end) / 2
        key = (showing(captions, middle), showing(banners, middle))
        path = drawn.get(key)
        if path is None:
            path = frames / f"{len(drawn):04d}.png"
            image = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            caption, banner = key
            if caption:
                _draw_block(draw, _wrap(draw, caption, cap_font, w - 2 * side),
                            cap_font, w // 2, h - margin_v,
                            (255, 255, 255, 255), (0, 0, 0, 255))
            if banner:
                _draw_block(draw, _wrap(draw, banner, ban_font, w - 2 * side),
                            ban_font, w // 2, h - margin_v - int(cap_font.size * 3),
                            (255, 214, 10, 255), (16, 16, 16, 255))
            image.save(path)
            drawn[key] = path
        entries.append(f"file '{path.resolve()}'\nduration {end - start:.3f}")
        last = path

    if last is None:
        return None
    # The concat demuxer ignores the last entry's duration, so the final frame
    # is listed again to hold until the picture ends.
    entries.append(f"file '{last.resolve()}'")
    listfile = out_dir / "captions_overlay.txt"
    listfile.write_text("\n".join(entries) + "\n", encoding="utf-8")

    # Encoded to a constant frame rate track rather than handed to the render
    # as the concat list itself. overlay lines the two streams up by timestamp,
    # and the timestamps the concat demuxer derives from image durations do not
    # survive that against a real video: whole captions silently fail to
    # appear. A plain CFR track composites exactly as listed.
    track = out_dir / "captions_track.mov"
    fps = cfg["video"]["fps"]
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(listfile), "-vf", f"fps={fps},format=rgba",
         "-c:v", "qtrle", "-r", str(fps), str(track)],
        check=True,
    )
    shutil.rmtree(frames, ignore_errors=True)
    listfile.unlink(missing_ok=True)
    print(f"  captions: {len(drawn)} frames drawn with Pillow")
    return track


def _music_track() -> Path | None:
    music_dir = Path(__file__).resolve().parent.parent / "assets" / "music"
    tracks = sorted(
        p for p in music_dir.glob("*")
        if p.suffix.lower() in {".mp3", ".wav", ".m4a", ".ogg"}
    )
    return tracks[0] if tracks else None


def _normalized(narration: Path, lufs: int, out_dir: Path) -> Path:
    """Loudness-normalise the narration in its own pass, before the render.

    Running loudnorm inside the render's filtergraph stalls: its three second
    lookahead does not survive alongside the video encoder, and whole seconds
    of audio never reach the muxer — the voice drops out repeatedly while the
    picture keeps going. Handing the render a finished track it only has to
    encode avoids that. Measuring first also applies one static gain instead
    of riding it, which single-pass loudnorm would do.
    """
    out = out_dir / "narration_norm.wav"
    base = f"loudnorm=I={lufs}:TP=-1.5"
    probe = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostdin", "-i", str(narration),
         "-af", f"{base}:print_format=json", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    chain = base
    match = re.search(r"\{[^{}]*\"input_i\".*?\}", probe.stderr, re.S)
    if match:
        try:
            m = json.loads(match.group(0))
            keys = ("input_i", "input_lra", "input_tp",
                    "input_thresh", "target_offset")
            v = {k: float(m[k]) for k in keys}          # "-inf" on silence
            chain = (f"{base}:linear=true:measured_I={v['input_i']}"
                     f":measured_LRA={v['input_lra']}"
                     f":measured_TP={v['input_tp']}"
                     f":measured_thresh={v['input_thresh']}"
                     f":offset={v['target_offset']}")
        except (ValueError, KeyError):
            pass
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(narration),
         "-af", chain, "-ar", "44100", "-ac", "2", str(out)],
        check=True,
    )
    return out


def finalize(silent_video: Path, narration: Path, captions: Path | None,
             cfg: dict[str, Any], out_dir: Path) -> Path:
    final = out_dir / "final.mp4"
    log = out_dir / "ffmpeg_error.log"
    log.unlink(missing_ok=True)
    music = _music_track()
    # loudnorm is deliberately absent from the graph below; see _normalized.
    narration = _normalized(narration, cfg["video"]["narration_lufs"], out_dir)

    def command(with_captions: bool) -> list[str]:
        inputs = ["-i", str(silent_video), "-i", str(narration)]
        index = 2
        music_index = 0
        if music:
            inputs += ["-stream_loop", "-1", "-i", str(music)]
            music_index = index
            index += 1
        if with_captions:
            inputs += ["-i", str(captions)]
            graph = f"[0:v][{index}:v]overlay=0:0:shortest=1[v]"
        else:
            graph = "[0:v]null[v]"

        if music:
            vol = cfg["video"]["music_volume"]
            graph += (
                f";[{music_index}:a]aformat=fltp:44100:stereo,volume={vol},"
                f"afade=t=in:st=0:d=2[bed];"
                f"[bed][1:a]sidechaincompress=threshold=0.05:ratio=8"
                f":attack=5:release=400[duck];"
                f"[1:a][duck]amix=inputs=2:duration=first:dropout_transition=0[aout]"
            )
            audio_map = "[aout]"
        else:
            audio_map = "1:a"

        return [
            "ffmpeg", "-y", "-loglevel", "error", "-nostdin",
            *inputs,
            "-filter_complex", graph,
            "-map", "[v]", "-map", audio_map,
            "-c:v", "libx264", "-preset", "medium", "-crf", "19",
            "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
            "-movflags", "+faststart", "-shortest", str(final),
        ]

    # A video without captions still beats no video, so a failed overlay costs
    # the captions rather than the render.
    for with_captions in ([True, False] if captions else [False]):
        args = command(with_captions)
        try:
            subprocess.run(args, check=True, capture_output=True, text=True)
            if captions and not with_captions:
                print(f"  WARNING: rendered WITHOUT captions — see {log}")
            elif captions:
                captions.unlink(missing_ok=True)   # regenerated per render
            return final
        except subprocess.CalledProcessError as exc:
            with log.open("a", encoding="utf-8") as fh:
                fh.write("COMMAND:\n" + " ".join(args) + "\n\nSTDERR:\n"
                         + (exc.stderr or "").strip() + "\n\n")
            if with_captions:
                print("  caption overlay failed; retrying without captions")

    raise SystemExit(
        "Final render failed even without captions. Full ffmpeg output:\n  "
        + str(log)
    )
