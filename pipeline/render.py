"""Video assembly with ffmpeg.

One shot per `max_shot_seconds` of narration, so a 40-second scene gets six
different visuals instead of one long static clip. That single detail does more
for retention than anything else in this file.

Captions are burned in as ASS. Roughly 70% of YouTube pet content is watched
with sound off at some point; unburned captions lose those viewers.
"""
from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any

from .visuals import fetch_clip

FONT = "Poppins"


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


def _ass_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


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


def build_captions(scenes: list[dict[str, Any]], durations: list[float],
                   cfg: dict[str, Any], out_dir: Path) -> Path:
    """Chunk each scene's narration into caption lines, timed by word count."""
    w, h = cfg["video"]["width"], cfg["video"]["height"]
    v = cfg["video"]
    font = v.get("font", FONT)
    cap_size = v.get("caption_size", 58)
    ban_size = v.get("banner_size", 92)
    margin_v = v.get("caption_margin_v", 90)
    side = v.get("side_margin", 160)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font},{cap_size},&H00FFFFFF,&H00000000,&H90000000,1,3,0,2,2,{side},{side},{margin_v},1
Style: Banner,{font},{ban_size},&H0000E5FF,&H00101010,&H00000000,1,1,5,2,8,{side},{side},{margin_v + 40},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines: list[str] = []
    t = 0.0
    for scene, dur in zip(scenes, durations):
        words = scene["narration"].split()
        if not words:
            t += dur
            continue
        chunks: list[list[str]] = []
        current: list[str] = []
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
            text = " ".join(chunk).replace("\n", " ")
            lines.append(
                f"Dialogue: 0,{_ass_time(cursor)},{_ass_time(cursor + span)},"
                f"Caption,,0,0,0,,{text}"
            )
            cursor += span

        banner = (scene.get("on_screen_text") or "").strip()
        if banner:
            lines.append(
                f"Dialogue: 1,{_ass_time(t + 0.2)},{_ass_time(min(t + 3.2, t + dur))},"
                f"Banner,,0,0,0,,{banner.upper()}"
            )
        t += dur

    path = out_dir / "captions.ass"
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return path


def _music_track() -> Path | None:
    music_dir = Path(__file__).resolve().parent.parent / "assets" / "music"
    tracks = sorted(
        p for p in music_dir.glob("*")
        if p.suffix.lower() in {".mp3", ".wav", ".m4a", ".ogg"}
    )
    return tracks[0] if tracks else None


def finalize(silent_video: Path, narration: Path, captions: Path,
             cfg: dict[str, Any], out_dir: Path) -> Path:
    final = out_dir / "final.mp4"
    music = _music_track()
    lufs = cfg["video"]["narration_lufs"]

    # Captions go inside the complex graph rather than through -vf. Mixing
    # -vf with -filter_complex in one command is rejected outright by newer
    # ffmpeg builds, so everything lives in one graph and nothing is implicit.
    # ffmpeg 8 rejects the quoted shorthand `ass='/path'` with "No option name
    # near ..." — the option has to be named explicitly, with the path escaped
    # rather than quoted. This form is accepted by ffmpeg 6, 7 and 8 alike.
    ass = str(captions)
    for ch in ("\\", "'", ":", ",", "[", "]", ";"):
        ass = ass.replace(ch, "\\" + ch)
    video_chain = f"[0:v]ass=filename={ass}[v]"

    if music:
        vol = cfg["video"]["music_volume"]
        graph = (
            f"{video_chain};"
            f"[1:a]aformat=fltp:44100:stereo,loudnorm=I={lufs}:TP=-1.5[nar];"
            f"[2:a]aformat=fltp:44100:stereo,volume={vol},"
            f"afade=t=in:st=0:d=2[bed];"
            f"[bed][nar]sidechaincompress=threshold=0.05:ratio=8:attack=5:release=400[duck];"
            f"[nar][duck]amix=inputs=2:duration=first:dropout_transition=0[aout]"
        )
        inputs = ["-i", str(silent_video), "-i", str(narration),
                  "-stream_loop", "-1", "-i", str(music)]
    else:
        graph = (
            f"{video_chain};"
            f"[1:a]aformat=fltp:44100:stereo,loudnorm=I={lufs}:TP=-1.5[aout]"
        )
        inputs = ["-i", str(silent_video), "-i", str(narration)]

    args = [
        "ffmpeg", "-y", "-loglevel", "error", "-nostdin",
        *inputs,
        "-filter_complex", graph,
        "-map", "[v]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "19",
        "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
        "-movflags", "+faststart", "-shortest", str(final),
    ]
    try:
        subprocess.run(args, check=True, capture_output=True, text=True)
        return final
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or "").strip()
        (out_dir / "ffmpeg_error.log").write_text(
            "COMMAND:\n" + " ".join(args) + "\n\nSTDERR:\n" + detail,
            encoding="utf-8",
        )
        print("  caption burn-in failed; retrying with the subtitles filter")

    # Some ffmpeg builds ship libass under the `subtitles` filter only, or
    # reject `ass` outright. Try that, then fall back to no burned captions —
    # a video without captions beats no video, and the .ass file is still there.
    for attempt, chain in enumerate(
        (f"[0:v]subtitles=filename={ass}[v]", "[0:v]null[v]"), start=1
    ):
        retry_graph = graph.replace(video_chain, chain, 1)
        retry = list(args)
        retry[retry.index("-filter_complex") + 1] = retry_graph
        try:
            subprocess.run(retry, check=True, capture_output=True, text=True)
            if attempt == 2:
                print("  WARNING: rendered WITHOUT burned-in captions — "
                      "your ffmpeg cannot burn subtitles. See ffmpeg_error.log")
            return final
        except subprocess.CalledProcessError as exc:
            last = (exc.stderr or "").strip()
            with (out_dir / "ffmpeg_error.log").open("a", encoding="utf-8") as fh:
                fh.write(f"\n\n--- retry {attempt} ---\n{last}")

    raise SystemExit(
        "Final render failed even without captions. Full ffmpeg output:\n  "
        + str(out_dir / "ffmpeg_error.log")
    )
