"""Narration.

Default engine is Piper: free, offline, MIT-licensed, runs fine on an M-series
Mac mini and costs nothing per video. Voice models download once (~60MB).

`edge` uses Microsoft's free Edge-TTS voices — noticeably more natural, still
free, but it needs internet on every run and is an undocumented endpoint that
can change without notice. Piper is the safer default for an unattended job.

`estimate` produces correctly-timed silence. It exists so you can test the
rendering pipeline without any audio setup.
"""
from __future__ import annotations

import re
import shutil
import time
import subprocess
import sys
from pathlib import Path
from typing import Any

VOICES_DIR = Path(__file__).resolve().parent.parent / "assets" / "voices"


def ensure_piper_voice(name: str) -> Path:
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    model = VOICES_DIR / f"{name}.onnx"
    if model.exists():
        return model
    print(f"  downloading voice model {name} (one time, ~60MB)...")
    subprocess.run(
        [sys.executable, "-m", "piper.download_voices", name,
         "--download-dir", str(VOICES_DIR)],
        check=True, capture_output=True, text=True,
    )
    if not model.exists():
        raise SystemExit(f"Voice model {name} did not download to {VOICES_DIR}")
    return model


def _piper(text: str, out: Path, cfg: dict[str, Any]) -> None:
    model = ensure_piper_voice(cfg["voice"]["piper_voice"])
    subprocess.run(
        [
            "piper",
            "-m", str(model),
            "-f", str(out),
            "--length-scale", str(cfg["voice"].get("piper_speed", 1.0)),
            "--sentence-silence", str(cfg["voice"].get("sentence_silence", 0.4)),
        ],
        input=text.encode("utf-8"),
        check=True,
        stdout=subprocess.DEVNULL,
    )


# No speech is intelligible above this rate, so a clip shorter than
# len(text) / this is proof the endpoint dropped audio, not a fast read.
_MAX_CHARS_PER_SECOND = 25.0

_SENTENCE_BREAK = re.compile(r"(?<=[.!?…])\s+")


def _sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_BREAK.split(text) if p.strip()]
    return parts or [text.strip()]


_SILENCE_EDGE = (
    "silenceremove=start_periods=1:start_duration=0:start_threshold=-50dB"
    ":detection=peak"
)


def _trim_silence(path: Path) -> None:
    """Strip leading and trailing silence from a clip.

    edge-tts pads what it returns with roughly a quarter second of silence at
    the head and close to a second at the tail. Joined end to end those pads
    stack into over a second of dead air, which is heard as the narration
    stopping and restarting rather than as a pause between sentences.
    """
    trimmed = path.with_name(path.stem + "_trim.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
         "-af", f"{_SILENCE_EDGE},areverse,{_SILENCE_EDGE},areverse",
         "-ar", "22050", "-ac", "1", str(trimmed)],
        check=True,
    )
    trimmed.replace(path)


def _pad_tail(path: Path, seconds: float) -> None:
    """Give a scene a deliberate tail so the next one does not run into it."""
    padded = path.with_name(path.stem + "_pad.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
         "-af", f"apad=pad_dur={seconds}",
         "-ar", "22050", "-ac", "1", str(padded)],
        check=True,
    )
    padded.replace(path)


def _cap_pauses(path: Path, seconds: float) -> None:
    """Shorten every pause inside a clip to at most `seconds`.

    edge-tts rests about a quarter second at each comma. Vietnamese narration
    is comma-heavy, so a scene picks up a pause every couple of seconds and the
    voice reads as repeatedly stopping rather than as speaking in phrases. The
    pause still has to exist, so it is shortened rather than removed.
    """
    capped = path.with_name(path.stem + "_cap.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
         "-af", f"silenceremove=stop_periods=-1:stop_duration={seconds}"
                f":stop_threshold=-50dB:detection=peak",
         "-ar", "22050", "-ac", "1", str(capped)],
        check=True,
    )
    capped.replace(path)


def _edge_sentence(text: str, out: Path, cfg: dict[str, Any]) -> None:
    """One sentence through Microsoft's free endpoint, verified on arrival.

    The endpoint streams the clip back in chunks and will sometimes end the
    stream early, leaving a file that plays fine but is missing its tail. That
    truncation is what is audible as narration cutting out mid-sentence, so a
    clip too short for its text is rejected and asked for again.
    """
    # Invoked through the interpreter so it always resolves to the venv's copy.
    # `--rate=-4%` uses the '=' form: as a separate argument, a value starting
    # with '-' is parsed as another option.
    mp3 = out.with_suffix(".mp3")
    cmd = [
        sys.executable, "-m", "edge_tts",
        f"--voice={cfg['voice']['edge_voice']}",
        f"--rate={cfg['voice'].get('edge_rate', '+0%')}",
        f"--pitch={cfg['voice'].get('edge_pitch', '+0Hz')}",
        f"--text={text}",
        f"--write-media={mp3}",
    ]
    floor = len(text) / _MAX_CHARS_PER_SECOND
    delays = [0, 3, 8, 20]
    last_error = ""
    best = out.with_name(out.stem + "_best.wav")
    best_seconds = 0.0
    for attempt, wait in enumerate(delays, start=1):
        if wait:
            time.sleep(wait)
        mp3.unlink(missing_ok=True)
        out.unlink(missing_ok=True)
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            if not (mp3.exists() and mp3.stat().st_size > 500):
                last_error = "empty audio returned"
            else:
                subprocess.run(
                    ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
                     "-ar", "22050", "-ac", "1", str(out)],
                    check=True,
                )
                _trim_silence(out)
                # Checked before pauses are capped: capping removes silence,
                # which would make a healthy clip look too fast to be real.
                have = duration(out)
                if have >= floor:
                    _cap_pauses(out, cfg["voice"].get("phrase_pause", 0.12))
                    mp3.unlink(missing_ok=True)
                    best.unlink(missing_ok=True)
                    return
                if have > best_seconds:
                    best_seconds = have
                    shutil.copyfile(out, best)
                last_error = "truncated audio returned"
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            last_error = detail.splitlines()[-1] if detail else f"exit {exc.returncode}"
        if attempt < len(delays):
            print(f"      retrying ({last_error[:70]})")
    mp3.unlink(missing_ok=True)
    if best_seconds > 0:
        # The length test cannot tell a clipped sentence from a naturally
        # brisk one, so it decides which attempt to prefer rather than whether
        # the build may continue. Losing a whole video to it would be worse
        # than narrating one sentence slightly short.
        best.replace(out)
        _cap_pauses(out, cfg["voice"].get("phrase_pause", 0.12))
        print(f"      every attempt came back short; keeping the longest "
              f"({best_seconds:.1f}s): {text[:40]}...")
        return
    raise RuntimeError(f"edge-tts failed after {len(delays)} tries: {last_error}")


def _join(parts: list[Path], out: Path, gap: float) -> None:
    """Concatenate clips with `gap` seconds of silence between them."""
    if len(parts) == 1:
        shutil.move(str(parts[0]), str(out))
        return
    work = parts[0].parent
    silence = work / "_gap.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "anullsrc=r=22050:cl=mono", "-t", f"{gap}", str(silence)],
        check=True,
    )
    entries: list[str] = []
    for i, part in enumerate(parts):
        if i:
            entries.append(f"file '{silence.resolve()}'")
        entries.append(f"file '{part.resolve()}'")
    listfile = work / "_join.txt"
    listfile.write_text("\n".join(entries), encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(listfile), "-c", "copy", str(out)],
        check=True,
    )


def _edge(text: str, out: Path, cfg: dict[str, Any]) -> None:
    """Microsoft's free endpoint, one sentence per request.

    A whole scene in one request is a long stream, and the longer the stream
    the likelier the endpoint ends it early. Sentence-sized requests largely
    avoid that, confine a retry to the sentence that needs one, and give the
    pause between sentences a real duration rather than whatever the endpoint
    happened to leave behind.
    """
    work = out.parent / f"_{out.stem}_parts"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    try:
        parts = []
        for i, sentence in enumerate(_sentences(text)):
            part = work / f"{i:02d}.wav"
            _edge_sentence(sentence, part, cfg)
            parts.append(part)
        _join(parts, out, cfg["voice"].get("sentence_silence", 0.25))
        _pad_tail(out, cfg["voice"].get("scene_gap", 0.18))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _estimate(text: str, out: Path, cfg: dict[str, Any]) -> None:
    words = max(1, len(text.split()))
    seconds = round(words / (cfg["script"]["words_per_minute"] / 60), 2) + 0.4
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", f"anullsrc=r=22050:cl=mono", "-t", str(seconds), str(out)],
        check=True,
    )


ENGINES = {"piper": _piper, "edge": _edge, "estimate": _estimate}

# If the configured engine fails, these are tried in order before giving up.
# Piper needs a one-time 60MB model download; edge needs none but needs
# internet on every run. Between them, one almost always works.
FALLBACKS = ["edge", "piper"]


def pick_engine(cfg: dict[str, Any], out_dir: Path) -> str:
    """Find an engine that actually produces audio on this machine.

    Done once per build with a short test phrase, so a broken engine fails in
    two seconds instead of twenty scenes in.
    """
    configured = cfg["voice"]["engine"]
    if configured == "estimate":
        return configured

    order = [configured] + [e for e in FALLBACKS if e != configured]
    probe = out_dir / "_voice_probe.wav"
    errors = []

    for name in order:
        engine = ENGINES.get(name)
        if not engine:
            continue
        try:
            probe.unlink(missing_ok=True)
            engine("Testing one two three.", probe, cfg)
            if probe.exists() and probe.stat().st_size > 1000:
                probe.unlink(missing_ok=True)
                if name != configured:
                    print(f"  '{configured}' voice unavailable — using '{name}' instead")
                else:
                    print(f"  voice engine: {name}")
                return name
            errors.append(f"{name}: produced no audio")
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            last = detail.splitlines()[-1] if detail else f"exit {exc.returncode}"
            errors.append(f"{name}: {last[:250]}")
        except Exception as exc:
            first = str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc)
            errors.append(f"{name}: {first[:250]}")

    probe.unlink(missing_ok=True)
    raise SystemExit(
        "No working text-to-speech engine.\n  "
        + "\n  ".join(errors)
        + "\n\nPiper needs a one-time model download from huggingface.co; edge "
        "needs internet on each run. If both fail, check your connection, then "
        "try: .venv/bin/python -m piper.download_voices en_US-lessac-medium "
        "--download-dir assets/voices"
    )


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def narrate_scenes(scenes: list[dict[str, Any]], cfg: dict[str, Any],
                   out_dir: Path) -> list[float]:
    """Render one wav per scene. Returns each scene's duration in seconds."""
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    if cfg["voice"]["engine"] not in ENGINES:
        raise SystemExit(f"Unknown voice engine: {cfg['voice']['engine']}")

    name = pick_engine(cfg, audio_dir)
    engine = ENGINES[name]
    backup = next(
        (ENGINES[e] for e in FALLBACKS if e != name and e in ENGINES), None
    )

    durations = []
    for i, scene in enumerate(scenes):
        wav = audio_dir / f"{i:03d}.wav"
        if cfg.get("_refresh_audio") or not wav.exists():
            try:
                engine(scene["narration"], wav, cfg)
            except Exception as exc:
                # One scene should not cost you the whole build.
                if backup is None:
                    raise SystemExit(f"Scene {i} narration failed: {exc}")
                print(f"    scene {i}: {str(exc)[:90]}")
                print(f"    falling back to the other engine for this scene")
                try:
                    backup(scene["narration"], wav, cfg)
                except Exception as exc2:
                    raise SystemExit(
                        f"Scene {i} narration failed on both engines.\n"
                        f"  primary: {exc}\n  backup:  {exc2}"
                    )
        durations.append(duration(wav))
        print(f"  scene {i + 1}/{len(scenes)}: {durations[-1]:.1f}s")
    return durations


def concat_narration(scenes_count: int, out_dir: Path) -> Path:
    audio_dir = out_dir / "audio"
    listfile = audio_dir / "list.txt"
    # Absolute paths: ffmpeg's concat demuxer resolves relative entries against
    # the list file's own directory, which silently doubles the path.
    listfile.write_text(
        "\n".join(
            f"file '{(audio_dir / f'{i:03d}.wav').resolve()}'"
            for i in range(scenes_count)
        ),
        encoding="utf-8",
    )
    full = out_dir / "narration.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(listfile), "-c", "copy", str(full)],
        check=True,
    )
    return full
