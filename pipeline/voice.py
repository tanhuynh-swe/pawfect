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


def _edge(text: str, out: Path, cfg: dict[str, Any]) -> None:
    """Microsoft's free endpoint, which rate-limits a rapid run of requests.

    A 20-scene script fires 20 calls in under a minute and the service starts
    refusing partway through, so each scene gets a few patient retries before
    the build is allowed to fail.
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
    delays = [0, 3, 8, 20]
    last_error = ""
    for attempt, wait in enumerate(delays, start=1):
        if wait:
            time.sleep(wait)
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            if mp3.exists() and mp3.stat().st_size > 500:
                break
            last_error = "empty audio returned"
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            last_error = detail.splitlines()[-1] if detail else f"exit {exc.returncode}"
        if attempt < len(delays):
            print(f"      retrying ({last_error[:70]})")
    else:
        raise RuntimeError(f"edge-tts failed after {len(delays)} tries: {last_error}")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
         "-ar", "22050", "-ac", "1", str(out)],
        check=True,
    )
    mp3.unlink(missing_ok=True)


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
