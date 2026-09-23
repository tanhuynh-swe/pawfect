"""Put clips made by hand in Google Flow into a slot's shots.

Flow has no API, so its clips arrive as downloads - one per scene, eight
seconds each. `render.build_shots` cuts a scene into several shots and looks
each one up as `media/<index>.mp4`, so this cuts every scene's clip the same
way the render will, in order, and writes the pieces under those indices.
A scene with no clip keeps whatever it had, and the next build draws it.

    python flow_import.py <slug> <folder>

The folder holds `scene1.mp4`, `scene2.mp4` ... (Flow's own file names are
fine too, as long as each contains its scene number after "scene").
Run `python run.py build <slug>` afterwards; narration is not redone.
"""
import math
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import script as script_mod, voice
from pipeline.config import load_config, slot_dir
from run import apply_format

slug, folder = sys.argv[1], Path(sys.argv[2]).expanduser()
out = slot_dir(slug)
data = script_mod.load_script(out)
cfg = load_config()
apply_format(cfg, data)
max_shot = float(cfg["video"]["max_shot_seconds"])

clips = {}
for f in folder.glob("*.mp4"):
    m = re.search(r"scene\D*(\d+)", f.name, re.I)
    if m:
        clips[int(m.group(1))] = f

index = 0
for i, scene in enumerate(data["scenes"]):
    wav = out / "audio" / f"{i:03d}.wav"
    if not wav.exists():
        raise SystemExit(f"{wav} is missing - run `run.py build {slug}` once "
                         f"first so the narration fixes the shot lengths")
    dur = voice.duration(wav)
    n = max(1, math.ceil(dur / max_shot))
    per = dur / n
    clip = clips.get(i + 1)
    if clip:
        length = voice.duration(clip)
        for k in range(n):
            # A clip a little shorter than its scene gives its last piece an
            # earlier start rather than a piece that runs out of picture.
            start = max(0.0, min(k * per, length - per))
            for old in (out / "media").glob(f"{index + k:03d}.*"):
                old.unlink()
            (out / "shots" / f"{index + k:04d}.mp4").unlink(missing_ok=True)
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}",
                 "-i", str(clip), "-t", f"{per:.3f}", "-an",
                 "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                 str(out / "media" / f"{index + k:03d}.mp4")],
                check=True)
        print(f"scene {i + 1}: {clip.name} -> shots {index}..{index + n - 1}"
              f" ({n} x {per:.1f}s, clip {length:.1f}s)")
    else:
        print(f"scene {i + 1}: no clip, keeps shots {index}..{index + n - 1}")
    index += n
