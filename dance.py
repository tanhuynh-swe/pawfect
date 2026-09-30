"""Make a dance video of the dogs: one continuous generated take, synced to a song, no cuts.

    python dance.py dances/<name>.json            # every step that isn't done yet
    python dance.py dances/<name>.json music      # fetch the song preview, find the beat, loop it long enough
    python dance.py dances/<name>.json clips [--chain T]  # generate the chains on Kaggle (waits; resumable)
    python dance.py dances/<name>.json clips --ltx  # or add LTX-2.3 links on ZeroGPU until the quota runs out
    python dance.py dances/<name>.json edit [--chain B] [--from 0.5]
    python dance.py dances/<name>.json sheet [clip.mp4]   # contact sheets for a frame-by-frame check

The spec (checked in under dances/) says what to make: the song to search on iTunes, the length, the dog's
description, the moves and the chains. With `sections` in place of `moves` the dance follows the song rather
than cycling through moves: each section names how many bars it lasts, what the music does there and the move
for it, becomes one link sized to those bars, and the edit starts each link on its section's first beat. Everything generated goes to workspace/<name>/ (music/, chain/,
qa/, final.mp4 with _silent and _click copies), and the finished video is copied to tiktok_ready/.
Shared settings live in config.yaml under `dance:`.

A chain is one take: each clip starts from the last frame of the clip before it. Several chains with
different start photos and seeds are made so the best one can be used whole - there is never a cut between
chains. A spec never has a script.json, so a dance slot is never queued for upload.
"""
import json
import shutil

import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from pipeline import dance
from pipeline.config import load_config, slot_dir


def load_spec(path: str) -> tuple[str, dict]:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = ROOT / path
    return p.stem, json.loads(p.read_text(encoding="utf-8"))


def prompt(spec: dict, move: str, cfg: dict) -> str:
    style = " ".join((spec.get("style") or cfg["dance"]["style"]).split())
    return f"{spec['subject'].strip()} {move.strip()} {style}"


def links(spec: dict, ch: dict | None = None) -> list[str]:
    moves = [s["move"] for s in spec["sections"]] if "sections" in spec else spec["moves"]
    n = (ch or {}).get("links", spec.get("links", len(moves)))
    return [moves[i % len(moves)] for i in range(n)]


def link_frames(spec: dict, cfg: dict, grid: dict | None) -> list[int]:
    """Frames per link: a section's link lasts its bars at normal speed (Wan wants 4k+1 frames, and a
    chained link loses its first, repeated frame); without sections every link is `dance.link_frames`."""
    if "sections" not in spec:
        return [int(cfg["dance"]["link_frames"])] * len(links(spec))
    fps = float(cfg["visuals"]["kaggle"]["fps"])
    return [4 * round(s["bars"] * 4 * grid["beat"] * fps / 4) + 1 for s in spec["sections"]]


def clip_path(dest: Path, chain: str, i: int) -> Path:
    return dest / f"{chain}{i}.mp4"


def step_music(spec: dict, cfg: dict, out: Path) -> dict:
    music = out / "music"
    grid_file = music / "grid.json"
    if grid_file.exists():
        return json.loads(grid_file.read_text())
    preview = music / "preview.m4a"
    info = dance.itunes_preview(spec["song"], cfg["dance"]["itunes_country"], preview)
    print(f"song: {info['track']} - {info['artist']} ({info['explicitness']})")
    grid = dance.beat_grid(preview)
    grid.update(info)
    grid.update(dance.loop_track(preview, grid, float(spec["seconds"]), music / "track.wav"))
    print(f"  {grid['bpm']} BPM, downbeat {grid['downbeat']:.3f}s, harmony repeats every {grid['cycle_bars']} "
          f"bar(s); track {grid['track_seconds']:.2f}s ({grid['loops']} loop joins)")
    grid_file.write_text(json.dumps(grid, indent=2))
    return grid


def step_clips_kaggle(spec: dict, cfg: dict, out: Path, grid: dict | None = None, only: str | None = None) -> None:
    """Each chain is its own Kaggle job, so two chains generate at once (Kaggle runs two GPU sessions).

    A chain with `"anchor": {"soften": 0.7}` resets each link's start frame to the real photo's tone and
    takes the edge off the sharpening before it goes back in - without it, a chain of a dozen links drifts
    to a painted look (kaggle/wan_job.py, `anchored`).
    """
    from concurrent.futures import ThreadPoolExecutor
    import hf_gen
    import kaggle_gen

    frames = link_frames(spec, cfg, grid)
    base = spec.get("kernel", f"pawfect-{out.name}")
    todo = [ch for ch in spec["chains"] if len(chain_clips(spec, out, ch["name"])) < len(links(spec, ch))
            and only in (None, ch["name"])]
    if not todo:
        print("clips: all chains are here")
        return

    def one(ch: dict) -> None:
        jobs = [{"name": f"{ch['name']}{i}", "seed": ch["seed"] + i, "prompt": prompt(spec, move, cfg),
                 "image": ch["photo"], "frames": frames[i], "chain": i > 0, "anchor": ch.get("anchor")}
                for i, move in enumerate(links(spec, ch))]
        images = {ch["photo"]: hf_gen.model_sized(ROOT / cfg["visuals"]["hf"]["photos"][ch["photo"]])}
        kaggle_gen.run(jobs, images, cfg, out / "chain" / ch["name"], kernel=f"{base}-{ch['name'].lower()}")

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(one, todo))


def flf_clip(client, first: Path, last: Path, prompt_text: str, seconds: float, cfg: dict, seed: int,
             audio: Path | None = None):
    """One LTX-2.3 clip that starts on `first` and ends on `last` (visuals.ltx.flf_space).

    With `audio`, the section's own stretch of the song goes in as the clip's soundtrack, so the model
    moves the dog to that music rather than to a rhythm it imagines."""
    from gradio_client import handle_file

    ltx = cfg["visuals"].get("ltx") or {}
    style = (ltx.get("style") or "").strip()
    a, b = handle_file(str(first)), handle_file(str(last))
    width, height = client.predict(a, b, bool(ltx.get("high_res", True)), api_name="/on_image_upload")
    size = [v["value"] if isinstance(v, dict) else v for v in (width, height)]
    return client.predict(
        first_image=a, last_image=b, input_audio=handle_file(str(audio)) if audio else None,
        prompt=f"{prompt_text} {style}".strip(), duration=round(seconds, 1), enhance_prompt=False,
        seed=seed, randomize_seed=False, width=int(size[0]), height=int(size[1]), api_name="/generate_video")


def step_clips_ltx(spec: dict, cfg: dict, out: Path, grid: dict | None = None, only: str | None = None) -> None:
    """Add links on Lightricks' LTX-2.3 Space, chain by chain, until done or out of ZeroGPU quota.

    A chained link starts from the last frame of the one before, and LTX drifts on the first join (quilted
    fur, a harness where the orange spot was). With `home` in the spec, each link is made instead on the
    first/last-frame Space from the home frame back to the home frame: the links meet on the same picture
    and nothing is handed down to compound. A section with `clip` uses that footage, up to `to_frame`."""
    import os
    import hf_gen
    import httpx
    from gradio_client import Client

    dest = out / "chain_ltx"
    dest.mkdir(parents=True, exist_ok=True)
    client = Client((cfg["visuals"].get("ltx") or {}).get("space", "Lightricks/LTX-2-3"), verbose=False,
                    token=os.environ.get("HF_TOKEN") or None, httpx_kwargs={"timeout": httpx.Timeout(120.0)})
    home = ROOT / spec["home"] if spec.get("home") else None
    flf = None
    starts = np.cumsum([0] + [s["bars"] * 4 * grid["beat"] for s in spec.get("sections", [])])
    for ch in spec["chains"]:
        if only not in (None, ch["name"]):
            continue
        for i, move in enumerate(links(spec, ch)):
            sec = spec["sections"][i] if "sections" in spec else {}
            if sec.get("clip") and not clip_path(dest, ch["name"], i).exists():
                cut = f"select=lte(n\\,{int(sec['to_frame'])})" if "to_frame" in sec else "null"
                dance.run("ffmpeg", "-v", "error", "-y", "-i", ROOT / sec["clip"], "-vf", f"{cut},setpts=N/FRAME_RATE/TB",
                          "-an", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", clip_path(dest, ch["name"], i))
                continue
            # A section's link lasts its bars; otherwise every link is `dance.ltx_seconds`.
            seconds = (spec["sections"][i]["bars"] * 4 * grid["beat"] if "sections" in spec
                       else float(cfg["dance"]["ltx_seconds"]))
            clip = clip_path(dest, ch["name"], i)
            if clip.exists():
                continue
            print(f"  {clip.name}: {move[:60]}...")
            try:
                if home:
                    if flf is None:
                        flf = Client(cfg["visuals"]["ltx"]["flf_space"], verbose=False,
                                     token=os.environ.get("HF_TOKEN") or None,
                                     httpx_kwargs={"timeout": httpx.Timeout(120.0)})
                    audio = None
                    if spec.get("audio"):
                        audio = dest / f"{clip.stem}.audio.wav"
                        dance.run("ffmpeg", "-v", "error", "-y", "-ss", f"{starts[i]:.4f}", "-t", f"{seconds:.4f}",
                                  "-i", out / "music" / "track.wav", audio)
                    video, _ = flf_clip(flf, home, home, prompt(spec, move, cfg), seconds + 1 / 24, cfg,
                                        ch["seed"] + i, audio)
                else:
                    start = (ROOT / cfg["visuals"]["hf"]["photos"][ch["photo"]] if i == 0
                             else hf_gen.last_frame(clip_path(dest, ch["name"], i - 1)))
                    video, _ = hf_gen.ltx_clip(client, start, {"query": prompt(spec, move, cfg)}, seconds, cfg,
                                               seed=ch["seed"] + i)
            except Exception as exc:              # quota, a busy Space: stop, the next run carries on here
                print(f"  stopped: {str(exc)[:200]}")
                return
            src = video["video"] if isinstance(video, dict) else video
            if i == 0:
                shutil.copy(src, clip)
            else:                                 # frame 0 repeats the last one: a hitch at the join
                dance.run("ffmpeg", "-v", "error", "-y", "-i", src, "-vf", "select=gte(n\\,1),setpts=N/FRAME_RATE/TB",
                          "-an", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", clip)


def chain_clips(spec: dict, out: Path, name: str) -> list[Path]:
    # chain/<name>/ (one Kaggle job per chain), chain/ (older runs: all chains in one job), chain_ltx/
    ch = next((c for c in spec["chains"] if c["name"] == name), None)
    for folder in (out / "chain" / name, out / "chain", out / "chain_ltx"):
        clips = [clip_path(folder, name, i) for i in range(len(links(spec, ch)))]
        if clips[0].exists():
            have = [c for c in clips if c.exists()]
            return have[:next((i for i, c in enumerate(clips) if not c.exists()), len(clips))]
    return []


def step_edit(spec: dict, cfg: dict, out: Path, grid: dict, chain: str | None, skip: float) -> Path:
    name = chain or spec.get("use_chain") or spec["chains"][0]["name"]
    # `use_links` stops the take before a chain's late links drift; `edit` in the spec overrides `dance:`
    # settings (speed, interp) for this video only.
    clips = chain_clips(spec, out, name)[:spec.get("use_links")]
    if not clips:
        raise SystemExit(f"no clips for chain {name} yet - run the clips step")
    cfg = {**cfg, "dance": {**cfg["dance"], **spec.get("edit", {})}}
    final = out / "final.mp4"
    # Sections: link i starts on the first beat of section i, so each move plays over its part of the song.
    anchors = {}
    if "sections" in spec:
        beat_at, t = 0, 0.0
        for sec, clip in zip(spec["sections"], clips):
            anchors[beat_at] = t
            beat_at += 4 * sec["bars"]
            t += dance.probe(clip)[3]
    report = dance.edit_take(clips, grid, out / "music" / "track.wav", float(spec["seconds"]), final, cfg, skip,
                             anchors)
    print(f"edit: chain {name} ({len(clips)} clips) -> {json.dumps(report)}")
    ready = ROOT / "tiktok_ready"
    ready.mkdir(exist_ok=True)
    for suffix in ("", "_silent"):
        shutil.copy(out / f"final{suffix}.mp4", ready / f"{out.name}{suffix}.mp4")
    print(f"  {final}\n  copied to tiktok_ready/{out.name}.mp4 - check qa/ before posting")
    return final


def step_sheet(out: Path, clips: list[Path]) -> None:
    for clip in clips:
        sheet = out / "qa" / f"{clip.parent.name}_{clip.stem}.jpg"
        print(f"  {sheet.relative_to(ROOT)}: {dance.contact_sheet(clip, sheet, every=3)}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    slug, spec = load_spec(sys.argv[1])
    args = sys.argv[2:]
    opt = lambda k, d=None: args[args.index(k) + 1] if k in args and args.index(k) + 1 < len(args) else d
    step = args[0] if args and not args[0].startswith("-") else "all"
    cfg = load_config()
    out = slot_dir(slug)

    grid = step_music(spec, cfg, out) if step in ("all", "music", "clips", "edit") else None
    if step in ("all", "clips"):
        if "--ltx" in args:
            step_clips_ltx(spec, cfg, out, grid, opt("--chain"))
        else:
            step_clips_kaggle(spec, cfg, out, grid, opt("--chain"))
    if step in ("all", "edit"):
        final = step_edit(spec, cfg, out, grid, opt("--chain"), float(opt("--from", 0)))
        step_sheet(out, [final])
    if step == "sheet":
        named = [Path(a) for a in args[1:] if a.endswith(".mp4")]
        step_sheet(out, named or sorted((out / "chain").glob("*.mp4")) + sorted((out / "chain_ltx").glob("*.mp4")))


if __name__ == "__main__":
    main()
