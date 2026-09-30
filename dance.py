"""Make a dance video of the dogs: one continuous generated take, synced to a song, no cuts.

    python dance.py dances/<name>.json            # every step that isn't done yet
    python dance.py dances/<name>.json music      # fetch the song preview, find the beat, loop it long enough
    python dance.py dances/<name>.json clips      # generate the chains on Kaggle (waits; resumable)
    python dance.py dances/<name>.json clips --ltx  # or add LTX-2.3 links on ZeroGPU until the quota runs out
    python dance.py dances/<name>.json edit [--chain B] [--from 0.5]
    python dance.py dances/<name>.json sheet [clip.mp4]   # contact sheets for a frame-by-frame check

The spec (checked in under dances/) says what to make: the song to search on iTunes, the length, the dog's
description, the moves and the chains. Everything generated goes to workspace/<name>/ (music/, chain/,
qa/, final.mp4 with _silent and _click copies), and the finished video is copied to tiktok_ready/.
Shared settings live in config.yaml under `dance:`.

A chain is one take: each clip starts from the last frame of the clip before it. Several chains with
different start photos and seeds are made so the best one can be used whole - there is never a cut between
chains. A spec never has a script.json, so a dance slot is never queued for upload.
"""
import json
import shutil
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


def links(spec: dict) -> list[str]:
    moves = spec["moves"]
    n = spec.get("links", len(moves))
    return [moves[i % len(moves)] for i in range(n)]


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


def step_clips_kaggle(spec: dict, cfg: dict, out: Path) -> None:
    """Each chain is its own Kaggle job, so two chains generate at once (Kaggle runs two GPU sessions).

    A chain with `"anchor": {"soften": 0.7}` resets each link's start frame to the real photo's tone and
    takes the edge off the sharpening before it goes back in - without it, a chain of a dozen links drifts
    to a painted look (kaggle/wan_job.py, `anchored`).
    """
    from concurrent.futures import ThreadPoolExecutor
    import hf_gen
    import kaggle_gen

    frames = int(cfg["dance"]["link_frames"])
    base = spec.get("kernel", f"pawfect-{out.name}")
    todo = [ch for ch in spec["chains"] if len(chain_clips(spec, out, ch["name"])) < len(links(spec))]
    if not todo:
        print("clips: all chains are here")
        return

    def one(ch: dict) -> None:
        jobs = [{"name": f"{ch['name']}{i}", "seed": ch["seed"] + i, "prompt": prompt(spec, move, cfg),
                 "image": ch["photo"], "frames": frames, "chain": i > 0, "anchor": ch.get("anchor")}
                for i, move in enumerate(links(spec))]
        images = {ch["photo"]: hf_gen.model_sized(ROOT / cfg["visuals"]["hf"]["photos"][ch["photo"]])}
        kaggle_gen.run(jobs, images, cfg, out / "chain" / ch["name"], kernel=f"{base}-{ch['name'].lower()}")

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(one, todo))


def step_clips_ltx(spec: dict, cfg: dict, out: Path) -> None:
    """Add links on Lightricks' LTX-2.3 Space, chain by chain, until done or out of ZeroGPU quota."""
    import os
    import hf_gen
    import httpx
    from gradio_client import Client

    dest = out / "chain_ltx"
    dest.mkdir(parents=True, exist_ok=True)
    client = Client((cfg["visuals"].get("ltx") or {}).get("space", "Lightricks/LTX-2-3"), verbose=False,
                    token=os.environ.get("HF_TOKEN") or None, httpx_kwargs={"timeout": httpx.Timeout(120.0)})
    seconds = float(cfg["dance"]["ltx_seconds"])
    for ch in spec["chains"]:
        for i, move in enumerate(links(spec)):
            clip = clip_path(dest, ch["name"], i)
            if clip.exists():
                continue
            start = (ROOT / cfg["visuals"]["hf"]["photos"][ch["photo"]] if i == 0
                     else hf_gen.last_frame(clip_path(dest, ch["name"], i - 1)))
            print(f"  {clip.name}: {move[:60]}...")
            try:
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
    for folder in (out / "chain" / name, out / "chain", out / "chain_ltx"):
        clips = [clip_path(folder, name, i) for i in range(len(links(spec)))]
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
    report = dance.edit_take(clips, grid, out / "music" / "track.wav", float(spec["seconds"]), final, cfg, skip)
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

    grid = step_music(spec, cfg, out) if step in ("all", "music", "edit") else None
    if step in ("all", "clips"):
        (step_clips_ltx if "--ltx" in args else step_clips_kaggle)(spec, cfg, out)
    if step in ("all", "edit"):
        final = step_edit(spec, cfg, out, grid, opt("--chain"), float(opt("--from", 0)))
        step_sheet(out, [final])
    if step == "sheet":
        named = [Path(a) for a in args[1:] if a.endswith(".mp4")]
        step_sheet(out, named or sorted((out / "chain").glob("*.mp4")) + sorted((out / "chain_ltx").glob("*.mp4")))


if __name__ == "__main__":
    main()
