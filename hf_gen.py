"""Animate the channel's own dog for a slot's shots, on a Hugging Face Space.

Wan 2.2 image-to-video starts every clip from a real photograph of the dog,
so the animal on screen is this one rather than a plausible stranger. It
runs on the Space's GPU; only the finished clips come back to this machine.

    python hf_gen.py <slug>            # on the Hugging Face Space
    python hf_gen.py <slug> --ltx      # LTX-2.3 on Lightricks' Space instead
    python hf_gen.py <slug> --kaggle   # on Kaggle's GPU (kaggle_gen.py)
    python hf_gen.py <slug> --flow     # by hand in Google Flow (see on_flow)

Each scene names a photo in `source_photo` (a key of visuals.hf.photos).
Consecutive shots of a scene that share a query share one clip of up to
five seconds - the model's ceiling - cut in order.

With `"continuity": "chain"` in script.json, a clip that follows another in
the same scene starts from that clip's last frame instead of from the photo,
so a ten-second scene plays as one unbroken take rather than two restarts
of the same pose. A scene with `"continue": true` also picks up where the
scene before it ended; any other scene starts fresh from its photo, which is
where a cut belongs anyway - a new place, or the other dog. ZeroGPU gives a few GPU minutes a day (more with HF_TOKEN
in .env), so this stops cleanly when the quota runs out; run it again later
and it carries on from the first clip it does not have yet.
Run `python run.py build <slug>` afterwards.
"""
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from pipeline import script as script_mod, voice
from pipeline.config import load_config, slot_dir
from run import apply_format

CLIP_MAX = 5.0


def chunks(data: dict, out: Path, max_shot: float) -> list[dict]:
    """Group each scene's shots into clips, the way build_shots will cut them.

    Each clip records whether it continues the one before it (`chain`).
    """
    chain = data.get("continuity") == "chain"
    plan, index = [], 0
    for i, scene in enumerate(data["scenes"]):
        n = max(1, math.ceil(voice.duration(out / "audio" / f"{i:03d}.wav") / max_shot))
        per = voice.duration(out / "audio" / f"{i:03d}.wav") / n
        queries = script_mod.scene_queries(scene)
        for k in range(n):
            pos = 0 if n == 1 else int((k + 0.5) / n * len(queries))
            query = queries[min(len(queries) - 1, pos)]
            last = plan[-1] if plan else None
            if (last and last["scene"] == i and last["query"] == query
                    and (len(last["shots"]) + 1) * per <= CLIP_MAX):
                last["shots"].append(index)
            else:
                # `source_photos` names a photo per query, in step with
                # `visual_queries`; `source_photo` is one for the whole scene.
                photos = scene.get("source_photos") or []
                q = queries.index(query)
                photo = (photos[q] if q < len(photos)
                         else scene.get("source_photo", "front"))
                follows = bool(last) and (last["scene"] == i or
                                          (k == 0 and bool(scene.get("continue"))))
                plan.append({"scene": i, "query": query, "per": per,
                             "photo": photo, "shots": [index],
                             "chain": chain and follows})
            index += 1
    return plan


def model_sized(path: Path) -> str:
    """The photo as base64 JPEG, no larger than the model will use.

    Kaggle refuses a job whose source passes 1 MB, and the photos travel
    inside it; Wan never sees more than 832 px on the long side anyway.
    """
    import base64
    import io
    from PIL import Image

    image = Image.open(path).convert("RGB")
    image.thumbnail((832, 832), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def last_frame(clip: Path) -> Path:
    """The final frame of `clip`, as the start image of the clip after it."""
    still = clip.with_suffix(".last.jpg")
    if not still.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-sseof", "-0.25", "-i", str(clip),
                        "-update", "1", "-q:v", "2", str(still)], check=True)
    return still


def ltx_clip(client, start_image: Path, chunk: dict, seconds: float, cfg: dict,
             seed: int):
    """One clip from Lightricks' LTX-2.3 Space, the same shot the Wan Space makes.

    Sharper than Wan (1024x1536 against 832 px), and holds the dog's face
    for the whole clip - but only with a prompt that pins it down. Asked to
    "bounce", it walked the dog into the lens and smeared; told the camera
    holds still and the dog stays in the same spot, it kept her in frame and
    in focus throughout. `visuals.ltx.style` carries that - without naming
    any gear, since "on a tripod" drew a tripod into the room.

    The frame size is the Space's own choice for the photo (it rounds every
    portrait to 2:3); render.py crops it to the video's frame.
    """
    from gradio_client import handle_file

    ltx = cfg["visuals"].get("ltx") or {}
    style = (ltx.get("style") or cfg["visuals"]["hf"]["style"]).strip()
    image = handle_file(str(start_image))
    width, height = client.predict(image, bool(ltx.get("high_res", True)),
                                   api_name="/on_image_upload")
    size = [v["value"] if isinstance(v, dict) else v for v in (width, height)]
    return client.predict(
        input_image=image, prompt=f"{chunk['query'].strip()} {style}",
        duration=round(seconds, 1), enhance_prompt=False, seed=seed,
        randomize_seed=False, width=int(size[0]), height=int(size[1]),
        api_name="/generate_video")


def on_kaggle(plan: list[dict], cfg: dict, dest: Path, slug: str) -> None:
    """Every clip still missing, as one Kaggle job; its output lands in dest."""
    import base64
    import kaggle_gen

    hf = cfg["visuals"]["hf"]
    fps = cfg["visuals"]["kaggle"]["fps"]
    jobs, images = [], {}
    for c, chunk in enumerate(plan):
        if (dest / f"c{c:02d}.mp4").exists():
            continue
        seconds = min(CLIP_MAX, len(chunk["shots"]) * chunk["per"] + 0.2)
        job = {
            "name": f"c{c:02d}", "seed": 42 + c,
            "prompt": f"{chunk['query']} {hf['style']}",
            "image": chunk["photo"],
            # Wan wants 4k+1 frames; the Space rounds the same way.
            "frames": 4 * max(2, min(20, round(seconds * fps / 4))) + 1,
        }
        if chunk["chain"]:
            prev = dest / f"c{c - 1:02d}.mp4"
            if prev.exists():
                # Made on an earlier run: its last frame travels as the image.
                job["image"] = f"last_c{c - 1:02d}"
                images[job["image"]] = model_sized(last_frame(prev))
            else:
                # Made earlier in this same job: wan_job hands the frame on,
                # and falls back to the photo if that clip failed.
                job["chain"] = True
        jobs.append(job)
        images.setdefault(chunk["photo"], model_sized(ROOT / hf["photos"][chunk["photo"]]))
    if not jobs:
        return
    print(f"kaggle: {len(jobs)} clips from {len(images)} photos")
    kaggle_gen.run(jobs, images, cfg, dest / "_kaggle", kernel=f"pawfect-{slug}")
    for clip in (dest / "_kaggle").glob("c*.mp4"):
        clip.replace(dest / clip.name)


def flow_takes(plan: list[dict], dest: Path, longest: float) -> list[list[int]]:
    """The missing clips, grouped into the takes Flow will be asked for.

    A chained clip picks up where the one before it stopped, and Flow's take
    is longer than a clip, so a run of chained clips goes to Flow as one
    take and is split back into clips afterwards - one unbroken piece of
    motion rather than a joint at every clip boundary. A take starts afresh
    at every clip that does not chain, and when the next clip would push it
    past `longest` seconds.
    """
    takes: list[list[int]] = []
    for c, chunk in enumerate(plan):
        if (dest / f"c{c:02d}.mp4").exists():
            continue
        take = takes[-1] if takes else None
        if (take and chunk["chain"] and take[-1] == c - 1
                and sum(clip_seconds(plan[k]) for k in take + [c]) <= longest):
            take.append(c)
        else:
            takes.append([c])
    return takes


def clip_seconds(chunk: dict) -> float:
    return min(CLIP_MAX, len(chunk["shots"]) * chunk["per"] + 0.2)


def on_flow(plan: list[dict], cfg: dict, dest: Path, out: Path) -> None:
    """Take in clips made by hand in Google Flow, and list the ones still missing.

    Flow has no API, so this is a hand-off rather than a call. Each run
    writes `flow_in/SHOTLIST.md` - one entry per take still missing, with
    its prompt and its start frame in `flow_in/start/` - and takes in any
    take downloaded into `flow_in/` under its name (`c03.mp4`, or
    `c03-redo.mp4`; the newest wins). A take is cut into the clips it covers
    as `hf_out/cNN.mp4`, the numbering the Space and Kaggle write too, so
    the three routes can fill one video between them.

    A take is cut to exactly the seconds its clips need, so the last frame
    the next chained take starts from is a frame that actually reaches the
    screen. A take that chains from one not made yet has no start frame, so
    it is listed as waiting and gets one on the run after its predecessor.
    """
    import shutil

    hf = cfg["visuals"]["hf"]
    flow = cfg["visuals"].get("flow") or {}
    style = (flow.get("style") or hf["style"]).strip()
    longest = float(flow.get("take_seconds", 8))
    inbox = out / "flow_in"
    starts = inbox / "start"
    starts.mkdir(parents=True, exist_ok=True)

    # Taken in one take at a time, in order: a take's grouping depends on
    # which clips exist, and the last frame of one feeds the next.
    for take in flow_takes(plan, dest, longest):
        files = sorted(inbox.glob(f"c{take[0]:02d}*.mp4"), key=lambda p: p.stat().st_mtime)
        if not files:
            continue
        # The files are named after a take's first clip, and a take is
        # recomputed each run, so the one downloaded may have been listed as
        # longer or shorter than this; it is cut clip by clip until it runs
        # out, and anything it does not reach stays missing.
        length, at = voice.duration(files[-1]), 0.0
        for c in take:
            seconds = clip_seconds(plan[c])
            if at + 1.0 > length:
                break
            # Flow's own soundtrack goes: narration and music are ours.
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{at:.3f}",
                            "-i", str(files[-1]), "-t", f"{seconds:.3f}", "-an",
                            "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                            str(dest / f"c{c:02d}.mp4")], check=True)
            dest.joinpath(f"c{c:02d}.last.jpg").unlink(missing_ok=True)
            at += len(plan[c]["shots"]) * plan[c]["per"]
        print(f"flow: took {files[-1].name} as clips "
              f"{', '.join(f'c{c:02d}' for c in take if (dest / f'c{c:02d}.mp4').exists())}")

    w, h = int(cfg["video"]["width"]), int(cfg["video"]["height"])
    lines = [f"# Flow shot list: {out.name}", "",
             f"Aspect **{'9:16' if h > w else '16:9'}**. For each take: Frames to "
             f"Video, start frame from `start/`, paste the prompt, and download "
             f"the result into this folder under the take's name (`c03.mp4`, or "
             f"`c03-redo.mp4` to replace it). Then run "
             f"`python hf_gen.py {out.name} --flow` again - takes marked waiting "
             f"get their start frame once the take before them is in.", ""]
    takes = flow_takes(plan, dest, longest)
    for stale in starts.iterdir():
        stale.unlink()
    waiting = 0
    for take in takes:
        first = plan[take[0]]
        seconds = sum(clip_seconds(plan[c]) for c in take)
        covers = f"c{take[0]:02d}" + (f"-c{take[-1]:02d}" if len(take) > 1 else "")
        head = f"## {covers} - scene {first['scene'] + 1}, needs {seconds:.1f}s"
        prev = dest / f"c{take[0] - 1:02d}.mp4"
        if first["chain"] and not prev.exists():
            waiting += 1
            lines += [head, "", f"Waiting: starts from the last frame of "
                      f"c{take[0] - 1:02d}. Make that take first.", ""]
            continue
        source = last_frame(prev) if first["chain"] else ROOT / hf["photos"][first["photo"]]
        start = starts / f"c{take[0]:02d}{source.suffix.lower()}"
        shutil.copyfile(source, start)
        origin = (f"last frame of c{take[0] - 1:02d}" if first["chain"]
                  else f"photo `{first['photo']}`")
        beats = [q.strip().rstrip(".") + "." for q in
                 dict.fromkeys(plan[c]["query"] for c in take)]
        action = " ".join([beats[0]] + ["Then " + b[0].lower() + b[1:] for b in beats[1:]])
        lines += [head, "", f"Start frame: `start/{start.name}` ({origin})", "",
                  "```", f"{action} {style}", "```", ""]
    (inbox / "SHOTLIST.md").write_text("\n".join(lines))
    if takes:
        print(f"flow: {len(takes)} takes to make ({waiting} waiting on the take "
              f"before them) - see {inbox / 'SHOTLIST.md'}")


def main() -> None:
    slug = sys.argv[1]
    use_kaggle = "--kaggle" in sys.argv[2:]
    use_flow = "--flow" in sys.argv[2:]
    use_ltx = "--ltx" in sys.argv[2:]
    cfg = load_config()
    hf = cfg["visuals"]["hf"]
    out = slot_dir(slug)
    data = script_mod.load_script(out)
    apply_format(cfg, data)

    if not all((out / "audio" / f"{i:03d}.wav").exists()
               for i in range(len(data["scenes"]))):
        print("narration")
        voice.narrate_scenes(data["scenes"], cfg, out)

    plan = chunks(data, out, float(cfg["video"]["max_shot_seconds"]))
    dest = out / "hf_out"
    dest.mkdir(exist_ok=True)

    if use_kaggle:
        on_kaggle(plan, cfg, dest, slug)

    client = None
    if not (use_kaggle or use_flow):
        from gradio_client import Client
        import httpx
        space = (cfg["visuals"].get("ltx") or {}).get("space", "Lightricks/LTX-2-3") \
            if use_ltx else hf["space"]
        client = Client(space, verbose=False, token=os.environ.get("HF_TOKEN") or None,
                        httpx_kwargs={"timeout": httpx.Timeout(120.0)})
    stopped = None
    for c, chunk in enumerate(plan):
        clip = dest / f"c{c:02d}.mp4"
        if clip.exists() or stopped or client is None:
            continue
        seconds = min(CLIP_MAX, len(chunk["shots"]) * chunk["per"] + 0.2)
        start_image = ROOT / hf["photos"][chunk["photo"]]
        prev = dest / f"c{c - 1:02d}.mp4"
        if chunk["chain"] and prev.exists():
            start_image = last_frame(prev)
        started = time.time()
        from gradio_client import handle_file
        try:
            if use_ltx:
                video, _ = ltx_clip(client, start_image, chunk, seconds, cfg, seed=42 + c)
            else:
                video, _ = client.predict(
                    input_image=handle_file(str(start_image)),
                    prompt=f"{chunk['query']} {hf['style']}",
                    steps=hf.get("steps", 6), duration_seconds=round(seconds, 1),
                    guidance_scale=1, guidance_scale_2=1, seed=42 + c,
                    randomize_seed=False, api_name="/generate_video")
        except Exception as exc:
            # Out of quota is the same answer for every clip until it resets,
            # so the first refusal ends the run instead of asking again.
            stopped = f"{type(exc).__name__}: {exc}"[:300]
            print(f"clip {c}: stopped - {stopped}")
            continue
        path = video["video"] if isinstance(video, dict) else video
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-an",
                        "-c:v", "copy", str(clip)], check=True)
        print(f"clip {c}: scene {chunk['scene'] + 1}, {seconds:.1f}s, "
              f"{time.time() - started:.0f}s wall")

    # After the other routes, so the shot list leaves out whatever the Space
    # or Kaggle has just made.
    if use_flow:
        on_flow(plan, cfg, dest, out)

    media = out / "media"
    media.mkdir(exist_ok=True)
    have = 0
    for c, chunk in enumerate(plan):
        clip = dest / f"c{c:02d}.mp4"
        if not clip.exists():
            continue
        length, per = voice.duration(clip), chunk["per"]
        for j, index in enumerate(chunk["shots"]):
            start = max(0.0, min(j * per, length - per))
            for old in media.glob(f"{index:03d}.*"):
                old.unlink()
            (out / "shots" / f"{index:04d}.mp4").unlink(missing_ok=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}",
                            "-i", str(clip), "-t", f"{per:.3f}", "-an",
                            "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                            str(media / f"{index:03d}.mp4")], check=True)
        have += 1
    print(f"\n{have}/{len(plan)} clips in place"
          + (f" - run again later for the rest" if have < len(plan) else ""))


if __name__ == "__main__":
    main()
