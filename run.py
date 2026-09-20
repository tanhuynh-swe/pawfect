#!/usr/bin/env python3
"""Pawfect Love Animals — pipeline CLI.

  python run.py topic                 pick the next topic, write the brief
  python run.py script  <slug>        write (or collect) the script
  python run.py build   <slug>        narrate, fetch visuals, render, thumbnail
  python run.py approve <slug>        mark a finished video as human-reviewed
  python run.py publish <slug>        upload to YouTube or TikTok (based on script)
  python run.py make                  topic -> script -> build, in one go
  python run.py auto                  fully unattended: everything + publish (YouTube/TikTok)
  python run.py resume                clear the hold counter after a pause
  python run.py voicetest             render the same lines in 8 voices
  python run.py queue                 scripts waiting to be made into videos
  python run.py reauth                sign in to YouTube again (new permissions)
  python run.py tiktok                build queued TikTok shorts (for manual upload)
  python run.py status                what is in the workspace right now

Scripts can control where they publish by setting "publish_to": "tiktok" or "youtube" (default).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

import traceback

from pipeline import (notify, qa, render, research, script as script_mod,
                      thumbnail, upload, voice, tiktok_upload)
from pipeline.config import ROOT, WORKSPACE, load_config, slot_dir


def cmd_topic(args, cfg) -> str:
    topic = research.pick_topic(cfg, args.title)
    out = slot_dir(topic["slug"])
    research.save_brief(topic, out)
    print(f"\nTopic: {topic['title']}")
    print(f"Slug:  {topic['slug']}")
    if topic.get("related_searches"):
        print("People also search:")
        for s in topic["related_searches"][:5]:
            print(f"  - {s}")
    print(f"\nBrief written to {out / 'brief.json'}")
    return topic["slug"]


def cmd_script(args, cfg) -> None:
    out = slot_dir(args.slug)
    brief = json.loads((out / "brief.json").read_text(encoding="utf-8"))
    path = script_mod.write_script(brief, cfg, out)
    data = script_mod.load_script(out)
    words = sum(len(s["narration"].split()) for s in data["scenes"])
    mins = words / cfg["script"]["words_per_minute"]
    print(f"\nScript ready: {path}")
    print(f"  title:  {data['title']}")
    print(f"  scenes: {len(data['scenes'])}")
    print(f"  length: ~{mins:.1f} min ({words} words)")


def apply_format(cfg, data) -> bool:
    """Switch the render to vertical when the script asks for it.

    The format lives in the script rather than a flag, so a queued Short and a
    queued long-form video can sit side by side and the scheduler does not need
    to know the difference.
    """
    lang = data.get("language", cfg["channel"]["target_language"])
    if lang != cfg["channel"]["target_language"]:
        voices = cfg["voice"].get("voices", {})
        if lang in voices:
            cfg["voice"] = {**cfg["voice"], "edge_voice": voices[lang]}
            print(f"  language: {lang} — voice {voices[lang]}")
        else:
            print(f"  ! no voice configured for language '{lang}', using default")
        cfg["_language"] = lang

    if data.get("format") != "vertical":
        return False
    cfg["video"] = {**cfg["video"], **cfg.get("shorts", {})}
    if lang == "vi" and cfg["video"].get("font_vi"):
        cfg["video"]["font"] = cfg["video"]["font_vi"]
    # Ask the stock libraries for portrait footage too, otherwise every shot is
    # a landscape clip centre-cropped to a sliver.
    cfg["visuals"] = {**cfg["visuals"], "orientation": "portrait"}
    cfg["_vertical"] = True
    return True


def cmd_build(args, cfg) -> None:
    out = slot_dir(args.slug)
    data = script_mod.load_script(out)
    scenes = data["scenes"]
    if apply_format(cfg, data):
        w, h = cfg["video"]["width"], cfg["video"]["height"]
        print(f"\nVertical short: {w}x{h}")

    print("\n[1/5] narration")
    durations = voice.narrate_scenes(scenes, cfg, out)
    narration = voice.concat_narration(len(scenes), out)
    total = sum(durations)
    print(f"  total runtime: {total / 60:.1f} min")

    if cfg.get("_vertical"):
        lo_s, hi_s = cfg["video"].get("target_seconds", [35, 60])
        if total > 180:
            print(f"  ! {total:.0f}s is over YouTube's 3-minute Shorts limit")
        elif not (lo_s <= total <= hi_s):
            print(f"  ! {total:.0f}s is outside the {lo_s}-{hi_s}s sweet spot")
    else:
        lo, hi = cfg["channel"]["target_minutes"]
        if total / 60 < lo * 0.7:
            print(f"  ! shorter than your {lo}-{hi} min target — consider more scenes")

    print("\n[2/5] visuals")
    shots = render.build_shots(scenes, durations, cfg, out)
    print(f"  {len(shots)} shots")

    print("\n[3/5] captions")
    overlay = render.build_caption_overlay(scenes, durations, cfg, out)
    srt = render.build_srt(scenes, durations, out)
    print(f"  {srt.name} written for YouTube's caption track")

    print("\n[4/5] render")
    silent = render.concat_shots(shots, out)
    final = render.finalize(silent, narration, overlay, cfg, out)
    size_mb = final.stat().st_size / 1e6
    print(f"  {final}  ({size_mb:.0f} MB)")

    print("\n[5/5] thumbnail")
    if cfg.get("_vertical"):
        # Shorts are browsed in a vertical feed, where YouTube shows a frame
        # from the video rather than a 16:9 thumbnail. Making one would just
        # be a letterboxed image nobody sees.
        thumb = None
        print("  skipped (Shorts use a frame from the video)")
    else:
        # Pull the still from the caption-free cut, not the final one.
        thumb = thumbnail.make_thumbnail(
            silent, data.get("thumbnail_text", data["title"]), out, cfg,
            at=min(12.0, max(2.0, total * 0.35)),
        )
        print(f"  {thumb}")

    (out / "review.json").write_text(
        json.dumps({"approved": False, "notes": ""}, indent=2), encoding="utf-8"
    )
    print(
        f"\nWatch it: {final}\n"
        f"If it is good:  python run.py approve {args.slug}\n"
        f"Then:           python run.py publish {args.slug}"
    )
    return {"final": final, "thumbnail": thumb, "runtime_s": total, "script": data}


def cmd_approve(args, cfg) -> None:
    out = slot_dir(args.slug)
    path = out / "review.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data["approved"] = True
    data["notes"] = args.notes or data.get("notes", "")
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Approved: {args.slug}")


def cmd_publish(args, cfg) -> None:
    out = slot_dir(args.slug)
    final = out / "final.mp4"
    if not final.exists():
        raise SystemExit(f"No rendered video at {final}. Run build first.")

    if cfg["safety"]["require_human_review"]:
        review = out / "review.json"
        approved = (
            review.exists()
            and json.loads(review.read_text(encoding="utf-8")).get("approved")
        )
        if not approved:
            raise SystemExit(
                f"{args.slug} has not been reviewed.\n"
                f"Watch {final}, then run: python run.py approve {args.slug}\n"
                "(Or set safety.require_human_review: false — read the README "
                "section on the inauthentic-content policy before you do.)"
            )

    data = script_mod.load_script(out)
    destination = data.get("publish_to", "youtube")

    if destination == "tiktok":
        print(f"\n[publish to TikTok]")
        print(f"Uploading: {data['title']}")
        if args.at:
            print("  note: a draft cannot be scheduled — post it from the app.")
        publish_id = tiktok_upload.send_to_inbox(final, data)
        tiktok_upload.record(publish_id, out)
        research.mark_used(args.slug)
    else:
        # YouTube
        taken = [
            json.loads(p.read_text(encoding="utf-8")).get("publish_at")
            for p in WORKSPACE.glob("*/published.json")
        ]
        publish_at = args.at or upload.next_slot(cfg, [t for t in taken if t])
        print(f"Scheduling for {publish_at}")

        video_id = upload.upload_video(
            final, data, cfg, publish_at, out / "thumbnail.jpg"
        )
        upload.record(video_id, publish_at, out)
        research.mark_used(args.slug)
        print(
            "\nNote: if your API project has not passed YouTube's audit yet, this "
            "video stays private until you flip it to public in YouTube Studio."
        )


def cmd_make(args, cfg) -> None:
    slug = cmd_topic(args, cfg)
    args.slug = slug
    try:
        cmd_script(args, cfg)
    except SystemExit as exc:
        print(exc)
        return
    cmd_build(args, cfg)


CANDIDATE_VOICES = [
    # Microsoft's newer multilingual generation — the most human-sounding.
    "en-US-AndrewMultilingualNeural",   # male, warm, conversational
    "en-US-AvaMultilingualNeural",      # female, natural, very clear
    "en-US-BrianMultilingualNeural",    # male, relaxed and casual
    "en-US-EmmaMultilingualNeural",     # female, friendly and light
    # British, which suits calm explanatory pet content well.
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    # The older generation, included so you can hear the difference.
    "en-US-JennyNeural",
    "en-US-AriaNeural",
]


def cmd_voicetest(args, cfg) -> None:
    """Render the same paragraph in every candidate voice, so you can choose."""
    out = slot_dir("voicetest")
    sample = args.text
    if not sample:
        slug_dir = slot_dir(args.slug) if args.slug else None
        if slug_dir and (slug_dir / "script.json").exists():
            data = json.loads((slug_dir / "script.json").read_text(encoding="utf-8"))
            sample = " ".join(s["narration"] for s in data["scenes"][:2])
        else:
            sample = (
                "You let your dog out into the garden, and within about four "
                "seconds he has his head down, tearing at the lawn. The "
                "explanation you have probably read is that he is making "
                "himself sick. When researchers measured it, the numbers came "
                "out backwards."
            )

    print(f"Sample ({len(sample.split())} words):\n  {sample[:120]}...\n")
    working = []
    for name in CANDIDATE_VOICES:
        dest = out / f"{name}.mp3"
        try:
            subprocess.run(
                [sys.executable, "-m", "edge_tts", "--voice", name,
                 "--rate", str(cfg["voice"].get("edge_rate", "+0%")),
                 "--pitch", str(cfg["voice"].get("edge_pitch", "+0Hz")),
                 "--text", sample, "--write-media", str(dest)],
                check=True, capture_output=True, text=True,
            )
            print(f"  ok    {name}")
            working.append(name)
        except subprocess.CalledProcessError as exc:
            last = (exc.stderr or "").strip().splitlines()[-1:] or ["failed"]
            print(f"  skip  {name} — {last[0][:90]}")

    print(f"\n{len(working)} samples in: {out}")
    print("Listen to them, then put the one you like in config.yaml under "
          "voice.edge_voice, and rebuild with --refresh-audio.")


def _destination(d: Path) -> str:
    try:
        data = json.loads((d / "script.json").read_text(encoding="utf-8"))
        return data.get("publish_to", "youtube")
    except Exception:
        return "youtube"


def _queue_slots(destination: str = "youtube") -> list[Path]:
    """Workspace slots with a script, not yet published, for one destination.

    The destination filter is what stops a Vietnamese TikTok short being
    auto-uploaded to the English YouTube channel by the scheduler. Anything
    without an explicit publish_to is treated as YouTube, so existing scripts
    behave exactly as before.
    """
    out = []
    for d in sorted(WORKSPACE.glob("*")):
        if not d.is_dir() or d.name == "voicetest":
            continue
        if not (d / "script.json").exists() or (d / "published.json").exists():
            continue
        if _destination(d) != destination:
            continue
        out.append(d)
    return out


def next_queued() -> str | None:
    slots = _queue_slots()
    return slots[0].name if slots else None


def remaining_queue() -> int:
    return max(0, len(_queue_slots()) - 1)


def cmd_queue(args, cfg) -> None:
    for dest in ("youtube", "tiktok"):
        slots = _queue_slots(dest)
        print(f"\n{dest.upper()}  ({len(slots)} queued)")
        if not slots:
            if dest == "youtube":
                print("  empty — auto runs will write their own scripts "
                      "(needs ANTHROPIC_API_KEY in .env)")
            else:
                print("  empty")
            continue
        for d in slots:
            data = json.loads((d / "script.json").read_text(encoding="utf-8"))
            words = sum(len(s["narration"].split()) for s in data["scenes"])
            built = "built" if (d / "final.mp4").exists() else "not built"
            shape = "vertical" if data.get("format") == "vertical" else "16:9"
            lang = data.get("language", "en")
            # Measured from real renders: the Vietnamese voice runs ~123 wpm,
            # the English one ~150. Dividing by wpm gives MINUTES — multiply
            # by 60 for the seconds a short is judged in.
            wpm = 123 if lang == "vi" else 150
            length = (f"~{words / wpm * 60:.0f}s" if shape == "vertical"
                      else f"~{words / wpm:.1f} min")
            print(f"  {d.name}")
            print(f"    {data['title']}")
            print(f"    {len(data['scenes'])} scenes, {length}, "
                  f"{shape}, {lang}, {built}")
    if _queue_slots("tiktok"):
        print("\nTikTok videos are built but never auto-uploaded — post them "
              "from the phone app so you can attach affiliate products.")


def cmd_tiktok(args, cfg) -> None:
    """Build every queued TikTok short and collect them for phone upload.

    Deliberately never uploads. TikTok Shop affiliate products are attached in
    the app at posting time, and an API post cannot do that — so the useful
    output here is a finished file plus its caption text, ready to post.
    """
    slots = _queue_slots("tiktok")
    if not slots:
        print("Nothing queued for TikTok.")
        return

    ready = ROOT / "tiktok_ready"
    ready.mkdir(exist_ok=True)
    done = []

    for d in slots:
        print(f"\n{'=' * 60}\n{d.name}")
        args.slug = d.name
        built = cmd_build(args, dict(cfg))   # fresh cfg: format flags must not leak
        data = built["script"]
        dest = ready / f"{d.name}.mp4"
        shutil.copy(built["final"], dest)
        caption = ready / f"{d.name}.txt"
        caption.write_text(
            f"{data['title']}\n\n{data['description']}\n", encoding="utf-8"
        )
        done.append((d.name, built["runtime_s"], dest))

    print(f"\n{'=' * 60}\nReady to post, in {ready}:\n")
    for name, secs, dest in done:
        print(f"  {dest.name}  ({secs:.0f}s)")
        print(f"  {dest.with_suffix('.txt').name}  — caption to paste")
    print(
        "\nAirDrop these to your phone, post from the TikTok app, and attach "
        "the affiliate products there.\nRemember the paid-partnership "
        "disclosure toggle — TikTok requires it and hides posts that skip it."
    )


def cmd_tiktokauth(args, cfg) -> None:
    """One-time TikTok login. Uploading is granted per account, not per app."""
    tiktok_upload.authorize(args.code)


def cmd_reauth(args, cfg) -> None:
    upload.reauthorise()


HOLD_COUNTER = WORKSPACE / "consecutive_holds.json"


def _holds(delta: int | None = None) -> int:
    n = 0
    if HOLD_COUNTER.exists():
        n = json.loads(HOLD_COUNTER.read_text(encoding="utf-8")).get("count", 0)
    if delta is not None:
        n = 0 if delta == 0 else n + delta
        HOLD_COUNTER.write_text(json.dumps({"count": n}), encoding="utf-8")
    return n


def cmd_auto(args, cfg) -> None:
    """Unattended: topic -> script -> render -> automated gate -> publish.

    Never raises. Every outcome ends in an email digest, because a cron job
    that fails silently is worse than no cron job.
    """
    report: dict = {"notes": []}
    try:
        if cfg["script"]["mode"] != "api" and not next_queued():
            raise SystemExit(
                "Nothing queued and script.mode is not 'api'.\n"
                "Either add scripts to the queue, or set script.mode: api "
                "with ANTHROPIC_API_KEY in .env."
            )

        limit = int(cfg["safety"].get("max_consecutive_holds", 3))
        if _holds() >= limit:
            report["error"] = (
                f"Paused: {limit} runs in a row were held. Something upstream is "
                "wrong. Review the held videos, then reset with:\n"
                "  python run.py resume"
            )
            notify.send_digest(report)
            return

        queued = next_queued()
        if queued:
            # A script written ahead of time. Using the queue first means the
            # schedule keeps running with no API key and nothing to pay for,
            # and it degrades gracefully to writing its own once the queue
            # empties.
            slug = queued
            args.slug = slug
            data_q = script_mod.load_script(slot_dir(slug))
            print(f"\nUsing queued script: {data_q['title']}")
            report["notes"].append(f"queued script used ({remaining_queue()} left)")
        else:
            slug = cmd_topic(args, cfg)
            args.slug = slug
            cmd_script(args, cfg)
        built = cmd_build(args, cfg)
        data = built["script"]

        print("\n[gate] automated review")
        if cfg["safety"].get("auto_review", True):
            passed, issues = qa.run_gate(
                data, built["runtime_s"], cfg, slug, slot_dir(slug)
            )
        else:
            passed, issues = True, ["Automated review disabled in config."]
            report["notes"].append("auto_review is off — nothing reviewed this video.")

        for issue in issues:
            print(f"  - {issue}")

        if not passed:
            _holds(+1)
            qa_data = json.loads((slot_dir(slug) / "qa.json").read_text(encoding="utf-8"))
            report["held"] = {
                "slug": slug, "title": data["title"],
                "issues": qa_data["blocking"] or issues,
                "worst_line": qa_data.get("worst_line", ""),
            }
            print(f"\nHELD — not published. See {slot_dir(slug) / 'qa.json'}")
            notify.send_digest(report)
            return

        print("  passed")
        _holds(0)

        print("\n[publish]")
        destination = data.get("publish_to", "youtube")

        if destination == "tiktok":
            print(f"Uploading to TikTok: {data['title']}")
            publish_id = tiktok_upload.send_to_inbox(built["final"], data)
            tiktok_upload.record(publish_id, slot_dir(slug))
            research.mark_used(slug)

            report["published"] = {
                "title": data["title"],
                "url": "draft in the TikTok app — attach products, then post",
                "platform": "tiktok",
                "runtime_min": round(built["runtime_s"] / 60, 1),
            }
        else:
            # YouTube
            video_id = upload.upload_video(
                built["final"], data, cfg, None, built["thumbnail"]
            )
            privacy = upload.check_privacy(video_id)
            upload.record(video_id, None, slot_dir(slug))
            research.mark_used(slug)

            report["published"] = {
                "title": data["title"],
                "url": f"https://youtu.be/{video_id}",
                "privacy": privacy,
                "runtime_min": round(built["runtime_s"] / 60, 1),
                "forced_private": privacy == "private",
            }
            if privacy == "private":
                print("\n  YouTube forced this to private (API project not audited yet).")

        notify.send_digest(report)

    except SystemExit as exc:
        report["error"] = str(exc)
        notify.send_digest(report)
    except Exception:
        report["error"] = traceback.format_exc()
        print(report["error"])
        notify.send_digest(report)


def cmd_resume(args, cfg) -> None:
    _holds(0)
    print("Hold counter reset. Auto runs will resume.")


def cmd_status(args, cfg) -> None:
    rows = []
    for d in sorted(WORKSPACE.glob("*")):
        if not d.is_dir():
            continue
        has = lambda n: "yes" if (d / n).exists() else "-"
        approved = "-"
        if (d / "review.json").exists():
            approved = "yes" if json.loads(
                (d / "review.json").read_text(encoding="utf-8")
            ).get("approved") else "no"
        published = "-"
        if (d / "published.json").exists():
            published = json.loads(
                (d / "published.json").read_text(encoding="utf-8")
            )["url"]
        rows.append((d.name, has("script.json"), has("final.mp4"), approved, published))

    if not rows:
        print("Workspace is empty. Start with: python run.py make")
        return
    print(f"{'slug':<42}{'script':<9}{'video':<8}{'approved':<10}published")
    for r in rows:
        print(f"{r[0]:<42}{r[1]:<9}{r[2]:<8}{r[3]:<10}{r[4]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("topic"); p.add_argument("--title", default=None)
    p = sub.add_parser("script"); p.add_argument("slug")
    p = sub.add_parser("build"); p.add_argument("slug")
    p.add_argument("--refresh-audio", action="store_true",
                   help="re-narrate every scene (use after changing the voice)")
    p.add_argument("--refresh", action="store_true",
                   help="re-download visuals and re-render shots, keeping narration")
    p = sub.add_parser("approve"); p.add_argument("slug"); p.add_argument("--notes", default="")
    p = sub.add_parser("publish"); p.add_argument("slug"); p.add_argument("--at", default=None)
    p = sub.add_parser("make"); p.add_argument("--title", default=None)
    p = sub.add_parser("auto"); p.add_argument("--title", default=None)
    p = sub.add_parser("voicetest")
    p.add_argument("--slug", default="why-does-my-dog-eat-grass")
    p.add_argument("--text", default="")
    sub.add_parser("queue")
    sub.add_parser("reauth")
    sub.add_parser("tiktok")
    p = sub.add_parser("tiktokauth"); p.add_argument("--code", default=None)
    sub.add_parser("resume")
    sub.add_parser("status")

    args = parser.parse_args()
    cfg = load_config(args.config)
    # Lets `build --refresh` bypass the media/shot caches without deleting
    # anything: files are overwritten in place.
    cfg["_refresh"] = bool(getattr(args, "refresh", False))
    cfg["_refresh_audio"] = bool(getattr(args, "refresh_audio", False))
    handlers = {
        "topic": cmd_topic, "script": cmd_script, "build": cmd_build,
        "approve": cmd_approve, "publish": cmd_publish, "make": cmd_make,
        "auto": cmd_auto, "resume": cmd_resume, "status": cmd_status,
        "voicetest": cmd_voicetest, "queue": cmd_queue, "reauth": cmd_reauth, "tiktok": cmd_tiktok,
        "tiktokauth": cmd_tiktokauth,
    }

    log = ROOT / "logs" / "last_run.log"
    log.parent.mkdir(exist_ok=True)
    try:
        handlers[args.cmd](args, cfg)
        log.write_text(
            f"{dt.datetime.now().isoformat()}  {args.cmd} OK\n", encoding="utf-8"
        )
    except SystemExit as exc:
        if exc.code not in (0, None):
            log.write_text(
                f"{dt.datetime.now().isoformat()}  {args.cmd} STOPPED\n\n{exc}\n",
                encoding="utf-8",
            )
        raise
    except Exception:
        log.write_text(
            f"{dt.datetime.now().isoformat()}  {args.cmd} CRASHED\n\n"
            + traceback.format_exc(),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    sys.exit(main())
