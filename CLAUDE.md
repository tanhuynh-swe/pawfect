# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Python pipeline that produces and publishes videos for the YouTube channel `@PawfectLoveAnimals` (pet care). It goes topic research → script → TTS narration → stock footage → ffmpeg render → thumbnail → YouTube upload. It can also produce vertical Shorts/TikToks, including Vietnamese ones. It runs locally on a Mac mini. `README.md` holds the channel strategy, setup steps and a troubleshooting table.

## Commands

Setup: `brew install ffmpeg python@3.11`, then `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`, then `cp .env.example .env` and fill in the keys.

Everything goes through `run.py` (use `.venv/bin/python`):

```bash
python run.py make                      # topic -> script -> build
python run.py topic [--title "..."]     # pick/force a topic, writes workspace/<slug>/brief.json
python run.py script <slug>             # api mode: calls Claude; manual mode: writes prompt.txt
python run.py build <slug> [--refresh-audio] [--refresh]   # narrate, visuals, render, thumbnail
python run.py approve <slug> [--notes ...]
python run.py publish <slug> [--at 2026-10-02T10:00:00Z]
python run.py auto                      # unattended: queue/topic -> build -> QA gate -> upload -> email digest
python run.py resume                    # reset the consecutive-hold counter after auto pauses
python run.py status | queue | voicetest | reauth | tiktok
```

There is no test suite, linter or build step. To check a change, run `build` on an existing workspace slug and inspect `workspace/<slug>/final.mp4`. ffmpeg failures are written to `workspace/<slug>/ffmpeg_error.log`.

Scheduling: `./install_schedule.sh` installs a launchd agent (`com.pawfect.autopublish`) that runs `run.py auto` Tue/Thu/Sat at 08:00 and logs to `logs/auto.log`. `./install_schedule.sh remove` uninstalls it.

## Architecture

- **`run.py`** is the whole CLI. It holds orchestration, queue logic and the auto-mode state machine. The modules in `pipeline/` are stage libraries that `run.py` calls; they do not call each other much.
- **`pipeline/config.py`** loads `config.yaml` and `.env` with its own small parser that does not override existing env vars. It also defines `ROOT` and `WORKSPACE`, and `slot_dir(slug)` creates the per-video directory.
- **Every video lives in `workspace/<slug>/`**, and its state is the set of files there: `brief.json` → `script.json` → `audio/`, `media/`, `shots/`, `narration.wav`, `captions.srt`, `video_silent.mp4` → `final.mp4`, `thumbnail.jpg` → `review.json` (`approved`), `qa.json` (auto gate result), `published.json`. `status` and the queue are worked out by checking which files exist.
- **Queue:** any slot that has `script.json` and no `published.json` is queued. `script.json`'s `publish_to` (default `youtube`, or `tiktok`) decides the destination. This filter is what keeps Vietnamese TikTok scripts from being auto-uploaded to the English channel. `auto` uses a pre-written queued script before it generates a new one. `tiktok` builds TikTok slots into `tiktok_ready/` and never uploads anything.
- **The script JSON drives the render, not CLI flags.** `script.json` needs `title`, `description`, `tags` and `scenes[]` (5 or more; each scene has `narration`, `visual_query` and `on_screen_text`). `script.validate()` enforces this. Optional `format: "vertical"` merges the `shorts:` block over `video:` in `apply_format()`. Optional `language` selects a voice from `voice.voices`. `apply_format` records its decisions as `cfg["_vertical"]` / `cfg["_language"]`, which later stages read.
- **Stages:**
  - `research.py`: seed topics come from `topics/seed_topics.yaml`, ranked by Google-suggest demand and grounded in Wikipedia facts. Used topics are recorded in `workspace/used_topics.json`.
  - `script.py`: builds the prompt, then either calls the Anthropic API or reads a manually pasted reply.
  - `voice.py`: `edge` TTS by default, `piper` offline, and it automatically falls back to the other engine. It narrates one wav per scene and measures its duration.
  - `visuals.py`: tries providers in order (Pexels → Pixabay → Wikimedia), rotates clips so none repeat, and falls back to a colored card.
  - `render.py`: cuts to a new shot every `max_shot_seconds`, draws the captions with Pillow and composites them as a video track (this ffmpeg has no libass, so neither `ass` nor `drawtext` exists), ducks the music, and normalizes the narration to −16 LUFS in a pass of its own — loudnorm inside the render's filtergraph stalls and drops seconds of audio.
  - `thumbnail.py`: Pillow, drawn over a frame grabbed from the render.
  - `upload.py`: YouTube Data API OAuth (`token.json`, client secret found by `_find_client_secret`) and scheduling slots.
- **Two publish gates.** Manual `publish` requires `review.json` `approved: true` when `safety.require_human_review` is set. `auto` skips the human gate and uses `qa.run_gate` instead. That gate holds a video when the LLM review returns HOLD or cannot run at all, when trigram similarity to previously published scripts is above `safety.max_repetition`, or when the runtime is far under target. `workspace/consecutive_holds.json` counts holds in a row, and after `max_consecutive_holds` auto pauses itself. `notify.py` emails a digest over Gmail SMTP after every auto run.

## Constraints to preserve

- The QA gate and the human-review gate exist so the channel stays within YouTube's "mass-produced / repetitive content" monetization policy. Do not weaken them, and do not let a failed check silently count as a pass.
- Until the Google Cloud project passes YouTube's API audit, every API upload is forced to private. That is expected behavior, not a bug.
- `.env`, `token.json`, `client_secret*.json` and `workspace/` hold secrets or generated state. Do not commit them or overwrite them casually.
