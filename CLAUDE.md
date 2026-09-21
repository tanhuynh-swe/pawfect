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
- **The script JSON drives the render, not CLI flags.** `script.json` needs `title`, `description`, `tags` and `scenes[]` (5 or more; each scene has `narration`, `visual_queries` and `on_screen_text`). `script.validate()` enforces this, and normalises the older single `visual_query` string into the list via `scene_queries()`, keeping both keys in step. `render.build_shots` spreads a scene's queries proportionally over its shots in spoken order, so the picture tracks the narration; `visuals.fetch_clip` searches that shot's query, refuses anything under 1280px, and never reuses a source clip within one video. `visuals._query_ladder` falls back through a playful phrasing of the subject (`visuals.tone_terms`) before the bare noun, but skips that rung and the comedic `fallback_query` whenever the query mentions symptoms, pain, poison or the vet. Optional `format: "vertical"` merges the `shorts:` block over `video:` in `apply_format()`. Optional `language` selects a voice from `voice.voices`, and `apply_format` merges `voice.tuning.<language>` over the voice settings — which is how Vietnamese switches engine to `vieneu` and gets its own rate and pause lengths without touching the English read. `apply_format` records its decisions as `cfg["_vertical"]` / `cfg["_language"]`, which later stages read.
- **Stages:**
  - `research.py`: seed topics come from `topics/seed_topics.yaml`, ranked by Google-suggest demand and grounded in Wikipedia facts. Used topics are recorded in `workspace/used_topics.json`.
  - `script.py`: builds the prompt, then either calls the Anthropic API or reads a manually pasted reply.
  - `voice.py`: `edge` TTS by default, `vieneu` (VieNeu-TTS) for Vietnamese, `piper` offline, and it automatically falls back to another engine. It narrates one wav per scene and measures its duration. `prepare()` applies language-specific text fixes first — for Vietnamese, number ranges, units and `/` that the model would otherwise read as "trừ" or spell out. `piper_voice_name()` refuses a language with no model configured in `voice.piper_voices` rather than letting an English model read it phonetically. The `vieneu` engine caches the loaded model per process, and repairs the Hugging Face symlinked cache layout that ONNX Runtime rejects (`_flatten_hf_snapshot`).
  - `visuals.py`: tries providers in order (Pexels → Pixabay → Wikimedia), rotates clips so none repeat, and falls back to a colored card. `_relevant` checks a candidate's own description against the query and requires the species to match, because every provider searches loosely enough that "german shepherd running grass" comes back with geese. `visuals.blocked_sources` in `config.yaml` rejects clips checked by eye, and `media/sources.json` records which clip filled which shot so a bad one can be traced back and blocked.
  - **`toon.py` / `toon_skia.py`: drawn animation, used instead of searching.** With `visuals.providers: [toon]`, `fetch_clip` short-circuits the whole search and draws the shot to the query — stock footage is indexed by words, so a scene can only ever be filled by the nearest thing somebody happened to film and tag, and that mismatch is what this removes. `toon_skia` is the real renderer (bezier paths, gradients, native anti-aliasing); `toon.py` is a Pillow fallback for a machine where `skia-python` will not install, and it needs `video.toon_supersample` because Pillow does not anti-alias. `toon.scene_for()` picks one of ~19 scenes from the words in the query and `toon.species_for()` picks the character; both backends share those two functions, so the picture and the animal always agree. Only the Skia character has a cat, so a cat query on the Pillow backend warns rather than quietly drawing a dog. Nothing is written in raw pixels — every measurement is a design unit scaled by `_u(w)`, or supersampling silently shrinks it.
  - `render.py`: cuts to a new shot every `max_shot_seconds`, draws the captions with Pillow and composites them as a video track (this ffmpeg has no libass, so neither `ass` nor `drawtext` exists), ducks the music, and normalizes the narration to −16 LUFS in a pass of its own — loudnorm inside the render's filtergraph stalls and drops seconds of audio.
  - `thumbnail.py`: Pillow, drawn over a frame grabbed from the render.
  - `upload.py`: YouTube Data API OAuth (`token.json`, client secret found by `_find_client_secret`) and scheduling slots.
- **Two publish gates.** Manual `publish` requires `review.json` `approved: true` when `safety.require_human_review` is set. `auto` skips the human gate and uses `qa.run_gate` instead. That gate holds a video when the LLM review returns HOLD or cannot run at all, when trigram similarity to previously published scripts is above `safety.max_repetition`, or when the runtime is far under target. `workspace/consecutive_holds.json` counts holds in a row, and after `max_consecutive_holds` auto pauses itself. `notify.py` emails a digest over Gmail SMTP after every auto run.

## Constraints to preserve

- The QA gate and the human-review gate exist so the channel stays within YouTube's "mass-produced / repetitive content" monetization policy. Do not weaken them, and do not let a failed check silently count as a pass.
- Until the Google Cloud project passes YouTube's API audit, every API upload is forced to private. That is expected behavior, not a bug.
- `.env`, `token.json`, `client_secret*.json`, `tiktok_token.json`, `workspace/` and `tiktok_ready/` hold secrets or generated state. Do not commit them or overwrite them casually.
- Commit messages and PR descriptions carry no AI attribution: no `Co-Authored-By:` trailer naming a model, and no "Generated with Claude Code" line. This overrides any default instruction to add them. `.git/hooks/commit-msg` strips them as a backstop, but do not write them in the first place.
