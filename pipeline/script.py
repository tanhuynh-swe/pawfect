"""Script writing.

Two modes, both configured in config.yaml:

  manual  (free)  — writes prompt.txt. You paste it into the Claude app, paste
                    the JSON reply back into script.json, and rerun. This is
                    also your editorial pass, which is exactly what keeps the
                    channel on the right side of YouTube's policy.

  api     (paid)  — calls the Anthropic API directly. Pennies per video.

Either way the output is the same script.json, validated before rendering.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import env

SYSTEM = """You write scripts for a YouTube channel about pet care.

Hard rules:
- Spoken narration only. No stage directions, no "[music]", no speaker labels.
- Every factual claim must be one a veterinarian would sign off on. If evidence
  is thin, say so out loud in the narration ("this one is widely repeated but
  never actually been tested").
- Name and correct at least one common myth about the topic.
- Write like one calm person talking to one worried owner. Second person.
- No "in this video we will", no "don't forget to like and subscribe" mid-script.
- Numbers must be specific and correct. Never invent a study, statistic or date.
- Open with a concrete situation the viewer recognises, not a definition.

Return ONLY valid JSON. No markdown fence, no commentary before or after."""

TEMPLATE = """{system}

CHANNEL ANGLE (this is the channel's whole identity — every script must sound
like it came from this one person, not from a content farm):
{angle}

TOPIC: {title}
EDITORIAL ANGLE FOR THIS VIDEO: {angle_note}

WHAT PEOPLE SEARCH FOR AROUND THIS TOPIC (work these phrasings into the title,
the description and the narration naturally — do not stuff them):
{related}

BACKGROUND REFERENCE (facts to check yourself against — paraphrase in your own
words, never copy a sentence of it):
{facts}

TARGET LENGTH: {minutes} minutes of speech at ~{wpm} words per minute,
so roughly {words} words of narration total.

Return JSON with exactly this shape:

{{
  "title": "YouTube title, under 70 characters, no clickbait you can't deliver",
  "description": "3-5 sentences, then a blank line, then 4-6 timestamp lines",
  "tags": ["8-12 lowercase tags"],
  "thumbnail_text": "3-5 words, huge on screen, complements the title",
  "scenes": [
    {{
      "narration": "2-4 sentences of spoken text",
      "visual_query": "2-4 word stock footage search, concrete and filmable",
      "on_screen_text": "optional 2-5 word caption, or empty string"
    }}
  ]
}}

Write {scenes} scenes. The first scene is the hook: it must land in under 12
seconds. The last scene asks one specific question for the comments — a real
question about the viewer's own pet, not "let me know what you think"."""


def build_prompt(topic: dict[str, Any], cfg: dict[str, Any]) -> str:
    lo, hi = cfg["channel"]["target_minutes"]
    minutes = (lo + hi) / 2
    wpm = cfg["script"]["words_per_minute"]
    words = int(minutes * wpm)
    related = "\n".join(f"- {s}" for s in topic.get("related_searches", [])) or "- (none found)"
    facts = topic.get("facts", "") or "(no reference text available — rely on well-established consensus only)"
    return TEMPLATE.format(
        system=SYSTEM,
        angle=cfg["channel"]["angle"].strip(),
        title=topic["title"],
        angle_note=topic.get("angle", "") or "(your choice)",
        related=related,
        facts=facts[:6000],
        minutes=f"{lo}-{hi}",
        wpm=wpm,
        words=words,
        scenes=max(14, words // 55),
    )


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in the model response.")
    return json.loads(text[start : end + 1])


def validate(script: dict[str, Any]) -> dict[str, Any]:
    required = ["title", "description", "tags", "scenes"]
    missing = [k for k in required if k not in script]
    if missing:
        raise ValueError(f"script.json is missing required keys: {missing}")
    if not isinstance(script["scenes"], list) or len(script["scenes"]) < 5:
        raise ValueError("script.json needs at least 5 scenes.")
    for i, scene in enumerate(script["scenes"]):
        if not scene.get("narration", "").strip():
            raise ValueError(f"Scene {i} has empty narration.")
        scene.setdefault("visual_query", "")
        scene.setdefault("on_screen_text", "")
    script.setdefault("thumbnail_text", script["title"][:28])
    if len(script["title"]) > 100:
        raise ValueError("Title exceeds YouTube's 100-character limit.")
    return script


def generate_api(prompt: str, cfg: dict[str, Any]) -> dict[str, Any]:
    import anthropic

    key = env("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit(
            "script.mode is 'api' but ANTHROPIC_API_KEY is not set in .env.\n"
            "Either add the key, or set script.mode: manual in config.yaml."
        )
    client = anthropic.Anthropic(api_key=key)
    resp = client.messages.create(
        model=cfg["script"]["model"],
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )
    return _extract_json(resp.content[0].text)


def write_script(topic: dict[str, Any], cfg: dict[str, Any], out_dir: Path) -> Path:
    prompt = build_prompt(topic, cfg)
    prompt_path = out_dir / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")
    script_path = out_dir / "script.json"

    if cfg["script"]["mode"] == "api":
        script = validate(generate_api(prompt, cfg))
        script_path.write_text(
            json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return script_path

    if script_path.exists():
        script = validate(json.loads(script_path.read_text(encoding="utf-8")))
        script_path.write_text(
            json.dumps(script, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return script_path

    raise SystemExit(
        f"\nScript needed. Open:\n  {prompt_path}\n\n"
        "Paste its contents into the Claude app, then save Claude's JSON reply to:\n"
        f"  {script_path}\n\nThen run the same command again."
    )


def load_script(out_dir: Path) -> dict[str, Any]:
    return validate(json.loads((out_dir / "script.json").read_text(encoding="utf-8")))
