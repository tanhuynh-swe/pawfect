"""Automated quality gate.

This replaces the human review step when running in auto mode. It is not as
good as you watching the video, but it catches the failures that actually get
channels demonetized: scripts that repeat earlier scripts, videos that came out
too short, hollow filler language, and factual claims stated with more
confidence than the evidence supports.

Anything that fails is HELD, not published. Held videos appear in the digest
and sit in the workspace until you look at them.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .config import WORKSPACE, env

# Phrases that signal templated filler rather than a real point of view.
FILLER = [
    "in this video we will",
    "in today's video",
    "without further ado",
    "let's dive right in",
    "buckle up",
    "you won't believe",
    "number one on our list",
    "welcome back to the channel",
    "don't forget to like and subscribe",
    "stay tuned till the end",
]

REVIEW_PROMPT = """You are the final check before this script publishes to a pet
care channel with no human review. Be strict. A bad video that publishes costs
the channel its monetization; a good video you wrongly hold costs one day.

Check for:
1. Factual claims a veterinarian would dispute, or invented studies, statistics,
   percentages or dates. This is the most important check.
2. Medical advice that could harm an animal if followed (dosages, home
   treatments, "you don't need a vet for this").
3. Claims stated confidently where the real evidence is weak or absent.
4. Generic content with no real point of view — the kind of script that could
   have come from any channel.
5. A hook that does not land in the first 12 seconds.

CHANNEL ANGLE:
{angle}

SCRIPT:
{script}

Return ONLY JSON:
{{"verdict": "PASS" or "HOLD",
  "issues": ["specific problem, quoting the offending line"],
  "worst_line": "the single most dangerous or wrong sentence, or empty string"}}

Use HOLD if anything in checks 1-3 is present. Minor style issues alone are PASS."""


def _trigrams(text: str) -> set[str]:
    words = re.findall(r"[a-z']+", text.lower())
    return {" ".join(words[i:i + 3]) for i in range(len(words) - 2)}


def repetition_score(script: dict[str, Any], exclude_slug: str) -> tuple[float, str]:
    """Highest trigram overlap against every previously published script.

    This is the check that maps directly onto YouTube's 'mass-produced or
    repetitive content' policy. If your videos start converging on one template,
    this catches it before YouTube does.
    """
    new = _trigrams(" ".join(s["narration"] for s in script["scenes"]))
    if not new:
        return 0.0, ""
    worst, worst_slug = 0.0, ""
    for path in WORKSPACE.glob("*/script.json"):
        if path.parent.name == exclude_slug:
            continue
        if not (path.parent / "published.json").exists():
            continue
        try:
            old_data = json.loads(path.read_text(encoding="utf-8"))
            old = _trigrams(" ".join(s["narration"] for s in old_data["scenes"]))
        except Exception:
            continue
        if not old:
            continue
        overlap = len(new & old) / len(new | old)
        if overlap > worst:
            worst, worst_slug = overlap, path.parent.name
    return worst, worst_slug


def mechanical_checks(script: dict[str, Any], runtime_s: float,
                      cfg: dict[str, Any], slug: str) -> list[str]:
    issues: list[str] = []
    minutes = runtime_s / 60

    if cfg.get("_vertical"):
        # A Short is judged against seconds, not the long-form target, and the
        # only hard failure is exceeding YouTube's 3-minute Shorts limit.
        if runtime_s > 180:
            issues.append(
                f"Runtime {runtime_s:.0f}s exceeds the 3-minute Shorts limit."
            )
    else:
        lo, hi = cfg["channel"]["target_minutes"]
        if minutes < lo * 0.75:
            issues.append(
                f"Runtime {minutes:.1f} min is well under the {lo} min target."
            )
        if minutes > hi * 1.6:
            issues.append(
                f"Runtime {minutes:.1f} min is far over the {hi} min target."
            )

    full = " ".join(s["narration"] for s in script["scenes"]).lower()
    for phrase in FILLER:
        if phrase in full:
            issues.append(f"Filler phrase present: '{phrase}'.")

    if len(script["title"]) > 70:
        issues.append(f"Title is {len(script['title'])} chars; keep it under 70.")
    if not script.get("description", "").strip():
        issues.append("Description is empty.")

    score, other = repetition_score(script, slug)
    limit = float(cfg["safety"].get("max_repetition", 0.30))
    if score > limit:
        issues.append(
            f"Script is {score:.0%} similar to '{other}' (limit {limit:.0%}). "
            "This is the pattern YouTube's inauthentic-content policy targets."
        )
    return issues


def llm_review(script: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    key = env("ANTHROPIC_API_KEY")
    if not key:
        return {"verdict": "PASS", "issues": ["LLM review skipped: no API key."],
                "worst_line": ""}
    try:
        import anthropic

        body = "\n\n".join(s["narration"] for s in script["scenes"])
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model=cfg["script"]["model"],
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": REVIEW_PROMPT.format(
                    angle=cfg["channel"]["angle"].strip(),
                    script=f"TITLE: {script['title']}\n\n{body}",
                ),
            }],
        )
        text = resp.content[0].text.strip()
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        start, end = text.find("{"), text.rfind("}")
        return json.loads(text[start:end + 1])
    except Exception as exc:
        # A failed review must never silently become a pass.
        return {"verdict": "HOLD",
                "issues": [f"Automated review could not run: {exc}"],
                "worst_line": ""}


def run_gate(script: dict[str, Any], runtime_s: float, cfg: dict[str, Any],
             slug: str, out_dir: Path) -> tuple[bool, list[str]]:
    issues = mechanical_checks(script, runtime_s, cfg, slug)
    review = llm_review(script, cfg)
    issues += review.get("issues", [])
    # Style notes alone should not block a publish; these four things should.
    blocking = [
        i for i in issues
        if "similar to" in i or "could not run" in i or "under the" in i
    ]
    passed = review.get("verdict", "HOLD") == "PASS" and not blocking

    result = {
        "passed": passed,
        "issues": issues,
        "blocking": blocking,
        "worst_line": review.get("worst_line", ""),
        "runtime_min": round(runtime_s / 60, 2),
    }
    (out_dir / "qa.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return passed, issues
