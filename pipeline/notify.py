"""Email digest.

One message per automated run: what published, what was held and why, what
broke. Sent through Gmail SMTP with an App Password — Gmail will not accept
your normal password for this.

Create one at https://myaccount.google.com/apppasswords
(requires 2-Step Verification to be on first).
"""
from __future__ import annotations

import smtplib
import traceback
from email.message import EmailMessage
from typing import Any

from .config import env


def _plain(report: dict[str, Any]) -> str:
    lines = [f"Pawfect Love Animals — automated run", ""]

    if report.get("published"):
        p = report["published"]
        lines += [
            "PUBLISHED",
            f"  {p['title']}",
            f"  {p['url']}",
            f"  status: {p['privacy']}",
            f"  runtime: {p['runtime_min']} min",
        ]
        if p.get("forced_private"):
            lines += [
                "",
                "  NOTE: YouTube forced this video to private because your API",
                "  project has not passed its compliance audit yet. Flip it to",
                "  public in YouTube Studio, and apply for the audit at",
                "  https://support.google.com/youtube/contact/yt_api_form",
            ]
        lines.append("")

    if report.get("held"):
        h = report["held"]
        lines += ["HELD — not published", f"  {h['title']}", f"  slug: {h['slug']}", ""]
        for issue in h["issues"]:
            lines.append(f"  - {issue}")
        if h.get("worst_line"):
            lines += ["", f"  worst line: \"{h['worst_line']}\""]
        lines += [
            "",
            "  Fix the script and rerun:",
            f"    python run.py build {h['slug']} && python run.py publish {h['slug']}",
            "",
        ]

    if report.get("error"):
        lines += ["RUN FAILED", "", report["error"], ""]

    if report.get("notes"):
        lines += ["NOTES"] + [f"  - {n}" for n in report["notes"]] + [""]

    return "\n".join(lines)


def send_digest(report: dict[str, Any]) -> bool:
    user = env("GMAIL_ADDRESS")
    password = env("GMAIL_APP_PASSWORD")
    to = env("DIGEST_TO") or user

    if not user or not password:
        print("  (no Gmail credentials in .env — digest not sent)")
        print(_plain(report))
        return False

    subject = "Pawfect: run failed"
    if report.get("published"):
        subject = f"Pawfect: published — {report['published']['title'][:60]}"
    elif report.get("held"):
        subject = f"Pawfect: HELD — {report['held']['title'][:60]}"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to
    msg.set_content(_plain(report))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as smtp:
            smtp.login(user, password)
            smtp.send_message(msg)
        print(f"  digest sent to {to}")
        return True
    except Exception:
        print("  digest failed to send:")
        traceback.print_exc()
        print(_plain(report))
        return False
