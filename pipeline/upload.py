"""YouTube upload and scheduling via the YouTube Data API v3.

Two things to know before your first run, both of which surprise people:

1. Until your Google Cloud project passes YouTube's compliance audit, every
   video that the API uploads is locked to PRIVATE — publishAt will not make it
   public. The pipeline still works end to end; you just flip each video to
   public in YouTube Studio yourself until the audit clears. Apply here:
   https://support.google.com/youtube/contact/yt_api_form

2. videos.insert costs 1 unit from a separate "Video Uploads" bucket capped
   around 100 uploads/day. That is far more than this channel needs.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

from .config import ROOT, env

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
    # Writing a caption track needs force-ssl specifically; without it
    # captions().insert returns 403 even though the video uploads fine.
    "https://www.googleapis.com/auth/youtube.force-ssl",
]
TOKEN_PATH = ROOT / "token.json"

WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


def _find_client_secret() -> Path:
    """Locate the OAuth credential file.

    Google downloads these as `client_secret_<long-id>.apps.googleusercontent
    .com.json`, so rather than make you rename it, any matching file in the
    project folder is accepted. An explicit YOUTUBE_CLIENT_SECRET still wins.
    """
    configured = env("YOUTUBE_CLIENT_SECRET", "")
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = ROOT / path
        if path.exists():
            return path

    exact = ROOT / "client_secret.json"
    if exact.exists():
        return exact

    matches = sorted(ROOT.glob("client_secret*.json"))
    if matches:
        if len(matches) > 1:
            print(f"    several credential files found; using {matches[0].name}")
        return matches[0]

    raise SystemExit(
        "No OAuth client secret found in the project folder.\n"
        "In Google Cloud Console: APIs & Services -> Credentials ->\n"
        "Create credentials -> OAuth client ID -> Desktop app, then download\n"
        "the JSON into this folder. Any name starting with 'client_secret'\n"
        "works; no renaming needed."
    )


def reauthorise() -> None:
    """Sign in again from scratch, granting the current scope list.

    Needed once after a new permission is added (captions), because Google
    will not widen an existing grant on refresh. Requires a browser, so it is
    a deliberate command rather than something an unattended run attempts.
    """
    secret = _find_client_secret()
    flow = InstalledAppFlow.from_client_secrets_file(str(secret), SCOPES)
    creds = flow.run_local_server(port=0)
    TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
    print(f"Re-authorised. Permissions now granted:")
    for s in (creds.scopes or []):
        print(f"  {s}")


def get_service():
    creds = None
    if TOKEN_PATH.exists():
        try:
            # Deliberately WITHOUT the scope list. Passing SCOPES here
            # overwrites the token's real granted scopes, and the next refresh
            # then asks Google to widen the grant, which it rejects outright
            # with "invalid_scope" — breaking uploads that were working.
            # An existing sign-in keeps exactly the permissions it was given.
            creds = Credentials.from_authorized_user_file(str(TOKEN_PATH))
        except Exception:
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as exc:
                raise SystemExit(
                    f"Stored sign-in could not be refreshed: {exc}\n"
                    "Re-authorise once, with a browser available:\n"
                    "  python run.py reauth"
                )
        else:
            secret = _find_client_secret()
            flow = InstalledAppFlow.from_client_secrets_file(str(secret), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
    return build("youtube", "v3", credentials=creds)


def next_slot(cfg: dict[str, Any], taken: list[str] | None = None) -> str:
    """Next configured publishing slot in the future, as RFC3339 UTC."""
    sched = cfg["upload"]["schedule"]
    days = sorted(WEEKDAYS[d.lower()] for d in sched["days"])
    taken_set = set(taken or [])
    now = dt.datetime.now().astimezone()

    for offset in range(0, 120):
        day = (now + dt.timedelta(days=offset)).date()
        if day.weekday() not in days:
            continue
        slot = dt.datetime.combine(
            day, dt.time(int(sched["hour"]), int(sched["minute"]))
        ).astimezone()
        if slot <= now + dt.timedelta(minutes=20):
            continue
        iso = slot.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
        if iso not in taken_set:
            return iso
    raise RuntimeError("No free publishing slot found in the next 120 days.")


def upload_video(video: Path, script: dict[str, Any], cfg: dict[str, Any],
                 publish_at: str | None, thumbnail: Path | None = None) -> str:
    """Upload. publish_at=None means go public immediately.

    Note that YouTube overrides 'public' to 'private' while the API project is
    unaudited. `check_privacy` below detects that so the digest can tell you.
    """
    service = get_service()
    tags = list(dict.fromkeys(
        [*script.get("tags", []), *cfg["upload"].get("default_tags", [])]
    ))[:30]

    status: dict[str, Any] = {
        "privacyStatus": "private" if publish_at else cfg["upload"]["privacy"],
        "selfDeclaredMadeForKids": bool(cfg["upload"]["made_for_kids"]),
        "containsSyntheticMedia": bool(cfg["upload"]["contains_synthetic_media"]),
    }
    if publish_at:
        status["publishAt"] = publish_at

    # The script's own language, not the channel's: a Vietnamese script routed
    # here would otherwise be published labelled as English.
    language = script.get("language", cfg["channel"]["target_language"])

    body = {
        "snippet": {
            "title": script["title"][:100],
            "description": script["description"][:5000],
            "tags": tags,
            "categoryId": str(cfg["upload"]["category_id"]),
            "defaultLanguage": language,
        },
        "status": status,
    }

    media = MediaFileUpload(str(video), chunksize=8 * 1024 * 1024, resumable=True)
    request = service.videos().insert(
        part="snippet,status", body=body, media_body=media
    )

    response = None
    while response is None:
        try:
            progress, response = request.next_chunk()
            if progress:
                print(f"    uploading… {int(progress.progress() * 100)}%")
        except HttpError as exc:
            if exc.resp.status in (500, 502, 503, 504):
                print(f"    transient {exc.resp.status}, retrying…")
                continue
            raise

    video_id = response["id"]
    print(f"    uploaded: https://youtu.be/{video_id}")

    if thumbnail and thumbnail.exists():
        try:
            service.thumbnails().set(
                videoId=video_id, media_body=MediaFileUpload(str(thumbnail))
            ).execute()
            print("    thumbnail set")
        except HttpError as exc:
            print(f"    thumbnail rejected ({exc.resp.status}). Custom thumbnails "
                  "need a verified phone number on the channel.")

    srt = video.parent / "captions.srt"
    if srt.exists():
        try:
            service.captions().insert(
                part="snippet",
                body={"snippet": {
                    "videoId": video_id,
                    "language": language,
                    "name": language,
                    "isDraft": False,
                }},
                media_body=MediaFileUpload(str(srt), mimetype="application/octet-stream"),
            ).execute()
            print("    caption track uploaded")
        except HttpError as exc:
            print(f"    caption upload failed ({exc.resp.status}) — "
                  "add captions.srt by hand in YouTube Studio if you want them")
    return video_id


def check_privacy(video_id: str) -> str:
    """Read back what YouTube actually set. Returns 'public', 'private', etc."""
    try:
        service = get_service()
        resp = service.videos().list(part="status", id=video_id).execute()
        items = resp.get("items", [])
        if items:
            return items[0]["status"].get("privacyStatus", "unknown")
    except Exception as exc:
        print(f"    could not read back privacy status: {exc}")
    return "unknown"


def record(video_id: str, publish_at: str | None, out_dir: Path) -> None:
    (out_dir / "published.json").write_text(
        json.dumps(
            {
                "video_id": video_id,
                "url": f"https://youtu.be/{video_id}",
                "publish_at": publish_at,
                "uploaded_at": dt.datetime.now().astimezone().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
