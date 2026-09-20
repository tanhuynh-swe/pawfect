"""TikTok Content Posting API (v2).

Sends a finished video to the account's TikTok drafts. It deliberately stops
there rather than posting: TikTok Shop affiliate products can only be attached
in the app, and they are the point of these videos, so the last step stays
manual. What this removes is the file transfer, not the editorial control.

Posting outright is a different product ("Direct Post", scope `video.publish`)
and a different endpoint. The app is set up for drafts, so that is what this
implements.

Auth is a user token: `video.upload` is granted by the account holder, so there
is no unattended client-credentials path. Run `python run.py tiktokauth` once;
the refresh token is then used to keep going without further logins.

Note: TikTok stamps API uploads with their own watermark, and the Content
Posting API has no custom thumbnail — TikTok generates one.
"""
from __future__ import annotations

import json
import math
import time
import urllib.parse
from pathlib import Path
from typing import Any

import requests

from pipeline.config import ROOT, env

AUTHORIZE_URL = "https://www.tiktok.com/v2/auth/authorize/"
TOKEN_ENDPOINT = "https://open.tiktokapis.com/v2/oauth/token/"
INBOX_INIT_ENDPOINT = "https://open.tiktokapis.com/v2/post/publish/inbox/video/init/"
STATUS_ENDPOINT = "https://open.tiktokapis.com/v2/post/publish/status/fetch/"

TOKEN_FILE = ROOT / "tiktok_token.json"

# Scope for landing a video in the account's drafts. Posting it outright would
# need video.publish, which the app is not set up for.
SCOPE = "video.upload"

# TikTok accepts one chunk up to 64MB; past that it wants the file split, and
# no chunk may be under 5MB.
MAX_SINGLE_CHUNK = 64 * 1024 * 1024
MIN_CHUNK = 5 * 1024 * 1024


def _redirect_uri() -> str:
    uri = env("TIKTOK_REDIRECT_URI")
    if not uri:
        raise SystemExit(
            "TIKTOK_REDIRECT_URI is not set in .env. It has to match a Redirect "
            "URI registered on the app exactly, for example\n"
            "  TIKTOK_REDIRECT_URI=https://tanhuynh-swe.github.io/pawfect/callback.html"
        )
    return uri


def _credentials() -> tuple[str, str]:
    key, secret = env("TIKTOK_CLIENT_KEY"), env("TIKTOK_CLIENT_SECRET")
    if not key or not secret:
        raise SystemExit(
            "TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET must be set in .env"
        )
    return key, secret


def _post_form(payload: dict[str, str]) -> dict[str, Any]:
    """The token endpoint takes form encoding; it rejects a JSON body."""
    resp = requests.post(
        TOKEN_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    body = resp.json() if resp.content else {}
    if resp.status_code != 200 or "access_token" not in body:
        raise SystemExit(f"TikTok token request failed ({resp.status_code}): {body}")
    return body


def authorize() -> None:
    """One-time login, exchanging the code the redirect page shows for tokens."""
    key, secret = _credentials()
    redirect = _redirect_uri()
    params = {
        "client_key": key,
        "scope": SCOPE,
        "response_type": "code",
        "redirect_uri": redirect,
        "state": "pawfect",
    }
    print("\nOpen this in a browser and approve the app:\n")
    print("  " + AUTHORIZE_URL + "?" + urllib.parse.urlencode(params))
    print(
        "\nTikTok then sends you to your redirect page, which shows the "
        "authorization code.\nPaste it here (it is the `code` value, and it "
        "expires within minutes)."
    )
    code = input("\ncode: ").strip()
    if not code:
        raise SystemExit("No code given.")
    # TikTok percent-encodes the code in the redirect; paste-back often keeps it.
    code = urllib.parse.unquote(code)

    body = _post_form({
        "client_key": key,
        "client_secret": secret,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": redirect,
    })
    _save_token(body)
    print(f"\nAuthorized. Token saved to {TOKEN_FILE}")
    print("Keep that file out of git — it grants upload access to the account.")


def _save_token(body: dict[str, Any]) -> None:
    body["obtained_at"] = time.time()
    TOKEN_FILE.write_text(json.dumps(body, indent=2), encoding="utf-8")
    TOKEN_FILE.chmod(0o600)


def _access_token() -> str:
    """A valid user access token, refreshed when it is close to expiring."""
    if not TOKEN_FILE.exists():
        raise SystemExit(
            "No TikTok token yet. Run:  python run.py tiktokauth\n"
            "TikTok grants upload access per account, so this needs a one-time "
            "login; there is no unattended client-credentials path for it."
        )
    token = json.loads(TOKEN_FILE.read_text())
    age = time.time() - token.get("obtained_at", 0)
    # Refresh a minute early rather than racing the expiry mid-upload.
    if age < token.get("expires_in", 0) - 60:
        return token["access_token"]

    key, secret = _credentials()
    refreshed = _post_form({
        "client_key": key,
        "client_secret": secret,
        "grant_type": "refresh_token",
        "refresh_token": token["refresh_token"],
    })
    _save_token(refreshed)
    return refreshed["access_token"]


def _chunking(size: int) -> tuple[int, int]:
    """Chunk size and count TikTok will accept for a file this big."""
    if size <= MAX_SINGLE_CHUNK:
        return size, 1
    count = math.ceil(size / MAX_SINGLE_CHUNK)
    chunk = size // count
    # An undersized trailing chunk is rejected, so drop a chunk and let the
    # others carry more rather than send one that is too small.
    while count > 1 and size - chunk * (count - 1) < MIN_CHUNK:
        count -= 1
        chunk = size // count
    return chunk, count


def send_to_inbox(video: Path, script: dict[str, Any]) -> str:
    """Upload `video` to the account's TikTok drafts. Returns the publish id."""
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video}")

    token = _access_token()
    size = video.stat().st_size
    chunk, count = _chunking(size)

    init = requests.post(
        INBOX_INIT_ENDPOINT,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=UTF-8",
        },
        json={"source_info": {
            "source": "FILE_UPLOAD",
            "video_size": size,
            "chunk_size": chunk,
            "total_chunk_count": count,
        }},
        timeout=60,
    )
    body = init.json() if init.content else {}
    data = body.get("data") or {}
    if init.status_code != 200 or "upload_url" not in data:
        raise SystemExit(f"TikTok upload init failed ({init.status_code}): {body}")

    upload_url, publish_id = data["upload_url"], data["publish_id"]

    with video.open("rb") as fh:
        for index in range(count):
            start = index * chunk
            end = size - 1 if index == count - 1 else start + chunk - 1
            fh.seek(start)
            payload = fh.read(end - start + 1)
            # The upload URL is pre-signed: an Authorization header on it is
            # rejected rather than ignored.
            put = requests.put(
                upload_url,
                data=payload,
                headers={
                    "Content-Type": "video/mp4",
                    "Content-Length": str(len(payload)),
                    "Content-Range": f"bytes {start}-{end}/{size}",
                },
                timeout=600,
            )
            if put.status_code not in (200, 201, 202, 206):
                raise SystemExit(
                    f"TikTok chunk {index + 1}/{count} failed "
                    f"({put.status_code}): {put.text[:300]}"
                )
            print(f"    uploaded {min(end + 1, size) * 100 // size}%")

    return _await_inbox(publish_id, token)


def _await_inbox(publish_id: str, token: str) -> str:
    """Wait until TikTok has finished ingesting, so failures surface here."""
    deadline = time.time() + 300
    while time.time() < deadline:
        resp = requests.post(
            STATUS_ENDPOINT,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=UTF-8",
            },
            json={"publish_id": publish_id},
            timeout=30,
        )
        data = (resp.json() if resp.content else {}).get("data") or {}
        status = data.get("status", "")
        if status in ("SEND_TO_USER_INBOX", "PUBLISH_COMPLETE"):
            return publish_id
        if status == "FAILED":
            raise SystemExit(f"TikTok rejected the upload: {data}")
        time.sleep(5)
    raise SystemExit(
        f"TikTok did not finish processing within 5 minutes (publish_id {publish_id}). "
        "It may still land in your drafts; check the app before re-sending."
    )


def record(publish_id: str, out_dir: Path) -> None:
    """Note the draft so the slot leaves the queue.

    No URL: the video is a draft until it is posted from the app, and it has no
    public address before then.
    """
    (out_dir / "published.json").write_text(
        json.dumps({
            "platform": "tiktok",
            "publish_id": publish_id,
            "status": "draft_in_app",
            "sent_at": time.time(),
        }, indent=2),
        encoding="utf-8",
    )
    print("  sent to your TikTok drafts")
    print("  open the app to attach affiliate products and post")
