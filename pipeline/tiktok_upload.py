"""TikTok Content Posting API integration.

Uploads videos to TikTok using the Content Posting API (v1).
Requires TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET in .env.

Note: TikTok's API adds their watermark to videos. Custom thumbnail upload is not
supported by the Content Posting API — TikTok auto-generates one.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests

from pipeline.config import env


# TikTok API endpoints
API_BASE = "https://open.tiktokapis.com/v1"
AUTH_ENDPOINT = "https://open.tiktokapis.com/v1/oauth/token/"
UPLOAD_ENDPOINT = f"{API_BASE}/post/publish/action/upload/"
PUBLISH_ENDPOINT = f"{API_BASE}/post/publish/action/publish/"


def _get_access_token() -> str:
    """Get a fresh access token using Client Credentials flow.

    TikTok's Content Posting API uses the client credentials grant, which doesn't
    require user login — perfect for unattended automation.
    """
    client_key = env("TIKTOK_CLIENT_KEY")
    client_secret = env("TIKTOK_CLIENT_SECRET")

    if not client_key or not client_secret:
        raise ValueError(
            "TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET must be set in .env"
        )

    payload = {
        "client_key": client_key,
        "client_secret": client_secret,
        "grant_type": "client_credentials",
    }

    resp = requests.post(AUTH_ENDPOINT, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if "data" not in data or "access_token" not in data["data"]:
        raise ValueError(f"Unexpected token response: {data}")

    return data["data"]["access_token"]


def upload_video(video_path: Path, script: dict[str, Any]) -> dict[str, Any]:
    """Upload a video file to TikTok and return the upload ID.

    Args:
        video_path: Path to the MP4 file to upload
        script: Script dict containing title, description, tags

    Returns:
        Dict with 'upload_id' and 'video_id' (once published)

    Raises:
        Requests exceptions on API errors
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    access_token = _get_access_token()

    # Upload the video file
    with open(video_path, "rb") as f:
        files = {"video": f}
        headers = {"Authorization": f"Bearer {access_token}"}

        resp = requests.post(UPLOAD_ENDPOINT, files=files, headers=headers, timeout=300)

    resp.raise_for_status()
    upload_data = resp.json()

    if "data" not in upload_data or "upload_id" not in upload_data["data"]:
        raise ValueError(f"Upload failed: {upload_data}")

    return {
        "upload_id": upload_data["data"]["upload_id"],
        "video_id": None,  # Set after publishing
    }


def publish_video(
    upload_id: str,
    script: dict[str, Any],
    publish_at: str | None = None,
) -> str:
    """Publish a video that has been uploaded.

    Args:
        upload_id: The upload ID returned from upload_video()
        script: Script dict with title, description, tags
        publish_at: ISO 8601 timestamp (e.g., "2026-10-15T17:00:00Z") or None for immediate

    Returns:
        The video ID

    Raises:
        Requests exceptions on API errors
    """
    access_token = _get_access_token()

    # TikTok's API expects these fields
    publish_data = {
        "upload_id": upload_id,
        "post_info": {
            "title": script.get("title", ""),
            "description": script.get("description", ""),
            "disable_comment": False,
            "disable_duet": False,
            "disable_stitch": False,
        },
    }

    # Add tags if present
    if script.get("tags"):
        tags_str = " ".join(f"#{tag}" for tag in script["tags"][:20])  # TikTok limit
        # Append tags to description (TikTok doesn't have a separate tags field)
        current_desc = publish_data["post_info"]["description"]
        publish_data["post_info"]["description"] = f"{current_desc}\n\n{tags_str}"

    # Scheduling: TikTok's API doesn't support scheduling for Content Posting.
    # If publish_at is provided, we log it but publish immediately.
    if publish_at:
        print(f"  note: TikTok Content Posting API publishes immediately.")
        print(f"        schedule '{publish_at}' is not supported.")

    headers = {"Authorization": f"Bearer {access_token}"}
    resp = requests.post(
        PUBLISH_ENDPOINT,
        json=publish_data,
        headers=headers,
        timeout=30,
    )

    resp.raise_for_status()
    pub_data = resp.json()

    if "data" not in pub_data or "video_id" not in pub_data["data"]:
        raise ValueError(f"Publish failed: {pub_data}")

    return pub_data["data"]["video_id"]


def record(video_id: str, out_dir: Path) -> None:
    """Record the published video ID in published.json.

    Args:
        video_id: The TikTok video ID
        out_dir: Workspace directory for this video
    """
    published_file = out_dir / "published.json"

    record_data = {
        "platform": "tiktok",
        "video_id": video_id,
        "published_at": time.time(),
        "url": f"https://www.tiktok.com/@pawfect.love.animals/video/{video_id}",
    }

    published_file.write_text(json.dumps(record_data, indent=2), encoding="utf-8")
    print(f"  recorded: {published_file}")
    print(f"  url: {record_data['url']}")
