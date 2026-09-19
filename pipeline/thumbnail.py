"""Thumbnail generation.

Pulls a still from the video, darkens one half, and sets large text on it.
Thumbnails decide most of your click-through rate, so treat the output as a
starting point and override it by hand whenever you have a better idea.
"""
from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/google-fonts/Poppins-ExtraBold.ttf",
    "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def _grab_frame(video: Path, dest: Path, at: float = 4.0) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(at), "-i", str(video),
         "-frames:v", "1", "-vf", "scale=1280:720", str(dest)],
        check=True,
    )
    return dest


def make_thumbnail(video: Path, text: str, out_dir: Path,
                   cfg: dict[str, Any], at: float = 4.0) -> Path:
    frame = _grab_frame(video, out_dir / "frame.jpg", at)
    img = Image.open(frame).convert("RGB").resize((1280, 720))
    img = ImageEnhance.Color(img).enhance(1.15)
    img = ImageEnhance.Brightness(img).enhance(0.85)

    # A soft horizontal falloff rather than a hard-edged panel — a visible
    # vertical seam down the middle of a thumbnail reads as amateur.
    mask = Image.new("L", img.size, 0)
    mpx = mask.load()
    solid_to, fade_to = 560, 900
    for x in range(img.size[0]):
        if x <= solid_to:
            a = 190
        elif x >= fade_to:
            a = 0
        else:
            a = int(190 * (1 - (x - solid_to) / (fade_to - solid_to)) ** 1.4)
        for y in range(img.size[1]):
            mpx[x, y] = a
    shade = Image.new("RGB", img.size, (8, 10, 16))
    img = Image.composite(shade, img, mask)

    draw = ImageDraw.Draw(img)
    words = text.strip().upper().split()
    lines = textwrap.wrap(" ".join(words), width=11)[:3]

    size = 118 if len(lines) <= 2 else 96
    font = _font(size)
    total_h = sum(
        draw.textbbox((0, 0), ln, font=font)[3] + 14 for ln in lines
    )
    y = (720 - total_h) // 2

    for line in lines:
        for dx, dy in ((-4, 0), (4, 0), (0, -4), (0, 4), (-3, -3), (3, 3)):
            draw.text((64 + dx, y + dy), line, font=font, fill=(0, 0, 0))
        draw.text((64, y), line, font=font, fill=(255, 214, 10))
        y += draw.textbbox((0, 0), line, font=font)[3] + 14

    draw.rectangle([(0, 690), (1280, 720)], fill=(255, 214, 10))

    dest = out_dir / "thumbnail.jpg"
    img.save(dest, "JPEG", quality=88, optimize=True)

    # YouTube rejects thumbnails over 2MB.
    quality = 88
    while dest.stat().st_size > 2_000_000 and quality > 50:
        quality -= 8
        img.save(dest, "JPEG", quality=quality, optimize=True)
    return dest
