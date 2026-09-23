"""Drawn animation, as an alternative to searching stock libraries.

Stock footage is indexed by words, so a scene can only ever be filled by the
nearest thing somebody happened to film and tag. That is the whole source of
the mismatch this module exists to avoid: here the picture is drawn to the
narration, so "the dog is asleep, then it sprints past the sofa" is a drawing
job that cannot come back with a cat, a jungle animal or a cartoon bulldog.

The trade is polish, so three things the drawing has to get right:

  Anti-aliasing. Pillow does not anti-alias anything, so drawn straight to
  1080x1920 every outline and curve comes out stair-stepped, which is most of
  what makes flat vector art look homemade. Frames are drawn at SUPERSAMPLE
  times the output size and scaled down with LANCZOS.

  Resolution independence. Because of the above, nothing may be written in
  raw pixels: every measurement is in design units scaled by `_u(w)`, which is
  1.0 at a 1080-wide frame. Hard-coded pixels silently shrink to a third of
  their intended size the moment supersampling is on.

  One character. Same proportions, same collar, same palette in every scene.
  A mascot that changes between cuts reads as clip art.

`render_clip` is the entry point. It reads the same `visual_queries` the stock
path uses, picks a scene from the words in them, and writes an mp4 of exactly
the length the narration needs.
"""
from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw

# One palette for the whole channel, kept in step with the Skia backend so a
# machine that falls back to Pillow does not also change the channel's
# colours. Off-white walls over a sage dado, pale oak boards, muted sage
# upholstery, terracotta and ochre held back as accents: the single warm ramp
# this started with dated the room, and on a grey dog it left the animal
# sharing a hue with everything behind it.
PAPER = (247, 243, 236)
WALL_TOP = (244, 240, 233)
WALL_BOT = (232, 227, 218)
DADO = (208, 218, 206)
DADO_DARK = (186, 200, 187)
RAIL = (250, 248, 243)
FLOOR = (233, 215, 190)
FLOOR_DARK = (205, 183, 154)
FLOOR_LINE = (220, 200, 172)
RUG = (172, 198, 192)
RUG_LINE = (240, 236, 227)
FUR = (212, 152, 96)
FUR_DARK = (176, 118, 70)
FUR_LIGHT = (236, 194, 148)
INK = (50, 44, 42)
ACCENT = (214, 106, 78)
OCHRE = (226, 166, 96)
LEAF = (128, 168, 126)
LEAF_DARK = (86, 128, 98)
SKY_TOP = (166, 208, 228)
SKY_BOT = (228, 242, 240)
SOFA = (150, 174, 162)
SOFA_DARK = (120, 146, 135)
CARD = (218, 188, 146)
CARD_DARK = (190, 156, 114)
POT = (234, 228, 216)
FRAME = (62, 58, 56)
BED = (118, 126, 138)
BED_DARK = (92, 100, 112)
BARK = (146, 118, 96)
NIGHT_TOP = (42, 50, 78)
NIGHT_BOT = (74, 82, 112)
NIGHT_FLOOR = (62, 64, 90)
NIGHT_SHADOW = (52, 52, 76)
LAMP = (252, 224, 158)

# Draw at this multiple of the output size, then downsample. 2 removes most of
# the stair-stepping; 3 is visibly cleaner on the thin ink lines.
SUPERSAMPLE = 3

# Backgrounds are otherwise redrawn every frame, and a vertical gradient is one
# line draw per row. Built once per size and pasted.
_BASES: dict[tuple, Image.Image] = {}


def _u(w: int) -> float:
    """Design units. 1.0 at a 1080-wide frame, 3.0 when supersampling at 3x."""
    return w / 1080.0


def _ink(s: float, mult: float = 1.0) -> int:
    return max(1, int(round(6 * s * mult)))


def _dim(night: bool, day: tuple, dark: tuple) -> tuple:
    return dark if night else day


def _vgrad(img: Image.Image, top: tuple, bottom: tuple, y0: int, y1: int) -> None:
    d = ImageDraw.Draw(img)
    span = max(1, y1 - y0)
    for y in range(y0, y1):
        f = (y - y0) / span
        d.line([(0, y), (img.width, y)],
               fill=tuple(int(top[i] + (bottom[i] - top[i]) * f) for i in range(3)))


def _shadow(d: ImageDraw.ImageDraw, x: float, y: float, width: float,
            colour: tuple = FLOOR_DARK) -> None:
    h = width * 0.22
    d.ellipse([x - width / 2, y - h / 2, x + width / 2, y + h / 2], fill=colour)


# --- character --------------------------------------------------------------

def _leg(d: ImageDraw.ImageDraw, hip: tuple[float, float], swing: float,
         s: float, colour: tuple) -> None:
    """Two-segment leg, so the stride bends instead of pivoting like a stick."""
    hx, hy = hip
    knee = (hx + swing * 0.55, hy + 44 * s)
    paw = (hx + swing, hy + 86 * s)
    w = _ink(s, 3.2)
    d.line([hip, knee, paw], fill=INK, width=w + _ink(s, 0.9), joint="curve")
    d.line([hip, knee, paw], fill=colour, width=w, joint="curve")
    r = 15 * s
    d.ellipse([paw[0] - r, paw[1] - r * 0.7, paw[0] + r, paw[1] + r * 0.8],
              fill=colour, outline=INK, width=_ink(s, 0.8))


def _head(d: ImageDraw.ImageDraw, hx: float, hy: float, s: float,
          ear_lift: float = 0.0, asleep: bool = False,
          blink: float = 0.0, mouth: float = 0.0) -> None:
    """Skull, cheek, muzzle and ear as overlapping forms.

    A circle with a rectangle stuck on it reads as a diagram. Overlapping a
    cheek into the muzzle, and letting the ear hang past the jaw, gives the
    head a silhouette rather than the outline of two shapes.
    """
    lw = _ink(s)
    ear_y = hy - 34 * s - ear_lift * s
    d.ellipse([hx - 84 * s, ear_y, hx - 14 * s, ear_y + 112 * s],
              fill=FUR_DARK, outline=INK, width=lw)
    d.ellipse([hx - 64 * s, hy - 58 * s, hx + 66 * s, hy + 60 * s],
              fill=FUR, outline=INK, width=lw)
    d.ellipse([hx + 6 * s, hy - 16 * s, hx + 78 * s, hy + 56 * s],
              fill=FUR, outline=INK, width=lw)
    d.ellipse([hx + 30 * s, hy + 2 * s, hx + 106 * s, hy + 50 * s],
              fill=FUR_LIGHT, outline=INK, width=lw)
    d.ellipse([hx + 26 * s, hy + 6 * s, hx + 76 * s, hy + 46 * s],
              fill=FUR_LIGHT)
    d.ellipse([hx + 84 * s, hy + 8 * s, hx + 112 * s, hy + 34 * s], fill=INK)
    d.ellipse([hx + 90 * s, hy + 12 * s, hx + 99 * s, hy + 20 * s],
              fill=(152, 132, 122))
    if mouth > 0.02:
        # jaw drops open, tongue showing: the difference between a dog that
        # is standing there and a dog that is shouting at something
        gap = 54 * s * mouth
        d.polygon([(hx + 36 * s, hy + 24 * s), (hx + 108 * s, hy + 20 * s),
                   (hx + 100 * s, hy + 24 * s + gap),
                   (hx + 44 * s, hy + 20 * s + gap * 0.8)],
                  fill=(74, 40, 42), outline=INK, width=_ink(s, 0.8))
        d.ellipse([hx + 56 * s, hy + 22 * s + gap * 0.45, hx + 94 * s,
                   hy + 22 * s + gap * 1.0], fill=(222, 122, 130))
        # a tooth, which is what sells it as a bark rather than a yawn
        d.polygon([(hx + 92 * s, hy + 22 * s), (hx + 102 * s, hy + 22 * s),
                   (hx + 96 * s, hy + 22 * s + gap * 0.36)],
                  fill=(255, 252, 246))
    else:
        d.arc([hx + 56 * s, hy + 20 * s, hx + 96 * s, hy + 54 * s],
              start=20, end=120, fill=INK, width=_ink(s, 0.7))
    d.arc([hx + 2 * s, hy - 46 * s, hx + 44 * s, hy - 12 * s],
          start=200, end=310, fill=INK, width=_ink(s, 0.85))
    if asleep:
        d.arc([hx + 6 * s, hy - 24 * s, hx + 42 * s, hy + 8 * s],
              start=200, end=340, fill=INK, width=_ink(s, 0.9))
    elif blink > 0.985:
        d.line([(hx + 12 * s, hy - 12 * s), (hx + 38 * s, hy - 12 * s)],
               fill=INK, width=_ink(s, 0.9))
    else:
        d.ellipse([hx + 12 * s, hy - 28 * s, hx + 38 * s, hy - 1 * s], fill=INK)
        d.ellipse([hx + 26 * s, hy - 24 * s, hx + 35 * s, hy - 15 * s],
                  fill=(255, 255, 255))
        d.ellipse([hx + 16 * s, hy - 10 * s, hx + 21 * s, hy - 5 * s],
                  fill=(255, 255, 255))


def _collar(d: ImageDraw.ImageDraw, x: float, y: float, s: float,
            tilt: float = 0.0) -> None:
    """The one prop that makes it the same dog in every scene."""
    d.rounded_rectangle([x - 20 * s, y - 30 * s + tilt, x + 26 * s,
                         y + 4 * s + tilt], radius=max(1, int(9 * s)),
                        fill=ACCENT, outline=INK, width=_ink(s, 0.8))
    d.ellipse([x - 2 * s, y + 0 * s + tilt, x + 20 * s, y + 22 * s + tilt],
              fill=LAMP, outline=INK, width=_ink(s, 0.7))


def _torso(d: ImageDraw.ImageDraw, x: float, y: float, s: float,
           lean: float = 0.0) -> None:
    """Haunch heavier than chest, which stops it reading as a capsule."""
    lw = _ink(s)
    d.ellipse([x - 112 * s, y - 80 * s, x + 26 * s, y + 12 * s],
              fill=FUR, outline=INK, width=lw)
    d.ellipse([x - 12 * s, y - 76 * s + lean, x + 108 * s, y + 10 * s + lean],
              fill=FUR, outline=INK, width=lw)
    d.polygon([(x - 66 * s, y - 74 * s), (x + 74 * s, y - 70 * s + lean),
               (x + 74 * s, y + 2 * s + lean), (x - 66 * s, y + 6 * s)],
              fill=FUR)
    d.ellipse([x + 8 * s, y - 54 * s + lean, x + 96 * s, y - 2 * s + lean],
              fill=FUR_LIGHT)


def dog_running(d: ImageDraw.ImageDraw, x: float, y: float, t: float,
                s: float = 1.0, shadow: tuple = FLOOR_DARK) -> None:
    """A gallop, not a trot.

    A small symmetric leg swing reads as walking however fast it cycles. What
    makes a sprint legible is the extremes: front legs thrown forward while
    the back legs still trail, a body that leans into the run, and an airborne
    beat where all four paws leave the floor.
    """
    phase = t * 13.0
    reach = math.sin(phase) * 58 * s
    trail = math.sin(phase + 2.4) * 58 * s
    airborne = max(0.0, math.sin(phase * 2)) * 28 * s
    ground = y + 96 * s
    y -= airborne
    lean = 14 * s

    _shadow(d, x + 10 * s, ground, (250 - airborne * 1.4) * s, shadow)

    tail = math.sin(phase * 1.4) * 32 * s
    tip = (x - 206 * s, y - 86 * s - tail)
    d.line([(x - 88 * s, y - 36 * s), (x - 152 * s, y - 70 * s - tail * 0.6),
            tip], fill=INK, width=_ink(s, 3.4), joint="curve")
    d.line([(x - 88 * s, y - 36 * s), (x - 150 * s, y - 69 * s - tail * 0.6),
            tip], fill=FUR_DARK, width=_ink(s, 2.2), joint="curve")
    d.ellipse([tip[0] - 20 * s, tip[1] - 20 * s, tip[0] + 20 * s,
               tip[1] + 20 * s], fill=FUR_LIGHT, outline=INK,
              width=_ink(s, 0.8))
    _leg(d, (x - 58 * s, y - 4 * s), trail - 26 * s, s, FUR_DARK)
    _leg(d, (x - 30 * s, y - 4 * s), trail, s, FUR_DARK)
    _torso(d, x, y, s, lean)
    _leg(d, (x + 56 * s, y - 4 * s + lean), reach, s, FUR)
    _leg(d, (x + 82 * s, y - 4 * s + lean), reach + 26 * s, s, FUR)
    _collar(d, x + 96 * s, y - 46 * s + lean, s, tilt=4 * s)
    _head(d, x + 126 * s, y - 100 * s + lean, s,
          ear_lift=-math.sin(phase) * 20)


def dog_sleeping(d: ImageDraw.ImageDraw, x: float, y: float, t: float,
                 s: float = 1.0, shadow: tuple = FLOOR_DARK,
                 snore: bool = False) -> None:
    lw = _ink(s)
    breathe = math.sin(t * 2.0) * (9 if snore else 5) * s
    _shadow(d, x, y + 52 * s, 330 * s, shadow)
    d.ellipse([x - 152 * s, y - 86 * s - breathe, x + 152 * s, y + 48 * s],
              fill=FUR, outline=INK, width=lw)
    d.ellipse([x - 98 * s, y - 40 * s, x + 112 * s, y + 42 * s], fill=FUR_LIGHT)
    d.arc([x - 174 * s, y - 40 * s, x - 16 * s, y + 72 * s],
          start=20, end=210, fill=INK, width=_ink(s, 3.2))
    d.arc([x - 170 * s, y - 38 * s, x - 20 * s, y + 68 * s],
          start=20, end=210, fill=FUR_DARK, width=_ink(s, 2.0))
    _head(d, x + 94 * s, y - 30 * s - breathe, s * 0.94, asleep=True)
    d.ellipse([x + 42 * s, y + 6 * s, x + 100 * s, y + 46 * s],
              fill=FUR_LIGHT, outline=INK, width=_ink(s, 0.8))
    for i in range(3):
        a = t * (1.5 if snore else 0.9) + i * 1.1
        drift = (a % 3.0) / 3.0
        zx = x - 30 * s + math.sin(a * 2) * 20 * s
        zy = y - 120 * s - drift * 250 * s
        size = (30 + i * 15) * s * (1.25 if snore else 1.0)
        d.line([(zx, zy), (zx + size, zy), (zx, zy + size), (zx + size, zy + size)],
               fill=INK, width=_ink(s, 1.1), joint="curve")


def dog_standing(d: ImageDraw.ImageDraw, x: float, y: float, t: float,
                 s: float = 1.0, shadow: tuple = FLOOR_DARK) -> None:
    wag = math.sin(t * 9.0) * 42 * s
    bob = math.sin(t * 2.2) * 4 * s
    _shadow(d, x + 6 * s, y + 92 * s, 250 * s, shadow)
    y -= bob
    tip = (x - 196 * s, y - 128 * s + wag)
    d.line([(x - 86 * s, y - 40 * s), (x - 150 * s, y - 96 * s + wag * 0.6),
            tip], fill=INK, width=_ink(s, 3.4), joint="curve")
    d.line([(x - 86 * s, y - 40 * s), (x - 148 * s, y - 95 * s + wag * 0.6),
            tip], fill=FUR_DARK, width=_ink(s, 2.2), joint="curve")
    d.ellipse([tip[0] - 20 * s, tip[1] - 20 * s, tip[0] + 20 * s,
               tip[1] + 20 * s], fill=FUR_LIGHT, outline=INK,
              width=_ink(s, 0.8))
    _leg(d, (x - 52 * s, y - 4 * s), 0, s, FUR_DARK)
    _leg(d, (x - 24 * s, y - 4 * s), 0, s, FUR_DARK)
    _torso(d, x, y, s)
    _leg(d, (x + 56 * s, y - 4 * s), 0, s, FUR)
    _leg(d, (x + 82 * s, y - 4 * s), 0, s, FUR)
    _collar(d, x + 98 * s, y - 50 * s, s)
    _head(d, x + 122 * s, y - 108 * s, s, ear_lift=bob * 2,
          blink=(math.sin(t * 1.7) + 1) / 2)


def dog_barking(d: ImageDraw.ImageDraw, x: float, y: float, t: float,
                s: float = 1.0, shadow: tuple = FLOOR_DARK) -> None:
    """Barking, with the recoil that makes it read as a bark.

    A dog does not bark from a still body: the chest snaps back on each one.
    The whole pose is driven off the same pulse as the jaw so the sound arcs,
    the head lift and the body recoil all land together.
    """
    pulse = max(0.0, math.sin(t * 9.0))
    recoil = pulse * 14 * s
    wag = math.sin(t * 11.0) * 36 * s
    _shadow(d, x + 6 * s, y + 92 * s, 250 * s, shadow)
    y -= pulse * 6 * s

    tip = (x - 190 * s - recoil, y - 120 * s + wag)
    d.line([(x - 86 * s, y - 40 * s), (x - 148 * s, y - 92 * s + wag * 0.6),
            tip], fill=INK, width=_ink(s, 3.4), joint="curve")
    d.line([(x - 86 * s, y - 40 * s), (x - 146 * s, y - 91 * s + wag * 0.6),
            tip], fill=FUR_DARK, width=_ink(s, 2.2), joint="curve")
    d.ellipse([tip[0] - 20 * s, tip[1] - 20 * s, tip[0] + 20 * s, tip[1] + 20 * s],
              fill=FUR_LIGHT, outline=INK, width=_ink(s, 0.8))
    _leg(d, (x - 52 * s - recoil, y - 4 * s), 0, s, FUR_DARK)
    _leg(d, (x - 24 * s - recoil, y - 4 * s), 0, s, FUR_DARK)
    _torso(d, x - recoil, y, s)
    _leg(d, (x + 56 * s, y - 4 * s), 0, s, FUR)
    _leg(d, (x + 82 * s, y - 4 * s), 0, s, FUR)
    _collar(d, x + 98 * s - recoil * 0.4, y - 50 * s, s)
    _head(d, x + 126 * s, y - 118 * s - pulse * 12 * s, s,
          ear_lift=-pulse * 16, mouth=pulse)

    # sound arcs, thrown from the muzzle
    for k in range(3):
        r = (80 + k * 62) * s * (0.55 + pulse * 0.8)
        cx, cy = x + 210 * s, y - 108 * s - pulse * 12 * s
        d.arc([cx - r, cy - r, cx + r, cy + r], start=-52, end=52,
              fill=INK, width=_ink(s, 1.5))


def dog_playbow(d: ImageDraw.ImageDraw, x: float, y: float, t: float,
                s: float = 1.0, shadow: tuple = FLOOR_DARK) -> None:
    """Chest on the floor, rear in the air: the universal invitation to play."""
    wag = math.sin(t * 13.0) * 52 * s
    bounce = math.sin(t * 3.0) * 6 * s
    _shadow(d, x + 6 * s, y + 92 * s, 260 * s, shadow)
    rear_y = y - 40 * s + bounce
    tip = (x - 176 * s, rear_y - 148 * s + wag)
    d.line([(x - 86 * s, rear_y - 60 * s), (x - 140 * s, rear_y - 120 * s),
            tip], fill=INK, width=_ink(s, 3.4), joint="curve")
    d.line([(x - 84 * s, rear_y - 60 * s), (x - 138 * s, rear_y - 118 * s),
            tip], fill=FUR_DARK, width=_ink(s, 2.2), joint="curve")
    d.ellipse([tip[0] - 20 * s, tip[1] - 20 * s, tip[0] + 20 * s, tip[1] + 20 * s],
              fill=FUR_LIGHT, outline=INK, width=_ink(s, 0.8))
    _leg(d, (x - 50 * s, rear_y - 30 * s), 0, s, FUR_DARK)
    _leg(d, (x - 22 * s, rear_y - 30 * s), 0, s, FUR_DARK)
    # body sloping down to the chest
    lw = _ink(s)
    d.polygon([(x - 112 * s, rear_y - 96 * s), (x + 40 * s, y - 34 * s),
               (x + 40 * s, y + 18 * s), (x - 112 * s, rear_y - 6 * s)],
              fill=FUR, outline=INK, width=lw)
    d.ellipse([x - 122 * s, rear_y - 100 * s, x + 16 * s, rear_y - 2 * s],
              fill=FUR, outline=INK, width=lw)
    d.ellipse([x - 16 * s, y - 40 * s, x + 96 * s, y + 22 * s],
              fill=FUR, outline=INK, width=lw)
    d.polygon([(x - 70 * s, rear_y - 88 * s), (x + 60 * s, y - 30 * s),
               (x + 60 * s, y + 10 * s), (x - 70 * s, rear_y - 12 * s)], fill=FUR)
    # front legs stretched flat on the floor
    d.line([(x + 44 * s, y - 10 * s), (x + 124 * s, y + 58 * s)],
           fill=INK, width=_ink(s, 4.0))
    d.line([(x + 44 * s, y - 10 * s), (x + 122 * s, y + 57 * s)],
           fill=FUR, width=_ink(s, 2.8))
    d.ellipse([x + 116 * s, y + 42 * s, x + 152 * s, y + 72 * s],
              fill=FUR, outline=INK, width=_ink(s, 0.8))
    _collar(d, x + 84 * s, y - 16 * s, s, tilt=6 * s)
    _head(d, x + 110 * s, y - 34 * s, s * 0.96, ear_lift=-8,
          blink=(math.sin(t * 2.0) + 1) / 2, mouth=0.5)


def dog_sniffing(d: ImageDraw.ImageDraw, x: float, y: float, t: float,
                 s: float = 1.0, shadow: tuple = FLOOR_DARK) -> None:
    """Nose down, working. The pose the whole mirror joke turns on."""
    cast = math.sin(t * 3.4) * 26 * s
    wag = math.sin(t * 7.0) * 30 * s
    _shadow(d, x + 6 * s, y + 92 * s, 250 * s, shadow)
    tip = (x - 190 * s, y - 118 * s + wag)
    d.line([(x - 86 * s, y - 40 * s), (x - 148 * s, y - 92 * s + wag * 0.6),
            tip], fill=INK, width=_ink(s, 3.4), joint="curve")
    d.line([(x - 86 * s, y - 40 * s), (x - 146 * s, y - 91 * s + wag * 0.6),
            tip], fill=FUR_DARK, width=_ink(s, 2.2), joint="curve")
    d.ellipse([tip[0] - 20 * s, tip[1] - 20 * s, tip[0] + 20 * s, tip[1] + 20 * s],
              fill=FUR_LIGHT, outline=INK, width=_ink(s, 0.8))
    _leg(d, (x - 52 * s, y - 4 * s), 0, s, FUR_DARK)
    _leg(d, (x - 24 * s, y - 4 * s), 0, s, FUR_DARK)
    _torso(d, x, y, s)
    _leg(d, (x + 56 * s, y - 4 * s), 0, s, FUR)
    _leg(d, (x + 82 * s, y - 4 * s), 0, s, FUR)
    _collar(d, x + 96 * s, y - 40 * s, s, tilt=14 * s)
    _head(d, x + 120 * s + cast * 0.3, y - 34 * s, s * 0.96, ear_lift=-14)
    # scent curls coming off the floor
    for k in range(3):
        a = t * 2.2 + k * 1.3
        sx = x + 190 * s + cast + k * 22 * s
        sy = y + 30 * s - (a % 2.0) * 90 * s
        r = (10 + k * 5) * s
        d.arc([sx - r, sy - r, sx + r, sy + r], start=120, end=330,
              fill=INK, width=_ink(s, 0.7))


def dog_closeup(d: ImageDraw.ImageDraw, x: float, y: float, t: float,
                s: float = 1.0, asleep: bool = False) -> None:
    """Head filling the frame. A minute at one camera distance goes flat."""
    tilt = math.sin(t * 1.4) * 8 * s
    _head(d, x, y + tilt, s, ear_lift=math.sin(t * 2.2) * 6,
          asleep=asleep, blink=(math.sin(t * 2.3) + 1) / 2)


# --- props ------------------------------------------------------------------

def _sofa(d: ImageDraw.ImageDraw, sx: float, floor_y: float, u: float,
          night: bool) -> None:
    body = _dim(night, SOFA, (130, 90, 106))
    dark = _dim(night, SOFA_DARK, (108, 74, 90))
    lw = max(1, int(6 * u))
    sy = floor_y - 260 * u
    d.rounded_rectangle([sx - 18 * u, sy + 128 * u, sx + 62 * u, floor_y + 20 * u],
                        radius=int(28 * u), fill=body, outline=INK, width=lw)
    d.rounded_rectangle([sx + 458 * u, sy + 128 * u, sx + 538 * u,
                         floor_y + 20 * u], radius=int(28 * u), fill=body,
                        outline=INK, width=lw)
    d.rounded_rectangle([sx, sy, sx + 520 * u, floor_y + 20 * u],
                        radius=int(40 * u), fill=body, outline=INK, width=lw)
    d.rounded_rectangle([sx + 28 * u, sy + 42 * u, sx + 492 * u, sy + 196 * u],
                        radius=int(28 * u), fill=dark)
    # cushion seam, so the sofa is not one flat block
    d.line([(sx + 260 * u, sy + 42 * u), (sx + 260 * u, sy + 196 * u)],
           fill=body, width=max(1, int(5 * u)))


def _window(d: ImageDraw.ImageDraw, wx: float, h: int, u: float,
            night: bool) -> None:
    fill = _dim(night, (206, 230, 240), (44, 50, 84))
    lw = max(1, int(6 * u))
    d.rounded_rectangle([wx, h * 0.19, wx + 300 * u, h * 0.43],
                        radius=int(20 * u), fill=fill, outline=INK, width=lw)
    d.line([(wx + 150 * u, h * 0.19), (wx + 150 * u, h * 0.43)],
           fill=INK, width=max(1, int(5 * u)))
    if night:
        d.ellipse([wx + 196 * u, h * 0.225, wx + 252 * u, h * 0.256], fill=LAMP)


def _picture(d: ImageDraw.ImageDraw, x: float, y: float, u: float,
             night: bool) -> None:
    frame = _dim(night, FRAME, (90, 76, 102))
    inner = _dim(night, (238, 226, 206), (118, 130, 148))
    lw = max(1, int(6 * u))
    d.rounded_rectangle([x, y, x + 220 * u, y + 170 * u], radius=int(10 * u),
                        fill=frame, outline=INK, width=lw)
    d.rounded_rectangle([x + 22 * u, y + 22 * u, x + 198 * u, y + 148 * u],
                        radius=int(6 * u), fill=inner)
    # a paw print in the picture, because the room belongs to a dog
    d.ellipse([x + 88 * u, y + 76 * u, x + 134 * u, y + 122 * u], fill=frame)
    for dx in (-30, 2, 34):
        d.ellipse([x + 92 * u + dx * u, y + 44 * u,
                   x + 118 * u + dx * u, y + 72 * u], fill=frame)


def _plant(d: ImageDraw.ImageDraw, x: float, floor_y: float, u: float,
           night: bool) -> None:
    leaf = _dim(night, LEAF_DARK, (72, 102, 80))
    leaf2 = _dim(night, LEAF, (90, 122, 94))
    pot = _dim(night, POT, (148, 94, 72))
    lw = max(1, int(6 * u))
    d.polygon([(x - 56 * u, floor_y - 8 * u), (x + 56 * u, floor_y - 8 * u),
               (x + 40 * u, floor_y + 100 * u), (x - 40 * u, floor_y + 100 * u)],
              fill=pot, outline=INK, width=lw)
    for a, r in ((-48, 160), (0, 200), (46, 160)):
        tipx = x + math.sin(math.radians(a)) * r * u
        tipy = floor_y - 8 * u - math.cos(math.radians(a)) * r * u
        d.line([(x, floor_y - 8 * u), (tipx, tipy)], fill=INK,
               width=max(1, int(20 * u)))
        d.line([(x, floor_y - 8 * u), (tipx, tipy)], fill=leaf,
               width=max(1, int(13 * u)))
        d.ellipse([tipx - 38 * u, tipy - 46 * u, tipx + 38 * u, tipy + 26 * u],
                  fill=leaf2, outline=INK, width=lw)


def _dog_bed(d: ImageDraw.ImageDraw, cx: float, cy: float, u: float,
             night: bool) -> None:
    body = _dim(night, BED, (118, 82, 110))
    rim = _dim(night, BED_DARK, (94, 64, 90))
    lw = max(1, int(6 * u))
    d.ellipse([cx - 380 * u, cy - 108 * u, cx + 380 * u, cy + 108 * u],
              fill=body, outline=INK, width=lw)
    d.ellipse([cx - 320 * u, cy - 72 * u, cx + 320 * u, cy + 88 * u], fill=rim)


def _tree(d: ImageDraw.ImageDraw, tx: float, ty: float, horizon: float,
          r: float, u: float) -> None:
    lw = max(1, int(6 * u))
    d.line([(tx, ty), (tx, horizon + 10 * u)], fill=INK, width=max(1, int(44 * u)))
    # BARK, not FUR_DARK: the trunk had been painted in a coat colour.
    d.line([(tx, ty), (tx, horizon + 10 * u)], fill=BARK,
           width=max(1, int(32 * u)))
    d.ellipse([tx - r, ty - r, tx + r, ty + r * 0.7], fill=LEAF_DARK,
              outline=INK, width=lw)
    d.ellipse([tx - r * 0.7, ty - r * 0.95, tx + r * 0.8, ty + r * 0.35],
              fill=LEAF)


# --- backgrounds ------------------------------------------------------------

def _base_room(w: int, h: int, night: bool = False) -> Image.Image:
    key = ("room", w, h, night)
    if key not in _BASES:
        u = _u(w)
        img = Image.new("RGB", (w, h), PAPER)
        floor_y = int(h * 0.70)
        _vgrad(img, NIGHT_TOP if night else WALL_TOP,
               NIGHT_BOT if night else WALL_BOT, 0, floor_y)
        d = ImageDraw.Draw(img)
        d.rectangle([0, floor_y, w, h], fill=NIGHT_FLOOR if night else FLOOR)
        board = _dim(night, FLOOR_LINE, (64, 64, 86))
        step = max(2, int(130 * u))
        for by in range(int(floor_y + 90 * u), h, step):
            d.line([(0, by), (w, by)], fill=board, width=max(1, int(5 * u)))
        d.rectangle([0, floor_y - 16 * u, w, floor_y],
                    fill=_dim(night, FLOOR_DARK, (56, 56, 78)))
        d.ellipse([w * 0.10, floor_y + 170 * u, w * 1.04, floor_y + 400 * u],
                  fill=_dim(night, FLOOR_DARK, (62, 62, 86)))
        _BASES[key] = img
    return _BASES[key]


def _base_park(w: int, h: int) -> Image.Image:
    key = ("park", w, h)
    if key not in _BASES:
        u = _u(w)
        img = Image.new("RGB", (w, h), SKY_BOT)
        horizon = int(h * 0.66)
        _vgrad(img, SKY_TOP, SKY_BOT, 0, horizon)
        d = ImageDraw.Draw(img)
        d.ellipse([w * 0.66, h * 0.07, w * 0.92, h * 0.21], fill=(250, 232, 176))
        d.ellipse([-w * 0.25, horizon - 210 * u, w * 0.6, horizon + 60 * u],
                  fill=LEAF_DARK)
        d.ellipse([w * 0.42, horizon - 150 * u, w * 1.3, horizon + 60 * u],
                  fill=LEAF_DARK)
        d.rectangle([0, horizon, w, h], fill=LEAF)
        d.rectangle([0, horizon, w, horizon + 18 * u], fill=LEAF_DARK)
        _BASES[key] = img
    return _BASES[key]


def _scroll(t: float, speed: float, span: float, offset: float = 0.0) -> float:
    return (offset - t * speed) % span - span * 0.25


# --- scenes -----------------------------------------------------------------

Scene = Callable[[Image.Image, ImageDraw.ImageDraw, int, int, float], None]


def _room_static(img, d, w, h, night=False):
    u = _u(w)
    img.paste(_base_room(w, h, night))
    floor_y = h * 0.70
    _picture(d, w * 0.10, h * 0.21, u, night)
    _window(d, w * 0.62, h, u, night)
    _plant(d, w * 0.92, floor_y, u, night)
    _sofa(d, w * 0.02, floor_y, u, night)


def scene_sleep(img, d, w, h, t):
    _room_static(img, d, w, h)
    dog_sleeping(d, w * 0.52, h * 0.745, t, s=1.9 * _u(w))


def scene_sleep_night(img, d, w, h, t):
    _room_static(img, d, w, h, night=True)
    dog_sleeping(d, w * 0.52, h * 0.745, t, s=1.9 * _u(w), shadow=NIGHT_SHADOW)


def scene_snore(img, d, w, h, t):
    _room_static(img, d, w, h, night=True)
    u = _u(w)
    _dog_bed(d, w * 0.50, h * 0.70 + 250 * u, u, night=True)
    dog_sleeping(d, w * 0.50, h * 0.70 + 170 * u, t, s=1.75 * u,
                 shadow=NIGHT_SHADOW, snore=True)


def scene_bed(img, d, w, h, t):
    _room_static(img, d, w, h)
    u = _u(w)
    _dog_bed(d, w * 0.50, h * 0.70 + 250 * u, u, night=False)
    dog_sleeping(d, w * 0.50, h * 0.70 + 170 * u, t, s=1.75 * u)


def _sprint(img, d, w, h, t, night=False):
    u = _u(w)
    img.paste(_base_room(w, h, night))
    floor_y = h * 0.70
    span = w + 980 * u
    for i in range(2):
        _sofa(d, _scroll(t, 780 * u, span, offset=i * span * 0.5), floor_y, u, night)
    for i in range(2):
        _window(d, _scroll(t, 780 * u, span, offset=i * span * 0.5 + 460 * u),
                h, u, night)
    for k in range(7):
        ly = h * 0.46 + k * 52 * u
        off = (t * 2500 * u + k * 210 * u) % (w + 480 * u) - 480 * u
        d.line([(off, ly), (off + 210 * u, ly)],
               fill=_dim(night, FLOOR_DARK, (102, 102, 130)),
               width=max(1, int(7 * u)))
    dog_running(d, w * 0.46, h * 0.765, t, s=1.75 * u,
                shadow=NIGHT_SHADOW if night else FLOOR_DARK)


def scene_sprint_room(img, d, w, h, t):
    _sprint(img, d, w, h, t)


def scene_sprint_night(img, d, w, h, t):
    _sprint(img, d, w, h, t, night=True)


def scene_sprint_park(img, d, w, h, t):
    u = _u(w)
    img.paste(_base_park(w, h))
    horizon = h * 0.66
    span = w + 980 * u
    for i in range(4):
        _tree(d, _scroll(t, 680 * u, span, offset=i * span * 0.25), h * 0.50,
              horizon, (150 + (i % 2) * 46) * u, u)
    dog_running(d, w * 0.46, h * 0.785, t, s=1.75 * u, shadow=LEAF_DARK)


def scene_stand_park(img, d, w, h, t):
    u = _u(w)
    img.paste(_base_park(w, h))
    horizon = h * 0.66
    _tree(d, w * 0.15, h * 0.47, horizon, 172 * u, u)
    _tree(d, w * 0.86, h * 0.50, horizon, 140 * u, u)
    dog_standing(d, w * 0.46, h * 0.80, t, s=1.8 * u, shadow=LEAF_DARK)


def scene_stand_room(img, d, w, h, t):
    _room_static(img, d, w, h)
    dog_standing(d, w * 0.50, h * 0.78, t, s=1.8 * _u(w))


def scene_closeup(img, d, w, h, t):
    _room_static(img, d, w, h)
    dog_closeup(d, w * 0.52, h * 0.47, t, s=3.4 * _u(w))


def scene_closeup_sleep(img, d, w, h, t):
    _room_static(img, d, w, h, night=True)
    dog_closeup(d, w * 0.52, h * 0.47, t, s=3.4 * _u(w), asleep=True)


def scene_box(img, d, w, h, t):
    u = _u(w)
    _room_static(img, d, w, h)
    floor_y = h * 0.70
    bx, by = w * 0.26, floor_y + 40 * u
    bw, bh = 560 * u, 320 * u
    lw = max(1, int(6 * u))
    _shadow(d, bx + bw / 2, by + bh + 12 * u, bw * 1.15)
    d.polygon([(bx - 76 * u, by - 76 * u), (bx + 128 * u, by - 76 * u),
               (bx + 62 * u, by), (bx, by)], fill=CARD_DARK, outline=INK, width=lw)
    d.polygon([(bx + bw + 76 * u, by - 76 * u), (bx + bw - 128 * u, by - 76 * u),
               (bx + bw - 62 * u, by), (bx + bw, by)], fill=CARD_DARK,
              outline=INK, width=lw)
    d.polygon([(bx, by), (bx + bw, by), (bx + bw - 34 * u, by + bh),
               (bx + 34 * u, by + bh)], fill=CARD, outline=INK, width=lw)
    bob = math.sin(t * 2.0) * 9 * u
    _head(d, bx + bw * 0.60, by - 34 * u + bob, 1.55 * u,
          blink=(math.sin(t * 2.1) + 1) / 2)


def _mirror_frame(d, w, h, u):
    lw = max(1, int(6 * u))
    mx0, my0 = w * 0.42, h * 0.27
    mx1, my1 = w * 0.98, h * 0.70 + 130 * u
    d.rounded_rectangle([mx0 - 22 * u, my0 - 22 * u, mx1 + 22 * u, my1 + 22 * u],
                        radius=int(24 * u), fill=FRAME, outline=INK, width=lw)
    d.rounded_rectangle([mx0, my0, mx1, my1], radius=int(14 * u),
                        fill=(212, 228, 236), outline=INK, width=lw)
    return (int(mx0), int(my0), int(mx1), int(my1)), mx0


def _mirror_scene(img, d, w, h, t, pose):
    """The dog, and the same drawing flipped, inside the glass.

    Mirroring the identical pose is the whole point of the gag: the thing in
    the mirror has to move exactly when the dog moves, because it is the dog.
    """
    u = _u(w)
    _room_static(img, d, w, h)
    box, plane = _mirror_frame(d, w, h, u)
    floor_y = h * 0.70
    px, py = w * 0.20, floor_y + 60 * u

    layer = Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    pose(ld, px, py, t, s=1.55 * u, shadow=(190, 208, 216))
    # A plain flip mirrors about the middle of the frame, which puts the
    # reflection at the wrong depth. Reflecting about the glass itself is
    # flip plus a shift of 2*plane - w, and then the nose in the mirror meets
    # the real nose at the glass, which is the whole joke.
    flipped = layer.transpose(Image.FLIP_LEFT_RIGHT)
    shifted = Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 0))
    shifted.paste(flipped, (int(2 * plane - w), 0))
    region = shifted.crop(box)
    img.paste(region, box, region)

    pose(d, px, py, t, s=1.55 * u)


def scene_mirror_bark(img, d, w, h, t):
    _mirror_scene(img, d, w, h, t, dog_barking)


def scene_mirror_bow(img, d, w, h, t):
    _mirror_scene(img, d, w, h, t, dog_playbow)


def scene_bark(img, d, w, h, t):
    _room_static(img, d, w, h)
    dog_barking(d, w * 0.42, h * 0.78, t, s=1.8 * _u(w))


def scene_playbow(img, d, w, h, t):
    _room_static(img, d, w, h)
    dog_playbow(d, w * 0.42, h * 0.78, t, s=1.8 * _u(w))


def scene_sniff(img, d, w, h, t):
    _room_static(img, d, w, h)
    dog_sniffing(d, w * 0.42, h * 0.78, t, s=1.8 * _u(w))


def scene_sniff_park(img, d, w, h, t):
    u = _u(w)
    img.paste(_base_park(w, h))
    horizon = h * 0.66
    _tree(d, w * 0.15, h * 0.47, horizon, 172 * u, u)
    _tree(d, w * 0.86, h * 0.50, horizon, 140 * u, u)
    dog_sniffing(d, w * 0.44, h * 0.80, t, s=1.8 * u, shadow=LEAF_DARK)


def scene_vet(img, d, w, h, t):
    u = _u(w)
    img.paste(_base_room(w, h))
    floor_y = h * 0.70
    lw = max(1, int(6 * u))
    d.rounded_rectangle([w * 0.04, floor_y + 20 * u, w * 0.96, floor_y + 210 * u],
                        radius=int(28 * u), fill=(228, 234, 236),
                        outline=INK, width=lw)
    cx, cy, arm = w * 0.74, h * 0.24, 96 * u
    d.rounded_rectangle([cx - arm / 3, cy - arm, cx + arm / 3, cy + arm],
                        radius=int(14 * u), fill=ACCENT, outline=INK, width=lw)
    d.rounded_rectangle([cx - arm, cy - arm / 3, cx + arm, cy + arm / 3],
                        radius=int(14 * u), fill=ACCENT, outline=INK, width=lw)
    dog_standing(d, w * 0.44, floor_y - 68 * u, t, s=1.75 * u,
                 shadow=(204, 212, 216))


SCENES: dict[str, Scene] = {
    "sleep": scene_sleep,
    "sleep_night": scene_sleep_night,
    "snore": scene_snore,
    "bed": scene_bed,
    "sprint_room": scene_sprint_room,
    "sprint_night": scene_sprint_night,
    "sprint_park": scene_sprint_park,
    "stand_park": scene_stand_park,
    "stand_room": scene_stand_room,
    "closeup": scene_closeup,
    "closeup_sleep": scene_closeup_sleep,
    "box": scene_box,
    "mirror": scene_mirror_bark,
    "mirror_bow": scene_mirror_bow,
    "bark": scene_bark,
    "playbow": scene_playbow,
    "sniff": scene_sniff,
    "sniff_park": scene_sniff_park,
    "vet": scene_vet,
}

KEYWORDS: list[tuple[tuple[str, ...], str]] = [
    (("vet", "veterinarian", "clinic", "sick", "pain", "checkup", "examined"),
     "vet"),
    (("mirror", "reflection", "glass"), "mirror"),
    (("bow", "playbow"), "playbow"),
    (("bark", "barking", "barks", "meow", "meowing", "meows", "yowl"),
     "bark"),
    (("sniff", "sniffing", "smell", "scent", "nose"), "sniff"),
    (("snore", "snoring", "snores"), "snore"),
    (("box", "cardboard", "carton", "litter"), "box"),
    (("bed", "basket", "cushion", "purr", "purring", "purrs", "kneading",
      "lap", "curled"), "bed"),
    (("closeup", "close", "face", "portrait", "camera", "staring", "looking"),
     "closeup"),
    (("sleep", "sleeping", "asleep", "resting", "tired", "lying", "rest"),
     "sleep"),
    (("sprint", "running", "runs", "run", "zoomies", "fast", "jumping",
      "chasing", "playing", "play"), "sprint_room"),
    (("fetch", "ball", "toy"), "sprint_park"),
    (("park", "field", "grass", "outdoors", "walk", "walking", "leash",
      "lawn", "garden"), "stand_park"),
]

NIGHT_WORDS = ("night", "dark", "3am", "2am", "evening", "midnight", "bedtime")

# Where the scene happens, as opposed to what happens in it. "running" and
# "park" sit on separate rungs of KEYWORDS and the activity rung is reached
# first, so without this "puppy running park" drew a dog sprinting through a
# living room. Night wins over these: there is no after-dark park scene.
OUTDOOR_WORDS = ("park", "field", "grass", "outdoors", "outside", "lawn",
                 "garden", "yard")


CAT_WORDS = (
    "cat", "cats", "kitten", "kittens", "kitty", "feline", "tabby",
    "siamese", "persian", "ragdoll", "calico", "tomcat", "meow", "purr",
    "purring", "litter", "hairball", "whiskers",
)


def species_for(query: str, dog: str = "dog") -> str:
    """"cat" when the query is clearly about a cat, otherwise `dog`.

    A dog is the default because the channel is dog-led and an unmarked
    query ("zoomies at 3am") is far more likely to be one. `dog` names which
    dog character to draw, so the channel picks its own animal once in
    config instead of in every script.
    """
    words = query.lower().replace("-", " ").split()
    return "cat" if any(w.startswith(CAT_WORDS) for w in words) else dog


def scene_for(query: str) -> str:
    words = query.lower().replace("-", " ").split()
    night = any(w in NIGHT_WORDS for w in words)
    outdoors = any(w in OUTDOOR_WORDS for w in words)
    for keys, name in KEYWORDS:
        if any(k in words for k in keys):
            if night and name == "sleep":
                return "sleep_night"
            if night and name == "sprint_room":
                return "sprint_night"
            if night and name == "closeup":
                return "closeup_sleep"
            if night and name == "bed":
                return "snore"
            if name == "mirror" and any(k in words for k in ("bow", "play")):
                return "mirror_bow"
            if outdoors and name == "sprint_room":
                return "sprint_park"
            if outdoors and name == "sniff":
                return "sniff_park"
            return name
    return "stand_park" if outdoors else "stand_room"


def render_clip(query: str, seconds: float, cfg: dict[str, Any],
                dest: Path) -> Path:
    """Draw `seconds` of animation for `query` and encode it to `dest`."""
    w = int(cfg["video"]["width"])
    h = int(cfg["video"]["height"])
    fps = int(cfg["video"]["fps"])
    ss = max(1, int(cfg["video"].get("toon_supersample", SUPERSAMPLE)))
    scene = SCENES[scene_for(query)]

    frames = dest.parent / f"_{dest.stem}_frames"
    if frames.exists():
        for old in frames.glob("*.png"):
            old.unlink()
    frames.mkdir(parents=True, exist_ok=True)

    bw, bh = w * ss, h * ss
    total = max(2, int(round(seconds * fps)))
    for i in range(total):
        t = i / fps
        big = Image.new("RGB", (bw, bh), PAPER)
        d = ImageDraw.Draw(big)
        scene(big, d, bw, bh, t)
        big.resize((w, h), Image.LANCZOS).save(frames / f"{i:04d}.png")

    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
         "-i", str(frames / "%04d.png"), "-t", f"{seconds:.3f}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
         "-pix_fmt", "yuv420p", str(dest)],
        check=True,
    )
    for old in frames.glob("*.png"):
        old.unlink()
    frames.rmdir()
    return dest
