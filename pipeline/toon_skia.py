"""Skia drawing backend for the cartoon renderer.

Pillow's ImageDraw was the ceiling on how good the drawn animation could get.
It has no bezier curves, so every organic form had to be faked by overlapping
ellipses; no gradients inside a shape, so fur and walls were flat; no control
over stroke joins, so outlines met at hard corners; and no anti-aliasing at
all, which is why frames had to be drawn at 3x and scaled down.

Skia is the renderer behind Chrome and Flutter. It gives all four back:

  Bezier paths, so the body is one closed curve with the haunch and chest in
  it, instead of three ellipses stacked to imply a shape.

  Gradients, so fur has a lit top and a shaded belly, and walls fall off
  toward the floor.

  Round joins and caps on strokes, so the ink outline reads as a drawn line.

  Real anti-aliasing, which removes the need for supersampling entirely. That
  makes it both better looking and about three times faster.
"""
from __future__ import annotations

import math
from typing import Any

import skia

# Same palette as the Pillow backend, so the two can be compared directly.
PAPER = (250, 238, 219)
WALL_TOP = (247, 232, 212)
WALL_BOT = (232, 208, 180)
FLOOR = (226, 199, 166)
FLOOR_DARK = (204, 172, 137)
FLOOR_LINE = (214, 187, 153)
FUR_LIT = (228, 172, 116)
FUR = (208, 148, 92)
FUR_SHADE = (176, 116, 68)
FUR_BELLY = (240, 204, 160)
INK = (54, 38, 30)
ACCENT = (222, 96, 68)
LEAF = (132, 172, 120)
LEAF_DARK = (96, 136, 90)
SKY_TOP = (176, 214, 232)
SKY_BOT = (226, 242, 242)
SOFA = (198, 120, 102)
SOFA_DARK = (166, 92, 78)
FRAME = (176, 132, 104)
POT = (188, 118, 88)
CARD = (214, 172, 120)
CARD_DARK = (184, 140, 92)
GLASS_TOP = (222, 236, 242)
GLASS_BOT = (196, 216, 228)
NIGHT_TOP = (48, 54, 90)
NIGHT_BOT = (82, 84, 120)
NIGHT_FLOOR = (68, 66, 92)
LAMP = (250, 218, 150)
TONGUE = (222, 122, 130)
MOUTH = (74, 40, 42)


def col(rgb: tuple[int, int, int], a: int = 255) -> int:
    return skia.Color(rgb[0], rgb[1], rgb[2], a)


def fill(rgb: tuple[int, int, int], a: int = 255) -> skia.Paint:
    return skia.Paint(AntiAlias=True, Color=col(rgb, a),
                      Style=skia.Paint.kFill_Style)


def stroke(rgb: tuple[int, int, int], width: float) -> skia.Paint:
    """Ink line with round joins and caps, so corners read as drawn."""
    return skia.Paint(AntiAlias=True, Color=col(rgb), Style=skia.Paint.kStroke_Style,
                      StrokeWidth=width, StrokeJoin=skia.Paint.kRound_Join,
                      StrokeCap=skia.Paint.kRound_Cap)


def grad(p0: tuple[float, float], p1: tuple[float, float],
         c0: tuple[int, int, int], c1: tuple[int, int, int]) -> skia.Paint:
    p = skia.Paint(AntiAlias=True, Style=skia.Paint.kFill_Style)
    p.setShader(skia.GradientShader.MakeLinear(
        points=[p0, p1], colors=[col(c0), col(c1)]))
    return p


def soft_shadow(canvas, cx: float, cy: float, w: float, h: float,
                rgb: tuple[int, int, int], sigma: float) -> None:
    """A blurred contact shadow. Pillow could only do a hard ellipse."""
    p = fill(rgb, 150)
    p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, sigma))
    canvas.drawOval(skia.Rect.MakeLTRB(cx - w / 2, cy - h / 2,
                                       cx + w / 2, cy + h / 2), p)


def ink(s: float, mult: float = 1.0) -> float:
    return max(1.0, 5.6 * s * mult)


# Cuteness in character design is mostly baby schema: a head large against the
# body, big eyes set low in the face, a short muzzle and stubby limbs. These
# live together so they can be tuned as a set rather than hidden in each pose.
# The channel covers cats as well as dogs, and a cat script drawn with a dog
# is the same mismatch this module exists to prevent. The difference between
# the two is small and local: ears, tail, muzzle and whiskers. Everything else
# - body, legs, poses, scenes, camera - is shared.
SPECIES = "dog"


def use_species(name: str) -> None:
    global SPECIES
    SPECIES = "cat" if name == "cat" else "dog"


def is_cat() -> bool:
    return SPECIES == "cat"


HEAD = 1.32      # head scale relative to the body
LEG = 0.76       # leg length against the old lanky ones
BLUSH = (238, 150, 140)


# --- character --------------------------------------------------------------

def _body_path(x: float, y: float, s: float, lean: float = 0.0) -> skia.Path:
    """One closed curve: haunch, back, shoulder, chest, belly.

    This is the shape Pillow could not draw. Three overlapping ellipses can
    imply a body, but they cannot give it a back line that flows into the
    neck or a belly that tucks under the ribs.
    """
    p = skia.Path()
    p.moveTo(x - 94 * s, y - 38 * s)                        # rump
    p.cubicTo(x - 104 * s, y - 98 * s, x - 44 * s, y - 102 * s,
              x - 4 * s, y - 92 * s + lean * 0.3)           # back
    p.cubicTo(x + 36 * s, y - 86 * s + lean * 0.7, x + 72 * s, y - 90 * s + lean,
              x + 92 * s, y - 62 * s + lean)                # shoulder
    p.cubicTo(x + 110 * s, y - 38 * s + lean, x + 106 * s, y + 0 * s + lean,
              x + 80 * s, y + 12 * s + lean)                # chest
    p.cubicTo(x + 34 * s, y + 28 * s + lean * 0.6, x - 32 * s, y + 30 * s,
              x - 74 * s, y + 18 * s)                       # belly
    p.cubicTo(x - 92 * s, y + 10 * s, x - 92 * s, y - 16 * s,
              x - 94 * s, y - 38 * s)
    p.close()
    return p


def _belly_path(x: float, y: float, s: float, lean: float = 0.0) -> skia.Path:
    p = skia.Path()
    p.moveTo(x - 60 * s, y + 8 * s)
    p.cubicTo(x - 10 * s, y + 20 * s, x + 50 * s, y + 16 * s + lean * 0.6,
              x + 88 * s, y - 2 * s + lean)
    p.cubicTo(x + 70 * s, y - 34 * s + lean, x + 10 * s, y - 28 * s,
              x - 60 * s, y + 8 * s)
    p.close()
    return p


def _ear_path(hx: float, hy: float, s: float, lift: float) -> skia.Path:
    """Floppy and hanging for a dog; upright and triangular for a cat.

    This single shape does most of the work of telling the two apart, which
    is why it is worth branching here rather than drawing a second character.
    """
    p = skia.Path()
    if is_cat():
        # Tall and vertical. The first pass drew a small ear angled back,
        # which on a long muzzle is a rodent's silhouette. The base sits at
        # hy - 24s so it is buried under the skull rather than perched on it.
        top = hy - 104 * s - lift * s * 0.4
        p.moveTo(hx - 58 * s, top + 80 * s)
        p.cubicTo(hx - 58 * s, top + 22 * s, hx - 46 * s, top + 2 * s,
                  hx - 22 * s, top + 20 * s)
        p.cubicTo(hx - 4 * s, top + 36 * s, hx - 8 * s, top + 62 * s,
                  hx - 16 * s, top + 84 * s)
        p.close()
        return p
    top = hy - 40 * s - lift * s
    p.moveTo(hx - 26 * s, top)
    p.cubicTo(hx - 84 * s, top + 6 * s, hx - 92 * s, top + 70 * s,
              hx - 60 * s, top + 118 * s)
    p.cubicTo(hx - 34 * s, top + 140 * s, hx - 12 * s, top + 96 * s,
              hx - 14 * s, top + 40 * s)
    p.close()
    return p


def _skull_path(hx: float, hy: float, s: float) -> skia.Path:
    """The head outline: a dog's muzzle projects, a cat's face is flat.

    Branching the ears alone was not enough. A cat drawn with the dog's
    snout, whiskers running off the tip of it, reads as a mouse - so the
    front of the face comes back from +118 to +84 and the cheek fills out,
    and every feature in front of the eyes moves back with it.
    """
    p = skia.Path()
    if is_cat():
        p.moveTo(hx - 62 * s, hy - 12 * s)
        p.cubicTo(hx - 64 * s, hy - 68 * s, hx + 10 * s, hy - 80 * s,
                  hx + 46 * s, hy - 56 * s)          # skull dome
        p.cubicTo(hx + 72 * s, hy - 36 * s, hx + 84 * s, hy - 12 * s,
                  hx + 84 * s, hy + 14 * s)          # flat front of the face
        p.cubicTo(hx + 84 * s, hy + 44 * s, hx + 52 * s, hy + 62 * s,
                  hx + 10 * s, hy + 62 * s)          # full cheek into the jaw
        p.cubicTo(hx - 34 * s, hy + 62 * s, hx - 60 * s, hy + 30 * s,
                  hx - 62 * s, hy - 12 * s)
        p.close()
        return p
    p.moveTo(hx - 58 * s, hy - 14 * s)
    p.cubicTo(hx - 58 * s, hy - 62 * s, hx + 16 * s, hy - 74 * s,
              hx + 48 * s, hy - 48 * s)
    p.cubicTo(hx + 78 * s, hy - 24 * s, hx + 84 * s, hy + 10 * s,
              hx + 104 * s, hy + 12 * s)             # into the muzzle
    p.cubicTo(hx + 118 * s, hy + 14 * s, hx + 118 * s, hy + 40 * s,
              hx + 96 * s, hy + 44 * s)
    p.cubicTo(hx + 62 * s, hy + 50 * s, hx + 52 * s, hy + 58 * s,
              hx + 10 * s, hy + 58 * s)
    p.cubicTo(hx - 34 * s, hy + 58 * s, hx - 58 * s, hy + 26 * s,
              hx - 58 * s, hy - 14 * s)
    p.close()
    return p


def _second_ear(canvas, hx: float, hy: float, s: float, lift: float) -> None:
    """Cats read front-on enough that the far ear should show."""
    if not is_cat():
        return
    top = hy - 100 * s - lift * s * 0.4
    p = skia.Path()
    p.moveTo(hx + 20 * s, top + 78 * s)
    p.cubicTo(hx + 22 * s, top + 20 * s, hx + 40 * s, top + 0 * s,
              hx + 62 * s, top + 20 * s)
    p.cubicTo(hx + 78 * s, top + 36 * s, hx + 72 * s, top + 60 * s,
              hx + 62 * s, top + 82 * s)
    p.close()
    canvas.drawPath(p, fill(FUR_SHADE))
    canvas.drawPath(p, stroke(INK, ink(s)))
    inner = skia.Path(p)
    m = skia.Matrix()
    m.setScale(0.56, 0.56, hx + 42 * s, top + 78 * s)
    inner.transform(m)
    canvas.drawPath(inner, fill(BLUSH, 170))


def head(canvas, hx: float, hy: float, s: float, lift: float = 0.0,
         asleep: bool = False, blink: float = 0.0, mouth: float = 0.0) -> None:
    lw = ink(s)
    ip = stroke(INK, lw)

    _second_ear(canvas, hx, hy, s, lift)
    ear = _ear_path(hx, hy, s, lift)
    canvas.drawPath(ear, fill(FUR_SHADE))
    canvas.drawPath(ear, ip)
    # Only the ear _second_ear draws gets a pink inner. This one sits behind
    # the skull, which is drawn over it, so tinting it showed a 50-pixel
    # sliver of pink and nothing more.

    skull = _skull_path(hx, hy, s)
    canvas.drawPath(skull, grad((hx, hy - 70 * s), (hx, hy + 60 * s),
                                FUR_LIT, FUR))
    canvas.drawPath(skull, ip)

    # muzzle, lighter, tucked under the skull curve
    if is_cat():
        canvas.drawOval(skia.Rect.MakeLTRB(hx + 8 * s, hy + 14 * s,
                                           hx + 70 * s, hy + 58 * s),
                        fill(FUR_BELLY))
        canvas.drawOval(skia.Rect.MakeLTRB(hx + 10 * s, hy + 6 * s,
                                           hx + 54 * s, hy + 42 * s),
                        fill(FUR_BELLY))
    muzzle = skia.Path()
    muzzle.moveTo(hx + 36 * s, hy + 8 * s)
    muzzle.cubicTo(hx + 62 * s, hy - 2 * s, hx + 90 * s, hy + 4 * s,
                   hx + 92 * s, hy + 26 * s)
    muzzle.cubicTo(hx + 94 * s, hy + 48 * s, hx + 58 * s, hy + 52 * s,
                   hx + 38 * s, hy + 42 * s)
    muzzle.close()
    if not is_cat():
        canvas.drawPath(muzzle, fill(FUR_BELLY))

    # cheek blush, the cheapest cuteness cue there is
    bl = fill(BLUSH, 95)
    bl.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 11 * s))
    bx = hx - 14 * s if is_cat() else hx
    canvas.drawOval(skia.Rect.MakeLTRB(bx + 4 * s, hy + 14 * s,
                                       bx + 46 * s, hy + 38 * s), bl)

    if mouth > 0.02:
        # A dog's jaw opens along the muzzle; a cat has no muzzle to open
        # along, so the whole mouth is shorter and sits under the nose.
        gap = (44 if is_cat() else 56) * s * mouth
        x0, x1 = (36, 82) if is_cat() else (44, 104)
        m = skia.Path()
        m.moveTo(hx + x0 * s, hy + 28 * s)
        m.cubicTo(hx + (x0 + x1) / 2 * s, hy + 24 * s, hx + x1 * s, hy + 26 * s,
                  hx + x1 * s, hy + 30 * s)
        m.cubicTo(hx + x1 * s, hy + 30 * s + gap,
                  hx + (x0 + x1) / 2 * s, hy + 26 * s + gap,
                  hx + x0 * s, hy + 28 * s)
        m.close()
        canvas.drawPath(m, fill(MOUTH))
        canvas.drawPath(m, stroke(INK, ink(s, 0.7)))
        t = skia.Path()
        t.addOval(skia.Rect.MakeLTRB(hx + (x0 + 14) * s, hy + 28 * s + gap * 0.35,
                                     hx + (x1 - 8) * s, hy + 28 * s + gap * 0.95))
        canvas.drawPath(t, fill(TONGUE))

    # nose
    if is_cat():
        nx, ny = hx + 48 * s, hy + 20 * s
        tri = skia.Path()
        tri.moveTo(nx - 15 * s, ny - 5 * s)
        tri.quadTo(nx, ny - 12 * s, nx + 15 * s, ny - 5 * s)
        tri.quadTo(nx + 13 * s, ny + 13 * s, nx, ny + 15 * s)
        tri.quadTo(nx - 13 * s, ny + 13 * s, nx - 15 * s, ny - 5 * s)
        tri.close()
        canvas.drawPath(tri, fill(BLUSH))
        canvas.drawPath(tri, stroke(INK, ink(s, 0.6)))
        if mouth <= 0.02:
            # the closed cat mouth: two small curves off the nose
            for side in (-1, 1):
                w = skia.Path()
                w.moveTo(nx, ny + 15 * s)
                w.quadTo(nx + side * 6 * s, ny + 30 * s,
                         nx + side * 19 * s, ny + 22 * s)
                canvas.drawPath(w, stroke(INK, ink(s, 0.6)))
        # Whiskers root beside the nose and stop short of the frame. On the
        # old snout they ran off the tip, which was half the rodent look.
        for dy, spread in ((-6, 14), (4, 2), (14, -12)):
            wk = skia.Path()
            wk.moveTo(nx + 10 * s, ny + dy * s)
            wk.quadTo(nx + 38 * s, ny + (dy + spread * 0.3) * s,
                      nx + 70 * s, ny + (dy + spread) * s)
            canvas.drawPath(wk, stroke(INK, ink(s, 0.45)))
    else:
        canvas.drawOval(skia.Rect.MakeLTRB(hx + 76 * s, hy + 12 * s,
                                           hx + 108 * s, hy + 40 * s), fill(INK))
        canvas.drawOval(skia.Rect.MakeLTRB(hx + 83 * s, hy + 17 * s,
                                           hx + 93 * s, hy + 25 * s),
                        fill((152, 134, 126)))

    # brow and eye. A cat's face is 34 units shorter, so the eye would
    # otherwise sit on the edge of it: everything here shifts back by `ex`.
    ex = hx - 16 * s if is_cat() else hx
    brow = skia.Path()
    brow.moveTo(ex + 6 * s, hy - 40 * s)
    brow.quadTo(ex + 26 * s, hy - 50 * s, ex + 44 * s, hy - 38 * s)
    canvas.drawPath(brow, stroke(INK, ink(s, 0.8)))

    if asleep:
        e = skia.Path()
        e.moveTo(ex + 10 * s, hy - 14 * s)
        e.quadTo(ex + 28 * s, hy + 2 * s, ex + 46 * s, hy - 14 * s)
        canvas.drawPath(e, stroke(INK, ink(s, 0.85)))
    elif blink > 0.995:
        e = skia.Path()
        e.moveTo(ex + 14 * s, hy - 14 * s)
        e.lineTo(ex + 42 * s, hy - 14 * s)
        canvas.drawPath(e, stroke(INK, ink(s, 0.85)))
    else:
        canvas.drawOval(skia.Rect.MakeLTRB(ex + 18 * s, hy - 28 * s,
                                           ex + 56 * s, hy + 12 * s), fill(INK))
        canvas.drawOval(skia.Rect.MakeLTRB(ex + 38 * s, hy - 22 * s,
                                           ex + 52 * s, hy - 8 * s),
                        fill((255, 255, 255)))
        canvas.drawOval(skia.Rect.MakeLTRB(ex + 24 * s, hy - 2 * s,
                                           ex + 33 * s, hy + 7 * s),
                        fill((255, 255, 255), 215))


def collar(canvas, x: float, y: float, s: float) -> None:
    r = skia.Rect.MakeLTRB(x - 22 * s, y - 16 * s, x + 24 * s, y + 12 * s)
    rr = skia.RRect.MakeRectXY(r, 9 * s, 9 * s)
    canvas.drawRRect(rr, fill(ACCENT))
    canvas.drawRRect(rr, stroke(INK, ink(s, 0.75)))
    canvas.drawOval(skia.Rect.MakeLTRB(x - 2 * s, y + 8 * s, x + 20 * s,
                                       y + 30 * s), fill(LAMP))
    canvas.drawOval(skia.Rect.MakeLTRB(x - 2 * s, y + 8 * s, x + 20 * s,
                                       y + 30 * s), stroke(INK, ink(s, 0.6)))


def leg(canvas, hx: float, hy: float, swing: float, s: float,
        shade: tuple[int, int, int]) -> None:
    p = skia.Path()
    p.moveTo(hx, hy)
    p.quadTo(hx + swing * 0.5, hy + 46 * s * LEG, hx + swing, hy + 86 * s * LEG)
    canvas.drawPath(p, stroke(INK, ink(s, 4.7)))
    canvas.drawPath(p, stroke(shade, ink(s, 3.4)))
    py0, py1 = hy + 70 * s * LEG, hy + 100 * s * LEG
    for paint in (fill(shade), stroke(INK, ink(s, 0.75))):
        canvas.drawOval(skia.Rect.MakeLTRB(hx + swing - 20 * s, py0,
                                           hx + swing + 20 * s, py1), paint)


def tail(canvas, x: float, y: float, s: float, tipx: float, tipy: float) -> None:
    p = skia.Path()
    p.moveTo(x - 88 * s, y - 40 * s)
    if is_cat():
        # Longer and higher than a dog's, but just as thick. Drawn thin and
        # hooked it was a mouse's tail, and no amount of ear fixed that.
        p.cubicTo(x - 152 * s, y - 72 * s, x - 176 * s, y - 134 * s,
                  tipx + 8 * s, tipy - 20 * s)
        canvas.drawPath(p, stroke(INK, ink(s, 3.4)))
        canvas.drawPath(p, stroke(FUR_SHADE, ink(s, 2.2)))
        return
    p.quadTo(x - 150 * s, y - 92 * s, tipx, tipy)
    canvas.drawPath(p, stroke(INK, ink(s, 3.6)))
    canvas.drawPath(p, stroke(FUR_SHADE, ink(s, 2.4)))
    for paint in (fill(FUR_BELLY), stroke(INK, ink(s, 0.75))):
        canvas.drawOval(skia.Rect.MakeLTRB(tipx - 21 * s, tipy - 21 * s,
                                           tipx + 21 * s, tipy + 21 * s), paint)


def _draw_body(canvas, x: float, y: float, s: float, lean: float = 0.0) -> None:
    body = _body_path(x, y, s, lean)
    canvas.drawPath(body, grad((x, y - 92 * s), (x, y + 20 * s),
                               FUR_LIT, FUR_SHADE))
    canvas.drawPath(_belly_path(x, y, s, lean), fill(FUR_BELLY))
    canvas.drawPath(body, stroke(INK, ink(s)))


def dog_standing(canvas, x: float, y: float, t: float, s: float = 1.0,
                 shade: tuple[int, int, int] = FLOOR_DARK) -> None:
    """Idle, but never still.

    A character standing on a perfectly fixed body reads as a cut-out. The
    ribcage rises and falls, the weight shifts slightly between the front and
    back legs, and the tail leads the whole thing.
    """
    wag = math.sin(lag(t * 9.0, 0.7)) * 44 * s
    bob = math.sin(t * 2.2) * 4 * s
    breath = 1.0 + math.sin(t * 1.9) * 0.018
    sway = math.sin(t * 1.4) * 3 * s
    canvas.save()
    canvas.translate(x + sway, y)
    canvas.scale(1.0, breath)
    canvas.translate(-x, -y)
    soft_shadow(canvas, x + 6 * s, y + 96 * s, 250 * s, 52 * s, shade, 14 * s)
    y -= bob
    tail(canvas, x, y, s, x - 196 * s, y - 126 * s + wag)
    leg(canvas, x - 54 * s, y - 6 * s, 0, s, FUR_SHADE)
    leg(canvas, x - 26 * s, y - 6 * s, 0, s, FUR_SHADE)
    _draw_body(canvas, x, y, s)
    leg(canvas, x + 54 * s, y - 6 * s, 0, s, FUR)
    leg(canvas, x + 80 * s, y - 6 * s, 0, s, FUR)
    collar(canvas, x + 70 * s, y - 34 * s, s * 1.1)
    head(canvas, x + 108 * s, y - 118 * s, s * HEAD, lift=bob * 2,
         blink=(math.sin(t * 1.15) + 1) / 2)
    canvas.restore()


def dog_running(canvas, x: float, y: float, t: float, s: float = 1.0,
                shade: tuple[int, int, int] = FLOOR_DARK) -> None:
    phase = t * 13.0
    reach = math.sin(phase) * 58 * s
    trail = math.sin(phase + 2.4) * 58 * s
    air = max(0.0, math.sin(phase * 2)) * 28 * s
    ground = y + 96 * s
    y -= air
    lean = 12 * s
    # squash when the paws are down, stretch at the top of the bound
    squash = 1.0 - 0.055 * math.cos(phase * 2)
    canvas.save()
    canvas.translate(x, ground)
    canvas.scale(2.0 - squash, squash)
    canvas.translate(-x, -ground)
    soft_shadow(canvas, x + 10 * s, ground, (250 - air * 1.4) * s,
                (52 - air * 0.3) * s, shade, 14 * s)
    tail(canvas, x, y, s, x - 206 * s,
         y - 88 * s - math.sin(lag(phase, 0.9) * 1.4) * 30 * s)
    leg(canvas, x - 58 * s, y - 6 * s, trail - 26 * s, s, FUR_SHADE)
    leg(canvas, x - 30 * s, y - 6 * s, trail, s, FUR_SHADE)
    _draw_body(canvas, x, y, s, lean)
    leg(canvas, x + 56 * s, y - 6 * s + lean, reach, s, FUR)
    leg(canvas, x + 80 * s, y - 6 * s + lean, reach + 26 * s, s, FUR)
    collar(canvas, x + 68 * s, y - 30 * s + lean, s * 1.1)
    head(canvas, x + 108 * s, y - 112 * s + lean, s * HEAD,
         lift=-math.sin(lag(phase, 0.55)) * 20)
    canvas.restore()


def dog_barking(canvas, x: float, y: float, t: float, s: float = 1.0,
                shade: tuple[int, int, int] = FLOOR_DARK) -> None:
    pulse = max(0.0, math.sin(t * 9.0))
    recoil = pulse * 14 * s
    wag = math.sin(t * 11.0) * 36 * s
    soft_shadow(canvas, x + 6 * s, y + 96 * s, 250 * s, 52 * s, shade, 14 * s)
    y -= pulse * 6 * s
    tail(canvas, x - recoil, y, s, x - 196 * s - recoil, y - 122 * s + wag)
    leg(canvas, x - 54 * s - recoil, y - 6 * s, 0, s, FUR_SHADE)
    leg(canvas, x - 26 * s - recoil, y - 6 * s, 0, s, FUR_SHADE)
    _draw_body(canvas, x - recoil, y, s)
    leg(canvas, x + 54 * s, y - 6 * s, 0, s, FUR)
    leg(canvas, x + 80 * s, y - 6 * s, 0, s, FUR)
    collar(canvas, x + 70 * s - recoil * 0.4, y - 34 * s, s * 1.1)
    head(canvas, x + 108 * s, y - 126 * s - pulse * 12 * s, s * HEAD,
         lift=-pulse * 16, mouth=pulse)
    for k in range(3):
        r = (86 + k * 64) * s * (0.55 + pulse * 0.8)
        cx, cy = x + 206 * s, y - 106 * s - pulse * 12 * s
        arc = skia.Path()
        arc.addArc(skia.Rect.MakeLTRB(cx - r, cy - r, cx + r, cy + r), -52, 104)
        canvas.drawPath(arc, stroke(INK, ink(s, 1.3)))


def dog_sleeping(canvas, x: float, y: float, t: float, s: float = 1.0,
                 shade: tuple[int, int, int] = FLOOR_DARK,
                 snore: bool = False) -> None:
    breathe = math.sin(t * 2.0) * (9 if snore else 5) * s
    soft_shadow(canvas, x, y + 56 * s, 340 * s, 64 * s, shade, 16 * s)
    curl = skia.Path()
    curl.moveTo(x - 150 * s, y + 10 * s)
    curl.cubicTo(x - 160 * s, y - 70 * s - breathe, x - 40 * s,
                 y - 96 * s - breathe, x + 40 * s, y - 76 * s - breathe)
    curl.cubicTo(x + 130 * s, y - 56 * s, x + 158 * s, y + 20 * s,
                 x + 100 * s, y + 44 * s)
    curl.cubicTo(x + 20 * s, y + 62 * s, x - 110 * s, y + 56 * s,
                 x - 150 * s, y + 10 * s)
    curl.close()
    canvas.drawPath(curl, grad((x, y - 96 * s), (x, y + 50 * s),
                               FUR_LIT, FUR_SHADE))
    canvas.drawPath(curl, stroke(INK, ink(s)))
    tuck = skia.Path()
    tuck.addOval(skia.Rect.MakeLTRB(x - 86 * s, y - 34 * s, x + 96 * s,
                                    y + 40 * s))
    canvas.drawPath(tuck, fill(FUR_BELLY))
    head(canvas, x + 58 * s, y - 36 * s - breathe, s * 0.92 * HEAD, asleep=True)
    for i in range(3):
        a = t * (1.5 if snore else 0.9) + i * 1.1
        drift = (a % 3.0) / 3.0
        zx = x - 40 * s + math.sin(a * 2) * 20 * s
        zy = y - 120 * s - drift * 250 * s
        size = (30 + i * 15) * s * (1.25 if snore else 1.0)
        z = skia.Path()
        z.moveTo(zx, zy)
        z.lineTo(zx + size, zy)
        z.lineTo(zx, zy + size)
        z.lineTo(zx + size, zy + size)
        canvas.drawPath(z, stroke(INK, ink(s, 1.0)))


def dog_playbow(canvas, x: float, y: float, t: float, s: float = 1.0,
                shade: tuple[int, int, int] = FLOOR_DARK) -> None:
    """Chest on the floor, rear in the air: the invitation to play."""
    wag = math.sin(t * 13.0) * 52 * s
    bounce = math.sin(t * 3.0) * 6 * s
    soft_shadow(canvas, x + 6 * s, y + 96 * s, 270 * s, 56 * s, shade, 14 * s)
    rear = y - 74 * s + bounce
    tail(canvas, x, rear - 10 * s, s, x - 168 * s, rear - 120 * s + wag)
    leg(canvas, x - 52 * s, rear - 30 * s, 0, s, FUR_SHADE)
    leg(canvas, x - 24 * s, rear - 30 * s, 0, s, FUR_SHADE)
    body = skia.Path()
    body.moveTo(x - 112 * s, rear - 40 * s)
    body.cubicTo(x - 116 * s, rear - 96 * s, x - 50 * s, rear - 104 * s,
                 x + 4 * s, rear - 76 * s)
    body.cubicTo(x + 56 * s, rear - 48 * s, x + 84 * s, y - 34 * s,
                 x + 96 * s, y + 4 * s)
    body.cubicTo(x + 70 * s, y + 24 * s, x + 10 * s, y + 24 * s,
                 x - 30 * s, y - 2 * s)
    body.cubicTo(x - 80 * s, rear - 6 * s, x - 108 * s, rear - 14 * s,
                 x - 112 * s, rear - 40 * s)
    body.close()
    canvas.drawPath(body, grad((x, rear - 104 * s), (x, y + 20 * s),
                               FUR_LIT, FUR_SHADE))
    canvas.drawPath(body, stroke(INK, ink(s)))
    front = skia.Path()
    front.moveTo(x + 52 * s, y - 6 * s)
    front.quadTo(x + 96 * s, y + 34 * s, x + 132 * s, y + 56 * s)
    canvas.drawPath(front, stroke(INK, ink(s, 3.9)))
    canvas.drawPath(front, stroke(FUR, ink(s, 2.7)))
    canvas.drawOval(skia.Rect.MakeLTRB(x + 120 * s, y + 42 * s,
                                       x + 156 * s, y + 72 * s), fill(FUR))
    canvas.drawOval(skia.Rect.MakeLTRB(x + 120 * s, y + 42 * s,
                                       x + 156 * s, y + 72 * s),
                    stroke(INK, ink(s, 0.75)))
    collar(canvas, x + 66 * s, y - 20 * s, s * 1.1)
    head(canvas, x + 96 * s, y - 48 * s, s * 0.96 * HEAD, lift=-8, mouth=0.45)


def dog_sniffing(canvas, x: float, y: float, t: float, s: float = 1.0,
                 shade: tuple[int, int, int] = FLOOR_DARK) -> None:
    """Nose down, working."""
    cast = math.sin(t * 3.4) * 24 * s
    wag = math.sin(t * 7.0) * 30 * s
    soft_shadow(canvas, x + 6 * s, y + 96 * s, 250 * s, 52 * s, shade, 14 * s)
    tail(canvas, x, y, s, x - 196 * s, y - 120 * s + wag)
    leg(canvas, x - 54 * s, y - 6 * s, 0, s, FUR_SHADE)
    leg(canvas, x - 26 * s, y - 6 * s, 0, s, FUR_SHADE)
    _draw_body(canvas, x, y, s)
    leg(canvas, x + 54 * s, y - 6 * s, 0, s, FUR)
    leg(canvas, x + 80 * s, y - 6 * s, 0, s, FUR)
    collar(canvas, x + 70 * s, y - 28 * s, s * 1.1)
    head(canvas, x + 104 * s + cast * 0.3, y - 38 * s, s * 0.96 * HEAD, lift=-16)
    for k in range(3):
        a = t * 2.2 + k * 1.3
        sx = x + 188 * s + cast + k * 24 * s
        sy = y + 34 * s - (a % 2.0) * 90 * s
        r = (12 + k * 6) * s
        arc = skia.Path()
        arc.addArc(skia.Rect.MakeLTRB(sx - r, sy - r, sx + r, sy + r), 120, 210)
        canvas.drawPath(arc, stroke(INK, ink(s, 0.7)))


def dog_closeup(canvas, x: float, y: float, t: float, s: float = 1.0,
                asleep: bool = False) -> None:
    tilt = math.sin(t * 1.4) * 8 * s
    head(canvas, x, y + tilt, s, lift=math.sin(t * 2.2) * 6, asleep=asleep,
         blink=(math.sin(t * 1.5) + 1) / 2)


# --- backgrounds and props --------------------------------------------------

def _u(w: int) -> float:
    return w / 1080.0


def room(canvas, w: int, h: int, night: bool = False) -> None:
    """Wall gradient, floor, boards. One draw call each, no cached bitmap."""
    u = _u(w)
    floor_y = h * 0.70
    canvas.drawRect(skia.Rect.MakeLTRB(0, 0, w, floor_y),
                    grad((0, 0), (0, floor_y),
                         NIGHT_TOP if night else WALL_TOP,
                         NIGHT_BOT if night else WALL_BOT))
    canvas.drawRect(skia.Rect.MakeLTRB(0, floor_y, w, h),
                    grad((0, floor_y), (0, h),
                         NIGHT_FLOOR if night else FLOOR,
                         (60, 58, 82) if night else FLOOR_DARK))
    canvas.drawRect(skia.Rect.MakeLTRB(0, floor_y - 16 * u, w, floor_y),
                    fill((54, 54, 76) if night else FLOOR_DARK))
    line = fill((62, 62, 84) if night else FLOOR_LINE)
    yy = floor_y + 90 * u
    while yy < h:
        canvas.drawRect(skia.Rect.MakeLTRB(0, yy, w, yy + 5 * u), line)
        yy += 130 * u


def park(canvas, w: int, h: int) -> None:
    u = _u(w)
    horizon = h * 0.66
    canvas.drawRect(skia.Rect.MakeLTRB(0, 0, w, horizon),
                    grad((0, 0), (0, horizon), SKY_TOP, SKY_BOT))
    canvas.drawOval(skia.Rect.MakeLTRB(w * 0.66, h * 0.07, w * 0.92, h * 0.21),
                    fill((250, 232, 176)))
    canvas.drawOval(skia.Rect.MakeLTRB(-w * 0.25, horizon - 210 * u,
                                       w * 0.6, horizon + 60 * u), fill(LEAF_DARK))
    canvas.drawOval(skia.Rect.MakeLTRB(w * 0.42, horizon - 150 * u,
                                       w * 1.3, horizon + 60 * u), fill(LEAF_DARK))
    canvas.drawRect(skia.Rect.MakeLTRB(0, horizon, w, h),
                    grad((0, horizon), (0, h), LEAF, LEAF_DARK))
    canvas.drawRect(skia.Rect.MakeLTRB(0, horizon, w, horizon + 18 * u),
                    fill(LEAF_DARK))


def _rr(canvas, l, t, r, b, rad, paint):
    canvas.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(l, t, r, b),
                                           rad, rad), paint)


def sofa(canvas, sx: float, floor_y: float, u: float, night: bool) -> None:
    body = (128, 88, 104) if night else SOFA
    dark = (106, 72, 88) if night else SOFA_DARK
    lw = 6 * u
    sy = floor_y - 260 * u
    for ax in (sx - 18 * u, sx + 458 * u):
        _rr(canvas, ax, sy + 128 * u, ax + 80 * u, floor_y + 20 * u, 28 * u,
            fill(body))
        _rr(canvas, ax, sy + 128 * u, ax + 80 * u, floor_y + 20 * u, 28 * u,
            stroke(INK, lw))
    _rr(canvas, sx, sy, sx + 520 * u, floor_y + 20 * u, 40 * u,
        grad((sx, sy), (sx, floor_y), body, dark))
    _rr(canvas, sx, sy, sx + 520 * u, floor_y + 20 * u, 40 * u, stroke(INK, lw))
    _rr(canvas, sx + 28 * u, sy + 42 * u, sx + 492 * u, sy + 196 * u, 28 * u,
        fill(dark))


def window(canvas, wx: float, h: int, u: float, night: bool) -> None:
    lw = 6 * u
    top, bot = h * 0.19, h * 0.43
    _rr(canvas, wx, top, wx + 300 * u, bot, 20 * u,
        grad((wx, top), (wx, bot),
             (40, 46, 80) if night else GLASS_TOP,
             (58, 62, 98) if night else GLASS_BOT))
    _rr(canvas, wx, top, wx + 300 * u, bot, 20 * u, stroke(INK, lw))
    canvas.drawRect(skia.Rect.MakeLTRB(wx + 147 * u, top, wx + 153 * u, bot),
                    fill(INK))
    if night:
        canvas.drawOval(skia.Rect.MakeLTRB(wx + 196 * u, h * 0.225,
                                           wx + 252 * u, h * 0.256), fill(LAMP))


def picture(canvas, x: float, y: float, u: float, night: bool) -> None:
    frame = (88, 74, 100) if night else FRAME
    inner = (116, 128, 146) if night else (240, 228, 208)
    lw = 6 * u
    _rr(canvas, x, y, x + 220 * u, y + 170 * u, 10 * u, fill(frame))
    _rr(canvas, x, y, x + 220 * u, y + 170 * u, 10 * u, stroke(INK, lw))
    _rr(canvas, x + 22 * u, y + 22 * u, x + 198 * u, y + 148 * u, 6 * u,
        fill(inner))
    canvas.drawOval(skia.Rect.MakeLTRB(x + 88 * u, y + 76 * u,
                                       x + 134 * u, y + 122 * u), fill(frame))
    for dx in (-30, 2, 34):
        canvas.drawOval(skia.Rect.MakeLTRB(x + 92 * u + dx * u, y + 44 * u,
                                           x + 118 * u + dx * u, y + 72 * u),
                        fill(frame))


def plant(canvas, x: float, floor_y: float, u: float, night: bool) -> None:
    leafc = (70, 100, 78) if night else LEAF_DARK
    leaf2 = (88, 120, 92) if night else LEAF
    potc = (146, 92, 70) if night else POT
    lw = 6 * u
    pot = skia.Path()
    pot.moveTo(x - 56 * u, floor_y - 8 * u)
    pot.lineTo(x + 56 * u, floor_y - 8 * u)
    pot.lineTo(x + 40 * u, floor_y + 100 * u)
    pot.lineTo(x - 40 * u, floor_y + 100 * u)
    pot.close()
    canvas.drawPath(pot, fill(potc))
    canvas.drawPath(pot, stroke(INK, lw))
    for a, r in ((-48, 160), (0, 200), (46, 160)):
        tx = x + math.sin(math.radians(a)) * r * u
        ty = floor_y - 8 * u - math.cos(math.radians(a)) * r * u
        stem = skia.Path()
        stem.moveTo(x, floor_y - 8 * u)
        stem.quadTo((x + tx) / 2 + 14 * u, (floor_y + ty) / 2, tx, ty)
        canvas.drawPath(stem, stroke(INK, 20 * u))
        canvas.drawPath(stem, stroke(leafc, 13 * u))
        canvas.drawOval(skia.Rect.MakeLTRB(tx - 38 * u, ty - 46 * u,
                                           tx + 38 * u, ty + 26 * u), fill(leaf2))
        canvas.drawOval(skia.Rect.MakeLTRB(tx - 38 * u, ty - 46 * u,
                                           tx + 38 * u, ty + 26 * u),
                        stroke(INK, lw))


def tree(canvas, tx: float, ty: float, horizon: float, r: float, u: float) -> None:
    trunk = skia.Path()
    trunk.moveTo(tx, ty)
    trunk.lineTo(tx, horizon + 10 * u)
    canvas.drawPath(trunk, stroke(INK, 44 * u))
    canvas.drawPath(trunk, stroke(FUR_SHADE, 32 * u))
    canvas.drawOval(skia.Rect.MakeLTRB(tx - r, ty - r, tx + r, ty + r * 0.7),
                    grad((tx, ty - r), (tx, ty + r * 0.7), LEAF, LEAF_DARK))
    canvas.drawOval(skia.Rect.MakeLTRB(tx - r, ty - r, tx + r, ty + r * 0.7),
                    stroke(INK, 6 * u))


def dog_bed(canvas, cx: float, cy: float, u: float, night: bool) -> None:
    body = (116, 80, 108) if night else (168, 126, 158)
    rim = (92, 62, 88) if night else (144, 104, 136)
    canvas.drawOval(skia.Rect.MakeLTRB(cx - 380 * u, cy - 108 * u,
                                       cx + 380 * u, cy + 108 * u), fill(body))
    canvas.drawOval(skia.Rect.MakeLTRB(cx - 380 * u, cy - 108 * u,
                                       cx + 380 * u, cy + 108 * u),
                    stroke(INK, 6 * u))
    canvas.drawOval(skia.Rect.MakeLTRB(cx - 320 * u, cy - 72 * u,
                                       cx + 320 * u, cy + 88 * u), fill(rim))


# --- scenes -----------------------------------------------------------------

NIGHT_SHADOW = (48, 48, 70)


def box(canvas, bx: float, by: float, bw: float, bh: float, u: float) -> None:
    """A cardboard box, open flaps toward the camera.

    Ported from the Pillow backend, where the Skia table had it aliased to
    the dog bed. A cat sitting in a box is the most cat-shaped scene there
    is, so on a cat script that alias was the mismatch showing up again.
    """
    lw = 6 * u
    soft_shadow(canvas, bx + bw / 2, by + bh + 12 * u, bw * 1.15, 60 * u,
                FLOOR_DARK, 16 * u)
    for flap in (
        [(bx - 76 * u, by - 76 * u), (bx + 128 * u, by - 76 * u),
         (bx + 62 * u, by), (bx, by)],
        [(bx + bw + 76 * u, by - 76 * u), (bx + bw - 128 * u, by - 76 * u),
         (bx + bw - 62 * u, by), (bx + bw, by)],
    ):
        p = skia.Path()
        p.moveTo(*flap[0])
        for pt in flap[1:]:
            p.lineTo(*pt)
        p.close()
        canvas.drawPath(p, fill(CARD_DARK))
        canvas.drawPath(p, stroke(INK, lw))
    front = skia.Path()
    front.moveTo(bx, by)
    front.lineTo(bx + bw, by)
    front.lineTo(bx + bw - 34 * u, by + bh)
    front.lineTo(bx + 34 * u, by + bh)
    front.close()
    canvas.drawPath(front, grad((bx, by), (bx, by + bh), CARD, CARD_DARK))
    canvas.drawPath(front, stroke(INK, lw))


def scene_box(canvas, w, h, t):
    """Only a head over the rim: the rest is hidden, which sells the box."""
    u = _u(w)
    _room_static(canvas, w, h)
    floor_y = h * 0.70
    bx, by = w * 0.26, floor_y + 40 * u
    bw, bh = 560 * u, 320 * u
    bob = math.sin(t * 2.0) * 9 * u
    # High enough that the whole face clears the rim. At the Pillow
    # backend's -34u the box front cut the muzzle off at the eyes.
    head(canvas, bx + bw * 0.60, by - 96 * u + bob, 1.55 * u * HEAD,
         blink=(math.sin(t * 2.1) + 1) / 2)
    box(canvas, bx, by, bw, bh, u)


def _room_static(canvas, w, h, night=False):
    u = _u(w)
    room(canvas, w, h, night)
    floor_y = h * 0.70
    picture(canvas, w * 0.10, h * 0.21, u, night)
    window(canvas, w * 0.62, h, u, night)
    plant(canvas, w * 0.92, floor_y, u, night)
    sofa(canvas, w * 0.02, floor_y, u, night)


def _scroll(t, speed, span, offset=0.0):
    return (offset - t * speed) % span - span * 0.25


def scene_stand_room(canvas, w, h, t):
    _room_static(canvas, w, h)
    dog_standing(canvas, w * 0.50, h * 0.78, t, s=1.8 * _u(w))


def scene_stand_park(canvas, w, h, t):
    u = _u(w)
    park(canvas, w, h)
    tree(canvas, w * 0.15, h * 0.47, h * 0.66, 172 * u, u)
    tree(canvas, w * 0.86, h * 0.50, h * 0.66, 140 * u, u)
    dog_standing(canvas, w * 0.46, h * 0.80, t, s=1.8 * u, shade=LEAF_DARK)


def scene_sleep(canvas, w, h, t):
    _room_static(canvas, w, h)
    dog_sleeping(canvas, w * 0.52, h * 0.745, t, s=1.9 * _u(w))


def scene_sleep_night(canvas, w, h, t):
    _room_static(canvas, w, h, night=True)
    dog_sleeping(canvas, w * 0.52, h * 0.745, t, s=1.9 * _u(w),
                 shade=NIGHT_SHADOW)


def scene_bed(canvas, w, h, t):
    u = _u(w)
    _room_static(canvas, w, h)
    dog_bed(canvas, w * 0.50, h * 0.70 + 250 * u, u, False)
    dog_sleeping(canvas, w * 0.50, h * 0.70 + 170 * u, t, s=1.75 * u)


def scene_snore(canvas, w, h, t):
    u = _u(w)
    _room_static(canvas, w, h, night=True)
    dog_bed(canvas, w * 0.50, h * 0.70 + 250 * u, u, True)
    dog_sleeping(canvas, w * 0.50, h * 0.70 + 170 * u, t, s=1.75 * u,
                 shade=NIGHT_SHADOW, snore=True)


def _sprint(canvas, w, h, t, night=False):
    u = _u(w)
    room(canvas, w, h, night)
    floor_y = h * 0.70
    span = w + 980 * u
    for i in range(2):
        sofa(canvas, _scroll(t, 780 * u, span, i * span * 0.5), floor_y, u, night)
    for i in range(2):
        window(canvas, _scroll(t, 780 * u, span, i * span * 0.5 + 460 * u),
               h, u, night)
    sl = fill((100, 100, 128) if night else FLOOR_DARK)
    for k in range(7):
        ly = h * 0.46 + k * 52 * u
        off = (t * 2500 * u + k * 210 * u) % (w + 480 * u) - 480 * u
        canvas.drawRect(skia.Rect.MakeLTRB(off, ly, off + 210 * u, ly + 7 * u), sl)
    dog_running(canvas, w * 0.46, h * 0.765, t, s=1.75 * u,
                shade=NIGHT_SHADOW if night else FLOOR_DARK)


def scene_sprint_room(canvas, w, h, t):
    _sprint(canvas, w, h, t)


def scene_sprint_night(canvas, w, h, t):
    _sprint(canvas, w, h, t, night=True)


def scene_sprint_park(canvas, w, h, t):
    u = _u(w)
    park(canvas, w, h)
    span = w + 980 * u
    for i in range(4):
        tree(canvas, _scroll(t, 680 * u, span, i * span * 0.25), h * 0.50,
             h * 0.66, (150 + (i % 2) * 46) * u, u)
    dog_running(canvas, w * 0.46, h * 0.785, t, s=1.75 * u, shade=LEAF_DARK)


def _push_in(canvas, factor: float, fx: float, fy: float,
             tx: float, ty: float) -> None:
    """Scale the scene and land the focal point at (tx, ty).

    Scaling about the focal point alone keeps it wherever it already was, so
    the head stayed near the floor with a wall of empty space above it. This
    maps it to where the shot wants it.
    """
    canvas.translate(tx, ty)
    canvas.scale(factor, factor)
    canvas.translate(-fx, -fy)


def scene_closeup(canvas, w, h, t):
    """A close-up is the camera moving in, not a head drawn on its own.

    Drawing just the head over a wide room left it hanging in mid-air with no
    neck or body under it. Pushing the camera into the whole dog keeps the
    chest and shoulder in frame and lets the body run out of the bottom,
    which is what a real close-up looks like.
    """
    u = _u(w)
    x, y = w * 0.42, h * 0.92
    s = 1.8 * u
    canvas.save()
    _push_in(canvas, 1.75, x + 108 * s, y - 118 * s, w * 0.46, h * 0.52)
    _room_static(canvas, w, h)
    dog_standing(canvas, x, y, t, s=s)
    canvas.restore()


def scene_closeup_sleep(canvas, w, h, t):
    u = _u(w)
    x, y = w * 0.46, h * 0.88
    s = 1.75 * u
    canvas.save()
    _push_in(canvas, 1.70, x + 58 * s, y - 36 * s, w * 0.48, h * 0.56)
    _room_static(canvas, w, h, night=True)
    dog_sleeping(canvas, x, y, t, s=s, shade=NIGHT_SHADOW)
    canvas.restore()


def scene_bark(canvas, w, h, t):
    _room_static(canvas, w, h)
    dog_barking(canvas, w * 0.42, h * 0.78, t, s=1.8 * _u(w))


def scene_playbow(canvas, w, h, t):
    _room_static(canvas, w, h)
    dog_playbow(canvas, w * 0.42, h * 0.78, t, s=1.8 * _u(w))


def scene_sniff(canvas, w, h, t):
    _room_static(canvas, w, h)
    dog_sniffing(canvas, w * 0.42, h * 0.78, t, s=1.8 * _u(w))


def scene_sniff_park(canvas, w, h, t):
    u = _u(w)
    park(canvas, w, h)
    tree(canvas, w * 0.15, h * 0.47, h * 0.66, 172 * u, u)
    tree(canvas, w * 0.86, h * 0.50, h * 0.66, 140 * u, u)
    dog_sniffing(canvas, w * 0.44, h * 0.80, t, s=1.8 * u, shade=LEAF_DARK)


def scene_vet(canvas, w, h, t):
    u = _u(w)
    _room_static(canvas, w, h)
    floor_y = h * 0.70
    _rr(canvas, w * 0.04, floor_y + 20 * u, w * 0.96, floor_y + 210 * u, 28 * u,
        fill((230, 236, 238)))
    _rr(canvas, w * 0.04, floor_y + 20 * u, w * 0.96, floor_y + 210 * u, 28 * u,
        stroke(INK, 6 * u))
    cx, cy, arm = w * 0.40, h * 0.22, 96 * u
    for l, tt, r, b in ((cx - arm / 3, cy - arm, cx + arm / 3, cy + arm),
                        (cx - arm, cy - arm / 3, cx + arm, cy + arm / 3)):
        _rr(canvas, l, tt, r, b, 14 * u, fill(ACCENT))
        _rr(canvas, l, tt, r, b, 14 * u, stroke(INK, 6 * u))
    dog_standing(canvas, w * 0.44, floor_y - 68 * u, t, s=1.75 * u,
                 shade=(200, 208, 212))


def _mirror_scene(canvas, w, h, t, pose):
    """The dog, and the same pose reflected about the glass.

    Skia clips natively, so the reflection is drawn straight into a clipped
    layer with a mirror transform instead of being rendered to a separate
    image, flipped and pasted.
    """
    u = _u(w)
    _room_static(canvas, w, h)
    mx0, my0 = w * 0.42, h * 0.27
    mx1, my1 = w * 0.98, h * 0.70 + 130 * u
    _rr(canvas, mx0 - 22 * u, my0 - 22 * u, mx1 + 22 * u, my1 + 22 * u, 24 * u,
        fill(FRAME))
    _rr(canvas, mx0 - 22 * u, my0 - 22 * u, mx1 + 22 * u, my1 + 22 * u, 24 * u,
        stroke(INK, 6 * u))
    glass = skia.Rect.MakeLTRB(mx0, my0, mx1, my1)
    _rr(canvas, mx0, my0, mx1, my1, 14 * u,
        grad((mx0, my0), (mx0, my1), GLASS_TOP, GLASS_BOT))

    px, py = w * 0.20, h * 0.70 + 60 * u
    canvas.save()
    canvas.clipRect(glass)
    canvas.translate(2 * mx0, 0)
    canvas.scale(-1, 1)
    pose(canvas, px, py, t, s=1.55 * u, shade=(188, 206, 214))
    canvas.restore()

    _rr(canvas, mx0, my0, mx1, my1, 14 * u, stroke(INK, 6 * u))
    pose(canvas, px, py, t, s=1.55 * u)


def scene_mirror(canvas, w, h, t):
    _mirror_scene(canvas, w, h, t, dog_barking)


def scene_mirror_bow(canvas, w, h, t):
    _mirror_scene(canvas, w, h, t, dog_playbow)


SCENES = {
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
    "mirror": scene_mirror,
    "mirror_bow": scene_mirror_bow,
    "bark": scene_bark,
    "playbow": scene_playbow,
    "sniff": scene_sniff,
    "sniff_park": scene_sniff_park,
    "vet": scene_vet,
}


def render_clip(query: str, seconds: float, cfg: dict[str, Any],
                dest) -> Any:
    """Draw `seconds` of animation for `query` and encode it to `dest`.

    Scene selection is shared with the Pillow backend so both read the same
    `visual_queries`. No supersampling: Skia anti-aliases natively.
    """
    import subprocess
    from pathlib import Path
    from . import toon

    dest = Path(dest)
    w = int(cfg["video"]["width"])
    h = int(cfg["video"]["height"])
    fps = int(cfg["video"]["fps"])
    use_species(toon.species_for(query))
    name = toon.scene_for(query)
    scene = SCENES.get(name) or SCENES["stand_room"]

    frames = dest.parent / f"_{dest.stem}_frames"
    if frames.exists():
        for old in frames.glob("*.png"):
            old.unlink()
    frames.mkdir(parents=True, exist_ok=True)

    text = str(cfg.get("_shot_text") or "")
    index = int(cfg.get("_shot_index") or 0)
    clock = float(cfg.get("_toon_offset") or 0.0)
    cam_phase = float(cfg.get("_toon_cam") or 0.0)

    surface = skia.Surface(w, h)
    total = max(2, int(round(seconds * fps)))
    for i in range(total):
        progress = i / max(1, total - 1)
        canvas = surface.getCanvas()
        canvas.clear(col(PAPER))
        canvas.save()
        camera(canvas, w, h, progress, index, cam_phase)
        scene(canvas, w, h, clock + i / fps)
        canvas.restore()
        banner(canvas, w, h, text, progress)
        surface.makeImageSnapshot().save(str(frames / f"{i:04d}.png"), skia.kPNG)

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


# --- motion and typography --------------------------------------------------

VI_FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
_TYPEFACE: Any = None


def _typeface():
    global _TYPEFACE
    if _TYPEFACE is None:
        _TYPEFACE = skia.Typeface.MakeFromFile(VI_FONT) or skia.Typeface('')
    return _TYPEFACE


def ease_out(p: float) -> float:
    """Decelerating. Motion that starts fast and settles reads as deliberate."""
    return 1 - (1 - p) ** 3


def ease_in_out(p: float) -> float:
    return 3 * p * p - 2 * p * p * p


def smooth(p: float) -> float:
    """Manim's default rate function.

    The cubic ease above still has a non-zero second derivative at its ends,
    so a move visibly "arrives". This quintic is flat in both the first and
    second derivative at 0 and 1, which is why motion eased with it seems to
    start and stop without a seam. Worth borrowing wholesale.
    """
    return p * p * p * (10 - 15 * p + 6 * p * p)


def there_and_back(p: float) -> float:
    """Out and back within one cycle, eased at both ends."""
    return smooth(2 * p) if p < 0.5 else smooth(2 * (1 - p))


def lag(phase: float, amount: float) -> float:
    """Follow-through: a part that trails the body rather than moving with it.

    Ears and tails are not rigidly attached. Driving them off the same phase
    as the body is what makes procedural animation look mechanical; delaying
    them by a fraction of a cycle is most of what fixes it.
    """
    return phase - amount


def camera(canvas, w: int, h: int, progress: float, index: int,
           cam_phase: float = 0.0) -> None:
    """A slow push or pull across the shot.

    A held camera on a flat illustration looks like a slide, not a shot. The
    direction alternates by shot so consecutive cuts do not drift the same
    way, and the move is eased so it never reads as a constant zoom.
    """
    span = 0.055
    e = smooth(progress)
    phase = (cam_phase + 0.5 * e) % 1.0
    zoom = 1.0 + span * there_and_back(phase)
    drift = (14 if index % 4 < 2 else -14) * e
    canvas.translate(w / 2 + drift, h / 2)
    canvas.scale(zoom, zoom)
    canvas.translate(-w / 2, -h / 2)


def _wrap(text: str, font, max_w: float) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if font.measureText(trial) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def banner(canvas, w: int, h: int, text: str, progress: float) -> None:
    """Vietnamese caption card in the empty upper third.

    The drawn style leaves the top of a 9:16 frame empty, and a short line of
    type is what that space is for. Drawn after the camera move so the type
    stays pinned while the picture drifts behind it.
    """
    if not text:
        return
    u = _u(w)
    size = 78 * u
    font = skia.Font(_typeface(), size)
    lines = _wrap(text.upper(), font, w * 0.82)
    line_h = size * 1.24

    pop = ease_out(min(1.0, progress / 0.12)) if progress < 0.12 else 1.0
    top = h * 0.085
    block_h = line_h * len(lines) + 46 * u
    block_w = max(font.measureText(l) for l in lines) + 84 * u

    canvas.save()
    canvas.translate(w / 2, top + block_h / 2)
    canvas.scale(0.86 + 0.14 * pop, 0.86 + 0.14 * pop)
    canvas.translate(-w / 2, -(top + block_h / 2))

    sh = fill(INK, 60)
    sh.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 14 * u))
    _rr(canvas, (w - block_w) / 2, top + 10 * u, (w + block_w) / 2,
        top + block_h + 10 * u, 26 * u, sh)
    _rr(canvas, (w - block_w) / 2, top, (w + block_w) / 2, top + block_h,
        26 * u, fill(PAPER))
    _rr(canvas, (w - block_w) / 2, top, (w + block_w) / 2, top + block_h,
        26 * u, stroke(INK, 6 * u))

    y = top + 34 * u + size * 0.78
    for line in lines:
        x = (w - font.measureText(line)) / 2
        blob = skia.TextBlob(line, font)
        canvas.drawTextBlob(blob, x, y, stroke(INK, 11 * u))
        canvas.drawTextBlob(blob, x, y, fill(ACCENT))
        y += line_h
    canvas.restore()
