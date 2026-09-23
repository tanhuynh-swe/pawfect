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

# The room palette. The first pass was a single warm ramp - tan wall, tan
# floor, brick sofa, terracotta pot - which is the palette of a 1970s
# children's book, and on a grey dog it also left the character sharing a hue
# with everything behind it. This one is the scheme a room gets photographed
# in now: warm off-white walls over a sage dado, pale oak boards, muted sage
# upholstery, and terracotta and ochre kept back as accents so they read as
# accents. It also separates the animal from its background, which the old
# one did not.
PAPER = (247, 243, 236)
WALL_TOP = (244, 240, 233)
WALL_BOT = (232, 227, 218)
DADO = (208, 218, 206)          # painted lower wall, the one green plane
DADO_DARK = (186, 200, 187)
RAIL = (250, 248, 243)          # dado rail and skirting, near-white trim
FLOOR = (233, 215, 190)
FLOOR_DARK = (205, 183, 154)
FLOOR_LINE = (220, 200, 172)
RUG = (172, 198, 192)
RUG_DARK = (144, 174, 168)
RUG_LINE = (240, 236, 227)
FUR_LIT = (228, 172, 116)
FUR = (208, 148, 92)
FUR_SHADE = (176, 116, 68)
FUR_BELLY = (240, 204, 160)
INK = (50, 44, 42)
ACCENT = (214, 106, 78)         # terracotta: collar, cushion, the vet cross
OCHRE = (226, 166, 96)
LEAF = (128, 168, 126)
LEAF_DARK = (86, 128, 98)
SKY_TOP = (166, 208, 228)
SKY_BOT = (228, 242, 240)
SOFA = (150, 174, 162)
SOFA_DARK = (120, 146, 135)
FRAME = (62, 58, 56)            # thin charcoal, not the old wide wood
MAT = (250, 247, 241)
POT = (234, 228, 216)           # speckled ceramic
POT_DARK = (208, 200, 186)
CARD = (218, 188, 146)
CARD_DARK = (190, 156, 114)
GLASS_TOP = (220, 238, 246)
GLASS_BOT = (194, 222, 234)
NIGHT_TOP = (42, 50, 78)
NIGHT_BOT = (74, 82, 112)
NIGHT_FLOOR = (62, 64, 90)
LAMP = (252, 224, 158)
BARK = (146, 118, 96)           # trees had been painted in a coat colour
TONGUE = (226, 128, 134)
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
# The channel's own dog is a grey village dog: black pricked ears, a dark
# crown and saddle, a white muzzle and chest, and a tail that curls over its
# back. The tan floppy-eared puppy this module started with is kept as its
# own character rather than replaced, so either can be used.
#
# The coat swaps by rebinding the FUR_* globals. Every drawing call reads
# them at call time and nothing captures them in a default argument, so one
# table recolours the whole character without touching the 39 draw sites.
TAN_COAT = ((228, 172, 116), (208, 148, 92), (176, 116, 68), (240, 204, 160))
# Warmed from the first pass. A sable coat photographs as grey but it is not
# neutral grey: every value has more red in it than blue, and mixed flat the
# animal came out the colour of the concrete it was standing on.
GREY_COAT = ((206, 199, 188), (178, 171, 160), (144, 137, 128), (238, 234, 226))
COATS = {"dog": TAN_COAT, "cat": TAN_COAT, "greydog": GREY_COAT}

# The markings, sampled off the reference photograph and lifted: the originals
# sit around (120, 116, 110) and read as mud once compressed to a phone feed.
# Ears, crown, the bridge of the muzzle, saddle and tail are the dark; the
# muzzle, chin and chest are the white; and the tan is the part that was
# missing - the eyebrow spots and the ring of lighter fur around each eye,
# which is what lets a dark eye sit on a dark mask and still read.
MARK_DARK = (74, 70, 68)
MARK_SOFT = (104, 99, 95)
MARK_WHITE = (245, 242, 236)
MARK_TAN = (190, 158, 112)
MARK_CREAM = (232, 225, 210)
IRIS = (104, 72, 46)

CHARACTERS = ("dog", "cat", "greydog")
SPECIES = "dog"


def use_species(name: str) -> None:
    global SPECIES, FUR_LIT, FUR, FUR_SHADE, FUR_BELLY
    SPECIES = name if name in CHARACTERS else "dog"
    FUR_LIT, FUR, FUR_SHADE, FUR_BELLY = COATS[SPECIES]


def is_cat() -> bool:
    return SPECIES == "cat"


def is_greydog() -> bool:
    """The channel's own dog. A dog everywhere except ears, tail and markings."""
    return SPECIES == "greydog"


def prick_ears() -> bool:
    """Upright and triangular, as opposed to hanging."""
    return SPECIES in ("cat", "greydog")


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
    if is_greydog():
        # Broader-based and larger than the cat's: a dog's prick ear reads as
        # a wide triangle, and these are the single clearest marker of which
        # dog this is.
        top = hy - 130 * s - lift * s * 0.4
        p.moveTo(hx - 56 * s, top + 96 * s)
        p.cubicTo(hx - 56 * s, top + 30 * s, hx - 44 * s, top + 2 * s,
                  hx - 16 * s, top + 24 * s)
        p.cubicTo(hx + 4 * s, top + 42 * s, hx - 2 * s, top + 72 * s,
                  hx - 10 * s, top + 100 * s)
        p.close()
        return p
    if is_cat():
        # Tall and vertical. The first pass drew a small ear angled back,
        # which on a long muzzle is a rodent's silhouette. The base sits at
        # hy - 24s so it is buried under the skull rather than perched on it.
        top = hy - 112 * s - lift * s * 0.4
        p.moveTo(hx - 58 * s, top + 88 * s)
        p.cubicTo(hx - 58 * s, top + 26 * s, hx - 46 * s, top + 2 * s,
                  hx - 20 * s, top + 22 * s)
        p.cubicTo(hx - 2 * s, top + 38 * s, hx - 6 * s, top + 66 * s,
                  hx - 14 * s, top + 92 * s)
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
    if is_greydog():
        # Between the two: a broader crown and a shorter, blunter muzzle than
        # the tan dog's, which came out longer and more pointed than the
        # reference animal's. Not the cat's flat face either - it still has a
        # muzzle, it is just a stubbier one.
        p.moveTo(hx - 64 * s, hy - 14 * s)
        p.cubicTo(hx - 64 * s, hy - 70 * s, hx + 14 * s, hy - 80 * s,
                  hx + 52 * s, hy - 50 * s)          # broad crown
        p.cubicTo(hx + 72 * s, hy - 30 * s, hx + 74 * s, hy + 6 * s,
                  hx + 86 * s, hy + 12 * s)          # into the muzzle
        p.cubicTo(hx + 96 * s, hy + 17 * s, hx + 96 * s, hy + 42 * s,
                  hx + 78 * s, hy + 47 * s)          # blunt front
        p.cubicTo(hx + 52 * s, hy + 53 * s, hx + 46 * s, hy + 62 * s,
                  hx + 6 * s, hy + 62 * s)           # jaw
        p.cubicTo(hx - 38 * s, hy + 62 * s, hx - 64 * s, hy + 26 * s,
                  hx - 64 * s, hy - 14 * s)
        p.close()
        return p
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


def _ear_inner(hx: float, hy: float, s: float, lift: float,
               far: bool) -> skia.Path:
    """The pink inside an ear, inset from its outline.

    Scaling the outer path about the ear's base put this where the skull
    covers it - the whole tint came to about fifty visible pixels. An inset
    path of its own sits up in the tip, which is the part that clears the
    head, and is what makes an ear read as an ear rather than a brown horn.
    """
    p = skia.Path()
    if is_greydog():
        if far:
            top = hy - 124 * s - lift * s * 0.4
            p.moveTo(hx + 38 * s, top + 86 * s)
            p.cubicTo(hx + 40 * s, top + 38 * s, hx + 54 * s, top + 20 * s,
                      hx + 68 * s, top + 40 * s)
            p.cubicTo(hx + 78 * s, top + 54 * s, hx + 72 * s, top + 70 * s,
                      hx + 66 * s, top + 88 * s)
        else:
            top = hy - 130 * s - lift * s * 0.4
            p.moveTo(hx - 44 * s, top + 92 * s)
            p.cubicTo(hx - 44 * s, top + 42 * s, hx - 34 * s, top + 22 * s,
                      hx - 15 * s, top + 42 * s)
            p.cubicTo(hx - 4 * s, top + 56 * s, hx - 8 * s, top + 72 * s,
                      hx - 14 * s, top + 94 * s)
        p.close()
        return p
    if far:
        top = hy - 108 * s - lift * s * 0.4
        p.moveTo(hx + 30 * s, top + 74 * s)
        p.cubicTo(hx + 32 * s, top + 32 * s, hx + 44 * s, top + 16 * s,
                  hx + 58 * s, top + 34 * s)
        p.cubicTo(hx + 68 * s, top + 46 * s, hx + 62 * s, top + 60 * s,
                  hx + 56 * s, top + 76 * s)
    else:
        top = hy - 112 * s - lift * s * 0.4
        p.moveTo(hx - 46 * s, top + 80 * s)
        p.cubicTo(hx - 46 * s, top + 34 * s, hx - 38 * s, top + 18 * s,
                  hx - 21 * s, top + 36 * s)
        p.cubicTo(hx - 10 * s, top + 48 * s, hx - 14 * s, top + 62 * s,
                  hx - 20 * s, top + 82 * s)
    p.close()
    return p


def _second_ear(canvas, hx: float, hy: float, s: float, lift: float) -> None:
    """A prick-eared head reads front-on enough that the far ear should show."""
    if not prick_ears():
        return
    if is_greydog():
        top = hy - 124 * s - lift * s * 0.4
        p = skia.Path()
        p.moveTo(hx + 26 * s, top + 92 * s)
        p.cubicTo(hx + 28 * s, top + 26 * s, hx + 48 * s, top + 2 * s,
                  hx + 72 * s, top + 26 * s)
        p.cubicTo(hx + 88 * s, top + 44 * s, hx + 82 * s, top + 70 * s,
                  hx + 72 * s, top + 96 * s)
        p.close()
        canvas.drawPath(p, fill(MARK_DARK))
        canvas.drawPath(p, stroke(INK, ink(s)))
        canvas.drawPath(_ear_inner(hx, hy, s, lift, far=True), fill(MARK_TAN))
        return
    top = hy - 108 * s - lift * s * 0.4
    p = skia.Path()
    p.moveTo(hx + 20 * s, top + 82 * s)
    p.cubicTo(hx + 22 * s, top + 22 * s, hx + 40 * s, top + 0 * s,
              hx + 62 * s, top + 22 * s)
    p.cubicTo(hx + 78 * s, top + 38 * s, hx + 72 * s, top + 62 * s,
              hx + 62 * s, top + 86 * s)
    p.close()
    canvas.drawPath(p, fill(FUR_SHADE))
    canvas.drawPath(p, stroke(INK, ink(s)))
    canvas.drawPath(_ear_inner(hx, hy, s, lift, far=True), fill(BLUSH, 190))


def _mask(canvas, hx: float, hy: float, s: float, skull: skia.Path) -> None:
    """The dark crown and nose bridge, the tan eye patch, the cream cheek.

    The first pass of this dog wore only a faint cap on its crown, on the
    grounds that a dark eye on a dark mask stops reading. The photograph
    solves that itself: the mask runs right down the bridge of the muzzle,
    and each eye sits in a ring of tan fur with a spot of the same tan over
    it. Those two marks are the animal's face - without them it is a grey
    dog, with them it is this grey dog - and they are what keeps the eye
    legible against the dark.

    Every mark is soft-edged and clipped to the skull. Sharp-edged they are
    stickers on a head; fur does not change colour along a line, and the
    blur is the whole difference between a marking and a paint job.
    """
    canvas.save()
    canvas.clipPath(skull, doAntiAlias=True)

    dark = skia.Path()
    dark.moveTo(hx - 72 * s, hy - 6 * s)
    dark.cubicTo(hx - 46 * s, hy - 26 * s, hx - 6 * s, hy - 28 * s,
                 hx + 30 * s, hy - 22 * s)          # crown, down to the brow
    dark.cubicTo(hx + 46 * s, hy - 14 * s, hx + 54 * s, hy - 2 * s,
                 hx + 68 * s, hy + 10 * s)          # down the bridge
    dark.cubicTo(hx + 80 * s, hy + 20 * s, hx + 86 * s, hy + 30 * s,
                 hx + 106 * s, hy + 40 * s)         # onto the nose
    dark.lineTo(hx + 130 * s, hy - 150 * s)
    dark.lineTo(hx - 90 * s, hy - 150 * s)
    dark.close()
    mp = fill(MARK_DARK, 245)
    mp.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 2.6 * s))
    canvas.drawPath(dark, mp)

    # cheek and jaw: the cream that carries on down into the chest
    ck = fill(MARK_CREAM)
    ck.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 7 * s))
    canvas.drawOval(skia.Rect.MakeLTRB(hx - 40 * s, hy + 12 * s,
                                       hx + 46 * s, hy + 68 * s), ck)

    # the tan ring the eye sits in, and the spot above it
    tan = fill(MARK_TAN)
    tan.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 4 * s))
    canvas.drawOval(skia.Rect.MakeLTRB(hx + 12 * s, hy - 30 * s,
                                       hx + 56 * s, hy + 20 * s), tan)
    spot = fill((214, 184, 134))
    spot.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 2.5 * s))
    canvas.drawOval(skia.Rect.MakeLTRB(hx + 14 * s, hy - 62 * s,
                                       hx + 50 * s, hy - 44 * s), spot)
    canvas.restore()


def head(canvas, hx: float, hy: float, s: float, lift: float = 0.0,
         asleep: bool = False, blink: float = 0.0, mouth: float = 0.0,
         pant: float = 0.0) -> None:
    lw = ink(s)
    ip = stroke(INK, lw)

    # A cat's head here is round and shows both ears, so it reads front-on,
    # and every feature has to sit symmetrically about the skull's own
    # centre. Each one had inherited the dog's forward bias independently:
    # the skull centres on hx + 11, but the eyes averaged hx - 4, the muzzle
    # pad sat at hx + 39, the nose at hx + 48 and the open mouth at hx + 59.
    # On a face that reads front-on that is not perspective, it is a squint.
    cx = hx + 11 * s

    _second_ear(canvas, hx, hy, s, lift)
    ear = _ear_path(hx, hy, s, lift)
    canvas.drawPath(ear, fill(MARK_DARK if is_greydog() else FUR_SHADE))
    canvas.drawPath(ear, ip)
    if prick_ears():
        canvas.drawPath(_ear_inner(hx, hy, s, lift, far=False),
                        fill(MARK_TAN if is_greydog() else BLUSH,
                             255 if is_greydog() else 190))

    skull = _skull_path(hx, hy, s)
    canvas.drawPath(skull, grad((hx, hy - 70 * s), (hx, hy + 60 * s),
                                FUR_LIT, FUR))

    # muzzle, lighter, tucked under the skull curve
    if is_cat():
        canvas.drawOval(skia.Rect.MakeLTRB(cx - 31 * s, hy + 14 * s,
                                           cx + 31 * s, hy + 58 * s),
                        fill(FUR_BELLY))
        canvas.drawOval(skia.Rect.MakeLTRB(cx - 22 * s, hy + 6 * s,
                                           cx + 22 * s, hy + 42 * s),
                        fill(FUR_BELLY))
    muzzle = skia.Path()
    if is_greydog():
        muzzle.moveTo(hx + 28 * s, hy + 20 * s)
        muzzle.cubicTo(hx + 52 * s, hy + 12 * s, hx + 78 * s, hy + 18 * s,
                       hx + 82 * s, hy + 34 * s)
        muzzle.cubicTo(hx + 84 * s, hy + 52 * s, hx + 52 * s, hy + 56 * s,
                       hx + 30 * s, hy + 48 * s)
    else:
        muzzle.moveTo(hx + 36 * s, hy + 8 * s)
        muzzle.cubicTo(hx + 62 * s, hy - 2 * s, hx + 90 * s, hy + 4 * s,
                       hx + 92 * s, hy + 26 * s)
        muzzle.cubicTo(hx + 94 * s, hy + 48 * s, hx + 58 * s, hy + 52 * s,
                       hx + 38 * s, hy + 42 * s)
    muzzle.close()
    if is_greydog():
        mz = fill(MARK_WHITE)
        mz.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 4 * s))
        canvas.drawPath(muzzle, mz)
    elif not is_cat():
        canvas.drawPath(muzzle, fill(FUR_BELLY))
    if is_greydog():
        _mask(canvas, hx, hy, s, skull)
    canvas.drawPath(skull, ip)

    # cheek blush, the cheapest cuteness cue there is
    bl = fill(BLUSH, 95)
    bl.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 11 * s))
    if is_cat():
        for side in (-1, 1):
            canvas.drawOval(
                skia.Rect.MakeLTRB(cx + side * 30 * s - 21 * s, hy + 12 * s,
                                   cx + side * 30 * s + 21 * s, hy + 36 * s), bl)
    else:
        canvas.drawOval(skia.Rect.MakeLTRB(hx + 4 * s, hy + 14 * s,
                                           hx + 46 * s, hy + 38 * s), bl)

    if mouth > 0.02:
        # A dog's jaw opens along the muzzle, so its mouth is drawn sideways.
        # A cat has no muzzle to open along and is seen front-on, so its
        # mouth is a symmetric shape that drops straight down from the nose.
        if is_cat():
            gap = 40 * s * mouth
            m = skia.Path()
            m.moveTo(cx - 23 * s, hy + 26 * s)
            m.quadTo(cx, hy + 21 * s, cx + 23 * s, hy + 26 * s)
            m.quadTo(cx + 15 * s, hy + 31 * s + gap, cx, hy + 33 * s + gap)
            m.quadTo(cx - 15 * s, hy + 31 * s + gap, cx - 23 * s, hy + 26 * s)
            m.close()
            canvas.drawPath(m, fill(MOUTH))
            canvas.drawPath(m, stroke(INK, ink(s, 0.7)))
            t = skia.Path()
            t.addOval(skia.Rect.MakeLTRB(cx - 13 * s, hy + 27 * s + gap * 0.40,
                                         cx + 13 * s, hy + 31 * s + gap * 1.0))
            canvas.save()
            canvas.clipPath(m, doAntiAlias=True)
            canvas.drawPath(t, fill(TONGUE))
            canvas.restore()
        else:
            gap = 56 * s * mouth
            x0, x1 = (38, 92) if is_greydog() else (44, 104)
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
            # Clipped to the mouth. Drawn free the tongue is an oval hanging
            # off the jaw with no lip around it, which is a tongue somebody
            # dropped rather than one inside a head.
            t = skia.Path()
            t.addOval(skia.Rect.MakeLTRB(hx + (x0 + 14) * s, hy + 28 * s + gap * 0.35,
                                         hx + (x1 - 8) * s, hy + 28 * s + gap * 0.95))
            canvas.save()
            canvas.clipPath(m, doAntiAlias=True)
            canvas.drawPath(t, fill(TONGUE))
            canvas.restore()

    if pant > 0.02 and not is_cat() and mouth <= 0.02:
        # The reference dog is almost never photographed with its mouth shut,
        # and a tongue over the lower lip is the whole difference between a
        # dog standing there and a dog enjoying standing there. Drawn as a
        # hang rather than an open jaw: the jaw only opens to bark.
        tw = 13 * s
        tx, ty = hx + 54 * s, hy + 44 * s
        drop = (26 + 8 * pant) * s
        lip = skia.Path()
        lip.moveTo(hx + 36 * s, hy + 40 * s)
        lip.quadTo(hx + 58 * s, hy + 48 * s, hx + 78 * s, hy + 38 * s)
        canvas.drawPath(lip, stroke(INK, ink(s, 0.62)))
        tongue = skia.Path()
        tongue.moveTo(tx - tw, ty - 4 * s)
        tongue.cubicTo(tx - tw - 5 * s, ty + drop * 0.72,
                       tx - tw * 0.4, ty + drop, tx + tw * 0.2, ty + drop)
        tongue.cubicTo(tx + tw * 0.9, ty + drop, tx + tw + 6 * s,
                       ty + drop * 0.6, tx + tw, ty - 6 * s)
        tongue.close()
        canvas.drawPath(tongue, fill(TONGUE))
        canvas.drawPath(tongue, stroke(INK, ink(s, 0.6)))
        crease = skia.Path()
        crease.moveTo(tx - 1 * s, ty + 4 * s)
        crease.lineTo(tx - 1 * s, ty + drop * 0.62)
        canvas.drawPath(crease, stroke((198, 98, 108), ink(s, 0.4)))

    # nose
    if is_cat():
        nx, ny = cx, hy + 20 * s
        tri = skia.Path()
        tri.moveTo(nx - 13 * s, ny - 5 * s)
        tri.quadTo(nx, ny - 11 * s, nx + 13 * s, ny - 5 * s)
        tri.quadTo(nx + 11 * s, ny + 11 * s, nx, ny + 13 * s)
        tri.quadTo(nx - 11 * s, ny + 11 * s, nx - 13 * s, ny - 5 * s)
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
        # Drawn as a real curve and thinner: three straight lines of even
        # weight read as wire, not hair.
        # (start offset, how far it fans by the tip). These have to diverge:
        # spreads that cancelled the start offset put all three tips on the
        # same point, which drew one spike instead of three hairs.
        for side in (1, -1):
            for dy, spread in ((2, -7), (10, -2), (18, 5)):
                wk = skia.Path()
                wk.moveTo(nx + side * 19 * s, ny + dy * s)
                wk.cubicTo(nx + side * 29 * s, ny + (dy + spread * 0.12) * s,
                           nx + side * 38 * s, ny + (dy + spread * 0.5) * s,
                           nx + side * 48 * s, ny + (dy + spread) * s)
                canvas.drawPath(wk, stroke(INK, ink(s, 0.44)))
        # Follicle dots. Cheap, and they do more for "cat" than the whiskers.
        for side in (1, -1):
            for ddx, ddy in ((21, -1), (25, 6), (21, 13)):
                canvas.drawOval(
                    skia.Rect.MakeLTRB(nx + side * ddx * s - 2.2 * s,
                                       ny + ddy * s - 2.2 * s,
                                       nx + side * ddx * s + 2.2 * s,
                                       ny + ddy * s + 2.2 * s),
                    fill(INK, 150))
    else:
        n0 = 60 if is_greydog() else 76
        canvas.drawOval(skia.Rect.MakeLTRB(hx + n0 * s, hy + 12 * s,
                                           hx + (n0 + 32) * s, hy + 40 * s),
                        fill(INK))
        canvas.drawOval(skia.Rect.MakeLTRB(hx + (n0 + 8) * s, hy + 17 * s,
                                           hx + (n0 + 15) * s, hy + 23 * s),
                        fill((150, 142, 140), 210))

    # brow and eye. A cat's face is 34 units shorter, so the eye would
    # otherwise sit on the edge of it: everything here shifts back by `ex`.
    ex = hx - 16 * s if is_cat() else hx
    if not (is_cat() or is_greydog()):
        brow = skia.Path()
        brow.moveTo(ex + 6 * s, hy - 40 * s)
        brow.quadTo(ex + 26 * s, hy - 50 * s, ex + 44 * s, hy - 38 * s)
        canvas.drawPath(brow, stroke(INK, ink(s, 0.8)))

    # The far eye, and a brow over it. A dog's long muzzle reads as a true
    # profile and one eye is right; a cat's face is round enough to read
    # front-on, and a single eye on it looks like a mistake rather than a
    # viewing angle. The far one is smaller and set back, which is what
    # turns the same head into a three-quarter view.
    if asleep or blink > 0.995:
        if is_cat():
            for side in (-1, 1):
                _eye_closed(canvas, cx + side * 26 * s, hy - 8 * s, 15 * s, s)
        else:
            _eye_closed(canvas, ex + 37 * s, hy - 8 * s, 16 * s, s)
    else:
        if is_cat():
            for side in (-1, 1):
                _eye(canvas, cx + side * 26 * s, hy - 8 * s, 18 * s, 19 * s, s)
        else:
            _eye(canvas, ex + 37 * s, hy - 8 * s, 19 * s, 20 * s, s)


def _eye_closed(canvas, cx: float, cy: float, rw: float, s: float,
                weight: float = 1.15) -> None:
    """A shut eye: a short, deep, downward curve centred on the open eye.

    Drawn wide and shallow it reads as a crease or a brow rather than a lid,
    which is what two long flat arcs high on the face were doing. Narrow and
    deep, on the same centre the open eye uses, reads as shut and content.
    """
    e = skia.Path()
    e.moveTo(cx - rw, cy - rw * 0.34)
    e.quadTo(cx, cy + rw * 0.76, cx + rw, cy - rw * 0.34)
    canvas.drawPath(e, stroke(INK, ink(s, weight)))


def _eye(canvas, cx: float, cy: float, rw: float, rh: float,
         s: float, alpha: int = 255) -> None:
    """One eye: a warm iris around a black pupil, lit twice.

    Two lights rather than one is most of what stops a flat disc reading as
    a dead button - the upper one is the light source, the lower a bounce.
    A single flat disc was still a button with lights on it, though. Real
    eyes are two tones, and putting an amber iris inside the pupil costs one
    more oval and is the difference between an eye and a bead - it is also
    what the reference dog's eyes actually are, warm brown right up to a
    dark rim.
    """
    canvas.drawOval(skia.Rect.MakeLTRB(cx - rw, cy - rh, cx + rw, cy + rh),
                    fill(INK, alpha))
    canvas.drawOval(skia.Rect.MakeLTRB(cx - rw * 0.86, cy - rh * 0.86,
                                       cx + rw * 0.86, cy + rh * 0.86),
                    fill(IRIS, alpha))
    canvas.drawOval(skia.Rect.MakeLTRB(cx - rw * 0.46, cy - rh * 0.50,
                                       cx + rw * 0.46, cy + rh * 0.50),
                    fill((28, 22, 20), alpha))
    canvas.drawOval(skia.Rect.MakeLTRB(cx + rw * 0.10, cy - rh * 0.72,
                                       cx + rw * 0.78, cy - rh * 0.10),
                    fill((255, 255, 255), alpha))
    canvas.drawOval(skia.Rect.MakeLTRB(cx - rw * 0.66, cy + rh * 0.22,
                                       cx - rw * 0.18, cy + rh * 0.70),
                    fill((255, 255, 255), min(alpha, 215)))


def collar(canvas, x: float, y: float, s: float) -> None:
    r = skia.Rect.MakeLTRB(x - 22 * s, y - 16 * s, x + 24 * s, y + 12 * s)
    rr = skia.RRect.MakeRectXY(r, 9 * s, 9 * s)
    canvas.drawRRect(rr, fill(ACCENT))
    canvas.drawRRect(rr, stroke(INK, ink(s, 0.75)))
    canvas.drawOval(skia.Rect.MakeLTRB(x - 2 * s, y + 8 * s, x + 20 * s,
                                       y + 30 * s), fill(LAMP))
    canvas.drawOval(skia.Rect.MakeLTRB(x - 2 * s, y + 8 * s, x + 20 * s,
                                       y + 30 * s), stroke(INK, ink(s, 0.6)))


def taper(canvas, pts: list[tuple[float, float]], w0: float, w1: float,
          rgb: tuple[int, int, int], lw: float) -> skia.Path:
    """A limb drawn as a shape that narrows, not a stroke of even width.

    Borrowed from the procedural-character work on GitHub, where tapered
    limbs are the one thing separating a drawn body from a pipe-cleaner
    one: a real leg is thick at the haunch and thin at the ankle, and a
    constant-width stroke cannot say that at any width. The path is sampled
    along a quadratic and offset by the local normal, so one outline covers
    both edges and the ink line follows the taper too.
    """
    (ax, ay), (bx, by), (cx, cy) = pts
    steps = 14
    left: list[tuple[float, float]] = []
    right: list[tuple[float, float]] = []
    for i in range(steps + 1):
        t = i / steps
        mt = 1 - t
        px = mt * mt * ax + 2 * mt * t * bx + t * t * cx
        py = mt * mt * ay + 2 * mt * t * by + t * t * cy
        dx = 2 * mt * (bx - ax) + 2 * t * (cx - bx)
        dy = 2 * mt * (by - ay) + 2 * t * (cy - by)
        n = math.hypot(dx, dy) or 1.0
        hw = (w0 + (w1 - w0) * t) / 2
        left.append((px - dy / n * hw, py + dx / n * hw))
        right.append((px + dy / n * hw, py - dx / n * hw))
    path = skia.Path()
    path.moveTo(*left[0])
    for pt in left[1:]:
        path.lineTo(*pt)
    for pt in reversed(right):
        path.lineTo(*pt)
    path.close()
    canvas.drawPath(path, fill(rgb))
    canvas.drawPath(path, stroke(INK, lw))
    return path


def inner_edge(canvas, path: skia.Path, dx: float, dy: float,
               rgb: tuple[int, int, int], alpha: int, width: float,
               sigma: float) -> None:
    """A light or dark band just inside an outline.

    Cel shading gets its depth from two marks: a rim along the lit edge and
    an occlusion band along the shaded one. Both are the same trick here -
    clip to the shape, then stroke the same shape shifted a little, so only
    the part of the stroke that falls inside the silhouette survives.
    """
    paint = skia.Paint(AntiAlias=True, Color=col(rgb, alpha),
                       Style=skia.Paint.kStroke_Style, StrokeWidth=width,
                       StrokeJoin=skia.Paint.kRound_Join,
                       StrokeCap=skia.Paint.kRound_Cap)
    paint.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, sigma))
    canvas.save()
    canvas.clipPath(path, doAntiAlias=True)
    canvas.translate(dx, dy)
    canvas.drawPath(path, paint)
    canvas.restore()


def leg(canvas, hx: float, hy: float, swing: float, s: float,
        shade: tuple[int, int, int]) -> None:
    knee = (hx + swing * 0.5, hy + 46 * s * LEG)
    paw = (hx + swing, hy + 86 * s * LEG)
    limb = taper(canvas, [(hx, hy), knee, paw], 38 * s, 25 * s, shade, ink(s))
    inner_edge(canvas, limb, -3.5 * s, -2 * s, FUR_LIT, 120, 7 * s, 4 * s)
    py0, py1 = hy + 70 * s * LEG, hy + 100 * s * LEG
    for paint in (fill(shade), stroke(INK, ink(s, 0.75))):
        canvas.drawOval(skia.Rect.MakeLTRB(hx + swing - 20 * s, py0,
                                           hx + swing + 20 * s, py1), paint)
    # Toe lines. A bare oval reads as the end of a tube; two short creases
    # read as a paw, and cost one stroke each.
    for tx in (-6.5, 6.5):
        toe = skia.Path()
        toe.moveTo(hx + swing + tx * s, py1 - (py1 - py0) * 0.52)
        toe.lineTo(hx + swing + tx * s, py1 - (py1 - py0) * 0.10)
        canvas.drawPath(toe, stroke(INK, ink(s, 0.5)))


def tail(canvas, x: float, y: float, s: float, tipx: float, tipy: float) -> None:
    p = skia.Path()
    p.moveTo(x - 88 * s, y - 40 * s)
    if is_greydog():
        # Curls up and forward over the back, and it is dark like the ears.
        # `tipx`/`tipy` still steer it so the wag in each pose still reads,
        # but the curve returns over the spine instead of trailing behind.
        p.cubicTo(x - 136 * s, y - 86 * s, x - 126 * s, y - 152 * s,
                  x - 66 * s, y - 156 * s)
        p.cubicTo(x - 32 * s, y - 158 * s, x - 14 * s, y - 136 * s,
                  x - 20 * s, y - 114 * s + (tipy - y) * 0.06)
        canvas.drawPath(p, stroke(INK, ink(s, 5.6)))
        canvas.drawPath(p, stroke(MARK_DARK, ink(s, 4.2)))
        return
    if is_cat():
        # Longer and higher than a dog's, but just as thick. Drawn thin and
        # hooked it was a mouse's tail, and no amount of ear fixed that.
        p.cubicTo(x - 152 * s, y - 72 * s, x - 176 * s, y - 134 * s,
                  tipx + 8 * s, tipy - 20 * s)
        canvas.drawPath(p, stroke(INK, ink(s, 5.4)))
        canvas.drawPath(p, stroke(FUR_SHADE, ink(s, 4.0)))
        return
    p.quadTo(x - 150 * s, y - 92 * s, tipx, tipy)
    canvas.drawPath(p, stroke(INK, ink(s, 3.6)))
    canvas.drawPath(p, stroke(FUR_SHADE, ink(s, 2.4)))
    for paint in (fill(FUR_BELLY), stroke(INK, ink(s, 0.75))):
        canvas.drawOval(skia.Rect.MakeLTRB(tipx - 21 * s, tipy - 21 * s,
                                           tipx + 21 * s, tipy + 21 * s), paint)


def _saddle(canvas, body: skia.Path, x: float, y: float, s: float,
            lean: float = 0.0) -> None:
    """The dark band down the back, clipped to whatever body encloses it.

    Clipping rather than tracing means one shape serves every pose: the
    saddle cannot leak past a silhouette it was not drawn for.
    """
    if not is_greydog():
        return
    canvas.save()
    canvas.clipPath(body, doAntiAlias=True)
    band = skia.Path()
    band.moveTo(x - 110 * s, y - 48 * s)
    band.cubicTo(x - 60 * s, y - 30 * s, x + 30 * s, y - 40 * s,
                 x + 100 * s, y - 62 * s + lean)
    band.lineTo(x + 120 * s, y - 150 * s + lean)
    band.lineTo(x - 120 * s, y - 150 * s)
    band.close()
    sp = grad((x, y - 110 * s), (x, y - 30 * s), MARK_DARK, MARK_SOFT)
    sp.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 9 * s))
    canvas.drawPath(band, sp)
    # the cream bib, up the chest and under the jaw
    bib = fill(MARK_CREAM)
    bib.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 12 * s))
    canvas.drawOval(skia.Rect.MakeLTRB(x + 44 * s, y - 56 * s + lean,
                                       x + 108 * s, y + 16 * s + lean), bib)
    canvas.restore()


def _draw_body(canvas, x: float, y: float, s: float, lean: float = 0.0) -> None:
    body = _body_path(x, y, s, lean)
    canvas.drawPath(body, grad((x, y - 92 * s), (x, y + 20 * s),
                               FUR_LIT, FUR_SHADE))
    _saddle(canvas, body, x, y, s, lean)
    canvas.drawPath(_belly_path(x, y, s, lean), fill(FUR_BELLY))
    # A lit edge along the back and a shaded one under the belly. Flat fills
    # with one gradient read as paper cut-outs; these two bands are what
    # give the shape a top and an underside.
    inner_edge(canvas, body, -4 * s, -7 * s, (255, 250, 240), 95, 13 * s, 7 * s)
    inner_edge(canvas, body, 3 * s, 9 * s, MARK_DARK if is_greydog()
               else FUR_SHADE, 70, 15 * s, 9 * s)
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
         blink=(math.sin(t * 1.15) + 1) / 2,
         pant=0.5 + 0.5 * math.sin(t * 5.2) if is_greydog() else 0.0)
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
    _saddle(canvas, curl, x, y - 24 * s, s)
    canvas.drawPath(curl, stroke(INK, ink(s)))
    tuck = skia.Path()
    tuck.addOval(skia.Rect.MakeLTRB(x - 86 * s, y - 34 * s, x + 96 * s,
                                    y + 40 * s))
    canvas.drawPath(tuck, fill(FUR_BELLY))
    head(canvas, x + 58 * s, y - 36 * s - breathe, s * 0.92 * HEAD, asleep=True)
    for i in range(3):
        a = t * (1.5 if snore else 0.9) + i * 1.1
        drift = (a % 3.0) / 3.0
        size = (30 + i * 15) * s * (1.25 if snore else 1.0)
        zx = x - 56 * s + math.sin(a * 2) * 20 * s
        # Offset by its own size: a Z is drawn downward from this point, so
        # anchoring the top put the smallest one across the ear.
        zy = y - 132 * s - drift * 250 * s - size
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
         blink=(math.sin(t * 1.5) + 1) / 2,
         pant=0.0 if asleep or not is_greydog()
         else 0.5 + 0.5 * math.sin(t * 5.2))


# --- backgrounds and props --------------------------------------------------

def _u(w: int) -> float:
    return w / 1080.0


def room(canvas, w: int, h: int, night: bool = False) -> None:
    """Wall, dado, skirting, floor. One draw call each, no cached bitmap.

    Drawn with an overscan margin. The close-ups push the camera in around a
    focal point near the floor, which maps the room's own bottom edge to
    about 85% of the frame height and left a bare band of canvas below it -
    15% of the picture, for as long as the shot ran. The canvas clips
    anything off-frame, so no other scene is affected, and the gradients stay
    anchored to the real frame so the colours inside it do not shift.

    The wall is two planes rather than one. A single flat wall behind a flat
    floor is the thing that makes a drawn room read as a backdrop; a painted
    lower wall with a rail on it puts a horizontal line at the character's
    shoulder and gives the room a near and a far surface.
    """
    u = _u(w)
    floor_y = h * 0.70
    dado_y = floor_y - h * 0.205
    over = h * 0.6
    canvas.drawRect(skia.Rect.MakeLTRB(-over, -over, w + over, dado_y),
                    grad((0, 0), (0, dado_y),
                         NIGHT_TOP if night else WALL_TOP,
                         NIGHT_BOT if night else WALL_BOT))
    canvas.drawRect(skia.Rect.MakeLTRB(-over, dado_y, w + over, floor_y),
                    grad((0, dado_y), (0, floor_y),
                         (56, 64, 92) if night else DADO,
                         (46, 54, 80) if night else DADO_DARK))
    # rail and skirting: thin trim, the two lines that sell a painted wall
    canvas.drawRect(skia.Rect.MakeLTRB(-over, dado_y - 9 * u, w + over, dado_y),
                    fill((78, 86, 116) if night else RAIL))
    canvas.drawRect(skia.Rect.MakeLTRB(-over, floor_y - 30 * u, w + over, floor_y),
                    fill((72, 80, 110) if night else RAIL))
    canvas.drawRect(skia.Rect.MakeLTRB(-over, floor_y - 34 * u, w + over,
                                       floor_y - 30 * u),
                    fill((52, 58, 84) if night else (216, 210, 198)))

    canvas.drawRect(skia.Rect.MakeLTRB(-over, floor_y, w + over, h + over),
                    grad((0, floor_y), (0, h),
                         NIGHT_FLOOR if night else FLOOR,
                         (54, 56, 80) if night else FLOOR_DARK))
    # Wide boards: fewer, softer lines than the old close-ruled ones, which
    # at phone size moired into a grey band.
    line = fill((60, 62, 86) if night else FLOOR_LINE)
    yy = floor_y + 104 * u
    while yy < h + over:
        canvas.drawRect(skia.Rect.MakeLTRB(-over, yy, w + over, yy + 4 * u), line)
        yy += 168 * u


def rug(canvas, cx: float, floor_y: float, u: float, night: bool) -> None:
    """A flatweave under the character.

    The floor is one flat plane, so a standing animal has nothing to stand
    on but a colour. A rug gives the pose a footprint, and it is the one
    place a cool tone can sit directly behind a grey dog's legs.
    """
    body = (74, 92, 108) if night else RUG
    edge = (58, 74, 90) if night else RUG_DARK
    stripe = (92, 108, 124) if night else RUG_LINE
    top, bot = floor_y + 120 * u, floor_y + 330 * u
    half_t, half_b = 470 * u, 620 * u
    p = skia.Path()
    p.moveTo(cx - half_t, top)
    p.lineTo(cx + half_t, top)
    p.lineTo(cx + half_b, bot)
    p.lineTo(cx - half_b, bot)
    p.close()
    canvas.drawPath(p, fill(body))
    canvas.save()
    canvas.clipPath(p, doAntiAlias=True)
    for f in (0.24, 0.74):
        y = top + (bot - top) * f
        canvas.drawRect(skia.Rect.MakeLTRB(cx - half_b, y, cx + half_b,
                                           y + 26 * u), fill(stripe, 180))
    canvas.restore()
    canvas.drawPath(p, stroke(edge, 6 * u))


def park(canvas, w: int, h: int) -> None:
    """Sky, sun, two ridges of hills, grass.

    The old version put two flat dark ovals on the horizon and called them
    trees. Layered ridges in separating tones read as distance, which is
    what an outdoor shot is for.
    """
    u = _u(w)
    horizon = h * 0.66
    canvas.drawRect(skia.Rect.MakeLTRB(0, 0, w, horizon),
                    grad((0, 0), (0, horizon), SKY_TOP, SKY_BOT))
    sun = skia.Paint(AntiAlias=True, Style=skia.Paint.kFill_Style)
    sun.setShader(skia.GradientShader.MakeRadial(
        center=(w * 0.78, h * 0.15), radius=h * 0.20,
        colors=[col((255, 246, 214), 220), col((255, 246, 214), 0)]))
    canvas.drawCircle(w * 0.78, h * 0.15, h * 0.20, sun)
    canvas.drawCircle(w * 0.78, h * 0.15, h * 0.058, fill((255, 244, 206)))
    for fy, rad, tone in ((0.055, 0.30, (168, 196, 176)),
                          (0.02, 0.24, (140, 176, 154))):
        far = skia.Path()
        far.moveTo(-w * 0.1, horizon + h * 0.02)
        far.cubicTo(w * 0.18, horizon - h * rad, w * 0.46, horizon - h * rad * 0.5,
                    w * 0.62, horizon - h * fy)
        far.cubicTo(w * 0.82, horizon - h * rad * 0.8, w * 1.0, horizon - h * 0.04,
                    w * 1.1, horizon + h * 0.02)
        far.close()
        canvas.drawPath(far, fill(tone))
    canvas.drawRect(skia.Rect.MakeLTRB(0, horizon, w, h),
                    grad((0, horizon), (0, h), LEAF, LEAF_DARK))
    canvas.drawRect(skia.Rect.MakeLTRB(0, horizon, w, horizon + 14 * u),
                    fill(LEAF_DARK))
    for fx, r in ((0.08, 0.030), (0.30, 0.021), (0.70, 0.024), (0.94, 0.032)):
        canvas.drawOval(skia.Rect.MakeLTRB(w * fx - h * r, horizon - h * r * 0.9,
                                           w * fx + h * r, horizon + h * r * 0.35),
                        fill(LEAF_DARK))


def _rr(canvas, l, t, r, b, rad, paint):
    canvas.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(l, t, r, b),
                                           rad, rad), paint)


def sofa(canvas, sx: float, floor_y: float, u: float, night: bool) -> None:
    """A low mid-century sofa: slim arms, splayed legs, two throw pillows.

    The old one was a single fat rounded rectangle sitting on the floor,
    which is a sofa the way a loaf is a sofa. Lifting it on legs lets the
    floor run underneath, and that gap is most of what dates or undates a
    drawn interior.
    """
    body = (102, 116, 150) if night else SOFA
    dark = (82, 94, 126) if night else SOFA_DARK
    seat = (118, 132, 166) if night else (176, 196, 184)
    lw = 6 * u
    sy = floor_y - 300 * u
    lift = floor_y - 46 * u          # underside of the frame
    legc = (74, 78, 104) if night else (150, 116, 84)

    for lx in (sx + 46 * u, sx + 474 * u):
        legp = skia.Path()
        legp.moveTo(lx - 14 * u, lift - 6 * u)
        legp.lineTo(lx + 14 * u, lift - 6 * u)
        legp.lineTo(lx + 24 * u, floor_y + 22 * u)
        legp.lineTo(lx + 8 * u, floor_y + 22 * u)
        legp.close()
        canvas.drawPath(legp, fill(legc))
        canvas.drawPath(legp, stroke(INK, lw))
    # back, then arms, then the seat in front of both
    _rr(canvas, sx + 24 * u, sy, sx + 496 * u, lift, 34 * u,
        grad((sx, sy), (sx, lift), body, dark))
    _rr(canvas, sx + 24 * u, sy, sx + 496 * u, lift, 34 * u, stroke(INK, lw))
    for ax in (sx - 6 * u, sx + 466 * u):
        _rr(canvas, ax, sy + 96 * u, ax + 62 * u, lift, 26 * u, fill(dark))
        _rr(canvas, ax, sy + 96 * u, ax + 62 * u, lift, 26 * u, stroke(INK, lw))
    _rr(canvas, sx + 8 * u, sy + 168 * u, sx + 512 * u, lift, 28 * u, fill(seat))
    _rr(canvas, sx + 8 * u, sy + 168 * u, sx + 512 * u, lift, 28 * u,
        stroke(INK, lw))
    canvas.drawLine(sx + 260 * u, sy + 176 * u, sx + 260 * u, lift - 10 * u,
                    stroke(INK, lw * 0.6))
    for px, c in ((sx + 66 * u, OCHRE), (sx + 366 * u, ACCENT)):
        tone = (96, 88, 124) if night else c
        _rr(canvas, px, sy + 58 * u, px + 118 * u, sy + 176 * u, 22 * u,
            fill(tone))
        _rr(canvas, px, sy + 58 * u, px + 118 * u, sy + 176 * u, 22 * u,
            stroke(INK, lw))


def window(canvas, wx: float, h: int, u: float, night: bool) -> None:
    """An arched window with a sill and something growing outside.

    A rectangle with a cross in it is a window from a child's drawing. The
    arch is the shape every interior shot has in it now, and a hedge and a
    sky behind the glass give the room an outside to sit in.
    """
    lw = 6 * u
    top, bot = h * 0.155, h * 0.475
    ww = 330 * u
    arch = skia.Path()
    arch.moveTo(wx, bot)
    arch.lineTo(wx, top + ww * 0.5)
    arch.arcTo(skia.Rect.MakeLTRB(wx, top, wx + ww, top + ww), 180, 180, False)
    arch.lineTo(wx + ww, bot)
    arch.close()
    canvas.drawPath(arch, grad((wx, top), (wx, bot),
                               (34, 42, 74) if night else GLASS_TOP,
                               (52, 58, 92) if night else GLASS_BOT))
    canvas.save()
    canvas.clipPath(arch, doAntiAlias=True)
    if night:
        canvas.drawCircle(wx + ww * 0.68, top + ww * 0.42, 34 * u, fill(LAMP))
        for dx, dy, r in ((0.22, 0.30, 5), (0.38, 0.55, 4), (0.80, 0.72, 5)):
            canvas.drawCircle(wx + ww * dx, top + ww * dy, r * u,
                              fill((236, 240, 255), 200))
    else:
        canvas.drawCircle(wx + ww * 0.74, top + ww * 0.34, 46 * u,
                          fill((255, 246, 214)))
        for fx, r in ((0.10, 120), (0.42, 96), (0.78, 108)):
            canvas.drawOval(skia.Rect.MakeLTRB(wx + ww * fx - r * u, bot - r * 1.5 * u,
                                               wx + ww * fx + r * u, bot + r * u),
                            fill(LEAF if fx == 0.42 else LEAF_DARK))
    canvas.restore()
    canvas.drawPath(arch, stroke(INK, lw))
    canvas.drawLine(wx + ww / 2, top + ww * 0.08, wx + ww / 2, bot,
                    stroke(INK, lw * 0.8))
    canvas.drawLine(wx + 6 * u, top + ww * 0.62, wx + ww - 6 * u,
                    top + ww * 0.62, stroke(INK, lw * 0.8))
    _rr(canvas, wx - 26 * u, bot, wx + ww + 26 * u, bot + 26 * u, 8 * u,
        fill((70, 78, 108) if night else RAIL))
    _rr(canvas, wx - 26 * u, bot, wx + ww + 26 * u, bot + 26 * u, 8 * u,
        stroke(INK, lw))


def picture(canvas, x: float, y: float, u: float, night: bool) -> None:
    """A pair of prints, hung off one baseline: an arch and a horizon.

    One frame with a dog's face in it was doing the job of a caption. Two
    abstract prints in thin charcoal frames read as a wall somebody
    decorated, and they do not compete with the animal for attention.
    """
    fr = (96, 102, 134) if night else FRAME
    inner = (108, 118, 146) if night else MAT
    lw = 5 * u
    _rr(canvas, x, y, x + 200 * u, y + 268 * u, 6 * u, fill(inner))
    _rr(canvas, x, y, x + 200 * u, y + 268 * u, 6 * u, stroke(fr, lw * 1.6))
    canvas.drawCircle(x + 74 * u, y + 96 * u, 44 * u,
                      fill((132, 142, 170) if night else OCHRE))
    a = skia.Path()
    a.moveTo(x + 40 * u, y + 216 * u)
    a.lineTo(x + 40 * u, y + 148 * u)
    a.arcTo(skia.Rect.MakeLTRB(x + 40 * u, y + 88 * u, x + 160 * u, y + 208 * u),
            180, 180, False)
    a.lineTo(x + 160 * u, y + 216 * u)
    a.close()
    canvas.drawPath(a, fill((122, 132, 160) if night else ACCENT, 225))
    canvas.drawRect(skia.Rect.MakeLTRB(x + 30 * u, y + 216 * u, x + 170 * u,
                                       y + 226 * u),
                    fill((96, 104, 136) if night else (104, 92, 86)))

    bx, by = x + 228 * u, y + 96 * u
    _rr(canvas, bx, by, bx + 168 * u, by + 172 * u, 6 * u, fill(inner))
    _rr(canvas, bx, by, bx + 168 * u, by + 172 * u, 6 * u, stroke(fr, lw * 1.6))
    hill = skia.Path()
    hill.moveTo(bx + 18 * u, by + 132 * u)
    hill.quadTo(bx + 66 * u, by + 58 * u, bx + 104 * u, by + 132 * u)
    hill.quadTo(bx + 130 * u, by + 92 * u, bx + 150 * u, by + 132 * u)
    hill.lineTo(bx + 18 * u, by + 132 * u)
    hill.close()
    canvas.drawPath(hill, fill((118, 128, 156) if night else LEAF_DARK, 210))


def _leaf(canvas, cx: float, cy: float, angle: float, length: float,
          half: float, face: tuple[int, int, int], vein: tuple[int, int, int],
          lw: float) -> None:
    """One leaf blade, drawn along its own axis with a midrib.

    The first pass cut monstera notches into the outline by sampling the
    half-width and dropping it inside a notch band. At the size a house
    plant occupies in frame those slits are three pixels wide and read as
    noise on the edge, not as a split leaf, so the blade is smooth and the
    veins carry the detail instead.
    """
    a = math.radians(angle)
    ux, uy = math.sin(a), -math.cos(a)
    nx, ny = -uy, ux
    tipx, tipy = cx + ux * length, cy + uy * length
    p = skia.Path()
    p.moveTo(cx, cy)
    for side in (1, -1):
        p.cubicTo(cx + ux * length * 0.22 + nx * half * 1.05 * side,
                  cy + uy * length * 0.22 + ny * half * 1.05 * side,
                  cx + ux * length * 0.74 + nx * half * 0.92 * side,
                  cy + uy * length * 0.74 + ny * half * 0.92 * side,
                  tipx, tipy)
        if side == 1:
            p.moveTo(tipx, tipy)
    p.close()
    canvas.drawPath(p, fill(face))
    # Veins are clipped to the blade. Drawn free they overshoot the outline
    # wherever the leaf narrows faster than the vein fans, and a leaf with
    # lines coming out of its edge reads as a web.
    canvas.save()
    canvas.clipPath(p, doAntiAlias=True)
    rib = skia.Path()
    rib.moveTo(cx, cy)
    rib.lineTo(tipx, tipy)
    canvas.drawPath(rib, stroke(vein, lw * 0.7))
    for f in (0.34, 0.62):
        bx, by = cx + ux * length * f, cy + uy * length * f
        for side in (1, -1):
            v = skia.Path()
            v.moveTo(bx, by)
            v.quadTo(bx + ux * length * 0.08 + nx * half * 0.45 * side,
                     by + uy * length * 0.08 + ny * half * 0.45 * side,
                     cx + ux * length * (f + 0.17) + nx * half * 0.62 * side,
                     cy + uy * length * (f + 0.17) + ny * half * 0.62 * side)
            canvas.drawPath(v, stroke(vein, lw * 0.42))
    canvas.restore()
    canvas.drawPath(p, stroke(INK, lw))


def plant(canvas, x: float, floor_y: float, u: float, night: bool) -> None:
    """A low, full house plant in a speckled pot on a stand.

    Tall thin stems with one blade on the end read as a spider, and at the
    right-hand edge of the frame half of it was outside the picture anyway.
    Short stems and wide blades give the corner a mass instead of a fringe.
    """
    leafc = (64, 96, 82) if night else LEAF_DARK
    leaf2 = (82, 116, 98) if night else LEAF
    vein = (52, 80, 68) if night else (72, 110, 84)
    potc = (130, 136, 164) if night else POT
    potd = (108, 114, 142) if night else POT_DARK
    lw = 6 * u
    rim = floor_y - 56 * u
    for angle, length, half, near in ((-58, 176, 60, False), (-24, 206, 66, True),
                                      (6, 194, 62, False), (38, 164, 56, True),
                                      (66, 126, 48, False)):
        sx = x + math.sin(math.radians(angle)) * 26 * u
        stem = skia.Path()
        stem.moveTo(x, rim)
        stem.quadTo(sx, rim - length * 0.35 * u, 
                    sx + math.sin(math.radians(angle)) * length * 0.30 * u,
                    rim - math.cos(math.radians(angle)) * length * 0.30 * u)
        canvas.drawPath(stem, stroke(INK, 13 * u))
        canvas.drawPath(stem, stroke(leafc, 8 * u))
        _leaf(canvas, sx + math.sin(math.radians(angle)) * length * 0.28 * u,
              rim - math.cos(math.radians(angle)) * length * 0.28 * u,
              angle, length * 0.78 * u, half * u,
              leaf2 if near else leafc, vein, lw * 0.9)
    pot = skia.Path()
    pot.moveTo(x - 66 * u, rim)
    pot.lineTo(x + 66 * u, rim)
    pot.lineTo(x + 46 * u, rim + 112 * u)
    pot.lineTo(x - 46 * u, rim + 112 * u)
    pot.close()
    canvas.drawPath(pot, grad((x, rim), (x, rim + 112 * u), potc, potd))
    canvas.drawPath(pot, stroke(INK, lw))
    canvas.save()
    canvas.clipPath(pot, doAntiAlias=True)
    for sx, sy in ((-30, 22), (-6, 60), (22, 16), (34, 56), (2, 88), (-38, 84)):
        canvas.drawCircle(x + sx * u, rim + sy * u, 3.4 * u, fill(INK, 70))
    canvas.restore()
    for lx in (-1, 1):
        lp = skia.Path()
        lp.moveTo(x + lx * 40 * u, rim + 104 * u)
        lp.lineTo(x + lx * 62 * u, floor_y + 92 * u)
        canvas.drawPath(lp, stroke(INK, 13 * u))
        canvas.drawPath(lp, stroke((74, 78, 104) if night else (150, 116, 84),
                                   8 * u))


def tree(canvas, tx: float, ty: float, horizon: float, r: float, u: float) -> None:
    """Trunk and a canopy of three blobs.

    The trunk used to be painted in FUR_SHADE, which is a coat colour: the
    moment the grey dog rebound it, every tree in the park turned grey.
    """
    trunk = skia.Path()
    trunk.moveTo(tx, ty)
    trunk.lineTo(tx, horizon + 10 * u)
    canvas.drawPath(trunk, stroke(INK, 42 * u))
    canvas.drawPath(trunk, stroke(BARK, 30 * u))
    # Unioned, not just added to one path: three ovals in a single path each
    # keep their own outline, so the canopy came out as a stroked Venn
    # diagram instead of one crown.
    canopy = skia.Path()
    for dx, dy, rr in ((-0.52, 0.22, 0.70), (0.52, 0.18, 0.66), (0.0, -0.18, 1.0)):
        blob = skia.Path()
        blob.addOval(skia.Rect.MakeLTRB(tx + dx * r - r * rr,
                                        ty + dy * r - r * rr,
                                        tx + dx * r + r * rr,
                                        ty + dy * r + r * rr * 0.82))
        canopy = skia.Op(canopy, blob, skia.PathOp.kUnion_PathOp)
    canvas.drawPath(canopy, grad((tx, ty - r), (tx, ty + r * 0.8), LEAF, LEAF_DARK))
    canvas.drawPath(canopy, stroke(INK, 6 * u))


def dog_bed(canvas, cx: float, cy: float, u: float, night: bool) -> None:
    """A bolster bed: charcoal shell, cream cushion.

    It was pink, which was the one saturated thing in the room and pulled
    the eye off the sleeping animal.
    """
    shell = (76, 82, 110) if night else (118, 126, 138)
    pad = (96, 102, 132) if night else (238, 232, 220)
    canvas.drawOval(skia.Rect.MakeLTRB(cx - 380 * u, cy - 108 * u,
                                       cx + 380 * u, cy + 108 * u), fill(shell))
    canvas.drawOval(skia.Rect.MakeLTRB(cx - 380 * u, cy - 108 * u,
                                       cx + 380 * u, cy + 108 * u),
                    stroke(INK, 6 * u))
    canvas.drawOval(skia.Rect.MakeLTRB(cx - 318 * u, cy - 68 * u,
                                       cx + 318 * u, cy + 92 * u), fill(pad))
    for k in range(-2, 3):
        canvas.drawLine(cx + k * 128 * u, cy - 58 * u, cx + k * 128 * u,
                        cy + 76 * u, stroke((208, 200, 186) if not night
                                            else (86, 92, 120), 4 * u))


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
        [(bx - 76 * u, by - 76 * u), (bx + 96 * u, by - 76 * u),
         (bx + 48 * u, by), (bx, by)],
        [(bx + bw + 76 * u, by - 76 * u), (bx + bw - 96 * u, by - 76 * u),
         (bx + bw - 48 * u, by), (bx + bw, by)],
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
    head(canvas, bx + bw * 0.50, by - 96 * u + bob, 1.55 * u * HEAD,
         blink=(math.sin(t * 2.1) + 1) / 2)
    box(canvas, bx, by, bw, bh, u)


def _room_static(canvas, w, h, night=False):
    """The room every indoor scene shares.

    Order is back to front: wall, then what hangs on it, then the floor rug,
    then the furniture standing on it. The rug goes down before the sofa so
    the sofa's legs sit on top of it.
    """
    u = _u(w)
    room(canvas, w, h, night)
    floor_y = h * 0.70
    picture(canvas, w * 0.08, h * 0.155, u, night)
    window(canvas, w * 0.60, h, u, night)
    rug(canvas, w * 0.52, floor_y, u, night)
    plant(canvas, w * 0.94, floor_y, u, night)
    sofa(canvas, w * 0.01, floor_y, u, night)


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


def clinic(canvas, w: int, h: int) -> None:
    """The consulting room: its own space, not the living room with a cross.

    The vet shot used to be the living room - sofa, prints, house plant -
    with a red cross floating over the wall art. A clinic is a cooler wall,
    a cabinet and a scrubbed floor, and the scene reads as somewhere else
    the moment those replace the furniture.
    """
    u = _u(w)
    floor_y = h * 0.70
    over = h * 0.6
    canvas.drawRect(skia.Rect.MakeLTRB(-over, -over, w + over, floor_y),
                    grad((0, 0), (0, floor_y), (240, 246, 244), (214, 228, 226)))
    canvas.drawRect(skia.Rect.MakeLTRB(-over, floor_y, w + over, h + over),
                    grad((0, floor_y), (0, h), (226, 232, 234), (198, 208, 212)))
    canvas.drawRect(skia.Rect.MakeLTRB(-over, floor_y - 26 * u, w + over, floor_y),
                    fill(RAIL))
    tile = fill((210, 220, 222))
    yy = floor_y + 120 * u
    while yy < h + over:
        canvas.drawRect(skia.Rect.MakeLTRB(-over, yy, w + over, yy + 4 * u), tile)
        yy += 190 * u

    # cabinet: three drawers and a worktop, pushed to the left
    cx0, cw = w * 0.02, 460 * u
    top = floor_y - 300 * u
    _rr(canvas, cx0, top, cx0 + cw, floor_y, 14 * u, fill((246, 248, 248)))
    _rr(canvas, cx0, top, cx0 + cw, floor_y, 14 * u, stroke(INK, 6 * u))
    _rr(canvas, cx0 - 12 * u, top - 22 * u, cx0 + cw + 12 * u, top, 8 * u,
        fill((186, 200, 202)))
    _rr(canvas, cx0 - 12 * u, top - 22 * u, cx0 + cw + 12 * u, top, 8 * u,
        stroke(INK, 6 * u))
    for k in range(3):
        dy = top + 30 * u + k * 88 * u
        _rr(canvas, cx0 + 24 * u, dy, cx0 + cw - 24 * u, dy + 68 * u, 10 * u,
            stroke(INK, 5 * u))
        canvas.drawLine(cx0 + cw * 0.40, dy + 34 * u, cx0 + cw * 0.60,
                        dy + 34 * u, stroke(INK, 7 * u))
    # a jar and a box on the worktop
    _rr(canvas, cx0 + 64 * u, top - 96 * u, cx0 + 132 * u, top - 22 * u, 10 * u,
        fill(LEAF))
    _rr(canvas, cx0 + 64 * u, top - 96 * u, cx0 + 132 * u, top - 22 * u, 10 * u,
        stroke(INK, 5 * u))
    _rr(canvas, cx0 + 168 * u, top - 70 * u, cx0 + 256 * u, top - 22 * u, 8 * u,
        fill(OCHRE))
    _rr(canvas, cx0 + 168 * u, top - 70 * u, cx0 + 256 * u, top - 22 * u, 8 * u,
        stroke(INK, 5 * u))

    # the cross, on clear wall above the table
    cx, cy, arm = w * 0.62, h * 0.20, 92 * u
    for l, tt, r, b in ((cx - arm / 3, cy - arm, cx + arm / 3, cy + arm),
                        (cx - arm, cy - arm / 3, cx + arm, cy + arm / 3)):
        _rr(canvas, l, tt, r, b, 14 * u, fill(ACCENT))
        _rr(canvas, l, tt, r, b, 14 * u, stroke(INK, 6 * u))


def scene_vet(canvas, w, h, t):
    u = _u(w)
    clinic(canvas, w, h)
    floor_y = h * 0.70
    top = floor_y + 30 * u
    # exam table: a padded top on a steel base, so the dog stands on
    # something rather than hovering over a slab
    for lx in (w * 0.30, w * 0.74):
        canvas.drawLine(lx, top + 44 * u, lx, floor_y + 250 * u,
                        stroke(INK, 22 * u))
        canvas.drawLine(lx, top + 44 * u, lx, floor_y + 250 * u,
                        stroke((188, 196, 200), 14 * u))
    _rr(canvas, w * 0.14, top, w * 0.90, top + 52 * u, 18 * u,
        fill((238, 242, 244)))
    _rr(canvas, w * 0.14, top, w * 0.90, top + 52 * u, 18 * u, stroke(INK, 6 * u))
    # Placed so the paws land on the tabletop rather than through it: the
    # pose puts them about 119 design units below the point it is given.
    dog_standing(canvas, w * 0.46, top - 112 * u, t, s=1.7 * u,
                 shade=(196, 206, 210))


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


_GRAIN: dict[tuple[int, int], Any] = {}


def grain(canvas, w: int, h: int, strength: int = 15) -> None:
    """A fixed film grain over the finished frame.

    Flat vector fills across a 1080x1920 frame band: the wall gradient alone
    covers a thousand rows in twenty steps of value, and h264 turns those
    steps into visible stripes. A little noise dithers the gradient and
    breaks the banding, and it is also what separates flat illustration that
    looks printed from flat illustration that looks like a slide.

    The pattern is generated once per frame size and reused, not reseeded
    per frame. Grain that crawls is expensive to encode - the codec has to
    spend bits on noise that changes every frame - and on a still drawing it
    reads as dirt on the lens rather than tooth in the paper.
    """
    img = _GRAIN.get((w, h))
    if img is None:
        import numpy as np
        # Generated at half size and doubled, so one grain is two pixels
        # across: at 1:1 on a 1080-wide frame it is finer than the encoder
        # can keep and comes out as mush. Doubled here rather than by
        # drawing it scaled, so the per-frame cost is a straight blit.
        gw, gh = max(1, w // 2), max(1, h // 2)
        rng = np.random.default_rng(1729)
        noise = rng.normal(128, 26, (gh, gw)).clip(0, 255).astype(np.uint8)
        noise = np.repeat(np.repeat(noise, 2, axis=0), 2, axis=1)[:h, :w]
        if noise.shape != (h, w):
            noise = np.pad(noise, ((0, h - noise.shape[0]),
                                   (0, w - noise.shape[1])), mode="edge")
        rgba = np.dstack([noise, noise, noise,
                          np.full((h, w), 255, dtype=np.uint8)])
        img = skia.Image.fromarray(np.ascontiguousarray(rgba))
        _GRAIN[(w, h)] = img
    canvas.drawImage(img, 0, 0, skia.SamplingOptions(),
                     skia.Paint(Alphaf=strength / 255.0,
                                BlendMode=skia.BlendMode.kOverlay))


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
    use_species(toon.species_for(
        query, str(cfg["visuals"].get("toon_dog", "greydog"))))
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
        grain(canvas, w, h, int(cfg["video"].get("toon_grain", 15)))
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


def spring(p: float, damping: float = 6.0, freq: float = 9.0) -> float:
    """A damped spring: overshoots the target and settles back onto it.

    The procedural-animation work this module has been borrowing from drives
    hair and clothing off damped springs rather than off eased interpolation,
    for the same reason a title card feels dead when it eases in and alive
    when it overshoots by a few per cent. Analytic rather than integrated,
    so it can be evaluated at any frame without carrying state between them.
    """
    if p >= 1.0:
        return 1.0
    return 1.0 - math.exp(-damping * p) * math.cos(freq * p)


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

    pop = spring(min(1.0, progress / 0.16)) if progress < 0.16 else 1.0
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
