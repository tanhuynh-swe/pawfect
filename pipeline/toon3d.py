"""GPU backend: the shot modelled and lit in 3D instead of drawn flat.

The Skia backend draws a side view. Every pose is a set of curves authored
for one camera angle, so the dog cannot turn its head, the room cannot be
looked at from anywhere else, and depth exists only as the order the shapes
are painted in. That ceiling is the reason for this module: here the room
and the animal are geometry, a camera looks at them, and one light decides
what is lit. Turning the camera is then a parameter rather than a redraw.

It stays a cartoon on purpose. The shading is a three-band ramp rather than
a smooth falloff, and every solid gets an ink outline from an inverted hull
- the same shape drawn slightly fattened, front faces culled, in ink, before
the shaded pass. Those two tricks are what keep a lit 3D object looking like
a drawing instead of a product render.

What it shares with the 2D backend rather than duplicating:

  `toon.scene_for` / `toon.species_for`, so a query picks the same scene and
  the same animal whichever backend draws it.

  `toon_skia`'s palette, so the room is the same room in both, and its
  `banner()`, `grain()` and easing curves, which are 2D work that belongs on
  top of a finished frame either way. The GL frame is handed to Skia as an
  image and the caption and grain go on over it.

Markings are computed in the fragment shader from the object-space position
of each vertex, which is the 3D form of `toon_skia._mask()`: the dark crown
and bridge, the tan eyebrow spot and eye ring, the cream muzzle and bib are
regions of the unit sphere the part was built from, not a painted texture.
Nothing has to be unwrapped, and the markings hold from any angle.
"""
from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any

import moderngl
import numpy as np
import skia

from . import toon
from . import toon_skia as t2d

# --- meshes -----------------------------------------------------------------
# Three primitives cover the whole channel: an ellipsoid for every soft part
# of an animal, a capsule for limbs and tails, and a box for the furniture.
# They are built once as unit shapes and placed by matrix, so the vertex data
# is uploaded to the GPU one time per process.


def _sphere(stacks: int = 20, slices: int = 32):
    v, idx = [], []
    for i in range(stacks + 1):
        phi = math.pi * i / stacks
        for j in range(slices + 1):
            th = 2 * math.pi * j / slices
            v.append((math.sin(phi) * math.cos(th), math.cos(phi),
                      math.sin(phi) * math.sin(th)))
    for i in range(stacks):
        for j in range(slices):
            a = i * (slices + 1) + j
            b = a + slices + 1
            idx += [a, a + 1, b, a + 1, b + 1, b]
    p = np.array(v, "f4")
    return p, p.copy(), np.array(idx, "i4")


def _capsule(slices: int = 20, cap: int = 6):
    """A cylinder from y=-1 to y=+1 with hemisphere ends, radius 1.

    Limbs drawn as bare cylinders end in a visible disc, and at the ankle
    that disc reads as a cut. The caps cost a few hundred triangles and mean
    a leg can be one primitive instead of three.
    """
    rings = []
    for i in range(cap + 1):                      # bottom hemisphere
        a = math.pi / 2 * i / cap
        rings.append((-1 - math.cos(a) * 0 - math.sin(math.pi / 2 - a),
                      math.cos(math.pi / 2 - a)))
    rings.append((-1.0, 1.0))
    rings.append((1.0, 1.0))
    for i in range(cap + 1):                      # top hemisphere
        a = math.pi / 2 * i / cap
        rings.append((1 + math.sin(a), math.cos(a)))
    v, n, idx = [], [], []
    for y, r in rings:
        for j in range(slices + 1):
            th = 2 * math.pi * j / slices
            v.append((math.cos(th) * r, y, math.sin(th) * r))
            ny = 0.0 if abs(y) <= 1.0 else (y - math.copysign(1.0, y))
            nn = np.array([math.cos(th) * r, ny, math.sin(th) * r], "f4")
            ln = np.linalg.norm(nn) or 1.0
            n.append(nn / ln)
    for i in range(len(rings) - 1):
        for j in range(slices):
            a = i * (slices + 1) + j
            b = a + slices + 1
            # Wound the opposite way round from the sphere: these rings run
            # bottom to top. Get this backwards and front-face culling swaps
            # the ink hull for the shaded surface, which paints every limb
            # solid black.
            idx += [a, b, a + 1, a + 1, b, b + 1]
    return np.array(v, "f4"), np.array(n, "f4"), np.array(idx, "i4")


def _box():
    f = [((0, 0, 1), [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]),
         ((0, 0, -1), [(1, -1, -1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1)]),
         ((1, 0, 0), [(1, -1, 1), (1, -1, -1), (1, 1, -1), (1, 1, 1)]),
         ((-1, 0, 0), [(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)]),
         ((0, 1, 0), [(-1, 1, 1), (1, 1, 1), (1, 1, -1), (-1, 1, -1)]),
         ((0, -1, 0), [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)])]
    v, n, idx = [], [], []
    for nrm, quad in f:
        base = len(v)
        for pt in quad:
            v.append(pt)
            n.append(nrm)
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    return np.array(v, "f4"), np.array(n, "f4"), np.array(idx, "i4")


def tube(stations, slices: int = 26):
    """A surface of revolution along the spine, with a varying radius.

    Three overlapping ellipsoids give a body three silhouettes and three
    shading gradients, and no amount of outline work hides that it is a
    string of sausages. One lathed surface gives the chest, the tucked waist
    and the haunch as a single form that lights as a single form - it is the
    difference between a model of a dog and a balloon sculpture of one.

    `stations` is a list of (x, y, ry, rz) from tail to nose. Normals come
    from the surface's own derivatives rather than being assumed radial, so
    the taper shades correctly where the body narrows.
    """
    n_st = len(stations)
    pts = np.zeros((n_st, slices + 1, 3), "f4")
    for i, (x, y, ry, rz) in enumerate(stations):
        for j in range(slices + 1):
            th = 2.0 * math.pi * j / slices
            pts[i, j] = (x, y + math.cos(th) * ry, math.sin(th) * rz)
    nrm = np.zeros_like(pts)
    for i in range(n_st):
        i0, i1 = max(0, i - 1), min(n_st - 1, i + 1)
        for j in range(slices + 1):
            j0, j1 = (j - 1) % slices, (j + 1) % slices
            du = pts[i1, j] - pts[i0, j]
            dv = pts[i, j1] - pts[i, j0]
            v = np.cross(dv, du)
            ln = np.linalg.norm(v)
            nrm[i, j] = v / ln if ln > 1e-8 else (0.0, 1.0, 0.0)
    verts = pts.reshape(-1, 3)
    norms = nrm.reshape(-1, 3)
    idx: list[int] = []
    for i in range(n_st - 1):
        for j in range(slices):
            a = i * (slices + 1) + j
            b = a + slices + 1
            # Wound to match the sphere. Backwards, and front-face culling
            # hands the ink hull the whole body: one solid black dog.
            idx += [a, a + 1, b, a + 1, b + 1, b]
    # caps, so the ends are closed rather than open pipes
    for end, station in ((0, stations[0]), (n_st - 1, stations[-1])):
        base = len(verts)
        centre = np.array([[station[0], station[1], 0.0]], "f4")
        verts = np.vstack([verts, centre])
        axis = np.array([[-1.0 if end == 0 else 1.0, 0.0, 0.0]], "f4")
        norms = np.vstack([norms, axis])
        for j in range(slices):
            a = end * (slices + 1) + j
            if end == 0:
                idx += [base, a + 1, a]
            else:
                idx += [base, a, a + 1]
    return verts.astype("f4"), norms.astype("f4"), np.array(idx, "i4")


def _grow_dirs(v: np.ndarray) -> np.ndarray:
    """Smooth outward directions for a faceted mesh, used by the ink hull."""
    ln = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, np.where(ln < 1e-6, 1.0, ln)).astype("f4")


def _wedge():
    """A flat triangle with thickness: two faces and three sides.

    Ears were cones, and an inverted-hull outline on a cone is a spike: every
    vertex at the apex shares a position but not a normal, so growing along
    the normal fans them into a starburst. A wedge's tip is an edge, not a
    point, so it grows cleanly.
    """
    t, b = 1.0, -1.0
    quads = [
        ((0, 0, 1), [(-1, b, 1), (1, b, 1), (0, t, 1), (0, t, 1)]),
        ((0, 0, -1), [(1, b, -1), (-1, b, -1), (0, t, -1), (0, t, -1)]),
        ((0, -1, 0), [(-1, b, -1), (1, b, -1), (1, b, 1), (-1, b, 1)]),
        ((-0.89, 0.45, 0), [(-1, b, 1), (0, t, 1), (0, t, -1), (-1, b, -1)]),
        ((0.89, 0.45, 0), [(1, b, -1), (0, t, -1), (0, t, 1), (1, b, 1)]),
    ]
    v, n, idx = [], [], []
    for nrm, quad in quads:
        base = len(v)
        for pt in quad:
            v.append(pt)
            n.append(nrm)
        idx += [base, base + 1, base + 2, base, base + 2, base + 3]
    return np.array(v, "f4"), np.array(n, "f4"), np.array(idx, "i4")


def _cone(slices: int = 24):
    """Radius 1 at y=-1, a point at y=+1. Ears, and the plant's pot."""
    v, n, idx = [], [], []
    for j in range(slices + 1):
        th = 2 * math.pi * j / slices
        c, s = math.cos(th), math.sin(th)
        v.append((c, -1.0, s))
        n.append((c * 0.7, 0.5, s * 0.7))
        v.append((0.0, 1.0, 0.0))
        n.append((c * 0.4, 0.9, s * 0.4))
    for j in range(slices):
        a = j * 2
        idx += [a, a + 2, a + 1]
    base = len(v)
    v.append((0.0, -1.0, 0.0))
    n.append((0.0, -1.0, 0.0))
    for j in range(slices + 1):
        th = 2 * math.pi * j / slices
        v.append((math.cos(th), -1.0, math.sin(th)))
        n.append((0.0, -1.0, 0.0))
    for j in range(slices):
        idx += [base, base + 1 + j + 1, base + 1 + j]
    return np.array(v, "f4"), np.array(n, "f4"), np.array(idx, "i4")


# --- matrices ---------------------------------------------------------------

def _ident() -> np.ndarray:
    return np.eye(4, dtype="f4")


def _trs(pos=(0, 0, 0), scale=(1, 1, 1), rot=(0, 0, 0)) -> np.ndarray:
    """Translate * Rz * Ry * Rx * Scale, the order a pose is easiest to think in."""
    sx, sy, sz = scale
    m = np.diag([sx, sy, sz, 1.0]).astype("f4")
    for axis, ang in enumerate(rot):
        if not ang:
            continue
        c, s = math.cos(ang), math.sin(ang)
        r = _ident()
        if axis == 0:
            r[1, 1], r[1, 2], r[2, 1], r[2, 2] = c, -s, s, c
        elif axis == 1:
            r[0, 0], r[0, 2], r[2, 0], r[2, 2] = c, s, -s, c
        else:
            r[0, 0], r[0, 1], r[1, 0], r[1, 1] = c, -s, s, c
        m = r @ m
    m[0, 3], m[1, 3], m[2, 3] = pos
    return m.astype("f4")


def _look_at(eye, target, up=(0, 1, 0)) -> np.ndarray:
    e = np.array(eye, "f4")
    f = np.array(target, "f4") - e
    f /= np.linalg.norm(f) or 1.0
    s = np.cross(f, np.array(up, "f4"))
    s /= np.linalg.norm(s) or 1.0
    u = np.cross(s, f)
    m = _ident()
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ e
    return m


def _persp(fov_y: float, aspect: float, near=0.1, far=120.0) -> np.ndarray:
    t = 1.0 / math.tan(math.radians(fov_y) / 2)
    m = np.zeros((4, 4), "f4")
    m[0, 0] = t / aspect
    m[1, 1] = t
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def _distance_for(half: float, fov_y: float, aspect: float) -> float:
    """How far back the camera must sit for the subject to fit in frame.

    Perspective is set by the vertical angle, so at the same fov a 9:16
    frame is narrow across and a 16:9 frame is short down. Solving for the
    vertical alone puts the subject half outside a Short; solving for the
    horizontal alone fills a landscape frame top to bottom with dog. Fitting
    it into whichever of the two angles is tighter composes both the same
    way from one number.
    """
    tan_half = math.tan(math.radians(fov_y) / 2)
    return half / max(0.02, tan_half * min(aspect, 1.0))


# --- materials --------------------------------------------------------------
# The shader reads `mark` to decide whether a surface is a flat colour or one
# of the marked parts of the channel's dog, and computes the markings from
# the object-space position. This is toon_skia._mask() as arithmetic.

PLAIN, HEAD, BODY, MUZZLE = 0, 1, 2, 3

# The face is not one flat dark. It is a grizzled mid tone over most of the
# skull with the crown, the bridge and the lips darker than that, which is
# what lets a black nose read against a dark muzzle at all.
MASK_MID = (96, 90, 82)


def _rgb(c: tuple[int, int, int]) -> tuple[float, float, float]:
    return (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)


VERT = """
#version 330
uniform mat4 viewproj;
uniform mat4 model;
uniform mat4 rootinv;
uniform mat3 nmat;
uniform float grow;
in vec3 in_p;
in vec3 in_n;
in vec3 in_g;
out vec3 v_obj;
out vec3 v_nrm;
out vec3 v_world;
out vec3 v_char;
void main() {
    // The outline hull is grown along the world normal, not the object one.
    // Grown in object space a 0.4-unit head and a 0.03-unit eye get outlines
    // in proportion to themselves, which means the small parts get none.
    v_obj = in_p;
    v_nrm = normalize(nmat * in_n);
    // The hull grows along in_g, not along the shading normal. A box or a
    // wedge carries one normal per face, so growing along those pushes the
    // six faces apart into a flat-pack and the part gets no outline at all;
    // in_g is a smooth per-vertex direction that inflates the shape whole.
    vec4 w = model * vec4(in_p, 1.0);
    w.xyz += normalize(nmat * in_g) * grow;
    v_world = w.xyz;
    v_char = (rootinv * vec4(w.xyz, 1.0)).xyz;
    gl_Position = viewproj * w;
}
"""

FRAG = """
#version 330
uniform vec3 base;
uniform vec3 mark_dark;
uniform vec3 mark_mid;
uniform vec3 mark_tan;
uniform vec3 mark_cream;
uniform vec3 ldir;
uniform vec3 eye;
uniform vec3 ink;
uniform int  mark;
uniform int  outline;
uniform float ambient;
uniform float alpha;
in vec3 v_obj;
in vec3 v_nrm;
in vec3 v_world;
in vec3 v_char;
out vec4 frag;

// The dog's markings, in the unit space of the sphere each part was built
// from: +x forward toward the nose, +y up, +z to the animal's left.
vec3 marked(vec3 c) {
    // smoothstep everywhere: fur does not change colour along a line, and a
    // hard edge on a lit 3D surface reads as a decal stuck to it.
    if (mark == 1) {                       // head
        // Head units now: x runs -1 at the back of the skull to +1.78 at the
        // nose, radius 1 at the cranium. The face is dark - crown, mask and
        // muzzle - with tan as the light marks and cream only under the jaw,
        // which is the pattern the reference photograph actually has.
        c = mark_mid;
        float top = smoothstep(-0.10, 0.40, v_obj.y);
        c = mix(c, mark_dark, top);
        c = mix(c, mark_dark, smoothstep(0.55, 1.00, v_obj.x));   // muzzle
        c = mix(c, mark_cream, smoothstep(-0.45, -0.80, v_obj.y)); // chin
        vec3 side = vec3(1.0, 1.0, sign(v_obj.z));
        // Tight. These are points on the face, not fields: the first pass
        // used a slow falloff and the cheek spread across the whole head,
        // turning a dark-faced dog into a tan-faced one.
        vec3 cheek_d = (v_obj - vec3(0.20, -0.30, 0.82) * side)
                       * vec3(1.10, 1.30, 1.00);
        c = mix(c, mark_tan, 1.0 - smoothstep(0.13, 0.27, length(cheek_d)));
        vec3 eye_d = (v_obj - vec3(0.38, 0.24, 0.80) * side)
                     * vec3(1.30, 1.35, 1.10);
        c = mix(c, mark_tan, 1.0 - smoothstep(0.11, 0.22, length(eye_d)));
        vec3 brow_d = (v_obj - vec3(0.12, 0.50, 0.70) * side)
                      * vec3(1.20, 1.30, 1.10);
        c = mix(c, mark_tan, 1.0 - smoothstep(0.09, 0.19, length(brow_d)));
    } else if (mark == 2) {                // torso, in character space
        // A saddle with an edge you can see. Tuned against the render, not
        // derived: the top of the barrel sits lower than the profile numbers
        // suggest once the spine pitches and the ribs breathe.
        float back = 0.74 - 0.20 * v_char.x;
        c = mix(c, mark_dark, smoothstep(0.0, 0.07, v_char.y - back));
        c = mix(c, mark_cream, smoothstep(0.60, 0.46, v_char.y));
        float bib = smoothstep(0.44, 0.70, v_char.x) *
                    (1.0 - smoothstep(0.68, 0.88, v_char.y));
        c = mix(c, mark_cream, bib);
    } else if (mark == 3) {                // muzzle: dark, like the real one
        c = mix(mark_mid, mark_dark, smoothstep(-0.45, 0.15, v_obj.y));
    }
    return c;
}

void main() {
    if (outline == 1) { frag = vec4(ink, alpha); return; }
    vec3 c = marked(base);
    vec3 n = normalize(v_nrm);
    float d = max(dot(n, normalize(ldir)), 0.0);
    // Three bands, not a ramp. A smooth falloff on a cartoon reads as
    // plastic; the steps are what make it look drawn.
    // Harder steps and a wider spread. Soft bands on a soft shape is what
    // made the first pass look like putty: the shading has to draw the form
    // as decisively as the outline draws the edge.
    float band = d > 0.58 ? 1.0 : (d > 0.22 ? 0.76 : 0.55);
    // Warm key, cool shadow. Shading one colour up and down its own value
    // ramp reads as dirty; splitting the hue is what makes it read as light.
    vec3 warm = vec3(1.05, 1.005, 0.94);
    vec3 cool = vec3(0.88, 0.93, 1.08);
    vec3 tint = mix(cool, warm, smoothstep(0.35, 0.85, band));
    vec3 lit = c * tint * (ambient + (1.0 - ambient) * band);
    // Rim light along the silhouette, the 3D form of toon_skia.inner_edge().
    float rim = pow(1.0 - max(dot(n, normalize(eye - v_world)), 0.0), 3.5);
    lit += rim * 0.16;
    frag = vec4(clamp(lit, 0.0, 1.0), alpha);
}
"""

SKY_VERT = """
#version 330
in vec2 in_p;
out vec2 v_uv;
void main() { v_uv = in_p * 0.5 + 0.5; gl_Position = vec4(in_p, 0.0, 1.0); }
"""

SKY_FRAG = """
#version 330
uniform vec3 top;
uniform vec3 bottom;
in vec2 v_uv;
out vec4 frag;
void main() { frag = vec4(mix(bottom, top, v_uv.y), 1.0); }
"""


class _GL:
    """One context, one program, one set of meshes, reused for every clip."""

    def __init__(self) -> None:
        self.ctx = moderngl.create_standalone_context()
        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self.sky = self.ctx.program(vertex_shader=SKY_VERT,
                                    fragment_shader=SKY_FRAG)
        quad = np.array([-1, -1, 3, -1, -1, 3], "f4")
        self.sky_vao = self.ctx.vertex_array(
            self.sky, [(self.ctx.buffer(quad), "2f", "in_p")])
        self.meshes: dict[str, Any] = {}
        faceted = ("box", "wedge")
        for name, data in (("sphere", _sphere()), ("capsule", _capsule()),
                           ("box", _box()), ("cone", _cone()),
                           ("wedge", _wedge())):
            v, n, i = data
            g = _grow_dirs(v) if name in faceted else n
            buf = self.ctx.buffer(np.hstack([v, n, g]).astype("f4").tobytes())
            ibo = self.ctx.buffer(i.tobytes())
            self.meshes[name] = self.ctx.vertex_array(
                self.prog, [(buf, "3f 3f 3f", "in_p", "in_n", "in_g")], ibo)
        self.fbo = None
        self.size = (0, 0)
        self._scratch: list[Any] = []

    def upload(self, data):
        """A one-frame VAO. Released after the frame so nothing accumulates."""
        v, n, i = data
        buf = self.ctx.buffer(np.hstack([v, n, n]).astype("f4").tobytes())
        ibo = self.ctx.buffer(i.tobytes())
        vao = self.ctx.vertex_array(
            self.prog, [(buf, "3f 3f 3f", "in_p", "in_n", "in_g")], ibo)
        self._scratch += [buf, ibo, vao]
        return vao

    def sweep(self):
        for obj in self._scratch:
            obj.release()
        self._scratch.clear()

    def target(self, w: int, h: int):
        if self.size != (w, h):
            samples = min(4, self.ctx.max_samples)
            self.ms = self.ctx.framebuffer(
                [self.ctx.renderbuffer((w, h), samples=samples)],
                self.ctx.depth_renderbuffer((w, h), samples=samples))
            self.fbo = self.ctx.framebuffer([self.ctx.renderbuffer((w, h))])
            self.size = (w, h)
        return self.ms, self.fbo


_gl: _GL | None = None


def _gpu() -> _GL:
    global _gl
    if _gl is None:
        _gl = _GL()
    return _gl


class Item:
    """One solid: a mesh, where it is, what colour, and whether it is inked."""

    __slots__ = ("mesh", "m", "colour", "mark", "outline", "alpha", "data")

    def __init__(self, mesh: str, m: np.ndarray, colour, mark: int = PLAIN,
                 outline: float = 0.0, alpha: float = 1.0, data=None) -> None:
        # `mesh` names a shared primitive; `data` carries one built for this
        # frame (the body, which changes shape with the pose).
        self.mesh = mesh
        self.data = data
        self.m = m
        self.colour = _rgb(colour)
        self.mark = mark
        self.outline = outline
        self.alpha = alpha


# --- the animal -------------------------------------------------------------
# One builder for both characters. A cat is a dog with a shorter muzzle,
# taller ears, a longer tail and no markings, which is the same split the 2D
# backend makes - keeping it one function is what stops the two drifting
# apart pose by pose.

# Unit-length directions in the head's own space, so `obj * scale`
# puts a solid on the skull's surface rather than buried in it, and the
# shader can centre the tan ring and the eyebrow spot on the same numbers.
# The eyebrow spot's centre, (0.46, 0.66, 0.52), lives only in the shader:
# nothing is placed there, so a Python copy of it could only ever drift.
EYE_OBJ = np.array([0.38, 0.24, 0.80], "f4")

HEAD_C, HEAD_S = (0.86, 1.12, 0.0), (0.40, 0.385, 0.36)
BODY_C, BODY_S = (0.0, 0.72, 0.0), (0.86, 0.46, 0.41)
LEG_HALF, PAW_DROP, PAW_R = 0.30, 0.60, 0.075

# Where the head ends up in each pose. A close-up has to aim at the animal's
# head, and the head is wherever the pose and the root rotation put it - in
# 2D the camera was a scale about a point on a flat drawing, here it is a
# real camera that has to be told what to look at.
HEAD_BY_POSE = {
    "stand": HEAD_C, "run": HEAD_C, "bark": (0.86, 1.28, 0.0),
    "sleep": (0.72, 0.40, 0.22), "bow": (0.98, 0.50, 0.0),
    "sniff": (1.00, 0.38, 0.0), "sit": (0.62, 1.16, 0.0),
}


def head_world(root: np.ndarray, pose: str) -> tuple[float, float, float]:
    h = HEAD_BY_POSE.get(pose, HEAD_C)
    v = root @ np.array([h[0], h[1], h[2], 1.0], "f4")
    return (float(v[0]), float(v[1]), float(v[2]))


def _on_head(obj: np.ndarray, flip: int = 1) -> tuple[float, float, float]:
    """Object-space point on the head sphere -> where it sits in the world.

    The shader draws the tan ring at EYE_OBJ and the eye itself is a solid
    placed here, so both read from the same constant. Sliding one without
    the other is how a character ends up with its markings beside its face.
    """
    return (HEAD_C[0] + obj[0] * HEAD_S[0],
            HEAD_C[1] + obj[1] * HEAD_S[1],
            HEAD_C[2] + obj[2] * HEAD_S[2] * flip)


def _rot2(px: float, py: float, cx: float, cy: float, ang: float):
    """Rotate a point about a pivot in the x/y plane (the dog's own side view)."""
    if not ang:
        return px, py
    c, sn = math.cos(ang), math.sin(ang)
    dx, dy = px - cx, py - cy
    return cx + dx * c - dy * sn, cy + dx * sn + dy * c


HIP_Y = 0.62
UPPER, LOWER, PAW_H = 0.17, 0.15, 0.06
# Where the paws end up under the hip, and therefore how far the root has to
# be lifted for them to stand on the floor rather than in it.
FOOT_DROP = 0.16 + UPPER + 0.30 + LOWER + PAW_H


def _limb(add, hx: float, hy: float, hz: float, swing: float, bend: float,
          coat, cream, s: float = 1.0) -> None:
    """One leg: thigh, shank, paw.

    A leg is not a tube. Two masses with a joint between them, the upper
    noticeably thicker than the lower, is the least it takes for a walk to
    read as a walk and for a standing animal to look like it is carrying its
    own weight. The thigh/shank contrast matters more than the joint angle:
    even limbs of one width read as furniture legs however they are posed.
    """
    ux, uy = _rot2(hx, hy - 0.16 - UPPER, hx, hy, swing)
    add("capsule", (ux, uy, hz), (0.120 * s, UPPER, 0.118 * s), coat,
        rot=(0, 0, swing))
    kx, ky = _rot2(hx, hy - 0.16 - UPPER * 2, hx, hy, swing)
    lx, ly = _rot2(kx, ky - LOWER, kx, ky, swing + bend)
    add("capsule", (lx, ly, hz), (0.068 * s, LOWER, 0.066 * s), coat,
        rot=(0, 0, swing + bend))
    fx, fy = _rot2(kx, ky - LOWER * 2 - PAW_H * 0.6, kx, ky, swing + bend)
    add("sphere", (fx + 0.035, fy, hz), (0.105, PAW_H, 0.092), cream)


# The head in its own units: x runs back-of-skull (-1) to nose (+1.75), and
# a radius of 1 is the widest part of the cranium. Built as one lathed
# surface for the same reason the body is: a sphere with a capsule stuck on
# the front is a ball with a beak, and no amount of shading hides the join.
# The shape a dog's head actually has is a dome, a brow, a pinch at the stop,
# and a muzzle that tapers to the nose - four features on one curve.
HEAD_PROFILE = [
    (-1.00, 0.02, 0.30, 0.28),
    (-0.88, 0.02, 0.62, 0.60),
    (-0.62, 0.02, 0.86, 0.84),
    (-0.30, 0.00, 0.99, 0.95),
    (0.02, -0.03, 1.00, 0.94),
    (0.26, -0.08, 0.90, 0.84),
    (0.46, -0.16, 0.68, 0.64),     # the stop: the pinch under the brow
    (0.70, -0.24, 0.53, 0.52),
    (1.05, -0.30, 0.47, 0.47),
    (1.40, -0.34, 0.43, 0.44),
    (1.66, -0.36, 0.33, 0.35),
    (1.78, -0.37, 0.16, 0.18),
]


def head_mesh():
    return tube(HEAD_PROFILE, slices=24)


def dog(items: list[Item], t: float, pose: str, root: np.ndarray,
        species: str = "greydog", pant: bool = True) -> None:
    """The character, posed at `t`. `species` picks which one.

    The torso is three masses on a spine - haunch, barrel, chest - rather
    than one ellipsoid. That is the difference between an animal and a
    beanbag with legs: the shoulder and the rump are where a quadruped
    carries its shape, and a single blob has neither.

    Markings belong to the channel's own dog only: the tan puppy and the cat
    share the geometry but take a flat coat, exactly as in the 2D backend.
    """
    cat = species == "cat"
    marked = species == "greydog"
    ink_g = 0.025
    coat = t2d.GREY_COAT[1] if marked else t2d.FUR
    dark = t2d.MARK_DARK if marked else t2d.FUR_SHADE
    tan = t2d.MARK_TAN if marked else t2d.FUR_BELLY
    cream = t2d.MARK_CREAM if marked else t2d.FUR_BELLY
    mark_head = HEAD if marked else PLAIN
    mark_body = BODY if marked else PLAIN
    mark_muz = MUZZLE if marked else PLAIN

    def add(mesh, pos, scale, colour, rot=(0, 0, 0), mark=PLAIN, outline=ink_g):
        items.append(Item(mesh, root @ _trs(pos, scale, rot), colour, mark,
                          outline))

    breath = 1.0 + math.sin(t * 2.1) * 0.020
    bob = math.sin(t * 2.2) * 0.018
    wag = math.sin(t * 8.0) * 0.55
    pitch = 0.0
    lift = 0.0
    squash = 1.0
    head_c = list(HEAD_C)
    head_pitch = 0.0
    mouth = 0.0
    swings = [0.0, 0.0, 0.0, 0.0]
    # Front knees fold back, rear hocks fold forward - that opposition is
    # the single thing that makes a quadruped's stance read as an animal's.
    bends = [-0.14, -0.14, 0.34, 0.34]
    curled = pose == "sleep"

    if pose == "stand":
        swings = [-0.05, -0.05, 0.18, 0.18]
        lift = bob
        head_c[1] += bob
        mouth = 0.16 + 0.10 * math.sin(t * 5.4) if pant else 0.0
    elif pose == "run":
        phase = t * 11.0
        lift = max(0.0, math.sin(phase * 2)) * 0.15
        pitch = -0.13
        reach = math.sin(phase) * 0.85
        trail = math.sin(phase + 2.3) * 0.85
        swings = [reach, reach + 0.28, trail, trail - 0.28]
        bends = [-0.35, -0.30, 0.45, 0.40]
        wag = math.sin(t * 9.0) * 0.35
        mouth = 0.34
        head_c[1] += lift
    elif pose == "bark":
        pulse = max(0.0, math.sin(t * 8.5))
        head_c = [0.84, 1.26 + pulse * 0.06, 0.0]
        head_pitch = -0.32 - pulse * 0.14
        pitch = -0.07
        mouth = pulse
        wag = math.sin(t * 11.0) * 0.7
    elif pose == "sleep":
        squash = 0.72
        lift = -0.30
        head_c = [0.70, 0.40 + math.sin(t * 2.0) * 0.012, 0.26]
        head_pitch = 0.22
    elif pose == "bow":
        pitch = 0.24
        lift = 0.02
        head_c = [0.96, 0.66, 0.0]
        head_pitch = 0.20
        swings = [0.62, 0.62, 0.0, 0.0]
        bends = [-0.50, -0.50, 0.16, 0.16]
        wag = math.sin(t * 12.0) * 0.8
        mouth = 0.40
    elif pose == "sniff":
        cast = math.sin(t * 3.2) * 0.09
        pitch = 0.18
        head_c = [0.98, 0.40, cast]
        head_pitch = 0.62
        wag = math.sin(t * 6.5) * 0.5
    elif pose == "sit":
        pitch = -0.38
        lift = -0.10
        head_c = [0.60, 1.14 + bob, 0.0]
        swings = [0.0, 0.0, 1.25, 1.25]
        mouth = 0.16 + 0.10 * math.sin(t * 5.0) if pant else 0.0

    spine_y = 0.70 + lift
    pivot = (0.0, spine_y)

    def on_spine(x, y):
        return _rot2(x, y, pivot[0], pivot[1], pitch)

    # --- tail, first so the haunch overlaps its root
    if not curled:
        rx, ry = on_spine(-0.60, spine_y + 0.12)
        ang = -0.26 + wag * 0.10
        for rad, half in ((0.080, 0.19), (0.070, 0.17), (0.058, 0.15)):
            ang += 0.50 + wag * 0.06
            dx, dy = math.sin(ang) * half, math.cos(ang) * half
            add("capsule", (rx + dx, ry + dy, 0.0), (rad, half, rad), dark,
                rot=(0, 0, -ang))
            rx += dx * 2
            ry += dy * 2
    else:
        add("capsule", (-0.46, 0.20, 0.44), (0.072, 0.28, 0.072), dark,
            rot=(0, 0.9, 1.5))

    # --- legs
    if curled:
        for hx, hz in ((0.30, 0.30), (0.30, -0.26), (-0.34, 0.30)):
            add("capsule", (hx, 0.16, hz), (0.095, 0.15, 0.095), coat,
                rot=(0, 0, 1.45))
            add("sphere", (hx + 0.20, 0.13, hz), (0.115, 0.055, 0.095), cream)
    else:
        for i, (hx, hz) in enumerate(((0.40, 0.25), (0.40, -0.25),
                                      (-0.42, 0.27), (-0.42, -0.27))):
            px, py = on_spine(hx, spine_y - 0.08)
            _limb(add, px, py, hz, swings[i], bends[i], coat, cream)

    # --- torso: one lathed surface from rump to neck
    # Radii sampled along the spine: deep chest, a waist that tucks, a round
    # haunch, and a neck that narrows into the skull. This is the shape the
    # three-ellipsoid version could only approximate from outside.
    prof = [(-0.64, 0.02, 0.17, 0.16), (-0.52, 0.03, 0.33, 0.31),
            (-0.34, 0.02, 0.41, 0.37), (-0.12, 0.00, 0.40, 0.355),
            (0.10, -0.01, 0.375, 0.34), (0.30, 0.00, 0.39, 0.35),
            (0.46, 0.02, 0.36, 0.33), (0.58, 0.08, 0.28, 0.26),
            (0.68, 0.17, 0.21, 0.20), (0.76, 0.26, 0.17, 0.165)]
    stations = []
    for sx, sy, ry, rz in prof:
        cx, cy = on_spine(sx, spine_y + sy)
        stations.append((cx, cy, ry * squash * (breath if sx > -0.3 else 1.0),
                         rz * (breath if sx > -0.3 else 1.0)))
    add("body", (0, 0, 0), (1, 1, 1), coat, mark=mark_body)
    items[-1].data = tube(stations)

    # --- head
    hs = HEAD_S if not cat else (0.40, 0.40, 0.40)
    skull = 0.40 if not cat else 0.42
    add("head", tuple(head_c), (skull, skull, skull), coat,
        rot=(0, 0, head_pitch), mark=mark_head)
    items[-1].data = head_mesh()
    hm = _trs(tuple(head_c), (skull, skull, skull), (0, 0, head_pitch))

    def at_head(obj, scale, colour, mesh="sphere", mark=PLAIN, rot=(0, 0, 0),
                outline=ink_g):
        """Place a solid in head units. The head is a unit-radius lathe now,
        so an object at |obj| = 1 sits on the skull's surface."""
        local = _trs(obj, tuple(v / skull for v in scale), rot)
        items.append(Item(mesh, root @ hm @ local, colour, mark, outline))

    # nose on the tip of the lathe, and the mouth line under it
    at_head((1.86, -0.36, 0.0), (0.085, 0.072, 0.082), (24, 20, 20))

    for flip in (1, -1):
        e = EYE_OBJ * np.array([1.0, 1.0, flip], "f4")
        if curled:
            at_head(e * 0.99, (0.055, 0.012, 0.055), (40, 34, 32), outline=0.0)
        else:
            # Almond rather than round, and darker: the reference eye is a
            # deep brown that reads as black until the light catches it.
            # It has to break the skull's surface to read at all, but only on
            # the near side - pushed proud on the far side it floats free of
            # the silhouette as a white bead hanging beside the head. The
            # camera sits on the animal's +z side in every scene here.
            # The steps between iris, pupil and catchlight are sized against
            # their own radii, not guessed: one unit of `e` is about 0.37 in
            # world, so a step of 0.035 moved the pupil 13 thousandths - it
            # sat entirely inside a sphere 0.076 across and never showed.
            # Only the near eye gets a pupil and a catchlight. Stacked on the
            # far side they clear the skull and hang in the air beside the
            # head; from three-quarters that eye is barely seen anyway.
            if flip > 0:
                at_head(e * 1.00, (0.070, 0.080, 0.070), (46, 30, 20),
                        outline=0.0)
                at_head(e * 1.10, (0.044, 0.049, 0.044), (16, 12, 11),
                        outline=0.0)
                at_head(e * 1.14 + np.array([0.04, 0.05, 0.0], "f4"),
                        (0.019, 0.021, 0.019), (255, 255, 255), outline=0.0)
            else:
                at_head(e * 0.96, (0.064, 0.072, 0.064), (26, 18, 14),
                        outline=0.0)
        ear_h = 0.34 if not cat else 0.30
        ear_rot = (0.30 * flip, -0.12 * flip, -0.14)
        at_head((-0.30, 0.86, 0.52 * flip), (0.125, ear_h, 0.042), dark,
                mesh="wedge", rot=ear_rot)
        at_head((-0.28, 0.84, 0.48 * flip), (0.082, ear_h * 0.78, 0.028),
                tan, mesh="wedge", rot=ear_rot, outline=0.0)

    # Whiskers. Three a side, and they cost almost nothing - the reference
    # dog has a face full of them and they are most of why a muzzle reads as
    # fur rather than moulded plastic.
    for flip in (1, -1):
        for wy, spread in ((-0.10, 0.34), (-0.26, 0.18), (-0.40, 0.02)):
            at_head((1.35, wy, 0.40 * flip), (0.115, 0.007, 0.007),
                    (46, 40, 38), rot=(0.0, -0.55 * flip, spread), outline=0.0)

    if mouth > 0.02:
        jaw = 1.30
        if mouth > 0.30:
            at_head((jaw, -0.62 - mouth * 0.40, 0.0),
                    (0.150, 0.035 + mouth * 0.095, 0.105), t2d.MOUTH,
                    rot=(0, 0, 0.16), outline=ink_g * 0.8)
            # A row of teeth along the upper lip: the reference dog's smile
            # is teeth and tongue together, and the tongue alone reads gummy.
            at_head((jaw + 0.22, -0.46 - mouth * 0.06, 0.0),
                    (0.100, 0.013, 0.082), (248, 244, 236),
                    rot=(0, 0, 0.14), outline=0.0)
        at_head((jaw + 0.30, -0.66 - mouth * 0.40, 0.0),
                (0.090, 0.026 + mouth * 0.035, 0.062), t2d.TONGUE,
                rot=(0, 0, 0.24), outline=ink_g * 0.55)

    if not curled:
        cx, cy = on_spine(0.56, spine_y + 0.22)
        add("sphere", (cx, cy, 0.0), (0.225, 0.05, 0.225), t2d.ACCENT,
            rot=(0, 0, pitch - 0.55 + math.pi / 2), outline=0.012)
        add("sphere", (cx + 0.10, cy - 0.20, 0.0), (0.06, 0.06, 0.035),
            t2d.LAMP, outline=0.010)


# --- sets -------------------------------------------------------------------
# The same three places the 2D backend has, built as geometry. Furniture is
# boxes and capsules: at this level of stylisation a sofa is a seat, a back,
# two arms and four legs, and modelling it any further would only fight the
# ink outline.

# The pose hangs its paws below its own origin; lifting the root by exactly
# that much is what puts them on the floor instead of through it or above it.
DOG_LIFT = FOOT_DROP - HIP_Y
INK = (34, 30, 28)
WALL_Z = -2.6


def contact_shadow(items: list[Item], x: float, z: float, rx: float = 0.95,
                   rz: float = 0.62, tone=(120, 104, 86)) -> None:
    """A soft dark patch on the floor under the character.

    There is no shadow map here and there does not need to be one: a single
    flattened, blended ellipse is what stops a lit object looking pasted on
    top of the floor, and it costs one draw.
    """
    items.append(Item("sphere", _trs((x, 0.014, z), (rx, 0.012, rz)), tone,
                      outline=0.0, alpha=0.26))


def _dog_root(x: float = 0.0, z: float = 0.0, yaw: float = -0.21) -> np.ndarray:
    """Place the dog, turned toward the camera.

    The default yaw is the one thing 2D could never do: the reference photo
    is a three-quarter view, and a flat side-on drawing cannot be one. Here
    it is a number.
    """
    return _trs((x, DOG_LIFT, z), rot=(0, yaw, 0))


def _floor(items, colour, size=8.0, y=0.0):
    items.append(Item("box", _trs((0, y - 0.1, 0), (size, 0.1, size)), colour))


def room(items: list[Item], night: bool = False) -> None:
    wall = t2d.NIGHT_BOT if night else t2d.WALL_BOT
    dado = (56, 64, 92) if night else t2d.DADO
    floor = t2d.NIGHT_FLOOR if night else t2d.FLOOR
    _floor(items, floor)
    items.append(Item("box", _trs((0, 2.1, WALL_Z), (10.0, 2.1, 0.1)), wall))
    items.append(Item("box", _trs((0, 0.62, WALL_Z + 0.11), (10.0, 0.62, 0.02)),
                      dado))
    items.append(Item("box", _trs((0, 1.26, WALL_Z + 0.13), (10.0, 0.035, 0.03)),
                      (70, 78, 108) if night else t2d.RAIL))
    items.append(Item("box", _trs((0, 0.06, WALL_Z + 0.14), (10.0, 0.06, 0.04)),
                      (70, 78, 108) if night else t2d.RAIL))
    # rug
    items.append(Item("box", _trs((0.25, 0.012, 0.45), (1.5, 0.012, 1.05)),
                      (74, 92, 108) if night else t2d.RUG))
    for dz in (-0.45, 0.45):
        items.append(Item("box", _trs((0.25, 0.026, 0.45 + dz),
                                      (1.5, 0.004, 0.14)),
                          (92, 108, 124) if night else t2d.RUG_LINE))
    sofa(items, -3.55, night)
    window(items, -1.05, night)
    for i, (x, y, w, hgt, col) in enumerate((
            (-2.60, 1.95, 0.30, 0.38, t2d.ACCENT),
            (-1.95, 1.88, 0.22, 0.24, t2d.LEAF_DARK))):
        items.append(Item("box", _trs((x, y, WALL_Z + 0.12),
                                      (w + 0.035, hgt + 0.035, 0.02)),
                          (96, 102, 134) if night else t2d.FRAME))
        items.append(Item("box", _trs((x, y, WALL_Z + 0.15), (w, hgt, 0.02)),
                          (108, 118, 146) if night else t2d.MAT))
        items.append(Item("box", _trs((x, y - hgt * 0.15, WALL_Z + 0.17),
                                      (w * 0.5, hgt * 0.45, 0.02)),
                          (122, 132, 160) if night else col))
    plant(items, 1.45, night)


def sofa(items: list[Item], x: float, night: bool) -> None:
    body = (102, 116, 150) if night else t2d.SOFA
    dark = (82, 94, 126) if night else t2d.SOFA_DARK
    z = WALL_Z + 0.75
    items.append(Item("box", _trs((x, 0.46, z), (0.92, 0.10, 0.42)), body,
                      outline=0.012))
    items.append(Item("box", _trs((x, 0.72, z - 0.34), (0.92, 0.34, 0.09)),
                      dark, outline=0.012))
    for dz in (-0.36, 0.36):
        items.append(Item("box", _trs((x, 0.56, z + dz), (0.92, 0.20, 0.07)),
                          dark, outline=0.012))
    for dx in (-0.78, 0.78):
        for dz in (-0.30, 0.30):
            items.append(Item("capsule",
                              _trs((x + dx, 0.18, z + dz), (0.038, 0.09, 0.038)),
                              (74, 78, 104) if night else (150, 116, 84),
                              outline=0.010))
    for dx, col in ((-0.40, t2d.OCHRE), (0.34, t2d.ACCENT)):
        items.append(Item("box", _trs((x + dx, 0.64, z - 0.22),
                                      (0.20, 0.17, 0.07),
                                      (0, 0, 0.12)),
                          (96, 88, 124) if night else col, outline=0.010))


def window(items: list[Item], x: float, night: bool) -> None:
    items.append(Item("box", _trs((x, 1.95, WALL_Z + 0.12), (0.80, 0.70, 0.03)),
                      (70, 78, 108) if night else t2d.RAIL))
    items.append(Item("box", _trs((x, 1.95, WALL_Z + 0.15), (0.73, 0.63, 0.02)),
                      (34, 42, 74) if night else t2d.GLASS_TOP, outline=0.0))
    items.append(Item("box", _trs((x, 1.95, WALL_Z + 0.17), (0.016, 0.63, 0.02)),
                      INK, outline=0.0))
    items.append(Item("box", _trs((x, 1.22, WALL_Z + 0.22), (0.88, 0.035, 0.11)),
                      (70, 78, 108) if night else t2d.RAIL))
    if night:
        items.append(Item("sphere", _trs((x + 0.22, 2.22, WALL_Z + 0.16),
                                         (0.09, 0.09, 0.01)), t2d.LAMP,
                          outline=0.0))
    else:
        items.append(Item("sphere", _trs((x + 0.36, 2.28, WALL_Z + 0.16),
                                         (0.14, 0.14, 0.01)), (255, 246, 214),
                          outline=0.0))
        items.append(Item("sphere", _trs((x - 0.16, 1.62, WALL_Z + 0.16),
                                         (0.36, 0.22, 0.01)), t2d.LEAF,
                          outline=0.0))


def plant(items: list[Item], x: float, night: bool) -> None:
    pot = (130, 136, 164) if night else t2d.POT
    leafc = (64, 96, 82) if night else t2d.LEAF_DARK
    leaf2 = (82, 116, 98) if night else t2d.LEAF
    z = WALL_Z + 0.9
    items.append(Item("capsule", _trs((x, 0.20, z), (0.23, 0.10, 0.23)), pot,
                      outline=0.012))
    for ang, ln, lean in ((0.3, 0.46, 0.5), (1.9, 0.40, 0.42), (3.4, 0.44, 0.55),
                          (4.8, 0.36, 0.38), (5.7, 0.30, 0.30)):
        dx, dz = math.cos(ang) * ln * 0.6, math.sin(ang) * ln * 0.6
        items.append(Item("sphere",
                          _trs((x + dx, 0.62 + ln * 0.6, z + dz),
                               (0.22, 0.06, 0.13), (0, -ang, lean)),
                          leaf2 if ang < 3 else leafc, outline=0.010))
        items.append(Item("capsule",
                          _trs((x + dx * 0.5, 0.48 + ln * 0.3, z + dz * 0.5),
                               (0.018, ln * 0.4, 0.018), (0, 0, -lean * 0.5)),
                          leafc, outline=0.008))


def park(items: list[Item]) -> None:
    _floor(items, t2d.LEAF, size=60.0)
    # Distant hills: a flat green plane meeting the sky is a green wall, and
    # the horizon is most of what says "outside" in a drawn landscape.
    for hx, hz, r, tone in ((-9.0, -22.0, 7.5, (150, 184, 162)),
                            (5.0, -26.0, 9.0, (166, 196, 176)),
                            (17.0, -21.0, 6.5, (150, 184, 162))):
        items.append(Item("sphere", _trs((hx, -0.6, hz), (r, r * 0.42, r)),
                          tone, outline=0.0))
    for x, z, s in ((-2.6, -3.4, 1.0), (2.6, -4.2, 0.86), (4.4, -1.8, 0.74),
                    (-5.2, -5.4, 1.15), (0.2, -7.5, 1.3)):
        items.append(Item("capsule", _trs((x, 0.62 * s, z),
                                          (0.10 * s, 0.62 * s, 0.10 * s)),
                          t2d.BARK, outline=0.014))
        for dx, dy, r in ((0, 1.55, 0.62), (-0.40, 1.28, 0.44), (0.42, 1.30, 0.46)):
            items.append(Item("sphere",
                              _trs((x + dx * s, dy * s, z + dx * 0.3 * s),
                                   (r * s, r * 0.86 * s, r * s)),
                              t2d.LEAF_DARK if dx else t2d.LEAF, outline=0.014))
    for x, z, r in ((-1.7, -2.2, 0.30), (2.1, -1.9, 0.26), (-3.6, -1.3, 0.22),
                    (3.4, -3.1, 0.24), (-0.6, -3.6, 0.20)):
        items.append(Item("sphere", _trs((x, r * 0.7, z), (r, r * 0.7, r)),
                          t2d.LEAF_DARK, outline=0.012))


def clinic(items: list[Item]) -> None:
    _floor(items, (226, 232, 234))
    items.append(Item("box", _trs((0, 2.1, WALL_Z), (10.0, 2.1, 0.1)),
                      (232, 240, 238)))
    items.append(Item("box", _trs((0, 0.06, WALL_Z + 0.12), (4.6, 0.06, 0.04)),
                      t2d.RAIL))
    # cabinet
    items.append(Item("box", _trs((-2.6, 0.55, WALL_Z + 0.55),
                                  (0.75, 0.55, 0.38)), (246, 248, 248),
                      outline=0.012))
    items.append(Item("box", _trs((-2.6, 1.13, WALL_Z + 0.55),
                                  (0.80, 0.035, 0.42)), (186, 200, 202),
                      outline=0.012))
    for k, col in enumerate((t2d.LEAF, t2d.OCHRE)):
        items.append(Item("box", _trs((-2.85 + k * 0.42, 1.26, WALL_Z + 0.55),
                                      (0.10, 0.12, 0.10)), col, outline=0.010))
    # the cross, on clear wall
    for sx, sy in ((0.09, 0.26), (0.26, 0.09)):
        items.append(Item("box", _trs((-0.6, 2.05, WALL_Z + 0.12),
                                      (sx, sy, 0.03)), t2d.ACCENT,
                          outline=0.010))
    # exam table
    items.append(Item("box", _trs((0.3, 0.60, 0.1), (1.15, 0.05, 0.62)),
                      (238, 242, 244), outline=0.012))
    for dx in (-0.95, 0.95):
        for dz in (-0.45, 0.45):
            items.append(Item("capsule", _trs((0.3 + dx, 0.28, 0.1 + dz),
                                              (0.046, 0.14, 0.046)),
                              (188, 196, 200), outline=0.012))


def dog_bed(items: list[Item], x: float, z: float, night: bool) -> None:
    shell = (76, 82, 110) if night else (118, 126, 138)
    pad = (96, 102, 132) if night else (238, 232, 220)
    items.append(Item("sphere", _trs((x, 0.13, z), (0.95, 0.13, 0.72)), shell,
                      outline=0.014))
    items.append(Item("sphere", _trs((x, 0.17, z), (0.78, 0.10, 0.56)), pad,
                      outline=0.0))


def carton(items: list[Item], x: float, z: float) -> None:
    w, d, h = 0.78, 0.62, 0.52
    for dx, dz, sx, sz in ((0, -d, w, 0.03), (0, d, w, 0.03),
                           (-w, 0, 0.03, d), (w, 0, 0.03, d)):
        items.append(Item("box", _trs((x + dx, h, z + dz), (sx, h, sz)),
                          t2d.CARD, outline=0.013))
    items.append(Item("box", _trs((x, 0.03, z), (w, 0.03, d)), t2d.CARD_DARK))
    for dz in (-1, 1):
        items.append(Item("box", _trs((x, h * 1.7, z + dz * (d + 0.18)),
                                      (w, 0.02, 0.22), (dz * 0.9, 0, 0)),
                          t2d.CARD_DARK, outline=0.012))


def mirror(items: list[Item], x: float, z: float) -> Any:
    """A standing mirror, and the plane to reflect the dog through.

    Real reflection, not a second drawing: the same pose is rendered again
    through a reflection matrix about the glass, which is the kind of thing
    that is free in 3D and was a separate clipped redraw in 2D.
    """
    for px, py, sx, sy in ((0, 1.92, 0.72, 0.06), (0, 0.04, 0.72, 0.06),
                           (-0.68, 0.98, 0.05, 0.95), (0.68, 0.98, 0.05, 0.95)):
        items.append(Item("box", _trs((x + px, py, z), (sx, sy, 0.06)),
                          t2d.FRAME, outline=0.012))
    refl = np.eye(4, dtype="f4")
    refl[2, 2] = -1.0
    refl[2, 3] = 2 * (z + 0.05)
    return refl, Item("box", _trs((x, 0.92, z + 0.06), (0.62, 0.86, 0.01)),
                      t2d.GLASS_TOP, outline=0.0, alpha=0.30)


# --- shots ------------------------------------------------------------------

class Shot:
    __slots__ = ("items", "target", "yaw", "elev", "fov", "half", "sky",
                 "bg", "reflect", "root")

    def __init__(self, items, target=(0.0, 0.62, 0.0), yaw=40.0, elev=12.0,
                 fov=34.0, half=1.35, sky=None, bg=None, reflect=None,
                 root=None):
        self.items = items
        self.target = target
        self.yaw = yaw
        self.elev = elev
        self.fov = fov
        self.half = half
        self.sky = sky
        self.bg = bg
        self.reflect = reflect
        # The character's own frame, so the shader can put the saddle and the
        # bib on the torso as a whole rather than on each mass separately.
        self.root = root


def _room_shot(t, species, pose, night=False, **kw) -> Shot:
    items: list[Item] = []
    room(items, night)
    root = _dog_root(0.15, 0.35)
    contact_shadow(items, 0.30, 0.30,
                   tone=(40, 44, 70) if night else (120, 104, 86))
    dog(items, t, pose, root, species)
    kw.setdefault("half", 1.55)
    kw.setdefault("target", (0.2, 0.66, 0.3))
    return Shot(items, root=root,
                bg=t2d.NIGHT_TOP if night else t2d.WALL_TOP, **kw)


def _park_shot(t, species, pose, **kw) -> Shot:
    items: list[Item] = []
    park(items)
    root = _dog_root(0.15, 0.6)
    contact_shadow(items, 0.30, 0.55, tone=(64, 96, 70))
    dog(items, t, pose, root, species)
    kw.setdefault("half", 1.6)
    kw.setdefault("target", (0.2, 0.66, 0.55))
    return Shot(items, root=root, sky=(t2d.SKY_TOP, t2d.SKY_BOT), **kw)


def scene_stand_room(t, sp):
    return _room_shot(t, sp, "stand")


def scene_stand_park(t, sp):
    return _park_shot(t, sp, "stand")


def scene_sprint_room(t, sp):
    return _room_shot(t, sp, "run", half=1.55)


def scene_sprint_night(t, sp):
    return _room_shot(t, sp, "run", night=True, half=1.55)


def scene_sprint_park(t, sp):
    return _park_shot(t, sp, "run", half=1.6)


def scene_bark(t, sp):
    return _room_shot(t, sp, "bark", yaw=36.0)


def scene_playbow(t, sp):
    return _room_shot(t, sp, "bow", half=1.5)


def scene_sniff(t, sp):
    return _room_shot(t, sp, "sniff", elev=16.0)


def scene_sniff_park(t, sp):
    return _park_shot(t, sp, "sniff", elev=16.0)


def scene_sleep(t, sp):
    return _room_shot(t, sp, "sleep", elev=17.0, half=1.3)


def scene_sleep_night(t, sp):
    return _room_shot(t, sp, "sleep", night=True, elev=17.0, half=1.3)


def scene_closeup(t, sp):
    head = head_world(_dog_root(0.15, 0.35), "stand")
    return _room_shot(t, sp, "stand", target=head, yaw=38.0, elev=4.0,
                      half=0.60)


def scene_closeup_sleep(t, sp):
    head = head_world(_dog_root(0.15, 0.35), "sleep")
    return _room_shot(t, sp, "sleep", night=True, target=head, yaw=38.0,
                      elev=8.0, half=0.62)


def scene_bed(t, sp, night=False):
    items: list[Item] = []
    room(items, night)
    dog_bed(items, 0.2, 0.45, night)
    bed_root = _trs((0.1, DOG_LIFT + 0.10, 0.45), rot=(0, -0.3, 0))
    dog(items, t, "sleep", bed_root, sp)
    return Shot(items, root=bed_root, elev=16.0, half=1.25,
                bg=t2d.NIGHT_TOP if night else t2d.WALL_TOP)


def scene_snore(t, sp):
    return scene_bed(t, sp, night=True)


def scene_box(t, sp):
    items: list[Item] = []
    room(items)
    box_root = _trs((0.16, DOG_LIFT - 0.26, 0.62), rot=(0, -0.24, 0))
    dog(items, t, "sit", box_root, sp)
    carton(items, 0.2, 0.5)
    return Shot(items, root=box_root, target=(0.35, 0.85, 0.5), elev=10.0,
                half=1.05,
                bg=t2d.WALL_TOP)


def scene_vet(t, sp):
    items: list[Item] = []
    clinic(items)
    vet_root = _trs((0.3, 0.65 + DOG_LIFT, 0.1), rot=(0, -0.22, 0))
    dog(items, t, "stand", vet_root, sp)
    return Shot(items, root=vet_root, target=(0.3, 1.30, 0.1), yaw=42.0,
                elev=8.0, half=1.30,
                bg=(232, 240, 238))


def _mirror_shot(t, sp, pose):
    items: list[Item] = []
    room(items)
    # The glass sits directly behind the dog in z, so the reflection lands
    # inside the frame rather than beside it.
    refl, glass = mirror(items, 0.55, -1.15)
    root = _dog_root(0.55, 0.30, yaw=0.95)
    reflected: list[Item] = []
    dog(reflected, t, pose, refl @ root, sp)
    items.extend(reflected)
    items.append(glass)
    dog(items, t, pose, root, sp)
    return Shot(items, root=root, target=(0.05, 0.84, -0.2), yaw=34.0,
                elev=9.0, half=1.5, bg=t2d.WALL_TOP)


def scene_mirror(t, sp):
    return _mirror_shot(t, sp, "bark")


def scene_mirror_bow(t, sp):
    return _mirror_shot(t, sp, "bow")


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


# --- drawing ----------------------------------------------------------------

LIGHT = (-0.42, 0.88, 0.62)


def _normal_matrix(m: np.ndarray) -> np.ndarray:
    return np.linalg.inv(m[:3, :3]).T.astype("f4")


def draw(shot: Shot, w: int, h: int, zoom: float = 1.0,
         drift: float = 0.0) -> bytes:
    gl = _gpu()
    ctx = gl.ctx
    ms, fbo = gl.target(w, h)
    ms.use()
    bg = shot.bg or t2d.PAPER
    ctx.clear(bg[0] / 255, bg[1] / 255, bg[2] / 255, 1.0)

    if shot.sky:
        ctx.disable(moderngl.DEPTH_TEST)
        gl.sky["top"].value = _rgb(shot.sky[0])
        gl.sky["bottom"].value = _rgb(shot.sky[1])
        gl.sky_vao.render()
    ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

    aspect = w / h
    fov = shot.fov / zoom
    dist = _distance_for(shot.half, fov, aspect)
    # In a 9:16 frame the subject wants to sit low, under the caption card;
    # in 16:9 it wants the middle. One expression covers both.
    lift = max(0.0, 0.55 * (1.0 - aspect)) * shot.half
    target = (shot.target[0], shot.target[1] + lift, shot.target[2])
    yaw = math.radians(shot.yaw + drift)
    el = math.radians(shot.elev)
    eye = (target[0] + dist * math.cos(el) * math.sin(yaw),
           target[1] + dist * math.sin(el),
           target[2] + dist * math.cos(el) * math.cos(yaw))
    viewproj = _persp(fov, aspect) @ _look_at(eye, target)

    prog = gl.prog
    prog["ldir"].value = LIGHT
    prog["eye"].value = eye
    prog["ink"].value = _rgb(INK)
    prog["ambient"].value = 0.46
    prog["mark_dark"].value = _rgb(t2d.MARK_DARK)
    prog["mark_mid"].value = _rgb(MASK_MID)
    prog["mark_tan"].value = _rgb(t2d.MARK_TAN)
    prog["mark_cream"].value = _rgb(t2d.MARK_CREAM)

    root_inv = np.linalg.inv(shot.root) if shot.root is not None else np.eye(4, dtype="f4")
    solid = [i for i in shot.items if i.alpha >= 1.0]
    glassy = [i for i in shot.items if i.alpha < 1.0]
    vaos = {id(i): (gl.upload(i.data) if i.data is not None
                    else gl.meshes[i.mesh]) for i in shot.items}

    def setup(item):
        prog["model"].write(item.m.T.tobytes())
        prog["nmat"].write(_normal_matrix(item.m).T.tobytes())
        prog["viewproj"].write(viewproj.T.astype("f4").tobytes())
        prog["rootinv"].write(root_inv.T.astype("f4").tobytes())
        prog["base"].value = item.colour
        prog["mark"].value = item.mark
        prog["alpha"].value = item.alpha
        # Near-even weight. Scaling the line with the part looked reasonable
        # in theory and in practice gave the ears and the muzzle a line too
        # fine to survive the encoder, so the ramp is narrow now.
        size = float(np.mean(np.linalg.norm(item.m[:3, :3], axis=0)))
        flip = np.linalg.det(item.m[:3, :3]) < 0
        return (item.outline * min(1.25, max(0.85, 0.75 + size * 0.35)), flip)

    # Every hull, then every surface - not hull-then-surface per part. A body
    # built from overlapping masses gets an ink line around each one if they
    # are interleaved, which draws the character as a string of sausages.
    # Run as two passes and each mass's hull is covered by the neighbour that
    # overlaps it, leaving ink only on the true silhouette.
    prog["outline"].value = 1
    for item in solid:
        if item.outline <= 0.0:
            continue
        grow, flip = setup(item)
        prog["grow"].value = grow
        ctx.cull_face = "back" if flip else "front"
        vaos[id(item)].render()
    prog["outline"].value = 0
    prog["grow"].value = 0.0
    for item in solid:
        _, flip = setup(item)
        prog["grow"].value = 0.0
        prog["outline"].value = 0
        ctx.cull_face = "front" if flip else "back"
        vaos[id(item)].render()

    for item in glassy:
        ctx.enable(moderngl.BLEND)
        grow, flip = setup(item)
        if item.outline > 0.0:
            prog["outline"].value = 1
            prog["grow"].value = grow
            ctx.cull_face = "back" if flip else "front"
            vaos[id(item)].render()
        prog["outline"].value = 0
        prog["grow"].value = 0.0
        ctx.cull_face = "front" if flip else "back"
        vaos[id(item)].render()
        ctx.disable(moderngl.BLEND)

    ctx.copy_framebuffer(fbo, ms)
    gl.sweep()
    return fbo.read(components=4)


def render_clip(query: str, seconds: float, cfg: dict[str, Any], dest) -> Any:
    """Draw `seconds` of 3D animation for `query` and encode it to `dest`.

    Same contract as the Skia backend, and the same scene and species
    lookup, so a script does not know or care which one drew it. The finished
    GL frame is handed to Skia for the caption card and the grain, which are
    2D jobs whichever way the picture was made.
    """
    dest = Path(dest)
    w = int(cfg["video"]["width"])
    h = int(cfg["video"]["height"])
    fps = int(cfg["video"]["fps"])
    species = toon.species_for(query, str(cfg["visuals"].get("toon_dog",
                                                             "greydog")))
    scene = SCENES.get(toon.scene_for(query)) or SCENES["stand_room"]

    frames = dest.parent / f"_{dest.stem}_frames"
    if frames.exists():
        for old in frames.glob("*.png"):
            old.unlink()
    frames.mkdir(parents=True, exist_ok=True)

    text = str(cfg.get("_shot_text") or "")
    clock = float(cfg.get("_toon_offset") or 0.0)
    cam_phase = float(cfg.get("_toon_cam") or 0.0)
    index = int(cfg.get("_shot_index") or 0)
    grain = int(cfg["video"].get("toon_grain", 15))

    surface = skia.Surface(w, h)
    total = max(2, int(round(seconds * fps)))
    for i in range(total):
        progress = i / max(1, total - 1)
        t = clock + i / fps
        # The same slow push and drift the 2D camera has, applied to a real
        # camera: the zoom is the field of view and the drift is the orbit
        # angle, so the parallax is true instead of a scaled bitmap.
        e = t2d.smooth(progress)
        phase = (cam_phase + 0.5 * e) % 1.0
        zoom = 1.0 + 0.055 * t2d.there_and_back(phase)
        drift = (3.0 if index % 4 < 2 else -3.0) * e
        pixels = draw(scene(t, species), w, h, zoom, drift)
        arr = np.frombuffer(pixels, np.uint8).reshape(h, w, 4)[::-1]
        img = skia.Image.fromarray(np.ascontiguousarray(arr))
        canvas = surface.getCanvas()
        canvas.drawImage(img, 0, 0)
        if grain:
            t2d.grain(canvas, w, h, grain)
        t2d.banner(canvas, w, h, text, progress)
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
