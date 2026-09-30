"""Paint burned-in text out of a video, so the footage underneath can be reused.

The channel's own TikToks carry captions baked into the picture ("Theo dõi kênh
con nha cô chú ^^" across Min's chest), and no caption may reach a new video.
A burned-in caption stays put while the camera and the dog move under it, so
its pixels are the ones that stay near-white in every frame: the mask is found
from the clip itself, not drawn by hand.

    python inpaint.py <in.mp4> <out.mp4> [--box x0,y0,x1,y1] [--thresh 225] [--grow 5]

`--box` limits the search to where the text is (a white wall that never moves
would otherwise count as text). Each frame is filled by LaMa (`inpaint.model`
in config.yaml) in a 512 px window around the mask - the model's native size,
so the fur it invents is at the footage's own resolution - and blended back
through a feathered mask, so nothing outside the text is touched. The mask is
saved next to the output as `<out>_mask.png` to check before using the clip.
"""
import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from pipeline.config import load_config

SIDE = 512  # LaMa's native input size


def probe(path: Path) -> tuple[int, int, str]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout.strip().split(",")
    return int(out[0]), int(out[1]), out[2]


def frames(path: Path, w: int, h: int):
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE)
    size = w * h * 3
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    proc.wait()


def static_text_mask(path: Path, w: int, h: int, box, thresh: int, grow: int) -> np.ndarray:
    """Pixels inside `box` that are near-white in every single frame, grown to take the text's edge and shadow."""
    floor = None
    for f in frames(path, w, h):
        lo = f.min(axis=2)
        floor = lo if floor is None else np.minimum(floor, lo)
    hit = np.zeros((h, w), bool)
    x0, y0, x1, y1 = box
    hit[y0:y1, x0:x1] = floor[y0:y1, x0:x1] >= thresh
    img = Image.fromarray(hit.astype(np.uint8) * 255)
    if grow:
        img = img.filter(ImageFilter.MaxFilter(2 * grow + 1))
    return np.asarray(img) > 127


def window(mask: np.ndarray, w: int, h: int) -> tuple[int, int, int]:
    """A square around the mask: SIDE px where it fits, larger (and scaled down for the model) where it does not."""
    ys, xs = np.nonzero(mask)
    side = max(SIDE, int(xs.max() - xs.min()) + 64, int(ys.max() - ys.min()) + 64)
    side = min(side, w, h)
    cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
    x = int(np.clip(cx - side // 2, 0, w - side))
    y = int(np.clip(cy - side // 2, 0, h - side))
    return x, y, side


def load_model(cfg: dict):
    import onnxruntime as ort
    icfg = cfg.get("inpaint", {})
    model = Path(icfg.get("model", "~/.cache/lama/lama_fp32.onnx")).expanduser()
    if not model.exists():
        url = icfg.get("model_url")
        if not url:
            sys.exit(f"no LaMa model at {model} and no inpaint.model_url to fetch it from")
        print(f"downloading LaMa to {model}")
        model.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, model)
    return ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])


def fill(sess, crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    img = crop.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    m = mask.astype(np.float32)[None, None]
    out = sess.run(None, {"image": img, "mask": m})[0][0].transpose(1, 2, 0)
    if out.max() <= 1.5:  # some exports return 0-1, others 0-255
        out = out * 255.0
    return np.clip(out, 0, 255)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--box", help="x0,y0,x1,y1 to search for the text (default: whole frame)")
    ap.add_argument("--thresh", type=int, default=225, help="darkest channel a text pixel keeps in every frame")
    ap.add_argument("--grow", type=int, default=5, help="px added round the text for its edge and shadow")
    args = ap.parse_args()

    w, h, rate = probe(args.src)
    box = tuple(int(v) for v in args.box.split(",")) if args.box else (0, 0, w, h)
    mask = static_text_mask(args.src, w, h, box, args.thresh, args.grow)
    if not mask.any():
        sys.exit("no burned-in text found - lower --thresh or widen --box")
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(args.dst.with_name(args.dst.stem + "_mask.png"))
    print(f"mask: {int(mask.sum())} px, bbox x {np.nonzero(mask)[1].min()}-{np.nonzero(mask)[1].max()}"
          f" y {np.nonzero(mask)[0].min()}-{np.nonzero(mask)[0].max()}")

    x, y, side = window(mask, w, h)
    m_crop = mask[y:y + side, x:x + side]
    m_model = np.asarray(Image.fromarray(m_crop.astype(np.uint8) * 255).resize((SIDE, SIDE), Image.NEAREST)) > 127
    # Blend weight: the mask itself fully replaced, fading out over a few px beyond it.
    alpha = np.asarray(Image.fromarray(m_crop.astype(np.uint8) * 255)
                       .filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(2)), np.float32) / 255.0
    alpha = np.maximum(alpha, m_crop)[..., None]

    sess = load_model(load_config())
    enc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", rate,
         "-i", "-", "-i", str(args.src), "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "14",
         "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", str(args.dst)],
        stdin=subprocess.PIPE)
    n = 0
    for f in frames(args.src, w, h):
        crop = f[y:y + side, x:x + side]
        small = crop if side == SIDE else np.asarray(Image.fromarray(crop).resize((SIDE, SIDE), Image.LANCZOS))
        filled = fill(sess, small, m_model)
        if side != SIDE:
            filled = np.asarray(Image.fromarray(filled.astype(np.uint8)).resize((side, side), Image.LANCZOS), np.float32)
        out = f.copy()
        out[y:y + side, x:x + side] = (crop * (1 - alpha) + filled * alpha).round().astype(np.uint8)
        enc.stdin.write(out.tobytes())
        n += 1
        if n % 60 == 0:
            print(f"  {n} frames")
    enc.stdin.close()
    enc.wait()
    print(f"wrote {args.dst} ({n} frames)")


if __name__ == "__main__":
    main()
