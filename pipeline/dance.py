"""Dance videos: one continuous generated take of a dog, synced to a song without a single cut.

The stages `dance.py` runs, each reading and writing `workspace/<slug>/`:

- music: the song's official 30 s iTunes preview (never a rip), its beat grid and downbeat found here, and
  the bars looped into a track at least as long as the video (`music/track.wav`, `music/grid.json`).
- clips: generation lives in `dance.py` (Kaggle Wan chains, or LTX on ZeroGPU); a chain is a list of clips
  where each starts on the frame the one before ended on, so the chain is one unbroken take.
- edit: the take is time-warped, not cut. It plays a little faster or slower (within `dance.speed`), easing
  from beat to beat, so the frames where the dog moves most land on the beats.
- sheet: a contact sheet of every Nth frame, for checking a clip frame by frame before anyone sees it.
"""
from __future__ import annotations

import json
import math
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np


def run(*cmd, **kw):
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def probe(path: Path) -> tuple[int, int, float, float]:
    """(width, height, fps, seconds) of a video."""
    out = run("ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
              "stream=width,height,r_frame_rate:format=duration", "-of", "json", path,
              capture_output=True, text=True).stdout
    info = json.loads(out)
    s = info["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    return int(s["width"]), int(s["height"]), num / den, float(info["format"]["duration"])


# ---------------------------------------------------------------- music

def itunes_preview(query: str, country: str, dest: Path) -> dict:
    """Download the first iTunes result's official preview; prefer a clean (non-explicit) version."""
    url = "https://itunes.apple.com/search?" + urllib.parse.urlencode(
        {"term": query, "entity": "song", "limit": 10, "country": country})
    results = json.loads(urllib.request.urlopen(url, timeout=30).read())["results"]
    results = [r for r in results if r.get("previewUrl")]
    if not results:
        raise SystemExit(f"iTunes has no preview for {query!r} in store {country!r}")
    track = next((r for r in results if r.get("trackExplicitness") != "explicit"), results[0])
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(track["previewUrl"], dest)
    return {"track": track["trackName"], "artist": track["artistName"],
            "explicitness": track.get("trackExplicitness"), "preview_url": track["previewUrl"]}


def beat_grid(song: Path) -> dict:
    """Tempo, downbeat and harmonic cycle of a song.

    The tracker's beats are fitted to one straight grid (a preview is too short for tempo drift). The
    downbeat is the beat phase where the kick and the change of harmony are strongest together - in
    hip-hop the kick lands on alternate beats, so kick alone leaves two candidates. The cycle is how many
    bars the harmony takes to repeat, which is where a loop can join without a hitch.
    """
    import librosa

    wav = song.with_suffix(".analysis.wav")           # libsndfile can't read the preview's AAC
    run("ffmpeg", "-v", "error", "-y", "-i", song, "-ac", "1", "-ar", "22050", wav)
    y, sr = librosa.load(str(wav), sr=None, mono=True)
    _, beats = librosa.beat.beat_track(y=y, sr=sr, units="time")
    k = np.arange(len(beats))
    period, phase = np.polyfit(k, beats, 1)
    phase -= math.floor(phase / period) * period         # first grid beat at or after 0
    n = int((len(y) / sr - phase) / period)
    grid = phase + np.arange(n) * period

    spec = np.abs(librosa.stft(y, n_fft=2048, hop_length=256))
    freqs = librosa.fft_frequencies(sr=sr)
    t_spec = librosa.frames_to_time(np.arange(spec.shape[1]), sr=sr, hop_length=256)
    kick = np.interp(grid, t_spec, spec[(freqs > 30) & (freqs < 120)].sum(0))
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    t_c = librosa.times_like(chroma, sr=sr, hop_length=512)

    def mean_chroma(a: float, b: float) -> np.ndarray:
        m = (t_c >= a) & (t_c < b)
        return chroma[:, m].mean(1) if m.any() else np.zeros(12)

    novelty = np.array([np.linalg.norm(mean_chroma(g, g + period) - mean_chroma(g - period, g))
                        for g in grid])
    z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
    by_phase = [float((z(kick) + z(novelty))[p::4].mean()) for p in range(4)]
    downbeat = float(grid[int(np.argmax(by_phase))])

    bar = 4 * period
    n_bars = int((len(y) / sr - downbeat) / bar)
    vecs = [mean_chroma(downbeat + i * bar, downbeat + (i + 1) * bar) for i in range(n_bars)]
    vecs = [v / (np.linalg.norm(v) + 1e-9) for v in vecs]
    # The cycle whose bars match their repeats best; a longer one has to win clearly (a loop beat is
    # nearly the same every bar, so the 2-bar pattern shows only as 0.99 against 1.00).
    match = {c: float(np.mean([vecs[i] @ vecs[i + c] for i in range(n_bars - c)]))
             for c in (1, 2, 4) if n_bars > c}
    cycle = 1
    for c in sorted(match):
        if match[c] > match[cycle] + 0.003:
            cycle = c
    return {"bpm": round(60 / period, 3), "beat": float(period), "downbeat": downbeat, "bars": n_bars,
            "cycle_bars": cycle, "duration": len(y) / sr}


def loop_track(song: Path, grid: dict, seconds: float, out: Path) -> dict:
    """The song from its downbeat for at least `seconds`, in whole bars.

    A preview is ~30 s; when the video needs more, the full bars run into a repeat of the first ones, joined
    on a bar line that is a multiple of the harmonic cycle, with a 20 ms crossfade.
    """
    bar = 4 * grid["beat"]
    bars_needed = math.ceil(seconds / bar - 1e-6)
    usable = grid["bars"] - grid["bars"] % grid["cycle_bars"]    # end the first pass on a whole cycle
    start = grid["downbeat"]
    pieces = [(start, start + min(usable, bars_needed) * bar)]
    left = bars_needed - min(usable, bars_needed)
    while left > 0:
        take = min(usable, left)
        pieces.append((start, start + take * bar))
        left -= take
    filters, labels = [], []
    for i, (a, b) in enumerate(pieces):
        tail = 0.02 if i < len(pieces) - 1 else 0.0              # overlap eaten by the crossfade
        filters.append(f"[0]atrim=start={a:.4f}:end={b + tail:.4f},asetpts=N/SR/TB[p{i}]")
        labels.append(f"p{i}")
    chain = labels[0]
    for i, lab in enumerate(labels[1:], 1):
        filters.append(f"[{chain}][{lab}]acrossfade=d=0.02:c1=tri:c2=tri[x{i}]")
        chain = f"x{i}"
    out.parent.mkdir(parents=True, exist_ok=True)
    run("ffmpeg", "-v", "error", "-y", "-i", song, "-filter_complex",
        ";".join(filters),
        "-map", f"[{chain}]", "-ar", "44100", "-ac", "2", out)
    return {"track_bars": bars_needed, "track_seconds": bars_needed * bar, "loops": len(pieces) - 1}


# ---------------------------------------------------------------- edit

def motion(path: Path) -> np.ndarray:
    """Mean absolute change between consecutive frames over the dog (the middle of the frame)."""
    raw = run("ffmpeg", "-v", "error", "-i", path, "-vf",
              "crop=iw*0.7:ih*0.8:iw*0.15:ih*0.05,scale=72:128,format=gray", "-f", "rawvideo", "-",
              capture_output=True).stdout
    f = np.frombuffer(raw, np.uint8).reshape(-1, 128, 72).astype(np.float32)
    m = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))
    return np.r_[m[0], m]


def beat_map(m: np.ndarray, rate: float, beat: float, n_beats: int, speed: tuple[float, float],
             skip: float = 0.0) -> np.ndarray:
    """Source frame shown at each of n_beats+1 beats, by dynamic programming.

    Consecutive beats are BEAT*speed apart in the source, speed within `speed`, and the speed changes only
    a little from one beat to the next, so the warp is never felt. The score is the motion at the frames
    that land on beats, after taking out the slow drift so only the rhythm counts.
    """
    m = np.convolve(m, np.ones(5) / 5, "same")
    win = max(1, int(rate))
    m = m - np.convolve(m, np.ones(win) / win, "same")
    n = len(m)
    steps = np.arange(int(round(beat * speed[0] * rate)), int(round(beat * speed[1] * rate)) + 1)
    neg = -1e18
    score = np.full((n_beats + 1, n, len(steps)), neg, np.float32)
    back = np.zeros((n_beats + 1, n, len(steps)), np.int32)
    first = int(skip * rate)
    for s in range(first, min(n, first + int(beat * rate))):
        score[0, s, :] = m[s]
    smooth = 0.6 * m.std()
    jumps = np.arange(len(steps))
    for k in range(1, n_beats + 1):
        prev = score[k - 1]
        for j, st in enumerate(steps):
            pen = prev - smooth * np.abs(jumps - j)[None, :]
            jb = pen.argmax(1)
            vb = pen[np.arange(n), jb]
            tgt = np.arange(st, n)
            score[k, tgt, j] = vb[tgt - st] + m[tgt]
            back[k, tgt, j] = jb[tgt - st]
    last = score[n_beats]
    s, j = np.unravel_index(last.argmax(), last.shape)
    if last[s, j] <= neg / 2:
        need = n_beats * beat * speed[0] + skip
        raise SystemExit(f"the take is too short: {n_beats} beats need at least {need:.1f} s of footage")
    path = [s]
    for k in range(n_beats, 0, -1):
        jp = back[k, s, j]
        s, j = s - steps[j], jp
        path.append(s)
    return np.array(path[::-1], float)


def edit_take(clips: list[Path], grid: dict, track: Path, seconds: float, out: Path, cfg: dict,
              skip: float = 0.0) -> dict:
    """Join `clips` into one take, time-warp it onto the beat, and write out.mp4, _silent and _click."""
    d = cfg["dance"]
    W, H, FPS, UP = d["width"], d["height"], d["fps"], d["interp_fps"]
    work = out.parent / "edit"
    work.mkdir(parents=True, exist_ok=True)
    beat = grid["beat"]
    n_beats = 4 * math.ceil(seconds / (4 * beat) - 1e-6)          # whole bars
    total = n_beats * beat

    # One take at a high frame rate, so a slowed stretch doesn't stutter. Clips of different rates (Wan's
    # 16 fps, LTX's 24) all come out at `interp_fps`.
    (work / "take.txt").write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    take = work / "take.mp4"
    run("ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", work / "take.txt", "-an", "-vf",
        f"fps={UP},scale=trunc(iw/2)*2:trunc(ih/2)*2" if d.get("interp") == "blend" else
        f"minterpolate=fps={UP}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
        "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", take)
    w, h, _, dur = probe(take)

    m = motion(take)
    if d.get("interp") == "blend":
        # Repeated frames read as a spike then zeros, which hides the moves from beat_map: give every copy
        # of a source frame the change that frame brought.
        r = max(1, round(UP / probe(clips[0])[2]))
        m = np.array([m[max(0, i - r + 1):i + r].max() for i in range(len(m))])
    marks =beat_map(m, UP, beat, n_beats, tuple(d["speed"]), skip)
    speeds = np.diff(marks) / (beat * UP)
    on_beat = float(m[marks.astype(int)].mean() / (m.mean() + 1e-9))

    # Tone lock: every frame's colour mean and spread pulled back to the opening second's (the take's most
    # faithful stretch, straight from the photo). A chain still hands a small tone step from link to link and
    # can creep in contrast; per-frame stats, lightly smoothed, take both out without touching the motion.
    lock = None
    if d.get("tone_lock"):
        sw, sh = 96, int(96 * h / w)
        raw = run("ffmpeg", "-v", "error", "-i", take, "-vf", f"scale={sw}:{sh}", "-f", "rawvideo",
                  "-pix_fmt", "rgb24", "-", capture_output=True).stdout
        fr = np.frombuffer(raw, np.uint8).reshape(-1, sh * sw, 3).astype(np.float32)
        mean, std = fr.mean(1), fr.std(1)
        k = max(1, int(UP * 0.1)) | 1
        smooth = lambda a: np.stack([np.convolve(np.pad(a[:, c], k // 2, mode="edge"), np.ones(k) / k, "valid")
                                     for c in range(3)], 1)
        mean, std = smooth(mean), smooth(std)
        lock = (mean, std, mean[:UP].mean(0), std[:UP].mean(0))

    n_out = int(round(total * FPS))
    src = np.interp(np.arange(n_out) / FPS / beat, np.arange(n_beats + 1), marks)
    src = np.clip(np.round(src).astype(int), 0, int(dur * UP) - 1)
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(take), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)   # stopped early on purpose
    warped = work / "warped.mp4"
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-vf",
                            f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H},"
                            f"setsar=1,{d['grade']}",
                            "-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p", str(warped)],
                           stdin=subprocess.PIPE)
    size, cur, frame = w * h * 3, -1, None
    for want in src:
        while cur < want:
            frame = dec.stdout.read(size)
            cur += 1
        if lock is None:
            enc.stdin.write(frame)
        else:
            mean, std, ref_mean, ref_std = lock
            f = np.frombuffer(frame, np.uint8).reshape(h, w, 3).astype(np.float32)
            f = (f - mean[want]) * (ref_std / (std[want] + 1e-6)) + ref_mean
            enc.stdin.write(np.clip(f, 0, 255).round().astype(np.uint8).tobytes())
    enc.stdin.close()
    enc.wait()
    dec.kill()
    dec.wait()

    song = work / "song.wav"
    run("ffmpeg", "-v", "error", "-y", "-i", track, "-t", f"{total:.4f}", "-af",
        f"loudnorm=I={d['loudness']}:TP=-1.5:LRA=11,afade=t=out:st={total - beat:.4f}:d={beat:.4f}",
        "-ar", "44100", "-ac", "2", song)
    stem = out.with_suffix("")
    run("ffmpeg", "-v", "error", "-y", "-i", warped, "-i", song, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out)
    run("ffmpeg", "-v", "error", "-y", "-i", warped, "-f", "lavfi", "-t", f"{total:.4f}",
        "-i", "anullsrc=r=44100:cl=stereo", "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
        "-shortest", "-movflags", "+faststart", f"{stem}_silent.mp4")
    clicks = (f"0.4*sin(2*PI*1000*t)*lt(mod(t\\,{beat:.6f})\\,0.04)"
              f"+0.5*sin(2*PI*1600*t)*lt(mod(t\\,{4 * beat:.6f})\\,0.04)")
    run("ffmpeg", "-v", "error", "-y", "-i", warped, "-i", song, "-f", "lavfi", "-t", f"{total:.4f}",
        "-i", f"aevalsrc={clicks}:s=44100", "-filter_complex",
        "[1]volume=0.5[s];[s][2]amix=inputs=2:normalize=0[a]", "-map", "0:v", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac", "-shortest", f"{stem}_click.mp4")
    return {"seconds": round(total, 2), "beats": n_beats, "take_seconds": round(dur, 2),
            "source_span": [round(marks[0] / UP, 2), round(marks[-1] / UP, 2)],
            "speed": [round(float(speeds.min()), 2), round(float(speeds.max()), 2)],
            "motion_on_beats": round(on_beat, 2)}


# ---------------------------------------------------------------- review

def contact_sheet(clip: Path, out: Path, every: int = 3, cols: int = 8, tile_w: int = 180) -> str:
    """Every `every`-th frame of `clip`, numbered with its time, for a frame-by-frame check."""
    from PIL import Image, ImageDraw

    w, h, fps, _ = probe(clip)
    th = round(tile_w * h / w)
    raw = run("ffmpeg", "-v", "error", "-i", clip, "-vf", f"scale={tile_w}:{th}", "-f", "rawvideo",
              "-pix_fmt", "rgb24", "-", capture_output=True).stdout
    size = tile_w * th * 3
    idx = list(range(0, len(raw) // size, every))
    rows = (len(idx) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, rows * th), "black")
    draw = ImageDraw.Draw(sheet)
    for k, i in enumerate(idx):
        x, y = (k % cols) * tile_w, (k // cols) * th
        sheet.paste(Image.frombytes("RGB", (tile_w, th), raw[i * size:(i + 1) * size]), (x, y))
        draw.text((x + 4, y + 4), f"{i} {i / fps:.2f}s", fill="yellow")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    return f"{len(raw) // size} frames at {fps:.2f} fps, {w}x{h}"
