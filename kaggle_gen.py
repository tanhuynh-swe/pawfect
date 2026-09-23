"""Run clip generation on Kaggle's free GPU and bring the clips back.

The job itself is kaggle/wan_job.py - Wan 2.2 14B image-to-video on a T4.
This fills it in, pushes it as a private Kaggle script, waits, and downloads
its output; only the finished clips (well under a megabyte each) come to
this machine. hf_gen.py calls it with `--kaggle`, and decides which clips to
ask for, so both backends fill the same list of shots:

    python hf_gen.py <slug> --kaggle

Kaggle's free allowance is 30 GPU hours a week; `.venv/bin/kaggle quota`
shows what is left. The job keeps running if this machine sleeps - run the
same command again and it picks the finished output up.
"""
import base64
import json
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KAGGLE = str(ROOT / ".venv" / "bin" / "kaggle")
KERNEL = "pawfect-wan"
POLL_SECONDS = 60


def kaggle(*args: str) -> str:
    done = subprocess.run([KAGGLE, *args], capture_output=True, text=True)
    if done.returncode:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{done.stderr or done.stdout}")
    return done.stdout


def username(settings: dict) -> str:
    if settings.get("username"):
        return settings["username"]
    for line in kaggle("config", "view").splitlines():
        if line.strip().lstrip("- ").startswith("username:"):
            return line.split(":", 1)[1].strip()
    raise SystemExit("Set visuals.kaggle.username in config.yaml")


def status(ref: str) -> str:
    return kaggle("kernels", "status", ref).lower()


def run(jobs: list[dict], images: dict[str, str], cfg: dict, dest: Path) -> dict:
    """Push `jobs` (name, prompt, image key, frames, seed), wait, download.

    `images` maps each key to a base64 JPEG. Each photo travels once, however
    many clips start from it: the job's whole source must stay under 1 MB.

    A job still running from an earlier push is waited on rather than
    replaced, so a sleeping Mac or a closed terminal loses nothing.
    """
    k = cfg["visuals"]["kaggle"]
    ref = f"{username(k)}/{KERNEL}"
    dest.mkdir(parents=True, exist_ok=True)

    try:
        state = status(ref)
    except SystemExit:
        state = ""                 # never pushed yet
    running = "running" in state or "queued" in state
    if "complete" in state and not (dest / "collected").exists():
        # Finished while nobody was waiting (the Mac slept, the terminal
        # closed): its clips are still Kaggle's latest output, so collect
        # them instead of paying GPU hours to make them again. `collected`
        # marks an output already taken, so the next run pushes afresh.
        print(f"  {ref} finished earlier - collecting its output")
        return collect(ref, dest)
    if running:
        print(f"  {ref} is already running - waiting for it instead of pushing")
    else:
        settings = {key: k[key] for key in
                    ("base", "gguf_repo", "gguf_high", "gguf_low", "steps", "fps")}
        enc = lambda obj: base64.b64encode(json.dumps(obj).encode()).decode()
        code = (ROOT / "kaggle" / "wan_job.py").read_text()
        code = (code.replace("__JOBS__", enc(jobs))
                    .replace("__IMAGES__", enc(images))
                    .replace("__SETTINGS__", enc(settings)))
        if len(code.encode()) >= 1_000_000:
            raise SystemExit(f"Kaggle job is {len(code.encode()) / 1e6:.2f} MB; "
                             f"Kaggle's limit is 1 MB - send fewer photos per run")
        build = dest / "_kernel"
        shutil.rmtree(build, ignore_errors=True)
        build.mkdir(parents=True)
        (build / "run.py").write_text(code)
        (build / "kernel-metadata.json").write_text(json.dumps({
            "id": ref, "title": KERNEL, "code_file": "run.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_internet": True,
            "machine_shape": k.get("accelerator", "NvidiaTeslaT4"),
            "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
        }, indent=2))
        print("  " + kaggle("kernels", "push", "-p", str(build)).strip())
        (dest / "collected").unlink(missing_ok=True)

    started = time.time()
    while True:
        time.sleep(POLL_SECONDS)
        state = status(ref)
        print(f"  {(time.time() - started) / 60:5.1f} min  {state.strip()}", flush=True)
        if "complete" in state or "error" in state or "cancel" in state:
            break
    return collect(ref, dest)


def collect(ref: str, dest: Path) -> dict:
    # The CLI skips any file it finds locally with a newer date, so a stale
    # log.json from the previous job would be read as this job's result.
    # Clear the last output and force the download.
    for old in [*dest.glob("*.mp4"), *dest.glob("*.json"), *dest.glob("*.log")]:
        old.unlink()
    kaggle("kernels", "output", ref, "-p", str(dest), "--force")
    (dest / "collected").touch()
    log_path = dest / "log.json"
    if not log_path.exists():
        raise SystemExit(f"Kaggle job ended without a log - "
                         f"see {dest} and the job page on kaggle.com")
    log = json.loads(log_path.read_text())
    for job in log.get("jobs", []):
        state_ = "ok" if job.get("ok") else f"FAILED {job.get('error', '')}"
        print(f"  {job['name']}: {state_} ({job['seconds']}s)")
    if log.get("stage") != "done":
        print(f"  job stopped during '{log.get('stage')}' - its log is in {dest}")
    return log
