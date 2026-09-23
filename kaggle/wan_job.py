"""Runs on Kaggle's GPU, not here: animate photos of Min with Wan 2.2 14B.

kaggle_gen.py fills in JOBS and SETTINGS and pushes this as a private Kaggle
script. The model is the same Wan 2.2 image-to-video 14B that the Hugging
Face Space runs, in a 4-bit GGUF build with the Lightning step-distillation
merged in, because the full-precision one does not fit a free T4 (16 GB).

The order is set by memory, not taste. The T5 text encoder alone is 11 GB, so
every prompt is encoded first on the CPU, the encoder is dropped and its files
deleted, and only then are the two 10 GB denoisers loaded - both would not
fit in RAM or on disk at once. Everything written to /kaggle/working is the
job's output, which is why all the caches live in /tmp.
"""
import base64
import gc
import io
import json
import os
import shutil
import subprocess
import sys
import time

os.environ["HF_HOME"] = "/tmp/hf"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
# The first real run lost its longest clip to fragmentation: 1.9 GB reserved
# but unusable when the VAE asked for 550 MB. Expandable segments give it back.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
JOBS = json.loads(base64.b64decode("__JOBS__").decode())
IMAGES = json.loads(base64.b64decode("__IMAGES__").decode())
SETTINGS = json.loads(base64.b64decode("__SETTINGS__").decode())
OUT = "/kaggle/working"
log = {"jobs": [], "stage": "install"}
t0 = time.time()


def save_log():
    with open(f"{OUT}/log.json", "w") as f:
        json.dump(log, f, indent=2)
    # Also to stdout, which `kaggle kernels logs` shows while the job runs.
    print(f"[{round(time.time() - t0)}s] {log['stage']}: "
          f"{len(log['jobs'])}/{len(JOBS)} clips", flush=True)


save_log()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "diffusers", "transformers", "accelerate", "ftfy", "gguf",
                "hf_transfer", "imageio", "imageio-ffmpeg", "sentencepiece"],
               check=True)

import numpy as np
import torch
from diffusers import (AutoencoderKLWan, GGUFQuantizationConfig,
                       WanImageToVideoPipeline, WanTransformer3DModel)
from diffusers.utils import export_to_video
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from transformers import AutoTokenizer, UMT5EncoderModel

BASE = SETTINGS["base"]
log["gpu"] = torch.cuda.get_device_name(0)

# 1. Every prompt to embeddings in bfloat16 - T5 overflows in float16. On the
#    GPU if it will run there (11 GB fits the T4 before anything else is
#    loaded); six prompts took 18 minutes on Kaggle's four CPU cores.
log["stage"] = "encode"
save_log()
tokenizer = AutoTokenizer.from_pretrained(BASE, subfolder="tokenizer")
encoder = UMT5EncoderModel.from_pretrained(BASE, subfolder="text_encoder",
                                           torch_dtype=torch.bfloat16)
device = "cpu"
try:
    encoder.to("cuda")
    probe = tokenizer(["a dog"], return_tensors="pt").to("cuda")
    with torch.no_grad():
        if torch.isfinite(encoder(**probe).last_hidden_state).all():
            device = "cuda"
except Exception as exc:
    log["encode_note"] = f"GPU encode unavailable: {type(exc).__name__}"
if device == "cpu":
    encoder.to("cpu")
    torch.cuda.empty_cache()
log["encode_device"] = device
embeds = {}
for job in JOBS:
    t = tokenizer([job["prompt"]], padding="max_length", max_length=512,
                  truncation=True, add_special_tokens=True,
                  return_attention_mask=True, return_tensors="pt").to(device)
    with torch.no_grad():
        e = encoder(t.input_ids, t.attention_mask).last_hidden_state
    # As WanPipeline does: keep the real tokens, zero the padding.
    n = int(t.attention_mask.sum())
    clean = torch.zeros_like(e)
    clean[:, :n] = e[:, :n]
    embeds[job["name"]] = clean.to("cpu", torch.float16)
del encoder
gc.collect()
torch.cuda.empty_cache()
shutil.rmtree("/tmp/hf/hub", ignore_errors=True)
log["encode_seconds"] = round(time.time() - t0)

# 2. The two denoisers, 4-bit. model_cpu_offload keeps one on the GPU at a
#    time; Wan hands over from the high-noise one to the low-noise one once
#    per clip, so the swap is two transfers a clip, not two a step.
log["stage"] = "load"
save_log()
quant = GGUFQuantizationConfig(compute_dtype=torch.float16)


def denoiser(filename, subfolder):
    path = hf_hub_download(SETTINGS["gguf_repo"], filename)
    return WanTransformer3DModel.from_single_file(
        path, quantization_config=quant, config=BASE, subfolder=subfolder,
        torch_dtype=torch.float16)


high = denoiser(SETTINGS["gguf_high"], "transformer")
low = denoiser(SETTINGS["gguf_low"], "transformer_2")
vae = AutoencoderKLWan.from_pretrained(BASE, subfolder="vae", torch_dtype=torch.float32)
# Decoding 69 frames in one piece is what ran out of memory; tiles do not.
vae.enable_tiling()
pipe = WanImageToVideoPipeline.from_pretrained(
    BASE, transformer=high, transformer_2=low, vae=vae,
    text_encoder=None, tokenizer=None, torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
log["load_seconds"] = round(time.time() - t0)


def fit(image, max_dim=832, min_dim=480, multiple=16):
    """The Space's resize_image: the same framing it gave the clips we liked."""
    w, h = image.size
    ratio = w / h
    if ratio < min_dim / max_dim:          # taller than 9:16.6 - crop height
        crop = int(round(w * max_dim / min_dim))
        top = (h - crop) // 2
        image, (tw, th) = image.crop((0, top, w, top + crop)), (min_dim, max_dim)
    elif ratio > max_dim / min_dim:
        crop = int(round(h * max_dim / min_dim))
        left = (w - crop) // 2
        image, (tw, th) = image.crop((left, 0, left + crop, h)), (max_dim, min_dim)
    elif w > h:
        tw, th = max_dim, int(round(max_dim / ratio))
    else:
        tw, th = int(round(max_dim * ratio)), max_dim
    tw = max(min_dim, min(max_dim, round(tw / multiple) * multiple))
    th = max(min_dim, min(max_dim, round(th / multiple) * multiple))
    return image.resize((tw, th), Image.LANCZOS)


log["stage"] = "generate"
for job in JOBS:
    entry = {"name": job["name"], "prompt": job["prompt"]}
    start = time.time()
    try:
        image = fit(Image.open(io.BytesIO(base64.b64decode(IMAGES[job["image"]]))).convert("RGB"))
        frames = pipe(
            image=image, prompt_embeds=embeds[job["name"]].to("cuda"),
            height=image.height, width=image.width, num_frames=job["frames"],
            num_inference_steps=SETTINGS["steps"], guidance_scale=1.0,
            guidance_scale_2=1.0,
            generator=torch.Generator("cuda").manual_seed(job.get("seed", 42)),
        ).frames[0]
        arr = np.asarray(frames, dtype=np.float32)
        if not np.isfinite(arr).all() or arr.mean() < 0.02:
            raise RuntimeError("blank or NaN frames - float16 overflow")
        export_to_video(frames, f"{OUT}/{job['name']}.mp4", fps=SETTINGS["fps"])
        entry["ok"] = True
    except Exception as exc:
        entry["ok"] = False
        entry["error"] = f"{type(exc).__name__}: {exc}"[:800]
    entry["seconds"] = round(time.time() - start)
    log["jobs"].append(entry)
    save_log()
    torch.cuda.empty_cache()

log["stage"] = "done"
log["total_seconds"] = round(time.time() - t0)
save_log()
print(json.dumps(log, indent=2))
