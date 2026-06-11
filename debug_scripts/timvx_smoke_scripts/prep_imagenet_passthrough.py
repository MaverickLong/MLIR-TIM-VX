#!/usr/bin/env python3
"""prep_imagenet_passthrough.py — generate quantized .bin files from
ORIGINAL imagenet_val/ JPEGs by replicating torchvision's
ResNet50_Weights.DEFAULT.transforms() pipeline end-to-end (geometric
+ ToTensor + Normalize), then quantizing with the model's input quant
params and packing in TIM-VX WHCN byte order (W innermost).

The resulting .bin files can be fed to the runner via:

    runner --eval-dir <bin_dir> --labels <labels.json> --passthrough

which uses CpuPassthroughPreProcessor — no libjpeg / mean-std /
quantize on the C++ side. This isolates "lowered model wrong" from
"C++ preproc wrong": feed identical bytes to both this and the TFLite
reference; whatever the runner reports IS the lowered model's accuracy
on the canonical preprocessing.

Quant params (must match the lowered MLIR's input tensor — verify with:
    grep arg0 example/lower_out/<base>/<base>.timvx.mlir | head -1
):
    scale = 0.15463575118234549
    zp    = 114  (= original i8 zp -14 + 128 from i8→u8 promotion)

Output layout (WHCN with W innermost, channel-planar RGB):
    byte[(c*H + h)*W + w] = uint8 value for channel c, row h, col w
    Total bytes per image = 3 * 224 * 224 = 150528

Usage:
    .venv/bin/python example/prep_imagenet_passthrough.py \\
        --src imagenet_val --dst imagenet_val_passthrough [--workers 8]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

# Match torchvision.models.ResNet50_Weights.DEFAULT.transforms()
RESIZE_SIZE = 232
CROP_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Match the lowered MLIR's input quant (u8 after i8→u8 promotion).
QUANT_SCALE = 0.15463575118234549
QUANT_ZP    = 114


def resize_shortest(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    if w < h:
        new_w = size
        new_h = int(round(size * h / w))
    else:
        new_h = size
        new_w = int(round(size * w / h))
    return img.resize((new_w, new_h), Image.BILINEAR)


def center_crop(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    left = (w - size) // 2
    top  = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def transform_one(args: tuple[Path, Path]) -> str:
    src, dst = args
    img = Image.open(src).convert("RGB")
    img = center_crop(resize_shortest(img, RESIZE_SIZE), CROP_SIZE)

    # ToTensor: HWC uint8 → CHW float32 in [0, 1].
    arr = np.asarray(img, dtype=np.uint8)              # (H, W, 3) RGB
    arr = arr.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))                 # (3, H, W)
    # Normalize per channel.
    arr = (arr - MEAN[:, None, None]) / STD[:, None, None]
    # Quantize: byte = clamp(round(x/S + Z), 0, 255).
    q = np.clip(np.round(arr / QUANT_SCALE + QUANT_ZP),
                0, 255).astype(np.uint8)
    # q.shape = (3, 224, 224) — numpy's C-order bytes == TIM-VX WHCN
    # with W innermost (verify: byte((c*H + h)*W + w) = q[c, h, w]).
    q.tofile(dst)
    return src.name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("imagenet_val"))
    ap.add_argument("--dst", type=Path, default=Path("imagenet_val_passthrough"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = all images")
    args = ap.parse_args()

    src_imgs = args.src / "images"
    if not src_imgs.is_dir():
        sys.exit(f"src images dir not found: {src_imgs}")
    dst_bins = args.dst / "images"
    dst_bins.mkdir(parents=True, exist_ok=True)

    for fn in ("labels.json", "manifest.json"):
        p = args.src / fn
        if p.exists():
            shutil.copy(p, args.dst / fn)

    work = [(p, dst_bins / (p.stem + ".bin"))
            for p in sorted(src_imgs.iterdir())
            if p.suffix.lower() in {".jpg", ".jpeg"}]
    if args.limit > 0:
        work = work[:args.limit]
    n = len(work)
    print(f"[prep] {n} images: {src_imgs} -> {dst_bins}/*.bin "
          f"(workers={args.workers})", flush=True)
    print(f"[prep] quant: scale={QUANT_SCALE} zp={QUANT_ZP} "
          f"(150528 bytes/image, WHCN)", flush=True)

    t0 = time.monotonic()
    with Pool(args.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(transform_one, work,
                                                  chunksize=64), 1):
            if i % 2000 == 0 or i == n:
                rate = i / (time.monotonic() - t0)
                eta = (n - i) / rate if rate else 0
                print(f"[prep] {i}/{n}  rate={rate:.0f} img/s  "
                      f"eta={eta:.0f}s", flush=True)
    print(f"[prep] done in {time.monotonic() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
