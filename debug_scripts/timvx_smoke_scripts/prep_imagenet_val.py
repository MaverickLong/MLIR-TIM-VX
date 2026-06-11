#!/usr/bin/env python3
"""prep_imagenet_val.py — pre-transform imagenet_val/ JPEGs with the
geometric prefix of torchvision's ResNet50_Weights.DEFAULT.transforms().

What we apply
-------------
torchvision.models.ResNet50_Weights.DEFAULT.transforms() is the
`ImageClassification` recipe used to train the IMAGENET1K_V2 checkpoint:

    Resize(232, BILINEAR, antialias=True)
    CenterCrop(224)
    ToTensor()
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

We only apply the first two (geometric) ops here. ToTensor + Normalize
produce float values in approximately [-2.12, 2.64] which cannot survive
a JPG round-trip; the C++ runner's CpuJpegPreProcessor already performs
the normalize + model-quant requantize on a uint8 RGB input. By pre-
applying Resize(232) + CenterCrop(224), we line the C++ pipeline up
exactly with torchvision's eval — its later cv::resize-to-224 becomes
a true no-op on the already-224 input.

Why this matters: the C++ preproc currently does cv::resize(src, 224x224)
unconditionally, a single-axis stretch that disagrees with torchvision's
resize-shortest-side-then-center-crop. On non-square ImageNet val images
(virtually all of them — most are 500xN or Nx500) that stretch costs
~2-3pp of top-1 accuracy.

Implementation note: we use PIL directly rather than `import torchvision`
because the venv doesn't have torch installed. For PIL inputs, torchvision's
`Resize(..., interpolation=BILINEAR, antialias=True)` and `CenterCrop`
both delegate to PIL.Image.resize / Image.crop with the same arithmetic
we do here, so the outputs match bit-for-bit.

Output: <dst>/images/<stem>.jpg (same stems as src), plus a verbatim
copy of labels.json and manifest.json from the source dir.

Usage:
    .venv/bin/python example/prep_imagenet_val.py \\
        --src imagenet_val --dst imagenet_val_224 [--workers 8]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

RESIZE_SIZE = 232
CROP_SIZE = 224


def resize_shortest(img: Image.Image, size: int) -> Image.Image:
    """Match torchvision.transforms.Resize(size, BILINEAR) for PIL inputs:
    rescale the shorter side to `size`, preserve aspect ratio."""
    w, h = img.size
    if w < h:
        new_w = size
        new_h = int(round(size * h / w))
    else:
        new_h = size
        new_w = int(round(size * w / h))
    return img.resize((new_w, new_h), Image.BILINEAR)


def center_crop(img: Image.Image, size: int) -> Image.Image:
    """Match torchvision.transforms.CenterCrop(size): crop centered square
    of side `size` from `img`. Assumes img.size >= (size, size)."""
    w, h = img.size
    left = (w - size) // 2
    top  = (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def transform_one(args: tuple[Path, Path]) -> str:
    src, dst = args
    img = Image.open(src).convert("RGB")
    img = center_crop(resize_shortest(img, RESIZE_SIZE), CROP_SIZE)
    img.save(dst, "JPEG", quality=95)
    return src.name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("imagenet_val"))
    ap.add_argument("--dst", type=Path, default=Path("imagenet_val_224"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    src_imgs = args.src / "images"
    if not src_imgs.is_dir():
        sys.exit(f"src images dir not found: {src_imgs}")
    dst_imgs = args.dst / "images"
    dst_imgs.mkdir(parents=True, exist_ok=True)

    for fn in ("labels.json", "manifest.json"):
        p = args.src / fn
        if p.exists():
            shutil.copy(p, args.dst / fn)

    work = [(p, dst_imgs / p.name)
            for p in sorted(src_imgs.iterdir())
            if p.suffix.lower() in {".jpg", ".jpeg"}]
    n = len(work)
    print(f"[prep] {n} images: {src_imgs} -> {dst_imgs}  "
          f"(workers={args.workers})", flush=True)

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
