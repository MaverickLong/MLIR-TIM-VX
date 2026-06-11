#!/usr/bin/env python3
"""dump_compare_outputs.py — byte-level diagnostics for the resnet18
NPU output. Aimed at the "argmax mismatch" bug: NPU and TFLite agree on
specific values but at different output indices.

Usage:
  # Build resnet18 first (no-fuse, since QRF still trips on it).
  .venv/bin/python example/lower_to_timvx.py --no-fuse \\
        resnet18_weights_v1.mlir
  # Capture NPU outputs for several inputs to a directory.
  .venv/bin/python debug_scripts.draft/dump_compare_outputs.py capture \\
        example/lower_out/resnet18_weights_v1/resnet18_weights_v1_runner \\
        --out-dir /tmp/resnet_outs
  # Compare them (top-K, multiset, input-independence).
  .venv/bin/python debug_scripts.draft/dump_compare_outputs.py analyze \\
        --out-dir /tmp/resnet_outs

The runner_main.cpp.tpl honors `TIMVX_DUMP_OUTPUT=<path>` to write the
dequantized fp32 1000-vector to a raw .bin file.
"""
from __future__ import annotations
import argparse, os, struct, subprocess, sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = REPO_ROOT / "example"
RUN_SCRIPT = EXAMPLE_DIR / "run_timvx.py"
VENV_PY = REPO_ROOT / ".venv" / "bin" / "python"
TIM_VX_LIB = REPO_ROOT / "TIM-VX" / "build" / "install" / "lib"
SDK_LIB = (Path.home() / "ufs" / "home" / "radxa" / "ai-sdk" /
            "unified-tina" / "timvx-sdk" / "lib")
VIV_SHIM = EXAMPLE_DIR / ".build" / "viv_sdk_shim"

# ImageNet hot classes for sanity checks
IMAGENET_NAMES = {
    0: "tench", 1: "goldfish", 2: "great_white_shark", 3: "tiger_shark",
    4: "hammerhead", 5: "electric_ray", 6: "stingray",
    51: "triceratops",  # the suspect
    151: "Chihuahua",
    281: "tabby_cat", 282: "tiger_cat", 283: "Persian_cat",
    284: "Siamese_cat", 285: "Egyptian_cat",
    457: "bow_tie",  # the suspect
    818: "spotlight",
}


def load_dump(path: Path) -> list[float]:
    raw = path.read_bytes()
    return list(struct.unpack(f"<{len(raw)//4}f", raw))


def synth_input(mode: str, dst: Path) -> None:
    """Materialize a synthetic uint8 NCHW [1,3,224,224] input.bin for
    bypass-image tests (gray/zero/ones). Bypasses run_timvx.py."""
    n = 1 * 3 * 224 * 224
    if mode == "gray":   buf = bytes([128]) * n
    elif mode == "zero": buf = bytes([0]) * n
    elif mode == "ones": buf = bytes([1]) * n
    else: sys.exit(f"unknown synth mode: {mode}")
    dst.write_bytes(buf)


def run_with_image(runner: Path, image: Path, dump: Path) -> None:
    env = {**os.environ, "TIMVX_DUMP_OUTPUT": str(dump)}
    cmd = [str(VENV_PY), str(RUN_SCRIPT), str(runner), str(image)]
    print("  $", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


def run_with_synth(runner: Path, mode: str, dump: Path) -> None:
    """Bypass run_timvx.py and pass a raw NCHW uint8 buffer directly."""
    inp = Path(f"/tmp/{mode}_input.bin")
    synth_input(mode, inp)
    env = {
        **os.environ,
        "TIMVX_DUMP_OUTPUT": str(dump),
        "LD_LIBRARY_PATH": f"{TIM_VX_LIB}:{SDK_LIB}:" +
                            os.environ.get("LD_LIBRARY_PATH", ""),
        "VIVANTE_SDK_DIR": str(VIV_SHIM),
    }
    print(f"  $ TIMVX_DUMP_OUTPUT={dump} {runner} {inp}")
    subprocess.run([str(runner), str(inp)], env=env, check=True)


def cmd_capture(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_with_image(args.runner, EXAMPLE_DIR / "cat105.jpg",
                    out_dir / "cat.bin")
    run_with_image(args.runner, EXAMPLE_DIR / "goldfish.jpg",
                    out_dir / "goldfish.bin")
    run_with_synth(args.runner, "gray", out_dir / "gray.bin")
    run_with_synth(args.runner, "zero", out_dir / "zero.bin")


def cmd_analyze(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    dumps: dict[str, list[float]] = {}
    for tag in ["cat", "goldfish", "gray", "zero"]:
        p = out_dir / f"{tag}.bin"
        if p.is_file(): dumps[tag] = load_dump(p)
    if not dumps:
        sys.exit(f"no .bin dumps found under {out_dir}; run `capture` first")

    print("\n=== top-10 per input ===")
    for tag, v in dumps.items():
        top10 = sorted(range(len(v)), key=lambda i: -v[i])[:10]
        names = [(i, IMAGENET_NAMES.get(i, "")) for i in top10]
        vals  = [round(v[i], 2) for i in top10]
        print(f"\n[{tag}] argmax={top10[0]} ({IMAGENET_NAMES.get(top10[0],'')})")
        for i, (idx, nm) in enumerate(names):
            print(f"   {i:>2}. idx {idx:>4} val={vals[i]:>7.3f}  {nm}")

    print("\n=== input-independence: indices in top-50 of 3+ inputs ===")
    sets = {tag: set(sorted(range(len(v)), key=lambda i: -v[i])[:50])
            for tag, v in dumps.items()}
    cnt = Counter()
    for s in sets.values():
        for i in s: cnt[i] += 1
    for idx, c in sorted(((k,v) for k,v in cnt.items() if v >= 3),
                          key=lambda kv: (-kv[1], kv[0])):
        per = " ".join(f"{tag}={dumps[tag][idx]:>6.2f}" for tag in dumps)
        print(f"  idx {idx:>4} (in {c}/{len(dumps)} top-50)  {per}  "
              f"{IMAGENET_NAMES.get(idx,'')}")

    print("\n=== multiset structure (cat vs goldfish) ===")
    if "cat" in dumps and "goldfish" in dumps:
        cat = sorted(dumps["cat"], reverse=True)
        gld = sorted(dumps["goldfish"], reverse=True)
        # Identical multiset → identical sorted lists.
        same = sum(1 for a, b in zip(cat, gld) if abs(a - b) < 1e-3)
        print(f"  shared sorted-value count (within 1e-3): {same}/1000")
        print(f"  cat  spread: {min(cat):.2f} .. {max(cat):.2f}")
        print(f"  gold spread: {min(gld):.2f} .. {max(gld):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("capture", help="Run NPU on cat/goldfish/gray/zero "
                                          "and dump 1000-vectors")
    pc.add_argument("runner", type=Path)
    pc.add_argument("--out-dir", required=True)
    pa = sub.add_parser("analyze", help="Top-K, input-independence, multiset")
    pa.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    {"capture": cmd_capture, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
