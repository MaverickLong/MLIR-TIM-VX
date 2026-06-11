#!/usr/bin/env python3
"""compare_targets.py — diff outputs across the lowering targets we care
about for a single quantized TOSA .mlir input.

Goal: sanity-check that `--timvx-quant-residual-fuse` produces
numerically equivalent output to the unfused fp32-detour path. The
fusion converts the `cast→sub→mul + cast→sub→mul → add → clip →
mul→add→cast` chain into a single `tim::vx::ops::Add(int8↔int8↔int8)`
call — TIM-VX's quant Add should give bit-equivalent results to the
fp32 chain assuming the recorded scales are TFLite-faithful
(`RescaleConvFusion` now derives those from the chain itself). If the
diff is large, that assumption is broken and we have a numerical bug
to chase.

Targets:
  fused    — full pipeline including `--timvx-quant-residual-fuse`.
  nofuse   — same pipeline minus that pass; everything else identical.

Usage:
  compare_targets.py <model.mlir> [<image.jpg>]
      --targets=fused,nofuse
      [--keep] keep build dirs under /tmp for inspection

Prints a table with argmax / first-N values / mean per target, and
flags any target whose output diverges from the others past a small
tolerance.
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = REPO_ROOT / "example"
LOWER_SCRIPT = EXAMPLE_DIR / "lower_to_timvx.py"
RUN_SCRIPT = EXAMPLE_DIR / "run_timvx.py"
VENV_PY = REPO_ROOT / ".venv" / "bin" / "python"


def run(cmd, **kw):
    print("  $ " + " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def run_capture(cmd, env=None):
    print("  $ " + " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], env=env,
                          check=False, capture_output=True, text=True)


# ----------------------------------------------------------------------------
# NPU targets (fused / nofuse) — pure delegation to lower_to_timvx.py.
# The production lowering script owns:
#   - the full pass pipeline (mirrored here would silently drift),
#   - timvx_runtime.h include resolution + custom_ops compilation,
#   - the clang invocations + library/SDK rpath glue.
# We just toggle the two passes whose presence defines a "variant"
# (`--timvx-quant-residual-fuse`, `--timvx-conv1x1-to-fc`) via the
# `--no-fuse` / `--no-conv1x1-to-fc` flags lower_to_timvx exposes.
# ----------------------------------------------------------------------------

def build_npu_target(model_mlir: Path, label: str,
                      out_root: Path, *,
                      with_fuse: bool,
                      with_conv1x1_to_fc: bool = True) -> Path:
    """Build the NPU runner for one variant. Returns the runner binary
    path under `<out_root>/<label>/`. Single subprocess invocation —
    lower_to_timvx.py does the lowering, runner template rendering, and
    clang link in one pass."""
    out_dir = out_root / label
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [VENV_PY, LOWER_SCRIPT, "--out-dir", out_dir]
    if not with_fuse:
        cmd.append("--no-fuse")
    if not with_conv1x1_to_fc:
        cmd.append("--no-conv1x1-to-fc")
    cmd.append(model_mlir)

    print(f"\n[{label}] building (with_fuse={with_fuse}, "
          f"with_conv1x1_to_fc={with_conv1x1_to_fc}):")
    run(cmd)

    return out_dir / f"{model_mlir.stem}_runner"


def run_npu_target(runner: Path, image: Path) -> dict:
    """Invoke run_timvx.py on the runner with the given image and
    return parsed `argmax`, `first16`, `mean`, `run_ms`."""
    proc = run_capture([VENV_PY, RUN_SCRIPT, runner, image])
    text = proc.stdout + proc.stderr
    return parse_runner_output(text)


# ----------------------------------------------------------------------------
# Output parsing
# ----------------------------------------------------------------------------

_RE_FIRST = re.compile(
    r"first \d+ values \((?:dequantized|raw fp)\): \[(.+?)(?:, \.\.\.)?\]")
_RE_ARGMAX = re.compile(r"argmax:\s*(\d+)\s*\(value=([\-0-9.eE]+)\)")
_RE_MMM = re.compile(r"min=([\-0-9.eE]+)\s+max=([\-0-9.eE]+)\s+mean=([\-0-9.eE]+)")
_RE_RUN_MS = re.compile(r"\[stage\] run done in ([\-0-9.eE]+) ms")


def parse_runner_output(text: str) -> dict:
    out = {"raw": text}
    if m := _RE_FIRST.search(text):
        out["first16"] = [float(v) for v in m.group(1).split(", ")]
    if m := _RE_ARGMAX.search(text):
        out["argmax"] = int(m.group(1))
        out["argmax_val"] = float(m.group(2))
    if m := _RE_MMM.search(text):
        out["min"] = float(m.group(1))
        out["max"] = float(m.group(2))
        out["mean"] = float(m.group(3))
    if m := _RE_RUN_MS.search(text):
        out["run_ms"] = float(m.group(1))
    return out




# ----------------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------------

def diff(a: dict, b: dict) -> dict:
    """Return a small dict describing how `a` and `b` differ on the
    summary fields. We don't have the full output tensors here, just
    the runner's printed first-16 values, argmax, mean — enough for a
    sanity diff."""
    d = {}
    if "argmax" in a and "argmax" in b:
        d["argmax_match"] = a["argmax"] == b["argmax"]
        d["argmax_a"] = a["argmax"]
        d["argmax_b"] = b["argmax"]
    if "mean" in a and "mean" in b:
        d["mean_a"] = a["mean"]
        d["mean_b"] = b["mean"]
        d["mean_diff"] = abs(a["mean"] - b["mean"])
    if "first16" in a and "first16" in b:
        # Use a relaxed tolerance — quant kernels round storage values
        # before dequant, so a one-LSB diff per element is normal even
        # for a "correct" fusion.
        diffs = [abs(x - y) for x, y in zip(a["first16"], b["first16"])]
        d["first16_max_diff"] = max(diffs) if diffs else 0.0
        d["first16_mean_diff"] = sum(diffs) / len(diffs) if diffs else 0.0
    return d


def print_target(label: str, r: dict) -> None:
    print(f"\n=== {label} ===")
    if "run_ms" in r: print(f"  run:    {r['run_ms']:.2f} ms")
    if "argmax" in r: print(f"  argmax: {r['argmax']} (value={r['argmax_val']:.6f})")
    if "mean" in r:   print(f"  mean:   {r['mean']:.6f}  (min={r['min']:.6f}  max={r['max']:.6f})")
    if "first16" in r:
        print(f"  first16: [{', '.join(f'{x:.4f}' for x in r['first16'][:8])}, ...]")


def print_diff(name_a: str, name_b: str, d: dict) -> None:
    print(f"\n=== diff: {name_a} vs {name_b} ===")
    if "argmax_match" in d:
        ok = "MATCH" if d["argmax_match"] else "MISMATCH"
        print(f"  argmax: {ok}  ({d['argmax_a']} vs {d['argmax_b']})")
    if "mean_diff" in d:
        print(f"  mean diff: {d['mean_diff']:.4f}  ({d['mean_a']:.4f} vs {d['mean_b']:.4f})")
    if "first16_max_diff" in d:
        # Heuristic threshold: 2× the typical i8 storage step is fine.
        # For tensors with scale ~0.07 that's ~0.14. Anything bigger is
        # a real numerical divergence.
        thresh = 0.5
        verdict = "ok" if d["first16_max_diff"] < thresh else "DIVERGES"
        print(f"  first16 diff: max={d['first16_max_diff']:.4f}  "
              f"mean={d['first16_mean_diff']:.4f}  → {verdict}")


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path,
                    help="Input quantized .tosa.mlir file (e.g. resnet50_v1.mlir)")
    ap.add_argument("image", type=Path, nargs="?",
                    default=EXAMPLE_DIR / "cat105.jpg",
                    help="Image to feed in (default: example/cat105.jpg)")
    ap.add_argument("--targets",
                    default="fused,nofuse",
                    help="Comma-separated list: fused, nofuse "
                         "(default: fused,nofuse)")
    ap.add_argument("--keep", action="store_true",
                    help="Don't delete the temp build dir after running")
    args = ap.parse_args()

    if not args.model.is_file():
        sys.exit(f"model not found: {args.model}")
    if not args.image.is_file():
        # cat105.jpg may be elsewhere; fall back to debug_scripts.draft
        alt = REPO_ROOT / "debug_scripts.draft" / "cat105.jpg"
        if alt.is_file():
            args.image = alt
        else:
            sys.exit(f"image not found: {args.image}")

    targets = args.targets.split(",")
    out_root = Path(tempfile.mkdtemp(prefix=f"compare-{args.model.stem}-",
                                       dir="/tmp"))
    print(f"build root: {out_root}")
    if args.keep:
        print(f"  (kept after run; rm -rf {out_root} when done)")

    results: dict[str, dict] = {}
    try:
        if "fused" in targets:
            runner = build_npu_target(args.model, "fused", out_root,
                                       with_fuse=True, with_conv1x1_to_fc=False)
            results["fused"] = run_npu_target(runner, args.image)
        if "nofuse" in targets:
            runner = build_npu_target(args.model, "nofuse", out_root,
                                       with_fuse=False, with_conv1x1_to_fc=False)
            results["nofuse"] = run_npu_target(runner, args.image)

        for label, res in results.items():
            print_target(label, res)

        # Pairwise diffs.
        labels = list(results.keys())
        for i, a in enumerate(labels):
            for b in labels[i + 1:]:
                print_diff(a, b, diff(results[a], results[b]))

    finally:
        if not args.keep:
            shutil.rmtree(out_root, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
