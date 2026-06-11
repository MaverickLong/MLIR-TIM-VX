#!/usr/bin/env python3
"""
Declaration: This file is mostly AI-generated and reviewed by human.
run_pipeline_bench.py — sweep the runner's 16 orchestration modes on a
JPEG input and print a comparison table.

Usage:
  run_pipeline_bench.py <runner_path> [<image.jpg>] [--iters N] [--warmup W]
                                       [--modes "cpu-pipeline,..."]

The runner binary already accepts `--mode <name> --bench <N> --warmup <W>
<image.jpg>` and emits `[bench MODE]` / `[topk MODE]` lines this script
parses.

What we measure
---------------
For each mode, we run the runner once with `--bench <iters> --warmup
<warmup>`. The runner prints:

  [bench MODE] n_ok=N n_fail=0 wall=W ms per-img=P ms throughput=T img/s
                infer-only per iter: mean=… median=… p99=… max=…
                e2e per iter:        mean=… median=… p99=… max=…
  [topk MODE] CLASS:SCORE CLASS:SCORE …

We collect those lines per mode and print:

  | mode | throughput | per-img | npu mean | npu p99 | e2e mean | e2e p99 |
    top-1 class | top-1 score |

The e2e numbers measure from `orch.submit()` (i.e. when the harness
hands the request off) to `future.get()` returning the top-K — they
include any time the request spent waiting in a BoundedQueue slot,
which is exactly what a real-time serving system would see under load.

That's the comparison table that tests the user's hypothesis (CPU
pre/post should fold into NPU time on the pipelined modes; PPU pre/post
should serialize on the shared dispatch queue). Run for 1000 iters /
100 warmup by default.
"""

from __future__ import annotations
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# Mode order: orchestration strategy outer, backend inner — so the
# four backends (cpu, cpu_zero_copy, ppu, ppu_zero_copy) for the SAME
# strategy are adjacent in the table.
DEFAULT_MODES = [
    "cpu-sequential",  "cpu_zero_copy-sequential",  "ppu-sequential",  "ppu_zero_copy-sequential",
    "cpu-pipeline",    "cpu_zero_copy-pipeline",    "ppu-pipeline",    "ppu_zero_copy-pipeline",
    "cpu-pool",        "cpu_zero_copy-pool",        "ppu-pool",        "ppu_zero_copy-pool",
    "cpu-hybrid",      "cpu_zero_copy-hybrid",      "ppu-hybrid",      "ppu_zero_copy-hybrid",
]


def find_timvx_lib_dir(tim_vx_build_dir: Path) -> Path:
    for cand in (tim_vx_build_dir / "install" / "lib",
                 tim_vx_build_dir / "lib",
                 tim_vx_build_dir):
        if (cand / "libtim-vx.so").is_file():
            return cand
    sys.exit(f"could not find libtim-vx.so under {tim_vx_build_dir}")


def apply_perf_governors() -> bool:
    """Set CPU + NPU devfreq governors to performance. Silently no-op
    when not root (or sysfs isn't mounted writable). Returns True if at
    least one CPU governor was successfully set."""
    ok = False
    for gov in Path("/sys/devices/system/cpu").glob("cpu*/cpufreq/scaling_governor"):
        try:
            gov.write_text("performance")
            ok = True
        except OSError:
            pass
    # NPU devfreq governor — sometimes lives under /sys/class/devfreq/.
    for dev in Path("/sys/class/devfreq").glob("*/governor"):
        try:
            dev.write_text("performance")
        except OSError:
            pass
    return ok


def wrap_runner_cmd(runner_argv: list[str], pin_cpus: str | None) -> list[str]:
    """Wrap the runner invocation with `taskset -c <cpus>` if available.
    Otherwise return the bare invocation.

    `pin_cpus` is whatever `taskset -c` accepts: a single core ("3"), a
    range ("0-7"), or a list ("6-7,0"). When None or empty, no pinning."""
    import shutil
    if shutil.which("taskset") is None or not pin_cpus:
        if pin_cpus:
            print("[rt]    taskset not found; running without CPU affinity",
                  file=sys.stderr)
        return runner_argv
    print(f"[rt]    pinned to cpus {pin_cpus}", file=sys.stderr)
    return ["taskset", "-c", pin_cpus, *runner_argv]


def make_env(repo_root: Path) -> dict:
    tim_vx_build = Path(os.environ.get(
        "TIM_VX_BUILD_DIR", repo_root / "TIM-VX" / "build" / "install"))
    timvx_lib = find_timvx_lib_dir(tim_vx_build)
    default_sdk = (Path(os.environ.get(
        "EXTERNAL_VIV_SDK",
        Path("/home/radxa/") / "ufs" / "home" / "radxa" / "ai-sdk" / "unified-tina"
        / "timvx-sdk")) / "lib")
    viv_sdk_lib = Path(os.environ.get("VIV_SDK_LIB_DIR", default_sdk))
    if not viv_sdk_lib.is_dir():
        sys.exit(f"VIV_SDK_LIB_DIR does not exist: {viv_sdk_lib}")
    # libCLC's JIT EVIS / CL shader compiler resolves
    # `#include "cl_viv_vx_ext.h"` against `$VIVANTE_SDK_DIR/include/CL/`.
    # The runtime unified-tina SDK doesn't ship that header, but TIM-VX's
    # prebuilt-sdk does — bundle it into a shim dir.
    viv_shim = repo_root / "example" / ".build" / "viv_sdk_shim"
    viv_prebuilt = (repo_root / "TIM-VX" / "prebuilt-sdk" / "x86_64_linux"
                    / "include" / "CL" / "cl_viv_vx_ext.h")
    target_hdr = viv_shim / "include" / "CL" / "cl_viv_vx_ext.h"
    if not target_hdr.is_file() and viv_prebuilt.is_file():
        target_hdr.parent.mkdir(parents=True, exist_ok=True)
        target_hdr.write_bytes(viv_prebuilt.read_bytes())

    ld = os.pathsep.join(filter(None, [
        str(timvx_lib), str(viv_sdk_lib),
        os.environ.get("LD_LIBRARY_PATH", "")]))
    return {**os.environ,
            "LD_LIBRARY_PATH": ld,
            "VIVANTE_SDK_DIR": str(viv_shim)}


_BENCH_RE = re.compile(
    r"\[bench (?P<mode>\S+)\]\s+n_ok=(?P<n_ok>\d+)\s+n_fail=(?P<n_fail>\d+)\s+"
    r"wall=(?P<wall>[\d.]+)\s*ms\s+per-img=(?P<per_img>[\d.]+)\s*ms\s+"
    r"throughput=(?P<thru>[\d.]+)\s*img/s")
# The single-line machine-parseable [stats MODE] summary from
# `benchmark.h::print_bench`. Captures mean / se / moe per metric so
# we can render the LaTeX-like table without re-parsing the per-block
# human-readable lines.
_STATS_RE = re.compile(r"\[stats (?P<mode>\S+)\] (?P<kv>.*)")
_TOPK_RE = re.compile(r"\[topk (?P<mode>\S+)\] (?P<list>.*)")


def benchmark_runner(runner: Path, mode: str, image: Path,
                     iters: int, warmup: int, timeout: int,
                     pin_cpus: str | None = None,
                     env: dict | None = None,
                     verbose: bool = True) -> dict | None:
    """
    Run a single benchmark mode and return structured results.

    Args:
        runner: Path to the model runner executable
        mode: Orchestration mode (e.g., "cpu-sequential", "ppu-pipeline")
        image: Path to JPEG image input
        iters: Number of benchmark iterations
        warmup: Number of warmup iterations
        timeout: Per-mode timeout in seconds
        pin_cpus: Optional CPU affinity string (e.g., "0-7" for taskset -c)
        env: Optional environment dict; defaults to make_env()
        verbose: Print progress messages to stderr

    Returns:
        Dict with parsed results (throughput, per_img_ms, infer_mean, infer_p99,
        e2e_mean, e2e_p99, topk list, etc.) or None on failure.
        Keys include: mode, n_ok, n_fail, wall_ms, per_img_ms, throughput,
        infer_mean, infer_se, infer_moe, infer_p99, e2e_mean, e2e_se, e2e_moe,
        e2e_p99, pre_mean, pre_se, pre_moe, prewait_mean, prewait_se, prewait_moe,
        postwait_mean, postwait_se, postwait_moe, post_mean, post_se, post_moe, topk
    """
    if env is None:
        repo_root = Path(os.environ.get("REPO_ROOT", runner.parent.parent.parent))
        env = make_env(repo_root)

    base = [str(runner), "--mode", mode,
            "--bench", str(iters), "--warmup", str(warmup), str(image)]
    cmd = wrap_runner_cmd(base, pin_cpus=pin_cpus)
    if verbose:
        print(f"=== running: {' '.join(cmd)}", file=sys.stderr, flush=True)
    try:
        cp = subprocess.run(cmd, env=env, timeout=timeout,
                            capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        if verbose:
            print(f"[timeout] {mode}", file=sys.stderr, flush=True)
        return None
    if cp.returncode != 0:
        if verbose:
            print(f"[rc={cp.returncode}] {mode}\n--- stdout ---\n{cp.stdout}\n"
                  f"--- stderr ---\n{cp.stderr}\n",
                  file=sys.stderr, flush=True)
        return None
    row = {"mode": mode}
    for line in cp.stdout.splitlines():
        if m := _BENCH_RE.search(line):
            row.update({
                "n_ok": int(m["n_ok"]), "n_fail": int(m["n_fail"]),
                "wall_ms": float(m["wall"]),
                "per_img_ms": float(m["per_img"]),
                "throughput": float(m["thru"]),
            })
        elif m := _STATS_RE.search(line):
            # `[stats MODE] key1=val1 key2=val2 ...` — float-keyed.
            for kv in m["kv"].split():
                k, _, v = kv.partition("=")
                if k and v:
                    try:
                        row[k] = float(v)
                    except ValueError:
                        pass
        elif m := _TOPK_RE.search(line):
            entries = []
            for e in m["list"].split():
                cls, sc = e.split(":")
                entries.append((int(cls), float(sc)))
            row["topk"] = entries
    return row


def run_mode(runner: Path, mode: str, image: Path,
             iters: int, warmup: int, env: dict, timeout: int,
             pin_cpus: str) -> dict | None:
    """Backward-compatible wrapper for benchmark_runner()."""
    return benchmark_runner(runner, mode, image, iters, warmup, timeout,
                           pin_cpus=pin_cpus, env=env, verbose=True)


def _parse_mode(mode: str) -> tuple[str, str, str]:
    """`cpu_zero_copy-pipeline` → ('CPU', 'Pipeline', 'Yes'). Unknown
    modes return ('?', '?', '?') so the table at least shows the cell."""
    back, _, strat = mode.partition("-")
    proc, zc = "?", "?"
    if back == "cpu":             proc, zc = "CPU", "No"
    elif back == "cpu_zero_copy": proc, zc = "CPU", "Yes"
    elif back == "ppu":           proc, zc = "PPU", "No"
    elif back == "ppu_zero_copy": proc, zc = "PPU", "Yes"
    return (proc, strat.capitalize(), zc)


def _row_order(rows: list[dict]) -> list[dict]:
    """Order rows for the LaTeX-like table: blocks of 4 parallelism
    strategies grouped by (processor, zerocopy). Matches the user's
    requested layout (CPU/Yes, CPU/No, PPU/Yes, PPU/No).

    When an orthogonal IO-transpose-fuse sweep is present (rows carry
    `io_fuse`), that's the OUTERMOST grouping — the full fused matrix
    prints first, then the full unfused matrix — so the two halves line
    up row-for-row for comparison."""
    # io_fuse outermost: "Yes" (fused) before "No" (unfused); rows with no
    # io_fuse tag (single-runner runs) all share one bucket.
    fuse_buckets: dict[str, list[dict]] = {}
    for r in rows:
        fuse_buckets.setdefault(r.get("io_fuse", ""), []).append(r)

    def order_within(bucket: list[dict]) -> list[dict]:
        blocks: dict[tuple[str, str], list[dict]] = {}
        for r in bucket:
            proc, strat, zc = _parse_mode(r.get("mode", ""))
            blocks.setdefault((proc, zc), []).append((strat, r))
        strat_rank = {"Sequential": 0, "Pipeline": 1, "Pool": 2, "Hybrid": 3}
        out: list[dict] = []
        for key in (("CPU", "Yes"), ("CPU", "No"),
                    ("PPU", "Yes"), ("PPU", "No")):
            if key not in blocks:
                continue
            for _, r in sorted(blocks[key],
                                key=lambda sr: strat_rank.get(sr[0], 99)):
                out.append(r)
        return out

    out: list[dict] = []
    for fuse_key in ("Yes", "No", ""):
        if fuse_key in fuse_buckets:
            out.extend(order_within(fuse_buckets[fuse_key]))
    return out


def _avg_cell(mean_key: str, se_key: str, moe_key: str,
              fmt: str = "%.3f") -> callable:
    """Format an "average" cell as `<mean> ± <MoE> (SE <SE>)`. Uses
    `_` to mark missing/uncomputed values so the LaTeX renders rather
    than NaN-ing."""
    def f(r: dict) -> str:
        m = r.get(mean_key)
        se = r.get(se_key)
        mo = r.get(moe_key)
        if m is None:
            return "—"
        cell = fmt % m
        if mo is not None:
            cell += " $\\pm$ " + (fmt % mo)
        if se is not None:
            cell += " (SE " + (fmt % se) + ")"
        return cell
    return f


def _plain_cell(key: str, fmt: str = "%.3f") -> callable:
    def f(r: dict) -> str:
        v = r.get(key)
        return (fmt % v) if v is not None else "—"
    return f


def print_table(rows: list[dict], with_io_fuse: bool = False) -> None:
    """Emit a LaTeX-like table. Columns match the user's request,
    with the new pre / pre_wait / post_wait / post columns appended at
    the end. Throughput / per-image / NPU mean / e2e and all the new
    columns carry mean $\\pm$ MoE_95% (SE …); the *_p99 columns are
    raw percentiles (no SE).

    `with_io_fuse` prepends an "IO Transpose Fuse" column — set when an
    orthogonal fuse/no-fuse sweep ran (rows carry `io_fuse`)."""
    rows = _row_order([r for r in rows if r])
    cols = []
    if with_io_fuse:
        cols.append(("IO Fuse", lambda r: r.get("io_fuse", "—")))
    cols += [
        ("Processor",                lambda r: _parse_mode(r["mode"])[0]),
        ("Parallelism",              lambda r: _parse_mode(r["mode"])[1]),
        ("Zero Copy",                lambda r: _parse_mode(r["mode"])[2]),
        ("Throughput (imgs/s)",      _avg_cell("thru_mean", "thru_se", "thru_moe",
                                                 fmt="%.2f")),
        ("Per-Image Average (ms)",   _avg_cell("perimg_mean", "perimg_se", "perimg_moe")),
        ("NPU Mean (ms)",            _avg_cell("infer_mean", "infer_se", "infer_moe")),
        ("NPU P99 (ms)",             _plain_cell("infer_p99")),
        ("E2E Latency (ms)",         _avg_cell("e2e_mean", "e2e_se", "e2e_moe")),
        ("E2E P99 (ms)",             _plain_cell("e2e_p99")),
        # Appended fields:
        ("Pre (ms)",                 _avg_cell("pre_mean", "pre_se", "pre_moe")),
        ("Pre Wait (ms)",            _avg_cell("prewait_mean", "prewait_se", "prewait_moe")),
        ("Post Wait (ms)",           _avg_cell("postwait_mean", "postwait_se", "postwait_moe")),
        ("Post (ms)",                _avg_cell("post_mean", "post_se", "post_moe")),
    ]
    # LaTeX-like preamble (no actual tabular spec — just the rows the
    # user asked for; they can paste this into a tabular environment).
    print()
    print(" \\hline")
    print(" " + " & ".join(name for name, _ in cols) + " \\\\")
    print(" \\hline")

    last_block: tuple[str, str, str] | None = None
    for r in rows:
        proc, strat, zc = _parse_mode(r.get("mode", ""))
        block = (r.get("io_fuse", ""), proc, zc)
        if last_block is not None and block != last_block:
            print(" \\hline")
        last_block = block
        cells = []
        for _, fn in cols:
            try:
                cells.append(str(fn(r)))
            except Exception:
                cells.append("?")
        print(" " + " & ".join(cells) + " \\\\")
    print(" \\hline")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runner", type=Path,
                    help="path to the model runner (e.g. lower_out/resnet50_v1/resnet50_v1_runner). "
                         "This is the FUSED build (default lowering folds the entry NCHW->WHCN "
                         "transpose and the trailing reshape/transpose tail).")
    ap.add_argument("--no-fuse-runner", type=Path, default=None,
                    help="Path to a SECOND runner built with "
                         "`--no-fold-input-transpose --no-fold-output-transpose`. When given, "
                         "run_pipeline_bench sweeps the full mode matrix on BOTH runners and "
                         "prints one table with an orthogonal 'IO Fuse' (Yes/No) axis — so you "
                         "can read off the perf delta of folding the IO transposes (and eyeball "
                         "any NPU-recompile numeric drift in the [topk] lines).")
    ap.add_argument("image", type=Path, nargs="?",
                    default=Path(__file__).resolve().parent / "cat105.jpg",
                    help="JPEG file to feed (default: example/cat105.jpg)")
    ap.add_argument("--iters",  type=int, default=1000,
                    help="benchmark iterations (default: 1000)")
    ap.add_argument("--warmup", type=int, default=100,
                    help="warmup iterations not included in stats (default: 100)")
    ap.add_argument("--modes",  type=str, default=",".join(DEFAULT_MODES),
                    help="comma-separated mode list (default: all 16)")
    ap.add_argument("--timeout", type=int, default=600,
                    help="per-mode timeout in seconds (default: 600)")
    ap.add_argument("--pin-cpus", type=str, default="0-7",
                    help="CPU set for taskset -c. On A733 the big cluster "
                         "is cores 6-7 (Cortex-A76, 2.0 GHz); the little "
                         "cluster is 0-5 (Cortex-A55, 1.8 GHz). Default "
                         "'0-7' lets pool/hybrid orchestrators spread "
                         "freely; use '6-7' to bias pre/post toward big "
                         "cores.")
    ap.add_argument("--no-governor", action="store_true",
                    help="Skip the cpufreq governor=performance step. Default "
                         "is to try; silently noop if not root.")
    args = ap.parse_args()

    if not args.runner.is_file() or not os.access(args.runner, os.X_OK):
        sys.exit(f"not executable: {args.runner}")
    if not args.image.is_file():
        sys.exit(f"image not found: {args.image}")
    # Orthogonal IO-transpose-fuse sweep: (runner, io_fuse_label) pairs.
    # The primary runner is the fused build; --no-fuse-runner adds the
    # unfused build as a second axis value.
    sweep = [(args.runner, "Yes")]
    if args.no_fuse_runner is not None:
        if (not args.no_fuse_runner.is_file()
                or not os.access(args.no_fuse_runner, os.X_OK)):
            sys.exit(f"not executable: {args.no_fuse_runner}")
        sweep.append((args.no_fuse_runner, "No"))

    script_dir = Path(__file__).resolve().parent
    repo_root = Path(os.environ.get("REPO_ROOT", script_dir.parent)).resolve()
    env = make_env(repo_root)

    # DVFS jitter is the biggest single source of bench-run-to-run variance
    # on this chip; pin frequencies up-front if possible. This is the
    # "set governors to performance" step from timvx_zerocopy_perf.bash.
    if not args.no_governor:
        if apply_perf_governors():
            print("[gov]   CPU/NPU governors set to performance",
                  file=sys.stderr)
        else:
            print("[gov]   could not set governors (re-run as root to "
                  "eliminate DVFS jitter)", file=sys.stderr)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    rows = []
    for runner, fuse_label in sweep:
        if len(sweep) > 1:
            print(f"[sweep] IO-transpose fuse = {fuse_label}  ({runner})",
                  file=sys.stderr, flush=True)
        for m in modes:
            row = run_mode(runner, m, args.image,
                           args.iters, args.warmup, env, args.timeout,
                           args.pin_cpus)
            if row is None:
                # placeholder so the table shows the gap
                rows.append({"mode": m, "io_fuse": fuse_label})
                continue
            row["io_fuse"] = fuse_label
            rows.append(row)
    print_table(rows, with_io_fuse=(len(sweep) > 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
