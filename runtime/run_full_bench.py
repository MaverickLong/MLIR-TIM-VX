#!/usr/bin/env python3
"""
Declaration: This file is mostly AI-generated and reviewed by human.
Comprehensive benchmark suite: orthogonal sweep across
  - Models: resnet{18, 50, 101, 152}
  - Native layout: on/off (input/output transpose folding)
  - Zero-copy: on/off (CPU_zero_copy vs CPU; PPU modes)
  - Parallelism: sequential, pipeline, pool, hybrid
  - Thread pool workers (pool/hybrid only): 1, 2, 4, 8, 16

For each combination, runs the runner and collects perf metrics,
then emits LaTeX-formatted tables grouped by model and T value.

Usage:
  python example/run_full_bench.py [--iters ITERS] [--warmup WARMUP] \\
    [--models MODEL,...] [--timeout TIMEOUT] [--no-governor]
    [--image IMAGE] [--continue] [--mode MODE_FILTER,...] [--skip-build]

Example:
  python example/run_full_bench.py --iters 2000 --warmup 100
    # Builds all 4 models × 2 layouts, runs full sweep (resumes from results.json by default)

  python example/run_full_bench.py --mode "resnet50,pool"
    # Resume from results.json, only run modes matching "resnet50" and "pool" that aren't already present

  python example/run_full_bench.py --overwrite --mode "resnet50,pool"
    # Fresh run: clear results.json and run only resnet50 + pool modes

  python example/run_full_bench.py --skip-build --iters 500
    # Use existing runners, run quick benchmark (500 iters, resume from JSON)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List

# Import the callable benchmark interface
sys.path.insert(0, str(Path(__file__).parent))
from run_pipeline_bench import benchmark_runner, make_env

@dataclass
class BenchResult:
    """One benchmark result row."""
    model: str              # resnet18, resnet50, ...
    native_layout: bool     # input/output transpose folding on
    processor: str          # CPU, PPU
    zero_copy: bool         # yes/no
    parallelism: str        # Sequential, Pipeline, Pool, Hybrid
    num_workers: int        # 1 (for sequential/pipeline), or 2-16 (pool/hybrid)
    throughput: Optional[float] = None
    throughput_se: Optional[float] = None
    per_img_ms: Optional[float] = None
    per_img_se: Optional[float] = None
    infer_mean: Optional[float] = None
    infer_p99: Optional[float] = None
    e2e_mean: Optional[float] = None
    e2e_p99: Optional[float] = None
    e2e_se: Optional[float] = None
    pre_mean: Optional[float] = None
    pre_se: Optional[float] = None
    prewait_mean: Optional[float] = None
    prewait_se: Optional[float] = None
    postwait_mean: Optional[float] = None
    postwait_se: Optional[float] = None
    post_mean: Optional[float] = None
    post_se: Optional[float] = None
    infer_se: Optional[float] = None
    topk: Optional[List[tuple]] = None  # [(class_id, score), ...]

    def to_dict(self) -> Dict:
        """Convert to dict for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "BenchResult":
        """Reconstruct from dict."""
        return cls(**d)

    def mode_id(self) -> str:
        """Generate canonical mode ID for filtering."""
        return (f"{self.model}:native={str(self.native_layout).lower()}:"
                f"{self.processor}:zero_copy={str(self.zero_copy).lower()}:"
                f"{self.parallelism.lower()}:T={self.num_workers}")


def load_results_json(path: Path) -> Dict[str, BenchResult]:
    """Load results from JSON file. Returns dict keyed by mode_id."""
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        result = {}
        for r in data:
            mode_id = r.pop("mode_id")  # Extract and remove mode_id from dict
            bench = BenchResult.from_dict(r)
            result[mode_id] = bench
        return result
    except Exception as e:
        print(f"[warn] failed to load {path}: {e}", file=sys.stderr)
        return {}


def save_results_json(path: Path, results: List[BenchResult]):
    """Save results to JSON file."""
    data = [{"mode_id": r.mode_id(), **r.to_dict()} for r in results]
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[warn] failed to save {path}: {e}", file=sys.stderr)


def matches_mode_filter(mode_id: str, filters: List[str]) -> bool:
    """Check if mode_id contains all filter substrings (case-insensitive)."""
    mode_lower = mode_id.lower()
    return all(f.lower() in mode_lower for f in filters)


def get_modes_to_run(base_modes: List[tuple], filters: Optional[List[str]] = None) -> List[tuple]:
    """
    Filter modes based on substring filters.
    Returns list of (mode_str, proc, par, zc, T) tuples.
    """
    result = []
    for mode_str, proc, par, zc, T in base_modes:
        # Build a mode_id-like string for filtering
        # Format: proc:zero_copy=...:parallelism:T=...
        test_id = f"{proc}:zero_copy={str(zc).lower()}:{par.lower()}:T={T}"

        # Check filters
        if filters and not matches_mode_filter(test_id, filters):
            continue

        result.append((mode_str, proc, par, zc, T))

    return result


def simplify_processor(proc_str: str) -> str:
    """Convert processor field to simple CPU/PPU."""
    if "ppu" in proc_str.lower():
        return "PPU"
    return "CPU"


def apply_perf_governors():
    """Set CPU/GPU governors to performance (non-root best-effort)."""
    try:
        subprocess.run(
            ["sudo", "bash", "-c",
             "echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor "
             "> /dev/null 2>&1 && "
             "echo performance > /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference "
             "> /dev/null 2>&1"],
            timeout=5, capture_output=True, text=True)
        return True
    except Exception:
        return False


def build_model(model: str, native_layout: bool, script_dir: Path,
                mlir_source: Optional[Path] = None,
                out_dir_override: Optional[Path] = None) -> Path:
    """Build a model runner. Returns the runner exe path.

    `mlir_source` overrides the default `<script_dir>/<model>_v1.mlir` lookup
    (use this to point at the tflite-tagged NEW MLIR at the repo root, while
    keeping the runtime/ copy as the OLD MLIR baseline).

    `out_dir_override` overrides the default `lower_out/<model>_v1_(native|nofuse)`
    path — useful when the canonical path is unwritable (e.g. owned by root
    from a prior sudo run) and you want to direct artefacts elsewhere.
    """
    mlir_file = mlir_source if mlir_source else (script_dir / f"{model}_v1.mlir")
    if not mlir_file.is_file():
        raise FileNotFoundError(f"MLIR file not found: {mlir_file}")

    if out_dir_override is not None:
        out_dir = out_dir_override
    else:
        out_dir = script_dir / "lower_out" / f"{model}_v1{'_native' if native_layout else '_nofuse'}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stale root-owned out_dir from a prior sudo-flavored run is by far the
    # most common cause of `lower_to_timvx.py` failing with `Permission
    # denied` on the `mlir-timvx-opt -o ...` step. Detect early and emit a
    # clear actionable error instead of an opaque CalledProcessError.
    import os
    try:
        if out_dir.exists() and not os.access(out_dir, os.W_OK):
            raise PermissionError(
                f"out_dir is not writable by current user: {out_dir}\n"
                f"  Likely owned by root from a prior sudo run. Fix with:\n"
                f"    sudo chown -R $USER:$USER {out_dir.parent}")
    except OSError as e:
        raise RuntimeError(f"Could not stat {out_dir}: {e}")

    cmd = [
        sys.executable, str(script_dir / "lower_to_timvx.py"),
        str(mlir_file),
        "--out-dir", str(out_dir),
    ]
    if not native_layout:
        cmd.extend(["--no-fold-input-transpose", "--no-fold-output-transpose"])

    print(f"  Building {model} (native={native_layout}) "
          f"from {mlir_file} -> {out_dir}...",
          file=sys.stderr, flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"Build failed for {model}:\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(f"Failed to build {model}")

    runner = out_dir / f"{model}_v1_runner"
    if not runner.is_file():
        raise FileNotFoundError(f"Runner not found: {runner}")
    return runner


def run_bench(runner: Path, model: str, native: bool, mode: str, image: Path,
              iters: int, warmup: int, timeout: int, env: Dict,
              proc: str, zc: bool, par: str, T: int) -> Optional[BenchResult]:
    """
    Run a single benchmark mode, collect results, return as BenchResult.

    Directly calls benchmark_runner() without subprocess overhead.
    Passes num_workers via TIMVX_POOL_WORKERS environment variable.
    """
    # Set thread pool worker count via environment variable
    # Create a clean environment dict with only string values
    run_env = {k: str(v) for k, v in env.items() if v is not None}
    run_env["TIMVX_POOL_WORKERS"] = str(T)

    # Verify all values are strings before passing to subprocess
    for k, v in run_env.items():
        if not isinstance(v, str):
            print(f"[error] env[{k!r}] is {type(v).__name__}, not str", file=sys.stderr)
            return None

    bench_dict = benchmark_runner(
        runner, mode, image, iters, warmup, timeout,
        pin_cpus=None,  # Already set globally
        env=run_env,
        verbose=False  # Suppress individual run logs
    )

    if not bench_dict:
        return None

    # Convert benchmark dict to BenchResult
    # Note: stats line uses different key names (thru_mean, perimg_mean, etc.)
    # Prefer stats values when available (higher precision)
    return BenchResult(
        model=model,
        native_layout=native,
        processor=proc,
        zero_copy=zc,
        parallelism=par.capitalize(),
        num_workers=T,
        throughput=bench_dict.get("thru_mean") or bench_dict.get("throughput"),
        throughput_se=bench_dict.get("thru_se"),
        per_img_ms=bench_dict.get("perimg_mean") or bench_dict.get("per_img_ms"),
        per_img_se=bench_dict.get("perimg_se"),
        infer_mean=bench_dict.get("infer_mean"),
        infer_p99=bench_dict.get("infer_p99"),
        infer_se=bench_dict.get("infer_se"),
        e2e_mean=bench_dict.get("e2e_mean"),
        e2e_p99=bench_dict.get("e2e_p99"),
        e2e_se=bench_dict.get("e2e_se"),
        pre_mean=bench_dict.get("pre_mean"),
        pre_se=bench_dict.get("pre_se"),
        prewait_mean=bench_dict.get("prewait_mean"),
        prewait_se=bench_dict.get("prewait_se"),
        postwait_mean=bench_dict.get("postwait_mean"),
        postwait_se=bench_dict.get("postwait_se"),
        post_mean=bench_dict.get("post_mean"),
        post_se=bench_dict.get("post_se"),
        topk=bench_dict.get("topk"),
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=2000,
                    help="Benchmark iterations (default: 2000)")
    ap.add_argument("--warmup", type=int, default=100,
                    help="Warmup iterations (default: 100)")
    ap.add_argument("--models", type=str, default="resnet18,resnet50,resnet101,resnet152",
                    help="Comma-separated model list (default: all 4 resnets)")
    ap.add_argument("--timeout", type=int, default=600,
                    help="Per-mode timeout in seconds (default: 600)")
    ap.add_argument("--no-governor", action="store_true",
                    help="Skip setting CPU governors to performance")
    ap.add_argument("--image", type=Path, default=Path(__file__).parent / "cat105.jpg",
                    help="JPEG image for benchmarking (default: cat105.jpg)")
    ap.add_argument("--overwrite", action="store_true", dest="overwrite",
                    help="Overwrite results.json; default is to resume from it (skip existing modes)")
    ap.add_argument("--mode", type=str, default=None,
                    help="Comma-separated mode filters (e.g. 'resnet50,pool,zero_copy'). "
                         "Default: only run missing modes. With --overwrite: only run these modes fresh.")
    ap.add_argument("--results-file", type=Path, default=Path(__file__).parent / "run_full_bench_results.json",
                    help="JSON file for persisting results (default: run_full_bench_results.json)")
    ap.add_argument("--skip-build", action="store_true",
                    help="Skip model compilation; use existing runners from lower_out/")
    ap.add_argument("--mlir-source", type=Path, default=None,
                    help="Override the source MLIR location (default: "
                         "<runtime/>/<model>_v1.mlir). Use this to point at "
                         "an alternate MLIR copy — e.g. the tflite-tagged "
                         "NEW MLIR at the repo root — without renaming files. "
                         "Only supported when --models names a single model.")
    ap.add_argument("--out-subdir", type=str, default=None,
                    help="Override the lower_out subdirectory tag (default: "
                         "'<model>_v1_(native|nofuse)'). Useful when running "
                         "the same model against two MLIR sources side by "
                         "side without clobbering each other's runner.")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    # Set governors globally if requested
    if not args.no_governor:
        if apply_perf_governors():
            print("[gov] CPU/NPU governors set to performance", file=sys.stderr)
        else:
            print("[gov] could not set governors (run as root to eliminate DVFS jitter)",
                  file=sys.stderr)

    # Build environment using shared make_env from run_pipeline_bench
    # Filter out any None values to avoid subprocess errors
    env_raw = make_env(repo_root)
    env = {k: v for k, v in env_raw.items() if v is not None}

    # Verify all env values are strings (subprocess requirement)
    bad_keys = [k for k, v in env.items() if not isinstance(v, str)]
    if bad_keys:
        print(f"[error] env dict has non-string values for keys: {bad_keys}", file=sys.stderr)
        print(f"[error] env types: {[(k, type(env[k]).__name__) for k in bad_keys]}", file=sys.stderr)
        return 1

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if (args.mlir_source or args.out_subdir) and len(models) != 1:
        print(f"[error] --mlir-source / --out-subdir require a single model "
              f"(--models has {len(models)}: {models})", file=sys.stderr)
        return 1

    # Load existing results if resuming
    existing_results: Dict[str, BenchResult] = {}
    if not args.overwrite:
        existing_results = load_results_json(args.results_file)
        print(f"[sweep] Loaded {len(existing_results)} existing results from {args.results_file}",
              file=sys.stderr, flush=True)

    # Build all variants (or skip if --skip-build)
    built: Dict[tuple, Path] = {}  # (model, native_layout) -> runner

    def out_dir_for(model: str, native: bool) -> Path:
        tag = args.out_subdir or f"{model}_v1{'_native' if native else '_nofuse'}"
        return script_dir / "lower_out" / tag

    if args.skip_build:
        print(f"[sweep] Using existing runners (--skip-build)", file=sys.stderr, flush=True)
        for model in models:
            for native in [True, False]:
                runner = out_dir_for(model, native) / f"{model}_v1_runner"
                if not runner.is_file():
                    print(f"[error] {model} (native={native}): runner not found at {runner}", file=sys.stderr)
                    return 1
                built[(model, native)] = runner
    else:
        print(f"[sweep] Building models: {', '.join(models)}", file=sys.stderr, flush=True)
        for model in models:
            for native in [True, False]:
                try:
                    runner = build_model(model, native, script_dir,
                                          mlir_source=args.mlir_source,
                                          out_dir_override=out_dir_for(model, native))
                    built[(model, native)] = runner
                except Exception as e:
                    print(f"[error] {model} (native={native}): {e}", file=sys.stderr)
                    return 1

    print(f"[sweep] Running benchmarks: {args.iters} iters, {args.warmup} warmup",
          file=sys.stderr, flush=True)

    # Benchmark matrix
    # Modes: cpu / cpu_zero_copy / ppu / ppu_zero_copy × sequential / pipeline / pool / hybrid
    # Pool/hybrid also sweep T in {1, 2, 4, 8, 16}
    base_modes = [
        ("cpu", False),
        ("cpu_zero_copy", True),
        ("ppu", False),
        ("ppu_zero_copy", True),
    ]
    parallelisms = ["sequential", "pipeline", "pool", "hybrid"]
    workers = [1, 2, 4, 8, 16]

    # Flatten the base mode matrix
    all_base_modes = []
    for proc, zc in base_modes:
        for par in parallelisms:
            if par in ("pool", "hybrid"):
                # These support thread pool sweeps
                for T in workers:
                    mode_str = f"{proc}-{par}"
                    all_base_modes.append((mode_str, proc, par, zc, T))
            else:
                # sequential/pipeline run with T=1 only
                mode_str = f"{proc}-{par}"
                all_base_modes.append((mode_str, proc, par, zc, 1))

    # Filter modes based on --mode argument. Filter substrings are matched
    # case-insensitively against the FULL mode_id
    # (`<model>:native=<bool>:<proc>:zero_copy=<bool>:<par>:T=<int>`), so a
    # filter like `"resnet50:native=true:cpu:zero_copy=false:sequential:T=1"`
    # picks one specific row. Comma-separating multiple filters keeps the
    # original AND-of-substrings semantics for less specific spec strings.
    mode_filters = None
    if args.mode:
        mode_filters = [m.strip() for m in args.mode.split(",") if m.strip()]

    # No pre-filter at the modes-level — apply the full-mode-id filter inside
    # the (model, native, mode) loop below where the full id is known.
    modes_to_run = all_base_modes

    # Run all (model, native, mode) combinations
    all_results: List[BenchResult] = []
    row_id = 0
    for model in models:
        for native in [True, False]:
            runner = built[(model, native)]
            for mode_str, proc, par, zc, T in modes_to_run:
                mode_key = mode_str if T == 1 else f"{mode_str}-T{T}"
                mode_id = f"{model}:native={str(native).lower()}:{proc}:zero_copy={str(zc).lower()}:{par.lower()}:T={T}"

                # Skip rows that don't match the --mode filter.
                if mode_filters and not matches_mode_filter(mode_id, mode_filters):
                    continue

                # Check if we should skip this mode (already have results and resuming)
                if (not args.overwrite) and mode_id in existing_results:
                    all_results.append(existing_results[mode_id])
                    print(f"[{row_id:3d}] {model:12s} native={native} {mode_key:30s} [cached]",
                          file=sys.stderr, flush=True)
                    row_id += 1
                    continue

                print(f"[{row_id:3d}] {model:12s} native={native} {mode_key:30s}",
                      end=" ", file=sys.stderr, flush=True)
                row_id += 1

                bench = run_bench(runner, model, native, mode_str, args.image,
                                  args.iters, args.warmup, args.timeout, env,
                                  proc, zc, par, T)
                if bench:
                    all_results.append(bench)
                    print(f"✓", file=sys.stderr, flush=True)
                    # Save after each result for checkpoint safety
                    save_results_json(args.results_file, all_results)
                else:
                    print(f"✗", file=sys.stderr, flush=True)

    print(f"[sweep] Complete. Collected {len(all_results)} results.",
          file=sys.stderr, flush=True)

    # Save final results
    save_results_json(args.results_file, all_results)
    print(f"[sweep] Results saved to {args.results_file}", file=sys.stderr, flush=True)

    # Format results as separate LaTeX tables, one per ResNet variant
    # Only include results that match the current run's model/mode filters
    print()

    def fmt(v: Optional[float], decimals: int = 2) -> str:
        """Format a metric value, handling None."""
        if v is None:
            return "—"
        return f"{v:.{decimals}f}"

    def fmt_with_ci(v: Optional[float], se: Optional[float], decimals: int = 2) -> str:
        """Format a metric with 95% confidence interval (mean ± 1.96*SE)."""
        if v is None:
            return "—"
        if se is None:
            return f"{v:.{decimals}f}"
        margin = 1.96 * se
        return f"{v:.{decimals}f}$\\pm${margin:.{decimals}f}"

    def parallelism_order(par: str) -> int:
        """Map parallelism to sort order: sequential, pipeline, pool, hybrid."""
        order = {"sequential": 0, "pipeline": 1, "pool": 2, "hybrid": 3}
        return order.get(par.lower(), 99)

    for model in models:
        model_results = [r for r in all_results if r.model == model]
        if not model_results:
            continue

        model_name = model.replace('resnet', 'ResNet-')

        # Sort by (native, processor, scheduler, zero_copy, T) for consistent ordering
        sorted_results = sorted(model_results,
                               key=lambda x: (not x.native_layout, x.processor,
                                             parallelism_order(x.parallelism),
                                             x.zero_copy, x.num_workers))

        # Create separate table for this model
        # Use resizebox for dynamic scaling to page width
        print(r"\begin{table*}[htb]")
        print(r"\centering")
        print(f"\\caption{{{model_name} Benchmark Results}}")
        print(r"\resizebox{\textwidth}{!}{%")
        print(r"\begin{tabular}{ |l|l|l|l|c|r|r|r|r|r|r|r|r|r|r| }")
        print(r" \hline")
        print(r" Processor & Scheduler & Layout & Zero Copy & T & " +
              r"Throughput (imgs/s) & Per-Image (ms) & Run Mean (ms) & Run P99 (ms) & " +
              r"E2E Mean (ms) & E2E P99 (ms) & Pre (ms) & Pre-NN Wait (ms) & NN-Post Wait (ms) & Post (ms) \\")
        print(r" \hline")

        prev_layout = None
        prev_proc = None
        prev_par = None
        for result in sorted_results:
            proc_simple = simplify_processor(result.processor)
            sched_str = result.parallelism.capitalize()
            layout_str = "Native" if result.native_layout else "Legacy"
            zc_str = "Yes" if result.zero_copy else "No"
            T_str = str(result.num_workers)

            # Add separator when layout, processor, or parallelism changes
            if (prev_layout is not None and
                (prev_layout != result.native_layout or
                 prev_proc != result.processor or
                 prev_par != result.parallelism)):
                print(r" \hline")

            print(f" {proc_simple} & {sched_str} & {layout_str} & {zc_str} & {T_str} & " +
                  f"{fmt_with_ci(result.throughput, result.throughput_se, decimals=2)} & " +
                  f"{fmt_with_ci(result.per_img_ms, result.per_img_se)} & " +
                  f"{fmt_with_ci(result.infer_mean, result.infer_se)} & " +
                  f"{fmt(result.infer_p99)} & " +
                  f"{fmt_with_ci(result.e2e_mean, result.e2e_se)} & " +
                  f"{fmt(result.e2e_p99)} & " +
                  f"{fmt_with_ci(result.pre_mean, result.pre_se)} & " +
                  f"{fmt_with_ci(result.prewait_mean, result.prewait_se)} & " +
                  f"{fmt_with_ci(result.postwait_mean, result.postwait_se)} & " +
                  f"{fmt_with_ci(result.post_mean, result.post_se)} \\\\")

            prev_layout = result.native_layout
            prev_proc = result.processor
            prev_par = result.parallelism

        print(r" \hline")
        print(r"\end{tabular}}")
        print(r"\end{table*}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
