#!/usr/bin/env python3
"""lower_sample.py — drive TOSA → TIM-VX → C++ → executable on one sample.

Stages:
  1. tosa.mlir       --tosa-const-fold + --tosa-to-timvx-->  <name>.timvx.mlir
                     (the const-fold collapses BatchNorm scalar chains so
                      they don't hit TIM-VX's broken same-rank broadcast)
  2. timvx.mlir      --timvx-to-emitc-->                     <name>.emitc.mlir
  3. emitc.mlir      --mlir-to-cpp---->                      <name>.func.cpp
  4. (alongside)                                             timvx_runtime.h
  5. (rendered)                                              runner_main.cpp
  6. clang++ stages 3+5  ----link---->                       <name>_runner

Stage 6 is skipped (with a friendly note) if a built TIM-VX is not pointed
at by env. Stages 1-5 always run.

Required env (each overridable):
  REPO_ROOT        — repo root              (default: parent of this script's dir)
  MLIR_OPT         — mlir-timvx-opt         (default: $REPO_ROOT/build/bin)
  MLIR_TRANSLATE   — mlir-translate         (default: $REPO_ROOT/llvm-project/build/bin)
  TIM_VX_DIR       — TIM-VX source tree     (default: $REPO_ROOT/TIM-VX)
  TIM_VX_BUILD_DIR — built TIM-VX (libtim-vx.so + deps); if absent, link skipped.
  VIV_SDK_LIB_DIR  — Verisilicon driver SDK lib/ (OpenVX / OpenVXU live here);
                     defaults to EXTERNAL_VIV_SDK/lib if EXTERNAL_VIV_SDK is set.
  CXX              — host C++ compiler      (default: clang++-16, then clang++)
  CXXFLAGS         — host C++ flags         (default: "-O0 -pipe"; set to
                     "-O2 -g" etc. for an optimized debug build — runner is a
                     thin tim::vx::Graph harness so -O0 is the right default)
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# MLIR scalar tag → tim::vx::DataType enum suffix. Mirrors the EmitC pass.
DTYPE_TO_VX = {
    "f16": "FLOAT16", "f32": "FLOAT32",
    "i1":  "BOOL8",
    "i8":  "INT8",   "ui8":  "UINT8",
    "i16": "INT16",  "ui16": "UINT16",
    "i32": "INT32",  "ui32": "UINT32",
    "i64": "INT64",
}

# tensor<DIMSxDTYPE> e.g. tensor<1x224x224x3xf32>; rank-0 e.g. tensor<i64>.
TENSOR_RE = re.compile(r"tensor<((?:[0-9]+x)*)([a-z][a-z0-9]*)>")


def must_exec(path: Path, label: str) -> Path:
    if not path.exists() or not os.access(path, os.X_OK):
        sys.exit(f"missing/non-exec {label}: {path}")
    return path


def run(cmd):
    print("  $ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def resolve_paths(args) -> dict:
    script_dir = Path(__file__).resolve().parent
    repo_root = Path(os.environ.get("REPO_ROOT", script_dir.parent)).resolve()

    mlir_opt = Path(os.environ.get(
        "MLIR_OPT", repo_root / "build" / "bin" / "mlir-timvx-opt"))
    mlir_translate = Path(os.environ.get(
        "MLIR_TRANSLATE",
        repo_root / "llvm-project" / "build" / "bin" / "mlir-translate"))
    tim_vx_dir = Path(os.environ.get("TIM_VX_DIR", repo_root / "TIM-VX"))

    must_exec(mlir_opt, "MLIR_OPT")
    must_exec(mlir_translate, "MLIR_TRANSLATE")

    if not args.input.is_file():
        sys.exit(f"not a file: {args.input}")

    base = args.input.name
    for suffix in (".mlir", ".tosa"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]

    out_dir = args.out_dir or (script_dir / "lower_out" / base)
    out_dir.mkdir(parents=True, exist_ok=True)

    return dict(
        script_dir=script_dir,
        repo_root=repo_root,
        mlir_opt=mlir_opt,
        mlir_translate=mlir_translate,
        tim_vx_dir=tim_vx_dir,
        base=base,
        out_dir=out_dir,
        timvx_mlir=out_dir / f"{base}.timvx.mlir",
        emitc_mlir=out_dir / f"{base}.emitc.mlir",
        func_cpp=out_dir / f"{base}.func.cpp",
        runner_cpp=out_dir / "runner_main.cpp",
        exec_bin=out_dir / f"{base}_runner",
        runtime_h=script_dir / "timvx_runtime.h",
    )


# ----------------------------------------------------------------------------
# Stages 1–3: lower
# ----------------------------------------------------------------------------

def lower(p: dict, input_path: Path) -> None:
    print(f"[1/6] tosa-const-fold + tosa-layout-tag + tosa-layout-to-whcn + "
          f"canonicalize + tosa-to-timvx + timvx-conv1x1-to-fc "
          f"-> {p['timvx_mlir']}")
    # `--verify-each=false`: TOSA's spatial-op verifiers expect NHWC shape
    # relationships; once `--tosa-layout-to-whcn` permutes to WHCN those
    # checks fail, but `--tosa-to-timvx` immediately consumes them.
    # `--canonicalize` after WHCN folds the boundary transpose against the
    # front-end's NCHW->NHWC transpose into one combined permute.
    # `--timvx-conv1x1-to-fc` runs after the lowering: collapses tflite's
    # final-classifier 1x1 conv2d into a real FullyConnected (better NN-
    # engine path than Conv2D on Vivante).
    cmd = [p["mlir_opt"], "--verify-each=false",
           "--tosa-const-fold", "--tosa-fold-avgpool-reduce",
           "--tosa-layout-tag", "--tosa-layout-to-whcn",
            "--canonicalize", "--tosa-to-timvx", "--timvx-conv1x1-to-fc",
            input_path, "-o", p["timvx_mlir"]]
    run(cmd)

    print(f"[2/6] timvx-to-emitc         -> {p['emitc_mlir']}")
    run([p["mlir_opt"], "--timvx-to-emitc", p["timvx_mlir"], "-o", p["emitc_mlir"]])

    print(f"[3/6] mlir-translate         -> {p['func_cpp']}")
    run([p["mlir_translate"], "--mlir-to-cpp", p["emitc_mlir"], "-o", p["func_cpp"]])


# ----------------------------------------------------------------------------
# Stage 4: timvx_runtime.h is now a checked-in file alongside this script;
# nothing to generate. Build adds `-I<this dir>` to make the include resolve.
# ----------------------------------------------------------------------------

def stage4(p: dict) -> None:
    if not p["runtime_h"].is_file():
        sys.exit(f"missing runtime header: {p['runtime_h']}")
    print(f"[4/6] timvx_runtime.h        -> {p['runtime_h']} (included via -I)")


# ----------------------------------------------------------------------------
# Stage 5: render runner_main.cpp from the template.
# ----------------------------------------------------------------------------

def extract_fn_name(func_cpp_path: Path) -> str:
    """Pull the lowered function name from the first declaration in the .cpp.
    The line is shaped like `... <name>(... ) {` — we want <name>. The
    timvx-to-emitc pass renames `main` to `timvx_main`, so reading from the
    .cpp (post-rename) is more reliable than parsing the timvx MLIR."""
    first_line = func_cpp_path.read_text().split("\n", 1)[0]
    m = re.search(r"\b([A-Za-z_]\w*)\s*\(", first_line)
    if not m:
        sys.exit(f"could not extract function name from {func_cpp_path}")
    return m.group(1)


_QUANT_ATTR_RE = re.compile(
    r"output_scale\s*=\s*([\d\.eE\+\-]+)\s*:\s*f64.*?"
    r"output_zp\s*=\s*(-?\d+)\s*:\s*i64",
    re.S,
)


def parse_input_specs(timvx_mlir_path: Path) -> list[dict]:
    """One InputSpec per tensor argument of the timvx-stage func.func, in
    declaration order. The graph SSA arg isn't added until timvx-to-emitc,
    so the timvx MLIR signature still reflects only the model inputs.

    For quantized inputs we also recover (scale, zp): the first
    `output_scale = ... : f64, output_zp = ... : i64` pair in the IR
    corresponds to the input's quant params, since the input's first
    consumer (typically timvx.pad / timvx.transpose) inherits its scale.
    Returns scale=0.0 / zp=0 when no quant attrs are present (FP32 model).
    """
    text = timvx_mlir_path.read_text()
    m = re.search(r"func\.func\s+@\w+\s*\((.*?)\)\s*->", text, re.S)
    if not m:
        sys.exit(f"could not find func.func signature in {timvx_mlir_path}")

    # First quant attr pair in the body is the input's scale/zp (the IR is
    # linear and the first quant-aware consumer is the input's pad).
    qm = _QUANT_ATTR_RE.search(text)
    quant_scale = float(qm.group(1)) if qm else 0.0
    quant_zp = int(qm.group(2)) if qm else 0

    specs: list[dict] = []
    for hit in TENSOR_RE.finditer(m.group(1)):
        dims_str, dtype_tag = hit.group(1), hit.group(2)
        dims = [d for d in dims_str.rstrip("x").split("x") if d]
        if dtype_tag not in DTYPE_TO_VX:
            sys.exit(f"unsupported tensor element type '{dtype_tag}' "
                     f"in arg {len(specs)} of {timvx_mlir_path}")

        # Promote asym int8 inputs to u8 with zp+128 — mirrors the
        # `shouldPromoteI8AsymToU8` rule applied to every spec on the
        # EmitC side. The runner's input tensor must match the
        # tim::vx::TensorSpec the lowered C++ uses for its first user
        # op, otherwise the runtime auto-DataConvert kicks in (and
        # COMPILE_FAILs at IO on this chip). For the byte data the user
        # feeds in, the runner-side conversion is the same bit-flip
        # (XOR 0x80) the `print_output` path applies in reverse.
        runner_dtype = DTYPE_TO_VX[dtype_tag]
        runner_scale = (quant_scale if dtype_tag in ("i8", "ui8") else 0.0)
        runner_zp    = (quant_zp    if dtype_tag in ("i8", "ui8") else 0)
        if dtype_tag == "i8" and quant_scale != 0.0:
            runner_dtype = "UINT8"
            runner_zp    = quant_zp + 128

        specs.append({
            "dims": dims,
            "dtype": runner_dtype,
            "quant_scale": runner_scale,
            "quant_zp": runner_zp,
        })
    return specs


def render_runner(p: dict) -> None:
    runner_tpl = p["script_dir"] / "runner_main.cpp.tpl"
    if not runner_tpl.is_file():
        sys.exit(f"missing template: {runner_tpl}")

    fn_name = extract_fn_name(p["func_cpp"])
    print(f"  (lowered fn: {fn_name})")

    specs = parse_input_specs(p["timvx_mlir"])
    if not specs:
        sys.exit("function takes no tensor arguments — runner has nothing to feed")

    # Each spec renders as `{shape, dtype, scale, zp}`. Trailing scale/zp
    # are zero when the model is FP32 (the runner skips Quantization() then).
    # f-string `{{` / `}}` are literal C++ braces.
    def _fmt(s: dict) -> str:
        return (f"    {{{{{', '.join(s['dims'])}}}, "
                f"tim::vx::DataType::{s['dtype']}, "
                f"{s.get('quant_scale', 0.0):.10g}, "
                f"{s.get('quant_zp', 0)}}},")
    input_specs = "\n".join(_fmt(s) for s in specs)
    input_args = ", ".join(f"inputs[{i}]" for i in range(len(specs)))

    body = (runner_tpl.read_text()
            .replace("__BASE__",        p["base"])
            .replace("__FUNC_NAME__",   fn_name)
            .replace("__INPUT_SPECS__", input_specs)
            .replace("__INPUT_ARGS__",  input_args))
    p["runner_cpp"].write_text(body)
    print(f"  ({len(specs)} non-graph input(s) wired)")


# ----------------------------------------------------------------------------
# Stage 6: compile + link.
# ----------------------------------------------------------------------------

def find_timvx_lib_dir(tim_vx_build_dir: Path) -> Path:
    candidates = [
        tim_vx_build_dir / "install" / "lib",
        tim_vx_build_dir / "lib",
        tim_vx_build_dir / "src" / "tim",
        tim_vx_build_dir,
    ]
    for cand in candidates:
        if (cand / "libtim-vx.so").is_file():
            return cand
    sys.exit(f"could not locate libtim-vx.so under {tim_vx_build_dir}")


def pick_cxx() -> str:
    cxx = os.environ.get("CXX") or shutil.which("clang++-16") or "clang++"
    return cxx


def _spawn(cmd):
    """Like run() but non-blocking — caller waits on the returned Popen."""
    print("  $ " + " ".join(str(c) for c in cmd))
    return subprocess.Popen([str(c) for c in cmd])


def build(p: dict) -> None:
    tim_vx_build_dir = Path(os.environ.get(
        "TIM_VX_BUILD_DIR", p["tim_vx_dir"] / "build"))
    if not tim_vx_build_dir.is_dir():
        print(
            f"[6/6] skipping host build: TIM_VX_BUILD_DIR does not exist:\n"
            f"      {tim_vx_build_dir}\n"
            f"      Build TIM-VX (or set TIM_VX_BUILD_DIR) to enable linking.\n",
            file=sys.stderr)
        print(f"\nartifacts in: {p['out_dir']}")
        return

    timvx_lib_dir = find_timvx_lib_dir(tim_vx_build_dir)

    # OpenVX/OpenVXU come from the Verisilicon driver SDK, NOT TIM-VX itself.
    default_sdk = (Path(os.environ.get("EXTERNAL_VIV_SDK",
                       Path.home() / "ufs" / "home" / "radxa" / "ai-sdk" /
                       "unified-tina" / "timvx-sdk")) / "lib")
    viv_sdk_lib_dir = Path(os.environ.get("VIV_SDK_LIB_DIR", default_sdk))
    if not viv_sdk_lib_dir.is_dir():
        sys.exit(
            f"VIV_SDK_LIB_DIR does not exist: {viv_sdk_lib_dir}\n"
            f"set VIV_SDK_LIB_DIR (or EXTERNAL_VIV_SDK) to point at your "
            f"driver SDK.")

    # CustomGemm extras: header dir + the .cc that carries the static
    # kernel_id_/kernel_name_ definitions. Lives under TIM-VX/samples/.
    custom_gemm_dir = p["tim_vx_dir"] / "samples" / "custom_op_test"
    custom_gemm_cc = custom_gemm_dir / "custom_gemm.cc"
    if not custom_gemm_cc.is_file():
        sys.exit(f"expected CustomGemm sample at {custom_gemm_cc}")

    cxx = pick_cxx()

    # The runner is a thin harness around tim::vx::Graph; the perf-critical
    # work runs on the NPU inside libtim-vx.so. The included <base>.func.cpp
    # is the whole model body as one giant TU and dominates wall time at
    # -O2, so default to -O0 for fast iteration. Override via CXXFLAGS env
    # (e.g. CXXFLAGS="-O2 -g") when you actually want an optimized binary.
    cxxflags = os.environ.get("CXXFLAGS", "-O0 -pipe").split()

    # lld links noticeably faster than bfd on this workload; fall back
    # silently if it isn't installed.
    link_extra = ["-fuse-ld=lld"] if shutil.which("ld.lld") else []

    compile_flags = [
        "-std=c++17", "-fPIC", *cxxflags,
        f"-I{p['tim_vx_dir']}/include",
        f"-I{p['script_dir']}",          # timvx_runtime.h
        f"-I{custom_gemm_dir}",          # custom_gemm.h
    ]

    runner_obj = p["out_dir"] / "runner_main.o"
    custom_obj = p["out_dir"] / "custom_gemm.o"

    # Compile both TUs in parallel — runner_main.cpp (with its #included
    # whole-model func.cpp) is by far the heavier one, but custom_gemm.cc
    # still costs a few seconds and there's no reason to serialize them.
    print(f"[6/6] {cxx} (parallel compile + link) -> {p['exec_bin']}")
    procs = [
        _spawn([cxx, *compile_flags, "-c", p["runner_cpp"], "-o", runner_obj]),
        _spawn([cxx, *compile_flags, "-c", custom_gemm_cc, "-o", custom_obj]),
    ]
    rcs = [pr.wait() for pr in procs]
    if any(rcs):
        sys.exit(f"compile failed (runner_main.o={rcs[0]}, "
                 f"custom_gemm.o={rcs[1]})")

    run([
        cxx, *cxxflags, *link_extra,
        runner_obj, custom_obj,
        f"-L{timvx_lib_dir}",
        f"-L{viv_sdk_lib_dir}",
        "-ltim-vx", "-lOpenVX", "-lOpenVXU",
        "-Wl,--unresolved-symbols=ignore-in-shared-libs",
        f"-Wl,-rpath,{timvx_lib_dir}",
        f"-Wl,-rpath,{viv_sdk_lib_dir}",
        "-o", p["exec_bin"],
    ])


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="Input .tosa.mlir file")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory "
                         "(default: debug_scripts/lower_out/<basename>)")
    ap.add_argument("--skip-build", action="store_true",
                    help="Stop after generating C++; skip clang link step")
    args = ap.parse_args()

    p = resolve_paths(args)

    lower(p, args.input)
    stage4(p)
    print(f"[5/6] generate runner_main   -> {p['runner_cpp']}")
    render_runner(p)

    if args.skip_build:
        print("[6/6] skipping host build (--skip-build)\n")
        print(f"artifacts in: {p['out_dir']}")
        return

    build(p)

    print("\ndone. artifacts:")
    print(f"  timvx mlir : {p['timvx_mlir']}")
    print(f"  emitc mlir : {p['emitc_mlir']}")
    print(f"  c++ source : {p['func_cpp']}")
    print(f"  runtime    : {p['runtime_h']}")
    print(f"  runner     : {p['runner_cpp']}")
    print(f"  executable : {p['exec_bin']}")


if __name__ == "__main__":
    main()
