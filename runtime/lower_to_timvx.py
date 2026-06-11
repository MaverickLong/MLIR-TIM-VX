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
    subprocess.run([str(c) for c in cmd], check=True, capture_output=True)


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

def lower(p: dict, input_path: Path, *,
          with_fuse: bool = True, with_conv1x1_to_fc: bool = True,
          with_dequant_fuse: bool = True,
          with_canonicalize_transpose: bool = True,
          with_fold_input_transpose: bool = True,
          with_fold_output_transpose: bool = True,
          with_arith_fold: bool = True,
          with_cse: bool = True) -> None:
    # The WHCN boundary is op-local: each `Conv2DOpConversion` /
    # `MaxPool2DConversion` / `AvgPool2DConversion` / `RescaleConvFusion`
    # emits explicit NHWC<->WHCN `timvx.transpose` pairs around its
    # `timvx.conv2d` / `timvx.pool2d`. The rest of the IR stays in MLIR-
    # row-major NHWC, so TOSA verifiers are satisfied at every pipeline
    # boundary.
    # `--tosa-quant-anchor` materialises absolute (S, Z) on every
    # quantized i8 tensor type as `!quant.uniform<i8:f32, S:Z>` so the
    # downstream lowerings have a single source of truth instead of
    # rederiving (and possibly disagreeing on) per-tensor scales.
    # `--timvx-promote-i8-to-u8` rewrites every i8-quantized tensor to its
    # u8 equivalent (zp += 128, bytes XOR 0x80, ±128 inserted at f32 cast
    # boundaries) once and for all. After that, every downstream pass
    # assumes u8 — no scattered "is this an i8 that needs promotion"
    # checks across `TimvxToEmitC` / `CastConversion` / `Strip*`.
    # `--timvx-quant-residual-fuse` (toggle via with_fuse): collapses the
    # `cast→sub→mul→add→clip→mul→add→cast` fp32 dequant detour around
    # every residual add into a single quantized `timvx.add`. Toggle off
    # to bisect QRF-introduced regressions against the unfused path.
    # `--timvx-conv1x1-to-fc` (toggle via with_conv1x1_to_fc): collapses
    # tflite's final-classifier 1x1 conv2d into a real FullyConnected
    # (better NN-engine path than Conv2D on Vivante). Toggle off to
    # keep the conv2d form (e.g. when chasing an FC-specific quirk).
    # QRF runs BEFORE `--timvx-promote-i8-to-u8`. Reasoning: QRF's
    # pattern matcher and cross-checks were written for the i8-space
    # TFLite chain shape (`cast(i8→f32) → sub(zp_i8) → mul(scale)`,
    # tail `mul(invScale) → add(zp_i8) → cast(f32→i8)`). Once promote
    # has injected the ±128 byte-shift compensation around every
    # f32↔u8 cast, the chain shape varies depending on whether the
    # zp const was uniquely-owned (folded in place) or shared (extra
    # `add(+128)` op inserted) — and QRF's matcher has no clean way
    # to discriminate. Running QRF first means it sees the canonical
    # i8 chain everywhere; promote then handles BOTH the fused
    # `timvx.add` (just bumps its `output_zp` by +128) AND the
    # un-fused dequant chains (inserts ±128 compensation as today)
    # with a single consistent rule.
    cmd = [p["mlir_opt"],
           "--tosa-const-fold", "--tosa-fold-avgpool-reduce",
            "--canonicalize", "--tosa-quant-anchor",
            "--tosa-to-timvx",
            "--canonicalize"]
    label = ("tosa-const-fold + tosa-fold-avgpool-reduce + canonicalize + "
             "tosa-quant-anchor + tosa-to-timvx + canonicalize")
    if with_fuse:
        cmd += ["--timvx-quant-residual-fuse", "--canonicalize"]
        label += " + timvx-quant-residual-fuse + canonicalize"
    cmd += ["--timvx-promote-i8-to-u8", "--canonicalize"]
    label += " + timvx-promote-i8-to-u8 + canonicalize"
    # `--timvx-dequant-fuse`: collapse any per-operand dequant chains
    # left behind by QRF (or never QRF candidates — e.g. the chain
    # feeding the global-avg-pool's `reduce_sum`) into single
    # `timvx.dataconvert` ops. Safe even when --no-fuse: if there are
    # no chains to fuse, this pass is a no-op.
    if with_dequant_fuse:
        cmd += ["--timvx-dequant-fuse", "--canonicalize"]
        label += " + timvx-dequant-fuse + canonicalize"
    # `--timvx-canonicalize-transpose`: push the NHWC<->WHCN boundary
    # transposes that `tosa-to-timvx` sandwiched around each spatial op
    # upward through see-through elementwise ops (cast / dataconvert /
    # clip / add / sub / multiply / slice). Pairs that bracket a pure
    # elementwise chain compose to identity and erase. Run after
    # promote-i8-to-u8 / QRF / dequant-fuse so the elementwise chain
    # shape is final (otherwise transposes would block in places those
    # passes were going to remove). Const operands are physically
    # permuted at compile time by the existing TransposeOfConstFold
    # canonicalizer; runtime sees no extra transposes.
    if with_canonicalize_transpose:
        cmd += ["--timvx-canonicalize-transpose", "--canonicalize"]
        label += " + timvx-canonicalize-transpose + canonicalize"
    # `--timvx-fold-input-transpose`: retire the one transpose
    # canonicalize-transpose can't — the entry NCHW→WHCN transpose
    # stranded on `%arg0` (func args are a propagation barrier). Folds
    # the permute into the arg's type (`1x3x224x224` → `224x224x3x1`)
    # and stamps `timvx.input_layout = "whcn"` on the arg. The runner's
    # preprocessor reads that marker and emits raw WHCN (channel-planar)
    # bytes directly, so the host-side layout-convert AND this device
    # transpose both vanish — they were mutually inverse. Runs after
    # canonicalize-transpose (which leaves the transpose sitting on the
    # arg) and is a no-op on models without the canonical entry transpose.
    if with_fold_input_transpose:
        cmd += ["--timvx-fold-input-transpose", "--canonicalize"]
        label += " + timvx-fold-input-transpose + canonicalize"
    # `--timvx-fold-output-transpose`: the output mirror — collapse the
    # trailing byte-preserving reshape→transpose→reshape→transpose chain
    # (classifier `{1000,1}` FC output → `{1,1000}` row-major result) into
    # the func result type. The chain only re-slots size-1 dims, so it's a
    # byte-level identity and top-K is layout-invariant — bit-identical
    # result, minus the (tiny) device transpose dispatches. No-op on models
    # whose output genuinely needs a spatial transpose.
    if with_fold_output_transpose:
        cmd += ["--timvx-fold-output-transpose", "--canonicalize"]
        label += " + timvx-fold-output-transpose + canonicalize"
    # `--timvx-arith-fold`: collapse `sub(sub(x, a), b)` / `add(add(x, a), b)`
    # / `multiply(multiply(x, a), b)` (and the mixed add/sub forms) into a
    # single op when both consts are 1-element f32 splats. Targets the head
    # (`sub(±128) → sub(zp)`) and tail (`add(Zout) → add(-128)`) of every
    # dequant/requant chain that `timvx-promote-i8-to-u8` leaves behind.
    if with_arith_fold:
        cmd += ["--timvx-arith-fold", "--canonicalize"]
        label += " + timvx-arith-fold + canonicalize"
    # `--timvx-cse`: structural common-subexpression elimination over pure
    # timvx ops. The fan-out propagation in `timvx-canonicalize-transpose`
    # materialises the same upstream `dataconvert` / `add` / `clip` per
    # downstream consumer; this pass collapses the duplicates back to one
    # op each. Uses `OperationEquivalence` so the (S, Z) on quant types
    # gates merging — two consts with identical data but different quant
    # context stay separate.
    if with_cse:
        cmd += ["--timvx-cse"]
        label += " + timvx-cse"
    if with_conv1x1_to_fc:
        cmd += ["--timvx-conv1x1-to-fc"]
        label += " + timvx-conv1x1-to-fc"
    cmd += [input_path, "-o", p["timvx_mlir"]]
    print(f"[1/6] {label} -> {p['timvx_mlir']}")
    run(cmd)

    # `extern-const-dir`: the EmitC pass writes any `timvx.const` whose
    # numel exceeds `extern-const-threshold` (default 1024) to a
    # `_timvx_const_<id>.bin` file under this directory, and emits a
    # `mmap_const(...)` initializer in place of the inline `static const
    # T[]`. We point it at the same out_dir that holds the runner; the
    # runtime's default search path is `dirname(/proc/self/exe)`, so the
    # bins resolve without any env tweaking.
    print(f"[2/6] timvx-to-emitc         -> {p['emitc_mlir']}")
    run([p["mlir_opt"],
         f"--timvx-to-emitc=extern-const-dir={p['out_dir']}",
         p["timvx_mlir"], "-o", p["emitc_mlir"]])

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


# Regex for an `output_scale = X : f64, output_zp = Y : i64` pair within
# a single op's `{...}` attribute dict. Used to recover an arg's quant
# context from its first user — only that op's attrs are looked at, not
# the function-wide search the previous version did. The character class
# allows `+`/`-` exponents (e.g. `1.0e+00`).
_OP_OUTPUT_QUANT_RE = re.compile(
    r"output_scale\s*=\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s*:\s*f64\s*,"
    r"\s*output_zp\s*=\s*(-?\d+)\s*:\s*i64"
)


# Regex to pluck `timvx.output_scale = X : f64, timvx.output_zp = Y : i64`
# from a func.func arg's attribute dict. Order may vary; we accept either
# `output_scale` first or `output_zp` first.
_ARG_QUANT_RE = re.compile(
    r"timvx\.output_scale\s*=\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s*:\s*f64"
    r".*?timvx\.output_zp\s*=\s*(-?\d+)\s*:\s*i64"
    r"|"
    r"timvx\.output_zp\s*=\s*(-?\d+)\s*:\s*i64"
    r".*?timvx\.output_scale\s*=\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\s*:\s*f64",
    re.S,
)


def _find_arg_quant(sig_arg_text: str, body: str,
                     arg_name: str) -> tuple[float, int]:
    """Recover (scale, zp) for a function argument.

    Two sources, in priority order:

    (1) `timvx.output_scale` / `timvx.output_zp` arg attrs declared on
        the func signature. This is the canonical source — it mirrors
        the discardable `output_scale` / `output_zp` that every quant
        timvx op carries on its result, so an arg attr declares "treat
        this BlockArgument as if it were the result of an op with these
        quant values". The compiler's `getProducerQuant` reads these
        same names, keeping the convention unified.

    (2) The first user op's `output_scale` / `output_zp` attrs. Used
        when the test/model didn't declare arg attrs but the arg flows
        through a quant-aware op whose attrs propagated the input scale
        through. Only valid for ops that pass quant through unchanged
        (cast i8→f32, transpose, reshape, ...). For ops that *change*
        the scale (timvx.add with output_scale != input_scale, conv,
        rescale), this would mis-report — so the test-and-model
        convention is to declare arg attrs explicitly when the first
        user op is one of those.

    If neither source yields a quant context, returns (0.0, 0) so the
    caller skips the i8→u8 promotion.
    """
    # (1) arg-attr lookup. sig_arg_text is the slice of the func
    # signature for this specific arg (`%argN: tensor<...> {attrs}`).
    qm = _ARG_QUANT_RE.search(sig_arg_text)
    if qm:
        # Either group set (1, 2) or group set (3, 4) is filled,
        # depending on whether scale appeared before zp.
        s = qm.group(1) or qm.group(4)
        z = qm.group(2) or qm.group(3)
        return float(s), int(z)

    # (2) first-user-op fallback. Find the first user line of `arg_name`
    # and pull its `output_scale` / `output_zp`.
    user_re = re.compile(
        r"^\s*%\w+\s*=\s*(?:\"[^\"]+\"|\S+)\s*[^\n]*?"
        + re.escape(arg_name)
        + r"(?:\s*,|\s*\)|\s+)[^\n]*$",
        re.M,
    )
    for line in user_re.finditer(body):
        qm = _OP_OUTPUT_QUANT_RE.search(line.group(0))
        if qm:
            return float(qm.group(1)), int(qm.group(2))
    return 0.0, 0


def _split_balanced(s: str, sep: str) -> list[str]:
    """Split `s` on top-level occurrences of `sep`, ignoring `sep`s
    that appear inside any of `<>`, `()`, `[]`, or `{}`. Used so we can
    walk a TOSA func arg list whose element types now include
    `!quant.uniform<i8:f32, S:Z>` (nested `<>` + embedded comma)."""
    out: list[str] = []
    depth = 0
    start = 0
    openers = {"<": ">", "(": ")", "[": "]", "{": "}"}
    closers = set(openers.values())
    stack: list[str] = []
    for i, c in enumerate(s):
        if c in openers:
            stack.append(openers[c])
            depth += 1
        elif c in closers:
            if stack and stack[-1] == c:
                stack.pop()
                depth -= 1
        elif depth == 0 and c == sep:
            out.append(s[start:i])
            start = i + 1
    out.append(s[start:])
    return out


def _split_arg_type_attrs(rest: str) -> tuple[str, str]:
    """Given `tensor<...> {attrs...}` or just `tensor<...>`, return
    (type_str, attrs_str). Uses bracket counting because the `tensor<>`
    can contain a nested `!quant.uniform<...>` whose `>` would confuse a
    naive split."""
    depth = 0
    end = -1
    for i, c in enumerate(rest):
        if c == "<": depth += 1
        elif c == ">":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return rest.strip(), ""
    type_str = rest[:end].strip()
    attrs_str = rest[end:].strip()
    return type_str, attrs_str


def parse_input_specs(timvx_mlir_path: Path) -> list[dict]:
    """One InputSpec per tensor argument of the timvx-stage func.func, in
    declaration order. The graph SSA arg isn't added until timvx-to-emitc,
    so the timvx MLIR signature still reflects only the model inputs.

    For each arg we also try to recover (scale, zp): we scan the body
    for the arg's first user op and read its `output_scale`/`output_zp`
    attrs. Most TIM-VX ops that respect quant (pad, transpose, conv,
    add, etc.) carry their input's scale forward as output_scale, so
    that's the arg's effective scale. If the first user is a numerical
    cast (i8->f32), no quant context exists for the arg and we leave
    scale=0 — the runner won't promote i8 to u8.
    """
    text = timvx_mlir_path.read_text()
    m = re.search(r"func\.func\s+@\w+\s*\((.*?)\)\s*->", text, re.S)
    if not m:
        sys.exit(f"could not find func.func signature in {timvx_mlir_path}")

    body_match = re.search(r"\{(.*)\}\s*\Z", text, re.S)
    body = body_match.group(1) if body_match else text

    specs: list[dict] = []
    # Args are parsed manually: the tensor element type can be
    # `!quant.uniform<i8:f32, S:Z>` which has nested `<>` and embedded
    # commas, so a single regex over the arg list isn't enough — we
    # walk the arg-list character by character with bracket counting.
    # Each arg's chunk is then split into (name, type, optional attrs).
    arg_chunks = _split_balanced(m.group(1), sep=",")
    for chunk in arg_chunks:
        chunk = chunk.strip()
        if not chunk: continue
        head, _, rest = chunk.partition(":")
        arg_name = head.strip()
        rest = rest.strip()
        # Split off optional `{attrs}` from the type using brace counting.
        type_str, arg_attrs = _split_arg_type_attrs(rest)
        if not type_str.startswith("tensor<") or not type_str.endswith(">"):
            sys.exit(f"unexpected arg type '{type_str}' for {arg_name} "
                     f"in {timvx_mlir_path}")
        tspec = type_str[len("tensor<"):-1]  # inside the outer <...>

        # tspec is e.g. "1x16xi8", "1x4x4x1xi8", or
        # "1x4x4x1x!quant.uniform<i8:f32, 0.0078125:4>". Split off
        # dimensions (leading `[0-9]+x` runs) from the element type.
        dim_run = re.match(r"((?:[0-9]+x)+)", tspec)
        if not dim_run:
            sys.exit(f"could not parse tensor type '{tspec}' in arg "
                     f"{len(specs)} of {timvx_mlir_path}")
        dims = dim_run.group(1).rstrip("x").split("x")
        elem_type = tspec[dim_run.end():]

        # Recover (dtype_tag, S, Z) from the element type.
        type_quant_scale = 0.0
        type_quant_zp = 0
        type_quant_present = False
        if elem_type.startswith("!quant.uniform<"):
            # `!quant.uniform<i8:f32, S:Z>` — pluck the storage type and
            # the (scale, zp) pair. Storage type is i8 / ui8 / u8.
            # Zero-point part `:Z` is omitted by MLIR's printer when Z=0.
            qm2 = re.match(
                r"!quant\.uniform<\s*(u?i\d+)\s*:\s*[a-z0-9]+\s*,\s*"
                r"([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)"
                r"(?:\s*:\s*(-?\d+))?\s*>$",
                elem_type)
            if not qm2:
                sys.exit(f"could not parse !quant.uniform element type "
                         f"'{elem_type}' in arg {len(specs)} of "
                         f"{timvx_mlir_path}")
            dtype_tag = qm2.group(1)
            type_quant_scale = float(qm2.group(2))
            type_quant_zp = int(qm2.group(3)) if qm2.group(3) else 0
            type_quant_present = True
        else:
            tm2 = re.match(r"([a-z][a-z0-9]*)$", elem_type)
            if not tm2:
                sys.exit(f"could not parse element type '{elem_type}' in "
                         f"arg {len(specs)} of {timvx_mlir_path}")
            dtype_tag = tm2.group(1)
        if dtype_tag not in DTYPE_TO_VX:
            sys.exit(f"unsupported tensor element type '{dtype_tag}' "
                     f"in arg {len(specs)} of {timvx_mlir_path}")

        if type_quant_present:
            # tosa-quant-anchor stamped (S, Z) on the type — that's the
            # source of truth.
            quant_scale, quant_zp = type_quant_scale, type_quant_zp
        else:
            quant_scale, quant_zp = _find_arg_quant(arg_attrs, body, arg_name)

        # `--timvx-promote-i8-to-u8` has rewritten every quantized i8
        # tensor (storage type and zp) to its u8 equivalent before the
        # signature got dumped, so by the time we read the timvx.mlir
        # signature here every quantized tensor is already u8 with the
        # +128-shifted zp. We just emit the spec verbatim.
        # The one residual case we still patch is dtype with no quant
        # context at all (e.g. a fp32 graph) — TIM-VX's cast op needs a
        # Quantization on the int side to compile, so we stamp identity
        # (scale=1.0, zp=0) when scale is unset.
        runner_dtype = DTYPE_TO_VX[dtype_tag]
        runner_scale = quant_scale if dtype_tag in ("i8", "ui8") else 0.0
        runner_zp    = quant_zp    if dtype_tag in ("i8", "ui8") else 0
        if dtype_tag in ("i8", "ui8") and quant_scale == 0.0:
            runner_scale = 1.0
            runner_zp    = 0

        # `timvx-fold-input-transpose` stamps `timvx.input_layout = "whcn"`
        # on an arg whose leading NCHW→WHCN transpose it folded into the
        # type. When present, `dims` are ALREADY TIM-VX innermost-first
        # (WHCN for 4D), so the runner must feed raw bytes in that order —
        # the preprocessor emits channel-planar (CHW) directly and skips
        # the MLIR-row-major→TIM-VX layout-convert.
        timvx_native = bool(re.search(
            r'timvx\.input_layout\s*=\s*"whcn"', arg_attrs))

        specs.append({
            "dims": dims,
            "dtype": runner_dtype,
            "quant_scale": runner_scale,
            "quant_zp": runner_zp,
            "timvx_native": timvx_native,
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
                f"{s.get('quant_zp', 0)}, "
                f"{'true' if s.get('timvx_native') else 'false'}}},")
    input_specs = "\n".join(_fmt(s) for s in specs)
    input_args = ", ".join(f"inputs[{i}]" for i in range(len(specs)))

    def render_one(tpl_path: Path, out_path: Path) -> None:
        body = (tpl_path.read_text()
                .replace("__BASE__",        p["base"])
                .replace("__FUNC_NAME__",   fn_name)
                .replace("__INPUT_SPECS__", input_specs)
                .replace("__INPUT_ARGS__",  input_args))
        out_path.write_text(body)

    render_one(runner_tpl, p["runner_cpp"])
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
                       Path("/") / "home"/ "radxa" / "ufs" / "home" / "radxa" / "ai-sdk" /
                       "unified-tina" / "timvx-sdk")) / "lib")
    viv_sdk_lib_dir = Path(os.environ.get("VIV_SDK_LIB_DIR", default_sdk))
    if not viv_sdk_lib_dir.is_dir():
        sys.exit(
            f"VIV_SDK_LIB_DIR does not exist: {viv_sdk_lib_dir}\n"
            f"set VIV_SDK_LIB_DIR (or EXTERNAL_VIV_SDK) to point at your "
            f"driver SDK.")

    # Project-owned custom OpenCL ops: header dir + .cc files carrying
    # the static kernel_id_/kernel_name_ definitions. Live under
    # example/custom_ops/ — see TIM-VX/docs/customized_op.md.
    custom_ops_dir = p["script_dir"] / "custom_ops"
    custom_gemm_cc = custom_ops_dir / "custom_gemm.cc"
    custom_reduce_sum_cc = custom_ops_dir / "custom_reduce_sum.cc"
    if not custom_gemm_cc.is_file():
        sys.exit(f"expected custom_gemm.cc at {custom_gemm_cc}")
    if not custom_reduce_sum_cc.is_file():
        sys.exit(f"expected custom_reduce_sum.cc at {custom_reduce_sum_cc}")

    cxx = pick_cxx()

    # The runner is a thin harness around tim::vx::Graph; the perf-critical
    # work runs on the NPU inside libtim-vx.so. The included <base>.func.cpp
    # is the whole model body as one giant TU and dominates wall time at
    # -O2, so default to -O0 for fast iteration. Override via CXXFLAGS env
    # (e.g. CXXFLAGS="-O2 -g") when you actually want an optimized binary.
    cxxflags = os.environ.get("CXXFLAGS", "-O2 -DNDEBUG").split()

    # lld links noticeably faster than bfd on this workload; fall back
    # silently if it isn't installed.
    link_extra = ["-fuse-ld=lld"] if shutil.which("ld.lld") else []

    # libjpeg-turbo headers / libs. The Radxa OS image installs the
    # shared libs at /usr/lib/aarch64-linux-gnu/libjpeg.so* but the dev
    # headers aren't in the default /usr/include search path — they
    # live under $REPO_ROOT/ufs/usr/include/ (with jconfig.h in the
    # aarch64 subdir). The runner's `pipeline/jpeg.h` wraps libjpeg
    # for the JPEG pre-processor; add the include / lib paths
    # unconditionally so the runner builds without extra env tweaks.
    jpeg_include = Path(os.environ.get(
        "LIBJPEG_INCLUDE", "/home/radxa/ufs/usr/include"))
    jpeg_lib = Path(os.environ.get(
        "LIBJPEG_LIB", "/home/radxa/ufs/usr/lib/aarch64-linux-gnu"))

    # OpenCV headers / libs. The CPU JPEG preprocessor uses cv::imdecode
    # + cv::cvtColor + cv::resize for parity with the vendor's reference
    # `class_pre.cpp` — same SIMD-optimised decode/resize path. Headers
    # live under $REPO_ROOT/ufs/usr/include/opencv4 on the Radxa image;
    # the .so files are alongside libjpeg in the aarch64 lib dir.
    opencv_include = Path(os.environ.get(
        "OPENCV_INCLUDE", "/home/radxa/ufs/usr/include/opencv4"))

    # OpenCV's transitive deps (libtbb.so.2 / libwebp.so.6 /
    # libtiff.so.5 / libIlmImf-2_5.so.25 / …) only exist as old major
    # versions in `{jpeg_lib}` — newer ones in /usr/lib are
    # ABI-incompatible. We need the dynamic linker to find those at
    # runtime. The cleanest way is to rpath `{jpeg_lib}` directly, but
    # that dir ALSO contains an OLDER libstdc++.so.6 (3.4.28) that
    # gets picked over the system's newer one (3.4.33) and the binary
    # fails to load with "GLIBCXX_3.4.29 not found". DT_RUNPATH search
    # order doesn't help here because libstdc++ then gets resolved via
    # OpenCV's transitive lookup which considers our rpath. The fix is
    # a curated shim directory: symlinks to every .so* in `{jpeg_lib}`
    # EXCEPT libstdc++ / libc / libm / libgcc_s (the runtime libs
    # that must come from the system). We rpath this shim dir; the
    # system C++ runtime resolves through /etc/ld.so.cache.
    opencv_shim = p["script_dir"] / ".build" / "opencv_shim"
    opencv_shim.mkdir(parents=True, exist_ok=True)
    # Wipe stale symlinks so we don't accumulate references to files
    # that may have moved.
    for old in opencv_shim.iterdir():
        if old.is_symlink():
            old.unlink()
    _SHIM_SKIP_PREFIXES = ("libstdc++.", "libc.", "libc-",
                            "libm.", "libm-", "libgcc_s.",
                            "libpthread.", "libdl.", "librt.",
                            "libresolv.", "ld-")
    # The OpenCV chain pulls deps from FOUR `$REPO_ROOT/ufs/` lib
    # dirs (libtbb / libopencv_* in usr/lib/aarch64; libgdal in
    # usr/lib; libpcre.so.3 in lib/aarch64; etc.). Glob all of them,
    # first-match-wins on duplicate names so the more specific
    # aarch64 subdir takes precedence over its parent.
    _shim_search_dirs = [
        jpeg_lib,                  # ufs/usr/lib/aarch64-linux-gnu
        jpeg_lib.parent,           # ufs/usr/lib
        Path("/home/radxa/ufs/lib/aarch64-linux-gnu"),
        Path("/home/radxa/ufs/lib"),
    ]
    for src_dir in _shim_search_dirs:
        if not src_dir.is_dir():
            continue
        for src in src_dir.glob("*.so*"):
            if any(src.name.startswith(p) for p in _SHIM_SKIP_PREFIXES):
                continue
            link = opencv_shim / src.name
            if link.exists() or link.is_symlink():
                continue
            link.symlink_to(src)

    compile_flags = [
        "-std=c++17", "-fPIC", "-pthread", *cxxflags,
        f"-I{p['tim_vx_dir']}/include",
        # TIM-VX internal headers — needed by `pipeline/preproc_rgb_op.h`,
        # which subclasses `BuiltinOp` to expose OVXLIB's
        # `VSI_NN_OP_PRE_PROCESS_RGB` op (not in TIM-VX's public C++
        # surface). `src/tim/vx/` gives us `op_impl.h`, etc.; the inner
        # `internal/include` is the OVXLIB API (vsi_nn_pub.h chain).
        f"-I{p['tim_vx_dir']}/src/tim/vx",
        f"-I{p['tim_vx_dir']}/src/tim/vx/internal/include",
        # The OVXLIB internal chain transitively pulls `<VX/vx_khr_cnn.h>`
        # (and friends) — those are part of the Verisilicon driver SDK,
        # not TIM-VX. Sibling to the driver's `lib/` we already use for
        # `-L`/rpath; resolve the include dir from the same default.
        f"-I{viv_sdk_lib_dir.parent}/include",
        f"-I{p['script_dir']}",          # timvx_runtime.h, custom_ops/*.h
        # `-idirafter` so the libjpeg path is searched ONLY for
        # otherwise-not-found headers (jpeglib.h, jerror.h, jconfig.h,
        # jmorecfg.h). Plain `-I` would shadow the system <stdlib.h>
        # via the dirs's stale glibc fragments — observed: a thicket of
        # "expected function body after function declarator" errors out
        # of /usr/include/stdlib.h before we ever hit our code.
        "-idirafter", str(jpeg_include),
        "-idirafter", str(jpeg_include / "aarch64-linux-gnu"),
        # Same idea for the OpenCV headers (they pull in their own
        # <opencv2/core/cvdef.h> chain — keep them after system
        # headers so glibc remains canonical).
        "-idirafter", str(opencv_include),
    ]

    runner_obj = p["out_dir"] / "runner_main.o"
    custom_obj = p["out_dir"] / "custom_gemm.o"
    custom_rs_obj = p["out_dir"] / "custom_reduce_sum.o"

    # Compile all TUs in parallel — runner_main.cpp (with its #included
    # whole-model func.cpp) is by far the heaviest, but custom_gemm.cc and
    # custom_reduce_sum.cc each cost a few seconds and there's no reason
    # to serialize them.
    print(f"[6/6] {cxx} (parallel compile + link) -> {p['exec_bin']}")
    parallel = [
        _spawn([cxx, *compile_flags, "-c", p["runner_cpp"], "-o", runner_obj]),
        _spawn([cxx, *compile_flags, "-c", custom_gemm_cc, "-o", custom_obj]),
        _spawn([cxx, *compile_flags, "-c", custom_reduce_sum_cc,
                 "-o", custom_rs_obj]),
    ]
    rcs = [pr.wait() for pr in parallel]
    if any(rcs):
        sys.exit(f"compile failed (rcs={rcs})")

    run([
        cxx, *cxxflags, *link_extra, "-pthread",
        runner_obj, custom_obj, custom_rs_obj,
        f"-L{timvx_lib_dir}",
        f"-L{viv_sdk_lib_dir}",
        f"-L{jpeg_lib}",
        "-ltim-vx", "-lOpenVX", "-lOpenVXU",
        # libjpeg is still pulled in by `pipeline/jpeg.h` (used by
        # the PPU preprocessor + the runner's JPEG-dim probe).
        # OpenCV's imgcodecs has its own libjpeg link, but keeping
        # ours explicit is harmless and means the include path stays
        # resolved.
        "-ljpeg",
        # OpenCV: cv::imdecode (imgcodecs), cv::cvtColor + cv::resize
        # (imgproc), plus cv::Mat (core). All three live in
        # /home/radxa/ufs/usr/lib/aarch64-linux-gnu/ alongside
        # libjpeg, so `-L{jpeg_lib}` covers them too.
        "-lopencv_core", "-lopencv_imgcodecs", "-lopencv_imgproc",
        "-Wl,--unresolved-symbols=ignore-in-shared-libs",
        # DT_RPATH (the LEGACY, pre-RUNPATH form) on purpose: DT_RPATH
        # is searched for the binary's DT_NEEDED AND for every
        # transitively-loaded library's DT_NEEDED too. DT_RUNPATH
        # only covers the binary's direct deps, which means libtbb /
        # libwebp / libtiff (loaded by libopencv_core, NOT by us
        # directly) would fall through to /etc/ld.so.cache and the
        # ABI-incompatible new majors would get picked.
        #
        # The `opencv_shim` dir is curated to NOT contain libstdc++ /
        # libc / libgcc_s / etc., so lookups for those fall through
        # past the rpath to /etc/ld.so.cache (system /lib version,
        # which matches what clang and libtim-vx were built against).
        "-Wl,--disable-new-dtags",
        f"-Wl,-rpath,{timvx_lib_dir}",
        f"-Wl,-rpath,{viv_sdk_lib_dir}",
        f"-Wl,-rpath,{opencv_shim}",
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
    # Pipeline knobs — defaults match the production lowering. compare_targets.py
    # passes --no-fuse to drive the fused/nofuse comparison without re-rolling
    # the rest of the pipeline.
    ap.add_argument("--no-fuse", action="store_true",
                    help="Skip --timvx-quant-residual-fuse (the residual-add "
                         "QRF fusion). Use to bisect QRF-introduced regressions "
                         "against the unfused fp32 dequant chain.")
    ap.add_argument("--no-conv1x1-to-fc", action="store_true",
                    help="Skip --timvx-conv1x1-to-fc (the 1x1-conv→FC "
                         "rewrite). Use to keep the conv2d form when chasing "
                         "an FC-specific kernel quirk.")
    ap.add_argument("--no-dequant-fuse", action="store_true",
                    help="Skip --timvx-dequant-fuse (the cast+sub+mul → "
                         "dataconvert peephole). Use to bisect dequant-fuse-"
                         "introduced regressions against the explicit "
                         "cast+sub+mul chain.")
    ap.add_argument("--no-canonicalize-transpose", action="store_true",
                    help="Skip --timvx-canonicalize-transpose (pushes "
                         "boundary transposes upward through elementwise "
                         "ops until sandwich pairs cancel). Use to bisect "
                         "transpose-elimination-introduced regressions.")
    ap.add_argument("--no-fold-input-transpose", action="store_true",
                    help="Skip --timvx-fold-input-transpose (folds the entry "
                         "NCHW→WHCN transpose into %(prog)s's input arg type "
                         "and marks it whcn-native, so the preprocessor emits "
                         "WHCN bytes directly). Use to keep the legacy "
                         "MLIR-row-major input + device transpose.")
    ap.add_argument("--no-fold-output-transpose", action="store_true",
                    help="Skip --timvx-fold-output-transpose (collapses the "
                         "trailing byte-preserving reshape/transpose chain "
                         "into the func result type). Use to keep the legacy "
                         "device output transposes.")
    ap.add_argument("--no-arith-fold", action="store_true",
                    help="Skip --timvx-arith-fold (chained scalar-const "
                         "add/sub/multiply fold). Use to bisect arith-"
                         "fold-introduced regressions.")
    ap.add_argument("--no-cse", action="store_true",
                    help="Skip --timvx-cse (structural CSE of pure timvx "
                         "ops). Use to bisect CSE-introduced regressions.")
    args = ap.parse_args()

    p = resolve_paths(args)

    lower(p, args.input,
          with_fuse=not args.no_fuse,
          with_conv1x1_to_fc=not args.no_conv1x1_to_fc,
          with_dequant_fuse=not args.no_dequant_fuse,
          with_canonicalize_transpose=not args.no_canonicalize_transpose,
          with_fold_input_transpose=not args.no_fold_input_transpose,
          with_fold_output_transpose=not args.no_fold_output_transpose,
          with_arith_fold=not args.no_arith_fold,
          with_cse=not args.no_cse)
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
