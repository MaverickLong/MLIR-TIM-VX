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
        target=getattr(args, "target", "runner"),
        timvx_mlir=out_dir / f"{base}.timvx.mlir",
        emitc_mlir=out_dir / f"{base}.emitc.mlir",
        func_cpp=out_dir / f"{base}.func.cpp",
        runner_cpp=out_dir / "runner_main.cpp",
        exec_bin=out_dir / f"{base}_runner",
        # NBG path (only used when target == "nbg"): a builder binary
        # that calls Graph::CompileToBinary and writes the .nbg blob,
        # plus a generic VIPLite runner.
        compile_nbg_cpp=out_dir / "compile_to_nbg.cpp",
        compile_nbg_bin=out_dir / f"{base}_compile_nbg",
        run_nbg_cpp=out_dir / "run_nbg.cpp",
        run_nbg_bin=out_dir / f"{base}_run_nbg",
        nbg_blob=out_dir / f"{base}.nbg",
        runtime_h=script_dir / "timvx_runtime.h",
    )


# ----------------------------------------------------------------------------
# Stages 1–3: lower
# ----------------------------------------------------------------------------

def lower(p: dict, input_path: Path) -> None:
    print(f"[1/6] tosa-const-fold + tosa-fold-avgpool-reduce + canonicalize "
          f"+ tosa-quant-anchor + tosa-to-timvx + timvx-quant-residual-fuse "
          f"+ canonicalize + timvx-conv1x1-to-fc -> {p['timvx_mlir']}")
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
    # `--timvx-conv1x1-to-fc` runs after the lowering: collapses tflite's
    # final-classifier 1x1 conv2d into a real FullyConnected (better NN-
    # engine path than Conv2D on Vivante).
    cmd = [p["mlir_opt"],
           "--tosa-const-fold", "--tosa-fold-avgpool-reduce",
            "--canonicalize", "--tosa-quant-anchor",
            "--tosa-to-timvx",
            "--timvx-quant-residual-fuse",
            # post-fuse canonicalize: DCE the orphaned fp32 dequant
            # chains that residual-fuse leaves behind once their last
            # consumer (the original final cast) has been replaced.
            "--canonicalize",
            "--timvx-conv1x1-to-fc",
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

        # Three cases for an i8 input:
        #   (1) Asymmetric quant context (zp != 0): promote to UINT8
        #       with zp+128. The harness/caller is expected to have
        #       XOR'd the input bytes by 0x80 so the unsigned
        #       interpretation aligns with the asymmetric int8
        #       semantics. This matches `shouldPromoteI8AsymToU8` on
        #       the EmitC side and avoids a runtime auto-DataConvert
        #       at the IO boundary (which COMPILE_FAILs on this chip).
        #   (2) Symmetric quant context (zp == 0, scale != 0): keep as
        #       INT8 but stamp Quant(scale, 0) on the runner tensor.
        #       Bytes are signed; TIM-VX's cast ops dequantize via the
        #       quant info, giving real_value = signed_byte * scale.
        #   (3) No quant context detected (cast-as-first-user, etc.):
        #       fall back to Quant(1.0, 0). Without this, TIM-VX's
        #       int->float cast graph fails to compile because the
        #       node setup requires a Quantization on the int side.
        runner_dtype = DTYPE_TO_VX[dtype_tag]
        runner_scale = quant_scale if dtype_tag in ("i8", "ui8") else 0.0
        runner_zp    = quant_zp    if dtype_tag in ("i8", "ui8") else 0
        if dtype_tag == "i8":
            if quant_scale != 0.0 and quant_zp != 0:
                # asymmetric → promote to UINT8
                runner_dtype = "UINT8"
                runner_zp    = quant_zp + 128
            elif quant_scale == 0.0:
                # no quant context → identity quant so TIM-VX cast works
                runner_scale = 1.0
                runner_zp    = 0

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

    def render_one(tpl_path: Path, out_path: Path) -> None:
        body = (tpl_path.read_text()
                .replace("__BASE__",        p["base"])
                .replace("__FUNC_NAME__",   fn_name)
                .replace("__INPUT_SPECS__", input_specs)
                .replace("__INPUT_ARGS__",  input_args))
        out_path.write_text(body)

    render_one(runner_tpl, p["runner_cpp"])
    print(f"  ({len(specs)} non-graph input(s) wired)")

    # NBG target additionally renders compile_to_nbg.cpp and run_nbg.cpp.
    # run_nbg.cpp.tpl is fully generic (no template subs needed) — it
    # reads input shapes from the NBG via vip_query_input — but we still
    # pass it through the substitutor for uniformity.
    if p.get("target") == "nbg":
        compile_tpl = p["script_dir"] / "compile_to_nbg.cpp.tpl"
        run_nbg_tpl = p["script_dir"] / "run_nbg.cpp.tpl"
        for tpl, dst in ((compile_tpl, p["compile_nbg_cpp"]),
                          (run_nbg_tpl, p["run_nbg_cpp"])):
            if not tpl.is_file():
                sys.exit(f"missing template: {tpl}")
            render_one(tpl, dst)
        print(f"  (NBG sources rendered: compile_to_nbg.cpp + run_nbg.cpp)")


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
    cxxflags = os.environ.get("CXXFLAGS", "-O0 -pipe").split()

    # lld links noticeably faster than bfd on this workload; fall back
    # silently if it isn't installed.
    link_extra = ["-fuse-ld=lld"] if shutil.which("ld.lld") else []

    compile_flags = [
        "-std=c++17", "-fPIC", *cxxflags,
        f"-I{p['tim_vx_dir']}/include",
        f"-I{p['script_dir']}",          # timvx_runtime.h, custom_ops/*.h
    ]

    runner_obj = p["out_dir"] / "runner_main.o"
    custom_obj = p["out_dir"] / "custom_gemm.o"
    custom_rs_obj = p["out_dir"] / "custom_reduce_sum.o"
    compile_nbg_obj = p["out_dir"] / "compile_to_nbg.o"

    # Compile all TUs in parallel — runner_main.cpp (with its #included
    # whole-model func.cpp) is by far the heaviest, but custom_gemm.cc and
    # custom_reduce_sum.cc each cost a few seconds and there's no reason
    # to serialize them. When --target=nbg, compile_to_nbg.cpp is also a
    # heavy TU (it #includes the same model body) — fan that out too.
    target = p.get("target", "runner")
    print(f"[6/6] {cxx} (parallel compile + link) -> {p['exec_bin']}")
    parallel = [
        _spawn([cxx, *compile_flags, "-c", p["runner_cpp"], "-o", runner_obj]),
        _spawn([cxx, *compile_flags, "-c", custom_gemm_cc, "-o", custom_obj]),
        _spawn([cxx, *compile_flags, "-c", custom_reduce_sum_cc,
                 "-o", custom_rs_obj]),
    ]
    if target == "nbg":
        parallel.append(_spawn([cxx, *compile_flags, "-c",
                                 p["compile_nbg_cpp"], "-o", compile_nbg_obj]))
    rcs = [pr.wait() for pr in parallel]
    if any(rcs):
        sys.exit(f"compile failed (rcs={rcs})")

    run([
        cxx, *cxxflags, *link_extra,
        runner_obj, custom_obj, custom_rs_obj,
        f"-L{timvx_lib_dir}",
        f"-L{viv_sdk_lib_dir}",
        "-ltim-vx", "-lOpenVX", "-lOpenVXU",
        "-Wl,--unresolved-symbols=ignore-in-shared-libs",
        f"-Wl,-rpath,{timvx_lib_dir}",
        f"-Wl,-rpath,{viv_sdk_lib_dir}",
        "-o", p["exec_bin"],
    ])

    if target == "nbg":
        # NBG compile binary: same TIM-VX/OVXLIB stack as the runner —
        # only the entrypoint differs (Graph::CompileToBinary instead
        # of Graph::Compile + Run).
        run([
            cxx, *cxxflags, *link_extra,
            compile_nbg_obj, custom_obj, custom_rs_obj,
            f"-L{timvx_lib_dir}",
            f"-L{viv_sdk_lib_dir}",
            "-ltim-vx", "-lOpenVX", "-lOpenVXU",
            "-Wl,--unresolved-symbols=ignore-in-shared-libs",
            f"-Wl,-rpath,{timvx_lib_dir}",
            f"-Wl,-rpath,{viv_sdk_lib_dir}",
            "-o", p["compile_nbg_bin"],
        ])

        # NBG runtime binary: VIPLite only — no tim-vx, no OVXLIB. The
        # SDK lib dir contains libVIPlite.so + libVIPuser.so + headers
        # under .../inc/. VIPLITE_SDK_DIR overrides the default search.
        viplite_dir = Path(os.environ.get(
            "VIPLITE_SDK_DIR",
            Path.home() / "ufs" / "home" / "radxa" / "ai-sdk" /
            "viplite-tina" / "lib" / "aarch64-none-linux-gnu" / "v1.13"))
        viplite_inc = viplite_dir / "inc"
        if not viplite_inc.is_dir():
            sys.exit(
                f"VIPLITE_SDK_DIR/inc does not exist: {viplite_inc}\n"
                f"set VIPLITE_SDK_DIR to a viplite-tina/.../v1.13/ directory.")
        run([
            cxx, *cxxflags, *link_extra,
            "-std=c++17",
            f"-I{viplite_inc}",
            p["run_nbg_cpp"],
            f"-L{viplite_dir}",
            "-lVIPlite", "-lVIPuser",
            "-Wl,--unresolved-symbols=ignore-in-shared-libs",
            f"-Wl,-rpath,{viplite_dir}",
            "-o", p["run_nbg_bin"],
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
    ap.add_argument("--target", choices=("runner", "nbg"), default="runner",
                    help="`runner` (default): build the OVXLIB-backed runner. "
                         "`nbg`: also build a TIM-VX -> NBG compile binary "
                         "and a VIPLite runtime binary; the latter "
                         "skips OVXLIB/OpenVX entirely at inference time.")
    ap.add_argument("--gen-nbg", action="store_true",
                    help="With --target=nbg, also run the compile binary "
                         "to materialise <base>.nbg next to the binaries. "
                         "One-time cost; reuse the .nbg afterwards.")
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

    if p.get("target") == "nbg" and args.gen_nbg:
        # Run the compile binary to materialise the .nbg blob. The
        # binary's DT_RPATH covers libtim-vx.so and libOpenVX*.so, but
        # libOpenVX.so transitively dlopens libVSC.so / libGAL.so /
        # libArchModelSw.so / libNNArchPerf.so by name, and rpath
        # doesn't propagate. Set LD_LIBRARY_PATH explicitly, mirroring
        # run_timvx.py.
        tim_vx_install = Path(os.environ.get(
            "TIM_VX_BUILD_DIR", p["tim_vx_dir"] / "build" / "install"))
        timvx_lib_dir = find_timvx_lib_dir(tim_vx_install)
        default_sdk = (Path(os.environ.get("EXTERNAL_VIV_SDK",
                            Path.home() / "ufs" / "home" / "radxa" /
                            "ai-sdk" / "unified-tina" / "timvx-sdk")) / "lib")
        viv_sdk_lib_dir = Path(os.environ.get("VIV_SDK_LIB_DIR", default_sdk))
        ld_path = os.pathsep.join(filter(None, [
            str(timvx_lib_dir),
            str(viv_sdk_lib_dir),
            os.environ.get("LD_LIBRARY_PATH", ""),
        ]))
        env = {**os.environ, "LD_LIBRARY_PATH": ld_path}
        print(f"\n[gen-nbg] {p['compile_nbg_bin']} {p['nbg_blob']}")
        print(f"  LD_LIBRARY_PATH += {timvx_lib_dir}:{viv_sdk_lib_dir}")
        subprocess.run([str(p["compile_nbg_bin"]), str(p["nbg_blob"])],
                       env=env, check=True)

    print("\ndone. artifacts:")
    print(f"  timvx mlir : {p['timvx_mlir']}")
    print(f"  emitc mlir : {p['emitc_mlir']}")
    print(f"  c++ source : {p['func_cpp']}")
    print(f"  runtime    : {p['runtime_h']}")
    print(f"  runner     : {p['runner_cpp']}")
    print(f"  executable : {p['exec_bin']}")
    if p.get("target") == "nbg":
        print(f"  nbg compile: {p['compile_nbg_bin']}")
        print(f"  nbg runtime: {p['run_nbg_bin']}")
        if args.gen_nbg:
            print(f"  nbg blob   : {p['nbg_blob']}")


if __name__ == "__main__":
    main()
