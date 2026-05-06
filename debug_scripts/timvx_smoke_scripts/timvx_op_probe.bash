#!/usr/bin/env bash
#
# timvx_op_probe.bash — build + run timvx_op_probe.cpp.
# Maps each (op, dtype) cell in the MLIR-TIMVX subset to compile/run
# success on the live NPU. See the cpp for what's tested.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$script_dir/../.." && pwd)}"
TIM_VX_DIR="${TIM_VX_DIR:-$REPO_ROOT/TIM-VX}"
TIM_VX_BUILD_DIR="${TIM_VX_BUILD_DIR:-$REPO_ROOT/TIM-VX/build/install}"
VIV_SDK_LIB_DIR="${VIV_SDK_LIB_DIR:-${EXTERNAL_VIV_SDK:-$HOME/ufs/home/radxa/ai-sdk/unified-tina/timvx-sdk}/lib}"

build="$script_dir/.build"
mkdir -p "$build"
patched_sdk_dir="$VIV_SDK_LIB_DIR"

# libCLC's JIT EVIS / CL shader compiler resolves `#include "cl_viv_vx_ext.h"`
# against `$VIVANTE_SDK_DIR/include/CL/`. Bundle from TIM-VX's prebuilt-sdk
# (the runtime unified-tina SDK doesn't ship it). Idempotent.
viv_shim="$build/viv_sdk_shim"
viv_prebuilt_header="$TIM_VX_DIR/prebuilt-sdk/x86_64_linux/include/CL/cl_viv_vx_ext.h"
if [[ ! -f "$viv_shim/include/CL/cl_viv_vx_ext.h" && -f "$viv_prebuilt_header" ]]; then
  mkdir -p "$viv_shim/include/CL"
  cp "$viv_prebuilt_header" "$viv_shim/include/CL/"
fi

timvx_lib_dir=""
for c in "$TIM_VX_BUILD_DIR/install/lib" "$TIM_VX_BUILD_DIR/lib" \
         "$TIM_VX_BUILD_DIR/src/tim" "$TIM_VX_BUILD_DIR"; do
  if [[ -f "$c/libtim-vx.so" ]]; then timvx_lib_dir="$c"; break; fi
done
[[ -n "$timvx_lib_dir" ]] || { echo "could not locate libtim-vx.so" >&2; exit 1; }

CXX="${CXX:-}"
if [[ -z "$CXX" ]]; then
  if command -v clang++-16 >/dev/null; then CXX=clang++-16
  else CXX=clang++; fi
fi

src="$script_dir/timvx_op_probe.cpp"
exe="$build/timvx_op_probe"
echo "[build] $CXX $src -> $exe" >&2
"$CXX" -std=c++17 -O2 \
  -I"$TIM_VX_DIR/include" \
  -I"$TIM_VX_DIR" \
  -L"$timvx_lib_dir" -L"$patched_sdk_dir" \
  "$src" \
  -ltim-vx -lOpenVX -lOpenVXU \
  -Wl,--unresolved-symbols=ignore-in-shared-libs \
  -Wl,-rpath,"$timvx_lib_dir" \
  -Wl,-rpath,"$patched_sdk_dir" \
  -o "$exe"

echo "[run]   $exe" >&2
echo "---" >&2
LD_LIBRARY_PATH="$timvx_lib_dir:$patched_sdk_dir:${LD_LIBRARY_PATH:-}" \
VIVANTE_SDK_DIR="$viv_shim" \
  "$exe"
