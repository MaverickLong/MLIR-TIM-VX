#!/usr/bin/env bash
#
# timvx_smoke.bash — build + run timvx_smoke.cpp with the same dynamic-link
# setup the generated runner uses. Used to bisect "Create tensor fail!"
# errors without the lowered model in the picture.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$script_dir/.." && pwd)}"
TIM_VX_DIR="${TIM_VX_DIR:-$REPO_ROOT/TIM-VX}"
TIM_VX_BUILD_DIR="${TIM_VX_BUILD_DIR:-$REPO_ROOT/TIM-VX/build/install}"
VIV_SDK_LIB_DIR="${VIV_SDK_LIB_DIR:-${EXTERNAL_VIV_SDK:-$HOME/ufs/home/radxa/ai-sdk/unified-tina/timvx-sdk}/lib}"

# # Reuse the cached patched mirror + glibc shim that run_sample.bash builds.
build="$script_dir/.build"
mkdir -p "$build"
# patched_sdk_dir="$build/viv-sdk-patched"
patched_sdk_dir="$VIV_SDK_LIB_DIR"

# libCLC's JIT EVIS / CL shader compiler resolves `#include "cl_viv_vx_ext.h"`
# against `$VIVANTE_SDK_DIR/include/CL/`. The unified-tina runtime SDK
# doesn't ship that header, but TIM-VX's prebuilt-sdk does — bundle a
# copy of it under .build/ so any op that needs a shader (e.g. Pow u8)
# can compile. Idempotent: skipped on re-runs once the file is in place.
viv_shim="$build/viv_sdk_shim"
viv_prebuilt_header="$TIM_VX_DIR/prebuilt-sdk/x86_64_linux/include/CL/cl_viv_vx_ext.h"
if [[ ! -f "$viv_shim/include/CL/cl_viv_vx_ext.h" && -f "$viv_prebuilt_header" ]]; then
  mkdir -p "$viv_shim/include/CL"
  cp "$viv_prebuilt_header" "$viv_shim/include/CL/"
fi
# shim_so="$build/libtimvx_glibc_compat.so"
# [[ -f "$shim_so" ]] || { echo "missing $shim_so — run debug_scripts/run_sample.bash once first to build the shim" >&2; exit 1; }
# [[ -d "$patched_sdk_dir" ]] || { echo "missing $patched_sdk_dir — run debug_scripts/run_sample.bash once first" >&2; exit 1; }

timvx_lib_dir=""
for c in "$TIM_VX_BUILD_DIR/install/lib" "$TIM_VX_BUILD_DIR/lib" \
         "$TIM_VX_BUILD_DIR/src/tim" "$TIM_VX_BUILD_DIR"; do
  if [[ -f "$c/libtim-vx.so" ]]; then timvx_lib_dir="$c"; break; fi
done
[[ -n "$timvx_lib_dir" ]] || { echo "could not locate libtim-vx.so under $TIM_VX_BUILD_DIR" >&2; exit 1; }

CXX="${CXX:-}"
if [[ -z "$CXX" ]]; then
  if command -v clang++-16 >/dev/null; then CXX=clang++-16
  else CXX=clang++; fi
fi

src="$script_dir/timvx_smoke.cpp"
exe="$build/timvx_smoke"
echo "[build] $CXX $src -> $exe" >&2
"$CXX" -std=c++17 -O2 \
  -I"$TIM_VX_DIR/include" \
  -L"$timvx_lib_dir" -L"$patched_sdk_dir" \
  "$src" \
  -ltim-vx -lOpenVX -lOpenVXU \
  -Wl,--unresolved-symbols=ignore-in-shared-libs \
  -Wl,-rpath,"$timvx_lib_dir" \
  -Wl,-rpath,"$patched_sdk_dir" \
  -o "$exe"

echo "[run]   $exe" >&2
echo "---" >&2
LD_PRELOAD="${LD_PRELOAD:+:$LD_PRELOAD}" \
LD_LIBRARY_PATH="$timvx_lib_dir:$patched_sdk_dir:${LD_LIBRARY_PATH:-}" \
VIVANTE_SDK_DIR="$viv_shim" \
  "$exe"
