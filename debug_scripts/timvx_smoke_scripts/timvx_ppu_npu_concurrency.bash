#!/usr/bin/env bash
#
# timvx_ppu_npu_concurrency.bash — build + run timvx_ppu_npu_concurrency.cpp.
# Measures whether the A733's PPU/shader path and NN-engine NPU can run two
# independent tim::vx graphs in parallel. See the .cpp header for what each
# mode measures and how the verdict is decided.
#
# Required at run time:
#   - $REPO_ROOT/example/lower_out/resnet50_v1/_timvx_const_*.bin
#     (mmap'd by resnet50_v1.func.cpp; we point $TIMVX_CONSTS_DIR at this).
#
# Build mirrors example/lower_to_timvx.py's pattern: the main TU #includes
# both timvx_runtime.h and resnet50_v1.func.cpp, and we additionally compile
# + link the custom OpenCL ops (custom_gemm.cc, custom_reduce_sum.cc) that
# timvx_runtime.h unconditionally pulls in.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$script_dir/../.." && pwd)}"
TIM_VX_DIR="${TIM_VX_DIR:-$REPO_ROOT/TIM-VX}"
TIM_VX_BUILD_DIR="${TIM_VX_BUILD_DIR:-$REPO_ROOT/TIM-VX/build/install}"
VIV_SDK_LIB_DIR="${VIV_SDK_LIB_DIR:-${EXTERNAL_VIV_SDK:-$HOME/ufs/home/radxa/ai-sdk/unified-tina/timvx-sdk}/lib}"
EXAMPLE_DIR="$REPO_ROOT/example"
RESNET_DIR="$EXAMPLE_DIR/lower_out/resnet50_v1"

[[ -d "$RESNET_DIR" ]] || {
  echo "missing $RESNET_DIR — generate it via example/lower_to_timvx.py first" >&2
  exit 1
}
[[ -f "$EXAMPLE_DIR/custom_ops/custom_gemm.cc" ]] || {
  echo "missing custom_ops/custom_gemm.cc under $EXAMPLE_DIR" >&2
  exit 1
}

build="$script_dir/.build"
mkdir -p "$build"

# libCLC's JIT EVIS / CL shader compiler resolves `#include "cl_viv_vx_ext.h"`
# against $VIVANTE_SDK_DIR/include/CL/. Bundle from TIM-VX's prebuilt-sdk
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
  if   command -v clang++-16 >/dev/null; then CXX=clang++-16
  elif command -v clang++    >/dev/null; then CXX=clang++
  else                                       CXX=g++; fi
fi

cxxflags=( -std=c++17 -O2 -fPIC -pthread
           -I"$TIM_VX_DIR/include"
           -I"$EXAMPLE_DIR"
           -I"$RESNET_DIR" )

custom_gemm_obj="$build/custom_gemm.o"
custom_rs_obj="$build/custom_reduce_sum.o"
main_obj="$build/timvx_ppu_npu_concurrency.o"
exe="$build/timvx_ppu_npu_concurrency"
src="$script_dir/timvx_ppu_npu_concurrency.cpp"

echo "[build] compile custom ops" >&2
"$CXX" "${cxxflags[@]}" -c "$EXAMPLE_DIR/custom_ops/custom_gemm.cc"       -o "$custom_gemm_obj"
"$CXX" "${cxxflags[@]}" -c "$EXAMPLE_DIR/custom_ops/custom_reduce_sum.cc" -o "$custom_rs_obj"

echo "[build] compile main TU (this includes the full ResNet50 body)" >&2
"$CXX" "${cxxflags[@]}" -c "$src" -o "$main_obj"

# Use lld if available; it's noticeably faster on this workload.
link_extra=()
if command -v ld.lld >/dev/null; then link_extra+=( -fuse-ld=lld ); fi

echo "[build] link -> $exe" >&2
"$CXX" -pthread "${link_extra[@]}" \
  "$main_obj" "$custom_gemm_obj" "$custom_rs_obj" \
  -L"$timvx_lib_dir" -L"$VIV_SDK_LIB_DIR" \
  -ltim-vx -lOpenVX -lOpenVXU \
  -Wl,--unresolved-symbols=ignore-in-shared-libs \
  -Wl,-rpath,"$timvx_lib_dir" \
  -Wl,-rpath,"$VIV_SDK_LIB_DIR" \
  -o "$exe"

# Lock CPU frequency governor to performance to reduce DVFS jitter.
# Silently skipped if not root / sysfs not present.
_gov_ok=0
for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  [[ -f "$gov" ]] && echo performance > "$gov" 2>/dev/null && _gov_ok=1
done
for dev in /sys/class/devfreq/*/governor; do
  [[ -f "$dev" ]] && echo performance > "$dev" 2>/dev/null || true
done
if (( _gov_ok )); then
  echo "[gov]   CPU/NPU governors set to performance" >&2
else
  echo "[gov]   could not set governors (re-run as root to eliminate DVFS jitter)" >&2
fi

# Wrap the executable with real-time scheduling + CPU affinity if available.
# SCHED_FIFO 99 ensures the waiting thread reschedules immediately after each
# NPU completion interrupt, rather than sitting in the run-queue behind softirqs.
runner=("$exe")
if command -v chrt >/dev/null 2>&1 && command -v taskset >/dev/null 2>&1; then
  if chrt -f 99 true 2>/dev/null; then
    runner=(chrt -f 99 taskset -c 3 "$exe")
    echo "[rt]    SCHED_FIFO 99 + pinned to cpu 3" >&2
  else
    echo "[rt]    chrt failed (need root/CAP_SYS_NICE); falling back to taskset only" >&2
    runner=(taskset -c 3 "$exe")
  fi
else
  echo "[rt]    chrt/taskset not found; running without RT priority or affinity" >&2
fi

echo "[run]   $exe" >&2
echo "        TIMVX_CONSTS_DIR=$RESNET_DIR" >&2
echo "---" >&2
LD_LIBRARY_PATH="$timvx_lib_dir:$VIV_SDK_LIB_DIR:${LD_LIBRARY_PATH:-}" \
TIMVX_CONSTS_DIR="$RESNET_DIR" \
VIVANTE_SDK_DIR="$viv_shim" \
  "$exe"
