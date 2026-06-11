#!/usr/bin/env bash
# timvx_zerocopy_perf.bash — build + run the perf bench. Same dynamic-link
# setup as timvx_zerocopy.bash.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$script_dir/../.." && pwd)}"
TIM_VX_DIR="${TIM_VX_DIR:-$REPO_ROOT/TIM-VX}"
TIM_VX_BUILD_DIR="${TIM_VX_BUILD_DIR:-$REPO_ROOT/TIM-VX/build/install}"
VIV_SDK_LIB_DIR="${VIV_SDK_LIB_DIR:-${EXTERNAL_VIV_SDK:-$HOME/ufs/home/radxa/ai-sdk/unified-tina/timvx-sdk}/lib}"

build="$script_dir/.build"
mkdir -p "$build"
patched_sdk_dir="$VIV_SDK_LIB_DIR"

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
[[ -n "$timvx_lib_dir" ]] || { echo "could not locate libtim-vx.so under $TIM_VX_BUILD_DIR" >&2; exit 1; }

CXX="${CXX:-}"
if [[ -z "$CXX" ]]; then
  if command -v clang++-16 >/dev/null; then CXX=clang++-16
  else CXX=clang++; fi
fi

src="$script_dir/timvx_zerocopy_perf.cpp"
exe="$build/timvx_zerocopy_perf"
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

echo "[run]   ${runner[*]}" >&2
echo "---" >&2
LD_LIBRARY_PATH="$timvx_lib_dir:$patched_sdk_dir:${LD_LIBRARY_PATH:-}" \
VIVANTE_SDK_DIR="$viv_shim" \
  "${runner[@]}"
