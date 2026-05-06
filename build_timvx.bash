#!/bin/bash
set -e
PROJECT_ROOT=$(dirname "$(realpath "$0")")

# Verisilicon driver SDK (aarch64). Layout expected by TIM-VX's
# cmake/local_sdk.cmake is `<sdk>/include` + `<sdk>/lib`; we use the staging
# directory under unified-tina/timvx-sdk that symlinks into the real SDK.
# Override by exporting EXTERNAL_VIV_SDK before running this script.
EXTERNAL_VIV_SDK="${EXTERNAL_VIV_SDK:-$HOME/ufs/home/radxa/ai-sdk/unified-tina/timvx-sdk}"
if [[ ! -e "$EXTERNAL_VIV_SDK/lib/libCLC.so" ]]; then
    echo "EXTERNAL_VIV_SDK does not look right (missing lib/libCLC.so):" >&2
    echo "  $EXTERNAL_VIV_SDK" >&2
    exit 1
fi

VIP_LITE_SDK="${VIP_LITE_SDK:-$HOME/ufs/home/radxa/ai-sdk/viplite-tina/lib/aarch64-none-linux-gnu/v2.0/inc}"

mkdir -p TIM-VX/build
cd TIM-VX/build
cmake .. \
   -DEXTERNAL_VIV_SDK="$EXTERNAL_VIV_SDK" \
   -DTIM_VX_ENABLE_TEST=ON \
   -DTIM_VX_ENABLE_CUSTOM_OP=ON \
   -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
   -DTIM_VX_ENABLE_TENSOR_CACHE=ON \
   -DSYSTEM_OPENSSL=ON
make -j8 && make install