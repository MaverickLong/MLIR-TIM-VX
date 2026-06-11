#!/bin/bash
# Declaration: This file is AI-generated and reviewed by human.
set -e
PROJECT_ROOT=$(dirname "$(realpath "$0")")
mkdir -p llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="Native" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
   -DLLVM_USE_LINKER=lld \
   -DLLVM_CCACHE_BUILD=ON \
   -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
   -DPython3_EXECUTABLE="$PROJECT_ROOT/.venv/bin/python" \
   -DCMAKE_C_VISIBILITY_PRESET=hidden \
   -DCMAKE_CXX_VISIBILITY_PRESET=hidden \
   -DCMAKE_VISIBILITY_INLINES_HIDDEN=ON \
   -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=timvx \
   -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build . --target check-mlir