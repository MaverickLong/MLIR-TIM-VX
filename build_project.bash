#!/bin/bash
set -e
cmake -S . -B build -G Ninja \
  -DMLIR_DIR="$(pwd)/llvm-project/build/lib/cmake/mlir" \
  -DLLVM_DIR="$(pwd)/llvm-project/build/lib/cmake/llvm" \
  -DCMAKE_C_COMPILER=clang-16 -DCMAKE_CXX_COMPILER=clang++-16 \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build