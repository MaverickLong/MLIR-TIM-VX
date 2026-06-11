#!/bin/bash
# Declaration: This file is AI-generated and reviewed by human.
set -e
cmake -S . -B build -G Ninja \
  -DMLIR_DIR="$(pwd)/llvm-project/build/lib/cmake/mlir" \
  -DLLVM_DIR="$(pwd)/llvm-project/build/lib/cmake/llvm" \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build