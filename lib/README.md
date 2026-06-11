# MLIR-TIM-VX C++ Source

This folder contains the major C++ sources of the defined timvx dialect and the optimisation passes.

The AI-assistance situation is mixed in here; due to many of the code written in early stage of the project, many files has gone through many revisions, out of which there are AI assisted bugfix passes. However, the novelty parts, e.g. `TimvxFoldInputTranspose.cpp` are relatively new and AI only involved in the document generation and refactoring for those.

The code in this folder is based on recommended template from the MLIR repository, and has largely been following the dialects in MLIR defaults.
