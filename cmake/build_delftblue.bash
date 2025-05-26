#!/usr/bin/env bash
set -e

# ----------- absolute paths (Note that these only work on delftblue) -----------------
CC=/beegfs/apps/generic/intel/oneapi/compiler/latest/linux/bin/icx
CXX=/beegfs/apps/generic/intel/oneapi/compiler/latest/linux/bin/icpx
MKL_CMAKE_DIR=/beegfs/apps/generic/intel/oneapi/mkl/latest/lib/intel64
# -------------------------------------------------------------

mkdir -p cmake-build-debug-docker && cd cmake-build-debug-docker

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DMKL_DIR="$MKL_CMAKE_DIR"

cmake --build . --target tri_solve --parallel

