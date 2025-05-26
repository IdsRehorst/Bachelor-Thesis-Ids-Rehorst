#!/usr/bin/env bash
set -e

# ----------- absolute paths (Note that these only work on delftblue) -----------------
CC=/beegfs/apps/generic/intel/oneapi/compiler/latest/linux/bin/icx
CXX=/beegfs/apps/generic/intel/oneapi/compiler/latest/linux/bin/icpx
MKL_CMAKE_DIR=/beegfs/apps/generic/intel/oneapi/mkl/latest/lib/intel64
HWLOC_ROOT=/apps/arch/2023r1/software/linux-rhel8-skylake_avx512/gcc-11.3.0/hwloc-2.8.0-h532kfkhl7jv76ds4ixmsnd6v6uwrrlf
# -------------------------------------------------------------

mkdir -p cmake-build-debug-docker && cd cmake-build-debug-docker

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_PREFIX_PATH="$MKLROOT;$HWLOC_ROOT" \
  -DMKL_DIR="$MKL_CMAKE_DIR"\
  -DHWLOC_INCLUDE_DIR=$HWLOC_ROOT/include \
  -DLIBHWLOC=$HWLOC_ROOT/lib/libhwloc.so

cmake --build . --target tri_solve --parallel

