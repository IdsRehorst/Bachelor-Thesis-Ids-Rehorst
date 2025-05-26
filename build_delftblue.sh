#!/bin/sh

module load intel/oneapi-all
module load 2023r1-gcc11
module load cmake
module load hwloc

echo $HWLOC_ROOT

./cmake/build_delftblue.bash
