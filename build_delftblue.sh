#!/bin/sh

module load 2023r1-gcc11

gcc -print-file-name=libstdc++.so.6

module load intel/oneapi-all
module load cmake
module load hwloc

export LIKWID_ROOT=$HOME/likwid-5.4

echo $HWLOC_ROOT

./cmake/build_delftblue.bash
