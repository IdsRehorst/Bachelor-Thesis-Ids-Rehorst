#!/bin/sh


module load 2024r1
module load intel/oneapi-all
module load cmake
module load hwloc

module use /scratch/iwdhrehorst/spack/share/spack/lmod/linux-rhel8-x86_64/Core
module load gcc/15.1.0

module use /home/iwdhrehorst/spack/share/spack/lmod/linux-rhel8-x86_64/Core/
module load kokkos/4.6.01
module load kokkos-kernels/4.6.01

export LIKWID_ROOT=$HOME/likwid-5.4
export TBB_ROOT=/beegfs/apps/generic/intel/oneapi_2022.3/tbb/latest

./cmake/build_delftblue.bash
