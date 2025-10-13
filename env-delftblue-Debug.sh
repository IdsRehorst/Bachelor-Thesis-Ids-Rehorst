#!/bin/bash
GCC_VERSION=15.2.0
module use /projects/unsupported/spack-gcc15/share/spack/lmod/linux-rhel8-x86_64/Core
module use /projects/unsupported/spack-gcc15/share/spack/lmod/linux-rhel8-x86_64/gcc/${GCC_VERSION}

module load gcc
module load cmake
module load hwloc
module load likwid
module load gdb

module load intel/oneapi-all
module load kokkos kokkos-kernels

export CMAKE_PREFIX_PATH=/projects/unsupported/install-gcc15-Debug/:${CMAKE_PREFIX_PATH}
export CMAKE_PREFIX_PATH=${MKLROOT}:${CMAKE_PREFIX_PATH}
export LD_LIBRARY_PATH=${TBBROOT}/lib/intel64/gcc4.8:${LD_LIBRARY_PATH}
