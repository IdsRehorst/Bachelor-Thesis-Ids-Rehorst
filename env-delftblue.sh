#!/bin/bash
GCC_VERSION=15.2.0
module use /projects/unsupported/spack-gcc15/share/spack/lmod/linux-rhel8-x86_64/Core
module use /projects/unsupported/spack-gcc15/share/spack/lmod/linux-rhel8-x86_64/gcc/${GCC_VERSION}

module load gcc/${GCC_VERSION}
module load gcc-runtime/${GCC_VERSION}
module load cmake
module load likwid

module load intel/oneapi-all
module load kokkos kokkos-kernels

export CMAKE_PREFIX_PATH=/projects/unsupported/install-gcc15/:${CMAKE_PREFIX_PATH}
export CMAKE_PREFIX_PATH=${MKLROOT}:${CMAKE_PREFIX_PATH}
