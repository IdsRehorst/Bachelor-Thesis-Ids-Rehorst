#!/bin/sh
#
#SBATCH --job-name="tri_solve_benchmark"
#SBATCH --partition=compute-p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --account=research-eemcs-diam

module load 2024r1
module load intel/oneapi-all
module load cmake
module load hwloc

module use /scratch/iwdhrehorst/spack/share/spack/lmod/linux-rhel8-x86_64/Core
module load gcc/15.1.0

module use /home/iwdhrehorst/spack/share/spack/lmod/linux-rhel8-x86_64/Core/
module load kokkos/4.6.01
module load kokkos-kernels/4.6.01

# Included to stop runtime issues, you have to change the name of the user here if you want to run it yourself
# This should be implemented in the CMakelist in the future.
RACE_LIB=$HOME/iwdhrehorst/Bachelor-Thesis-Ids-Rehorst/cmake-build-debug-docker/_deps/race-install/lib/RACE
export LD_LIBRARY_PATH="$RACE_LIB:$LD_LIBRARY_PATH"

export OMP_PROC_BIND=true
export OMP_PLACES=cores

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

srun --cpu-bind=ldoms ./cmake-build-debug-docker/tri_solve benchmark_1.csv
