#!/bin/sh
#
#SBATCH --job-name="tri_solve_benchmark"
#SBATCH --partition=compute-p2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --mem=0
#SBATCH --mem-per-cpu=1G
#SBATCH --qos=reservation
#SBATCH --reservation=perfctr

module load 2023r1-gcc11       
module load intel/oneapi-all     
module load hwloc
module load cmake

module load likwid

# Included to stop runtime issues, you have to change the name of the user here if you want to run it yourself
# This should be implemented in the CMakelist in the future.
RACE_LIB=$HOME/iwdhrehorst/Bachelor-Thesis-Ids-Rehorst/cmake-build-debug-docker/_deps/race-install/lib/RACE
export LD_LIBRARY_PATH="$RACE_LIB:$LD_LIBRARY_PATH"

export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_NUM_THREADS=32 

srun likwid-perfctr -C M0:0-31 -w MEM_DP ./cmake-build-debug-docker/tri_solve
