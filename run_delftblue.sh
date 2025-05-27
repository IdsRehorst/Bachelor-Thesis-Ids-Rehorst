#!/bin/sh
#
#SBATCH --job-name="tri_solve_benchmark"
#SBATCH --partition=compute-p1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --account=research-eemcs-diam

module load intel/oneapi-all
module load 2023r1-gcc11
module load hwloc

export OMP_PROC_BIND=true
export OMP_PLACES=cores

export OMP_NUM_THREADS=1

srun --cpu-bind=ldoms ./cmake-build-debug-docker/tri_solve