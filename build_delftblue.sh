#!/bin/sh
#
#SBATCH --job-name="Build Project"
#SBATCH --partition=compute-p1
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --account=research-eemcs-diam

module load intel/oneapi-all
module load 2023r1-intel
module load cmake

srun ./cmake/build_delftblue.bash