#!/bin/bash
#module load 2023r1-gcc11       
module load intel/oneapi-all     
module load hwloc
module load cmake

# Load the newer likwid module we compiled ourselves
module use $HOME/modules     
module load likwid/5.4

# Included to stop runtime issues, you have to change the name of the user here if you want to run it yourself
# This should be implemented in the CMakelist in the future.
RACE_LIB=$HOME/iwdhrehorst/Bachelor-Thesis-Ids-Rehorst/cmake-build-debug-docker/_deps/race-install/lib/RACE
export LD_LIBRARY_PATH="$RACE_LIB:$LD_LIBRARY_PATH"

#export OMP_PROC_BIND=true
#export OMP_PLACES=cores
#export OMP_NUM_THREADS=16

likwid-perfctr -C  S0:0-31 -g DATA -m -output DATA_32.txt  ./cmake-build-debug-docker/tri_solve 

