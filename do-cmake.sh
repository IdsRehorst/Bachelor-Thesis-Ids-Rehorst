#!/bin/bash
source ../env-delftblue.sh
cmake ..\
	-DCMAKE_BUILD_TYPE=Release \
	-DMKL_THREADING=gnu_thread

