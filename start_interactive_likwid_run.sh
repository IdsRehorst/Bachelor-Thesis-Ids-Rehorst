#!/bin/bash
srun --partition=compute-p2 --pty -N 1 -n 1 -c 64 -t 04:00:00 --mem-per-cpu=1G --qos=reservation --reservation=perfctr bash
