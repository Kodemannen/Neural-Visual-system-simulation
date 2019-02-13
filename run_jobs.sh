#!/bin/sh
# sets up folders and parameters, then runs jobscript.sh

#sbatch --array=0-7 n_jobs jobscript.sh
sbatch --array=0-7 jobscript.sh 
