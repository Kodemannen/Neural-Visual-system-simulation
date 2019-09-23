#!/bin/sh
# sets up folders and parameters, then runs jobscript.sh
cd simulation
params_path=$(python set_parameters.py abel)
export params_path
cd ..

sbatch --array=0-19 jobscript.sh
#sbatch --array=0 jobscript.sh 
