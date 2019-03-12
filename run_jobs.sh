#!/bin/sh
# sets up folders and parameters, then runs jobscript.sh
cd simulation
params_path=$(python set_parameters.py abel)
export params_path
cp main.py $params_path
cd ..

sbatch --array=0-127 jobscript.sh
