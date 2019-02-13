#!/bin/bash

# Job name:
#SBATCH --job-name=samuel
#
# Project:
#SBATCH --account=nn9565k
#
# Wall clock limit:
#SBATCH --time=04:30:01     
#
# Max memory usage:
#SBATCH --mem-per-cpu=2G


#SBATCH --ntasks=8


#SBATCH -o terminal_output.txt
#SBATCH -e errors.txt


## Set up job environment:
source /cluster/bin/jobsetup
set -o errexit # exit on errors

# ## Do some work:
cd simulation
# if [$SLURM_ARRAY_TASK_ID = 0]; then
#     python set_parameters.py abel 
#     echo "setting parameters"
# fi

python main.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT $params_path

# Run this file by: 
# sbatch --array=0-7 jobscript.sh abel

# output will be found at /work/users/samuelkk/output
