#!/bin/bash

# Job name:
#SBATCH --job-name=samueltest
#
# Project:
#SBATCH --account=nn9565k
#
# Wall clock limit:
#SBATCH --time=00:05:01     
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
python main.py $SLURM_ARRAY_TASK_ID abel

# Run this file by: 
# sbatch --array=0-7 jobscript.sh 

# output will be found at /work/users/samuelkk
