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
#SBATCH --mem-per-cpu=8G


#SBATCH -o terminal_output.txt
#SBATCH -e errors.txt


## Set up job environment:
source /cluster/bin/jobsetup
#module purge   # clear any inherited modules
set -o errexit # exit on errors


## Copy input files to the work directory:
cp simulation -r $SCRATCH

# ## Make sure the results are copied back to the submit directory (see Work Directory below):
#chkfile sim_output

unset $DISPLAY

# ## Do some work:
cd $SCRATCH/simulation
python main.py

# output will be found at /work/users/samuelkk

# Run this file by: sbatch jobscript.sh