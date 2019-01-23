#!/bin/bash

# Job name:
#SBATCH --job-name=samueltest
#
# Project:
#SBATCH --account=nn9565k
#
# Wall clock limit:
#SBATCH --time=00:00:01
#
# Max memory usage:
#SBATCH --mem-per-cpu=10M


#SBATCH -o utputt.txt
#SBATCH -e feilfil.txt


## Set up job environment:
source /cluster/bin/jobsetup
module purge   # clear any inherited modules
set -o errexit # exit on errors


## Copy input files to the work directory:
cp testsim -r $SCRATCH

# ## Make sure the results are copied back to the submit directory (see Work Directory below):
chkfile testsim/output

# ## Do some work:
cd $SCRATCH
python testsim/Prog/hello.py