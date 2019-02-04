# sets up folders and parameters, then runs jobscript.sh
cd simulation
python set_parameters.py abel
cd ..
#sbatch --array=0-7 n_jobs jobscript.sh
sbatch --array=0-7 jobscript.sh
