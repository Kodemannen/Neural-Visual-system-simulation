cd simulation
python set_parameters.py abel
sbatch --array=0-7 jobscript.sh