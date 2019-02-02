cd simulation
python set_parameters.py abel
cd ..
sbatch --array=0-7 jobscript.sh