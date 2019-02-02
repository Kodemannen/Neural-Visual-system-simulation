"""
                        Simulation mainframe
"""
import matplotlib 
matplotlib.use("Agg")
import numpy as np
import parameters as ps
import matplotlib.pyplot as plt
import time

import sys

############################
# Importing local scripts: #
############################
from set_parameters import Set_parameters
from kernel_creation.create_kernels import Create_kernels
from plot_kernels import Plot_kernels
from nest_simulation import Run_simulation
from calculate_LFP import Calculate_LFP
from plot_LFP import Plot_LFP
from save_LFP import Save_LFP

###########
### MPI ###
# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
# # run using: mpiexec -n 4 python script.py  for 4 nodes


##################################
# Getting simulation parameters: #
##################################
rank=int(sys.argv[1])
try:
    abel = sys.argv[2]
    if abel.lower() == "abel":
        abelrun = True
except IndexError:
    abelrun = False
if abelrun:
    params_path = "/work/users/samuelkk/output/out/params"
else:
    params_path = "../output/out/params"

if rank == 0 and abelrun == False:
    Set_parameters(abelrun)     # updating parameters file
#comm.barrier()
network_parameters = ps.ParameterSet(params_path)

################################################################
# Creating kernels for mapping population firing rates to LFP: #
################################################################

network_parameters["plots"] = False ## PLOTS CURRENTLY GIVING ERROR
if network_parameters.create_kernel:
    Create_kernels(network_parameters)
#Plot_kernels(network_parameters)
#Run_simulation([1.], [0.], network_parameters, 1)  # single run, no input

##############################
# Sinisoidal input from LGN: #
##############################
simtime = network_parameters.simtime    # simulation time (ms)
dt = network_parameters.dt

# oscillations = np.array([4,8,12,16]) # oscillations per simulation
# frequencies = oscillations/simtime      # kHz
# rate_times = np.arange(dt, simtime+dt, dt*10) # *10 because 10*dt is the resolution
                                                # of the LFP calculation
frequencies_Hz = np.array([4, 8, 12, 16, 24, 32, 64, 128])  # len == mpi size
frequencies = frequencies_Hz/1000.          # Hz
#frequencies /= 1000.                            # dividing by 1000 since Nest
                                                # uses ms and 4/ms = 4 kHz
rate_times = np.arange(dt, simtime+dt, dt*10)


A = 5.   # amplitude of rate oscillation
b = 15.  # mean rate
matr = np.outer(frequencies, rate_times)
rates = A*np.sin(2*np.pi*matr) + b      # each row is a time series


############################################
# Running point neuron simulation in Nest: #
############################################
t_start = time.time()
training_data_per_freq = 10             # number of simulations that are run per frequency
sim_index = int(training_data_per_freq*rank)

for j in range(training_data_per_freq):

    events = Run_simulation(rate_times,
                    rates[rank],
                    network_parameters,
                    simulation_index=sim_index)
    LFP = Calculate_LFP(events, network_parameters)
    Save_LFP(LFP, network_parameters, sim_index, frequencies_Hz[rank])
    #Plot_LFP(LFP, network_parameters, sim_index, class_label=frequencies[i])
    sim_index += 1
t_stop = time.time() - t_start

print(f"Run time = {t_stop/(60**2)} h")
print(f"Run time = {t_stop/(60)} min")
