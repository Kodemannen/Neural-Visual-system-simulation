"""
                        Simulation mainframe
"""
import matplotlib 
import numpy as np
import parameters as ps
import matplotlib.pyplot as plt
import time
import os
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
from save_population_rates import Save_population_rates

###########
### MPI ###
# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# size = comm.Get_size()
# # run using: mpiexec -n 4 python script.py  for 4 nodes


#################################
# Getting job array parameters: #
#################################
try:
    rank = int(sys.argv[1])         # job array index
    n_jobs = int(sys.argv[2])       # total number of jobs
    params_path = sys.argv[3]
    # abel = sys.argv[3]              # whether simulation is done on Abel or not
    # if abel.lower() == "abel":
    abelrun = True
    matplotlib.use("Agg")

except IndexError:
    abelrun = False
    n_jobs = 1
    rank = 0

# if (rank == 0):
#     network_parameters = Set_parameters(abelrun)     # updating parameters file

network_parameters = ps.ParameterSet(PS.params_path)

################################################################
# Creating kernels for mapping population firing rates to LFP: #
################################################################

network_parameters["plots"] = False ## PLOTS CURRENTLY GIVING ERROR
if network_parameters.create_kernel:
    Create_kernels(network_parameters)
Plot_kernels(network_parameters)

# ############################################
# # Part 1 and 2: Sinisoidal input from LGN: #
# ############################################
# # rank is between 0 and 7
# simtime = network_parameters.simtime    # simulation time (ms)
# dt = network_parameters.dt

# frequencies_Hz = np.array([4, 8, 12, 16, 24, 32, 64, 128])  
# frequencies = frequencies_Hz/1000.          # Hz

# rate_times = np.arange(dt, simtime+dt, dt*10)

# A = 5.   # amplitude of rate oscillation
# b = 15.  # mean rate
# matr = np.outer(frequencies, rate_times)
# rates = A*np.sin(2*np.pi*matr) + b      # each row is a time series

# ############################################
# # Running point neuron simulation in Nest: #
# ############################################
# t_start = time.time()
# training_data_per_freq = 1             # number of simulations that are run per frequency
# rank = 4    # 24 hz
# sim_index = int(training_data_per_freq*rank)

# for j in range(training_data_per_freq):

#     events = Run_simulation(rate_times,
#                     rates[rank],
#                     network_parameters,
#                     simulation_index=sim_index)
#     LFP, population_rates = Calculate_LFP(events, network_parameters)
#     Save_LFP(LFP, network_parameters, sim_index, class_label=frequencies_Hz[rank] )
#     Save_population_rates(population_rates, network_parameters, sim_index, class_label=frequencies[rank])
#     #Plot_LFP(LFP, network_parameters, sim_index, class_label=frequencies[rank])

#     sim_index += 1
# t_stop = time.time() - t_start

# print(f"Run time = {t_stop/(60**2)} h")
# print(f"Run time = {t_stop/(60)} min")



#############################################################
# Part 3: Sinusioidal input from LGN with varying amplitude #
#############################################################
simtime = network_parameters.simtime    # simulation time (ms)
dt = network_parameters.dt
frequencies_Hz = np.array([4, 24])      # one with linear effects and one with non linear
frequencies = frequencies_Hz/1000.  

step = 0.5
A = np.arange(0., 15+step, step=step)      # amplitudes Hz
b = 15                                     # mean rate

rate_times = np.arange(dt, simtime+dt, dt*10)   # times when input rate changes

states = []
for a in A:
    for f in frequencies:
        states.append((a,f))

############################################
# Running point neuron simulation in Nest: #
############################################
rank = rank     
n_jobs_ = n_jobs

n_sims_per_state = 5
n_states = len(states)

n_total_sims = n_sims_per_state*n_states

t_start = time.time()
sim_indices = np.arange(rank, n_total_sims, step=n_jobs)
for sim_index in sim_indices:

    state_index = sim_index % n_states 
    current_state = states[state_index]
    amplitude, freq = current_state
    
    rates = amplitude*np.sin(2*np.pi*freq*rate_times) + b        

    events = Run_simulation(rate_times, rates,
                            network_parameters,
                            simulation_index=sim_index,
                            class_label = str(current_state))
    LFP, population_rates = Calculate_LFP(events, network_parameters)
    Save_LFP(LFP, network_parameters, sim_index, class_label=str(current_state))
    Save_population_rates(population_rates, network_parameters, sim_index, class_label=str(current_state)) ### IMPLEMENT
    #Plot_LFP(LFP, network_parameters, sim_index, class_label=str(current_state))
    
    #plt.plot(population_rates[0])
    #plt.show()
    
    
t_stop = time.time() - t_start
print(f"Run time = {t_stop/(60**2)} h")
print(f"Run time = {t_stop/(60)} min")
