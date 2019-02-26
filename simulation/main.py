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

if abelrun==False:
    network_parameters = Set_parameters(abelrun)     # updating parameters file
    params_path = network_parameters.params_path

network_parameters = ps.ParameterSet(params_path)
################################################################
# Creating kernels for mapping population firing rates to LFP: #
################################################################

network_parameters["plots"] = False ## PLOTS CURRENTLY GIVING ERROR
if network_parameters.create_kernel:
    Create_kernels(network_parameters)
#Plot_kernels(network_parameters)
#exit("Kernel created")
# ############################################
# # Part 1 and 2: Sinisoidal input from LGN: #
# ############################################
# rank is between 0 and 7
simtime = network_parameters.simtime    # simulation time (ms)
dt = network_parameters.dt

frequencies_Hz = np.array([4, 8, 12, 16, 24, 32, 96])  
frequencies = frequencies_Hz/1000.          # Hz

rate_times = np.arange(dt, simtime+dt, dt*10)

amplitude = 3   # amplitude of rate oscillation
b = 0.  # mean rate
#matr = np.outer(frequencies, rate_times)
#rates = A*np.sin(2*np.pi*matr) + b      # each row is a time series

############################################
# Running point neuron simulation in Nest: #
############################################
t_start = time.time()
rank = rank    
n_jobs = n_jobs

#sim_index = int(training_data_per_freq*rank)
states = frequencies
n_sims_per_state = 20
n_states = len(states)

n_total_sims = n_sims_per_state*n_states

t_start = time.time()
sim_indices = np.arange(rank, n_total_sims, step=n_jobs)

threshold_rate_LGN = network_parameters.theta / (network_parameters.J_LGN* network_parameters.tauMem * network_parameters.C_LGN) * 1000     # * 1000 to get it in Hz


#for j in range(training_data_per_freq):
for sim_index in sim_indices:

    state_index = sim_index % n_states
    freq=frequencies[state_index]

    ###########################################
    # Setting new eta value to keep the mean: #
    eta_LGN = amplitude/2 / threshold_rate_LGN
    eta_bg = network_parameters.mean_eta - eta_LGN
    bg_rate = eta_bg*network_parameters.threshold_rate * 1000 # *1000 because nest uses Hz

    network_parameters.eta=eta_bg  
    network_parameters.background_rate=bg_rate

    rates = amplitude*(np.sin(2*np.pi*freq*rate_times)) + amplitude # avg rate = amplitude/2
    
    events = Run_simulation(rate_times,
                    rates,
                    network_parameters,
                    simulation_index=sim_index)
    LFP, population_rates = Calculate_LFP(events, network_parameters)
    Save_LFP(LFP, network_parameters, sim_index, class_label=frequencies_Hz[state_index] )
    Save_population_rates(population_rates, network_parameters, sim_index, class_label=frequencies_Hz[state_index])

    # ax = Plot_LFP(LFP)
    # plt.show(ax)
    # events_EX, events_IN, events_LGN = events
    # plt.scatter(events_EX["times"], events_EX["senders"],color="red", s=0.1)
    # plt.scatter(events_IN["times"], events_IN["senders"],color="green",s=0.1)
    # plt.scatter(events_LGN["times"], events_LGN["senders"],color="blue",s=0.1)
    # #plt.plot(population_rates[0])
    # plt.show()
    # exit("egg")
t_stop = time.time() - t_start
print(f"sims_per_job = {n_total_sims/n_jobs}" )
print(f"Run time = {t_stop/(60**2)} h")
print(f"Run time = {t_stop/(60)} min")



# #############################################################
# # Part 4: Sinusioidal input from LGN with varying amplitude #
# #############################################################
# simtime = network_parameters.simtime    # simulation time (ms)
# dt = network_parameters.dt
# frequencies_Hz = np.array([4, 12, 24, 36])      # one with linear effects and one with non linear

# frequencies = frequencies_Hz/1000.  

# step = 1
# A = np.arange(0., 25+step, step=step)      # amplitudes Hz
# b = 0                                     # mean rate

# rate_times = np.arange(dt, simtime+dt, dt*10)   # times when input rate changes

# states = []
# for a in A:
#     for f in frequencies:
#         states.append((a,f))

# ############################################
# # Running point neuron simulation in Nest: #
# ############################################
# rank = rank     
# n_jobs_ = n_jobs

# n_sims_per_state = 500
# n_states = len(states)

# n_total_sims = n_sims_per_state*n_states

# t_start = time.time()
# sim_indices = np.arange(rank, n_total_sims, step=n_jobs)

# # on average 0.5 min per simulation with simtime = 1001 ms
# # ca sim time per job = 0.5 * n_total_sims / n_jobs min
# #print(0.5*n_total_sims/32)


# ############################
# # With simtime = 10001 ms: #
# # n_total_sims = 11 for testing
# # running 11 sims took 22.913775674502055 min = 0.3818962612417009 h
# # dvs total simtime per job = 0.3819 * n_total_sims / n_jobs
# # setter n_jobs = 32


# #################################
# # reading some missing indices: #
# #################################

# nu_thr_LGN = network_parameters.theta/(network_parameters.J_LGN*network_parameters.tauMem*network_parameters.C_LGN)
# print(network_parameters.C_LGN)

# for sim_index in sim_indices:

#     state_index = sim_index % n_states 
#     current_state = states[state_index]
#     amplitude, freq = current_state
#     amplitude = 50
#     freq = 4
#     freq /= 1000
    
#     network_parameters.eta = 3.5 - amplitude/1
#     rates = amplitude*(np.sin(2*np.pi*freq*rate_times)) + amplitude
#     #plt.plot(rates)
#     plt.show()
#     events = Run_simulation(rate_times, rates,
#                             network_parameters,
#                             simulation_index=sim_index,
#                             class_label = str(current_state))



#     LFP, population_rates = Calculate_LFP(events, network_parameters)
#     plt.plot(population_rates[0])
#     plt.show()
#     Save_LFP(LFP, network_parameters, sim_index, class_label=str(current_state))
#     Save_population_rates(population_rates, network_parameters, sim_index, class_label=str(current_state)) ### IMPLEMENT
#     #ax = Plot_LFP(LFP, network_parameters, sim_index, class_label=str(current_state))
    
#     ax = Plot_LFP(LFP)
#     plt.show(ax)
#     events_EX, events_IN, events_LGN = events
#     plt.scatter(events_EX["times"], events_EX["senders"],color="red", s=0.1)
#     plt.scatter(events_IN["times"], events_IN["senders"],color="green",s=0.1)
#     plt.scatter(events_LGN["times"], events_LGN["senders"],color="blue",s=0.1)
#     #plt.plot(population_rates[0])
#     plt.show()
#     exit("egg")

# t_stop = time.time() - t_start
# print(f"sims_per_job = {n_total_sims/n_jobs}" )
# print(f"Run time = {t_stop/(60**2)} h")
# print(f"Run time = {t_stop/(60)} min")
