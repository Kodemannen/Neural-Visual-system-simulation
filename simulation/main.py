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
from scipy import signal

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



# # ############################################
# # # Part 1 rerun: Sinisoidal input from LGN: #
# # ############################################
# simtime = network_parameters.simtime    # simulation time (ms)
# dt = network_parameters.dt

# frequencies_Hz = np.array([4, 8, 12, 16, 24, 32, 96])  
# frequencies = frequencies_Hz/1000.          # Hz

# rate_times = np.arange(dt, simtime+dt, dt*10)

# amplitude = 3   # Hz, amplitude of rate oscillation
# b = 0.  # mean rate
# states = frequencies_Hz


# ####################################
# # Part 2 rerun: Varying amplitude: #
# ####################################

# simtime = network_parameters.simtime    # simulation time (ms)
# dt = network_parameters.dt

# frequencies = np.array([4, 12, 24, 36])

# rate_times = np.arange(dt, simtime+dt, dt*10)

# step = 1
# A = np.arange(1., 30+step, step=step)      # amplitudes Hz
# rate_times = np.arange(dt, simtime+dt, dt*10)   # times when input rate changes

# states = []
# for a in A:
#     for f in frequencies:
#         states.append((a,f))


# ############################
# # Part 3: Sawtooth signal #  
# ###########################

# simtime = network_parameters.simtime    # simulation time (ms)
# dt = network_parameters.dt

# frequencies = np.array([4, 12, 24, 36])

# rate_times = np.arange(dt, simtime+dt, dt*10)

# step = 1
# A = np.arange(1., 30+step, step=step)      # amplitudes Hz
# rate_times = np.arange(dt, simtime+dt, dt*10)   # times when input rate changes

# states = []
# for a in A:
#     for f in frequencies:
#         states.append((a,f))


# ###################################
# # Part 3b: Reverse awtooth signal #  
# ###################################

# simtime = network_parameters.simtime    # simulation time (ms)
# dt = network_parameters.dt

# #frequencies = np.array([4, 12, 24, 36])
# frequencies = np.array([4, 36])
# rate_times = np.arange(dt, simtime+dt, dt*10)

# step = 1
# #A = np.arange(1., 30+step, step=step)      # amplitudes Hz
# A = np.array([3,10,20,30])
# rate_times = np.arange(dt, simtime+dt, dt*10)   # times when input rate changes

# states = []
# for a in A:
#     for f in frequencies:
#         states.append((a,f))


#########################################################
# Part 4: Lower eta, because avg. poprate was too high. #
#########################################################


simtime = network_parameters.simtime    # simulation time (ms)
dt = network_parameters.dt

#frequencies = np.array([4, 12, 24, 36])
frequencies = np.array([4, 24])
rate_times = np.arange(dt, simtime+dt, dt*10)

step = 1
#A = np.arange(1., 30+step, step=step)      # amplitudes Hz
A = np.array([3])
rate_times = np.arange(dt, simtime+dt, dt*10)   # times when input rate changes

states = []
for a in A:
    for f in frequencies:
        states.append((a,f))


############################################
# Running point neuron simulation in Nest: #
############################################
# avg 1.1 min per sim with amplitude=3, mean_eta = 2.3
# simtime=1001 ms, 
rank = rank    
n_jobs = n_jobs

n_sims_per_state = 5000
n_states = len(states)
n_total_sims = n_sims_per_state*n_states
sim_indices = np.arange(rank, n_total_sims, step=n_jobs)


threshold_rate_LGN = network_parameters.theta / (network_parameters.J_LGN* network_parameters.tauMem * network_parameters.C_LGN) * 1000     # * 1000 to get it in Hz


t_start = time.time()

if rank == 0:
    with open(network_parameters.sim_output_dir + "/sim_info.txt", "w") as filen:
        filen.write("Part 3b: Reverse sawtooth signal \n")
        filen.write("n_jobs=" + str(n_jobs) + "\n")
        filen.write("n_sims_per_state=" + str(n_sims_per_state) + "\n")
        filen.write("n_total_sims="+ str(n_total_sims) + "\n")
        filen.write("states="+str(states) + "\n")

for sim_index in sim_indices:

    state_index = sim_index % n_states
    amplitude, freq=states[state_index]
    #freq=states[state_index]

    freq=freq/1000      # because rate_times is in ms

    ##################################################
    # Setting new eta value to keep the mean to 2.3: #
    ##################################################
    eta_LGN = amplitude / threshold_rate_LGN
    eta_bg = network_parameters.mean_eta - eta_LGN
    bg_rate = eta_bg*network_parameters.threshold_rate * 1000 # *1000 because nest uses Hz

    network_parameters.eta=eta_bg  
    network_parameters.background_rate=bg_rate

    rates = np.flip(amplitude*(signal.sawtooth(2*np.pi*freq*rate_times)) + amplitude) # avg rate = amplitude/2

    events = Run_simulation(rate_times,
                    rates,
                    network_parameters,
                    simulation_index=sim_index)
    LFP, population_rates = Calculate_LFP(events, network_parameters)
    Save_LFP(LFP, network_parameters, sim_index, class_label=str(states[state_index] ))
    Save_population_rates(population_rates, network_parameters, sim_index, class_label=str(states[state_index]))

    # ax = Plot_LFP(LFP)
    # plt.show(ax)
    # events_EX, events_IN, events_LGN = events
    # plt.scatter(events_EX["times"], events_EX["senders"],color="red", s=0.1)
    # plt.scatter(events_IN["times"], events_IN["senders"],color="green",s=0.1)
    # plt.scatter(events_LGN["times"], events_LGN["senders"],color="blue",s=0.1)
    # #plt.plot(population_rates[0])
    # print(amplitude, freq)
    # plt.show()
    # exit("egg")

t_stop = time.time() - t_start
print(f"sims_per_job = {n_total_sims/n_jobs}" )
print(f"Run time = {t_stop/(60**2)} h")
print(f"Run time = {t_stop/(60)} min")