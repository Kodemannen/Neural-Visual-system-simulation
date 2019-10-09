"""
                        Simulation mainframe
"""
import nest
import matplotlib 
import numpy as np
import parameters as ps
import matplotlib.pyplot as plt
import time
import os
import sys
from scipy import signal

# ############################
# # Importing local scripts: #
# ############################
from set_parameters import Set_parameters
#from kernel_creation.create_kernels import Create_kernels
from plot_kernels import Plot_kernels
from nest_simulation import Run_simulation
from calculate_LFP import Calculate_LFP
from plot_LFP import Plot_LFP
from save_LFP import Save_LFP
from save_population_rates import Save_population_rates
from LGNsimulation import Get_LGN_signal
# sys.path.append("/home/samknu/MyRepos/MasterProject/Results_and_analysis/Figure_setup")
# import plot_utils as pu
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


# # #######################################
# # # Part 1 : Sinisoidal input from LGN: #
# # #######################################
# Part = "Part 1: sinusoidal input"
# simtime = network_parameters.simtime    # simulation time (ms)
# dt = network_parameters.dt

# frequencies_Hz = np.array([4, 10, 25, 45, 70, 110]) 
# #frequencies_Hz = np.array([8]) 
# #frequencies = frequencies_Hz/1000.          # Hz

# rate_times = np.arange(dt, simtime+dt, dt*10)

# states = frequencies_Hz

# A = np.array([3])

# states = []
# for a in A:
#     for f in frequencies_Hz:
#         states.append((a,f))

# def rate_func(amp, freq_Hz, sim_index):
#     freq = freq_Hz/1000
#     rates = amp*np.sin(2*np.pi*freq*rate_times) + amp
#     return rates
# n_sims_per_state = 1


# # ##############################
# # # Part 2: Varying amplitude: #
# # ##############################
# Part = "Part 2b: varying amplitude"
# simtime = network_parameters.simtime    # simulation time (ms)
# dt = network_parameters.dt

# #frequencies = np.array([4, 10, 25, 70])
# frequencies = np.array([15, 20, 30, 40, 50, 60])

# frequencies = np.arange(2, 80, 2)

# rate_times = np.arange(dt, simtime+dt, dt*10)

# #step = 1
# #A = np.arange(1., 30+step, step=step)      # amplitudes Hz
# A = [1, 5, 10, 15, 20]
# rate_times = np.arange(dt, simtime+dt, dt*10)   # times when input rate changes

# states = []
# for a in A:
#     for f in frequencies:
#         states.append((a,f))

# def rate_func(amp, freq_Hz, sim_index):
#     freq = freq_Hz/1000
#     rates = amp*np.sin(2*np.pi*freq*rate_times) + amp
#     return rates

# n_sims_per_state = 500

# ############################
# # Part 3: Sawtooth signal #  
# ###########################
# Part = "Part 3: Sawtooth signals, redo"
# simtime = network_parameters.simtime    # simulation time (ms)
# dt = network_parameters.dt

# frequencies = np.array([4, 10, 25, 70])

# rate_times = np.arange(dt, simtime+dt, dt*10)

# step = 2
# A = np.arange(1., 30, step=step)      # amplitudes Hz

# rate_times = np.arange(dt, simtime+dt, dt*10)   # times when input rate changes

# states = []
# for a in A:
#     for f in frequencies:
#         states.append((a,f))
# print(len(states)/2)


# def rate_func(amp, freq_Hz, sim_index):
#     freq = freq_Hz/1000
#     if sim_index < n_total_sims/2:
#         rates = amp*signal.sawtooth(2*np.pi*freq*rate_times) + amp
#     else:
#         rates = np.flip(amp*signal.sawtooth(2*np.pi*freq*rate_times) + amp)
#     return rates

# n_sims_per_state = 500



# ########################################
# # Part 4: Classifying sawtooth signals #  
# ########################################
# Part = "Part 4: Classifying sawtooths"
# simtime = network_parameters.simtime    # simulation time (ms)
# dt = network_parameters.dt

# frequencies = np.array([4, 10, 25, 45, 70])

# rate_times = np.arange(dt, simtime+dt, dt*10)

# A = np.array([3])      # amplitudes Hz

# rate_times = np.arange(dt, simtime+dt, dt*10)   # times when input rate changes

# states = []
# for a in A:
#     for f in frequencies:
#         states.append((a,f))


# def rate_func(amp, freq_Hz, sim_index):
#     freq = freq_Hz/1000
#     if sim_index < n_total_sims/2:
#         rates = amp*signal.sawtooth(2*np.pi*freq*rate_times) + amp
#     else:
#         rates = np.flip(amp*signal.sawtooth(2*np.pi*freq*rate_times) + amp)
#     return rates

# n_sims_per_state = 20000


# ########################
# # Part 8: Using pyLGN: #
# # Make a new test set with g=5.2*0.9
# ########################
## En sim tar ca 0.712 min
Part = "6. Making test set with g=5.2*0.9"
simtime = network_parameters.simtime    # simulation time (ms)
dt = network_parameters.dt

#seq = np.random.choice(10, size=10, replace=False)
#seq = np.arange(10)
#rate = Get_LGN_signal(seq)
amplitude = 3
n_sims = 10000

rate_times = np.arange(dt, simtime+dt, dt*10)

def seq_to_string(s):
    """"Function for making a compact class label from sequence array"""
    string = ""
    for t in s:
        string+=str(t)
    return string


# ########################################
# # Calculating receptive field of pylgn #
# ########################################
# ## Only need this part:
# from LGNsimulation import Rf_heatmap
# Rf_heatmap(network_parameters, rank=rank, n_jobs=n_jobs)





############################################
# Running point neuron simulation in Nest: #
############################################
# avg 1.1 min per sim with amplitude=3, mean_eta = 2.3
# simtime=1001 ms, 
rank = rank    
n_jobs = n_jobs

#n_states = len(states)
n_total_sims = n_sims

sim_indices = np.arange(rank, n_total_sims, step=n_jobs)

threshold_rate_LGN = network_parameters.theta / (network_parameters.J_LGN* network_parameters.tauMem * network_parameters.C_LGN) * 1000     # * 1000 to get it in Hz


t_start = time.time()

if rank == 0:
    with open(network_parameters.sim_output_dir + "/sim_info.txt", "w") as filen:
        filen.write(Part + "\n")
        filen.write("Mean eta: " + str(network_parameters.mean_eta) + " \n") 
        filen.write("amplitude="+str(amplitude) + "\n")     
        filen.write("n_jobs=" + str(n_jobs) + "\n")
        filen.write("n_total_sims="+ str(n_total_sims) + "\n")
        #filen.write("states="+str(states) + "\n")


for sim_index in sim_indices:
    sim_index += 100000 
    seq = np.random.choice(10, size=10, replace=False)  # image sequence
    rates, mean = Get_LGN_signal(seq, amplitude=amplitude) 
    seq_label = seq_to_string(seq)

    
    # #############
    # # BULLSHIT #
    # rates = rates**6
    # rates = rates / np.max(rates)
    # rates = rates*amplitude
    # sim_index = 14
    # image_number = 1

    # ax = plt.subplot(111)
    # ax.plot(rates[250*4:250*5])
    # #plt.axis("off")
    #     # Hide the right and top spines
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)

    # #plt.plot(population_rates[0])
    # #plt.axis("off")
    # plt.xticks([])
    # plt.yticks([])
    # plt.savefig(network_parameters.sim_output_dir+"/LGN_signal.svg")
    # #exit("ballemos")

    # #print(np.mean(rates))
    #print(np.var(rates))
    
    #state_index = sim_index % n_states
    #state = states[state_index]
    
    
    ##################################################
    # Setting new eta value to keep the mean to 1.1: #
    ##################################################
    eta_LGN = float(mean) / threshold_rate_LGN
    eta_bg = network_parameters.mean_eta - eta_LGN
    if eta_bg < 0:
        eta_bg = 0  # cant have negative rates

    bg_rate = eta_bg*network_parameters.threshold_rate * 1000 # *1000 because nest uses Hz
    
    network_parameters.eta=eta_bg  
    network_parameters.background_rate=bg_rate

    events = Run_simulation(rate_times,
                    rates,
                    network_parameters,
                    simulation_index=sim_index)
    LFP, population_rates = Calculate_LFP(events, network_parameters)
    Save_LFP(LFP, network_parameters, sim_index, class_label=seq_label)
    Save_population_rates(population_rates, network_parameters, sim_index, class_label=seq_label)
    

    # # #######
    # # # junk #
    # LGN_spikes = events[2]["times"]
    # LGN_ids = events[2]["senders"]

    
    # spikes_LGN = LGN_spikes*(LGN_spikes > 250*(image_number+2))
    # spikes_LGN = spikes_LGN*(spikes_LGN<250*(image_number+3))
    # indices_LGN = np.argwhere(spikes_LGN==LGN_spikes)
    
    # LGN_spikes = LGN_spikes[indices_LGN]
    # LGN_ids = LGN_ids[indices_LGN]
    
    # LGN_spikes = LGN_spikes.reshape(-1)
    # LGN_ids = LGN_ids.reshape(-1)

    # print(LGN_ids.shape)

    # inds = np.argwhere(LGN_ids==LGN_ids[0])
    # inds = inds.reshape(-1)
    

    # def Plot_action_potentials(spike_times__, ax):
    #     dt = 0.01
    #     N = 1000
    #     amp = 0.1
    #     spike_times = spike_times__ - 250*(image_number+2)
    #     t = [0]
    #     done = False
    #     inds__ = [0]
    #     for s in spike_times:

    #         t.append(s-dt*200); inds__.append(0)            
    #         t.append(s-dt); inds__.append(amp/3)
    #         t.append(s); inds__.append(amp)
    #         t.append(s+dt); inds__.append(0)

    #     ax.plot(t, inds__, color="k")
    #     #plt.show()
    #     #exit("hoe")
    


    # pu.figure_setup()
    # fig, axes = plt.subplots(2,1, sharex=True, gridspec_kw = {'height_ratios':[1,2.5]})
    # fig_size = pu.get_fig_size(14,6.5)
    # fig.set_size_inches(fig_size)
    # #axes[0].scatter(LGN_spikes[inds], LGN_ids[inds], s=0.1, color="k")
    # Plot_action_potentials(LGN_spikes[inds],axes[0])

    # axes[1].plot(rates[250*(image_number+2):250*(image_number+3)])
    # #axes[1].axis("off")
    # axes[1].set_xticks([])
    # axes[1].set_yticks([])
    # #axes[0].axis("off")
    # axes[0].set_xticks([])
    # axes[0].set_yticks([])
    # axes[0].set_ylabel("Membrane\n potential (V)", va="bottom")
    # axes[1].set_xlabel("Time (ms)")
    # #axes[0].set_title("Spike train")
    # #axes[1].set_title("Rate profile")
    # axes[1].set_ylabel("Fire rate (Hz)")
    # #axes[1].set_ylabel("$\lambda_{LGN}(t)$")
    # axes[0].spines["top"].set_visible(False)
    # axes[0].spines["right"].set_visible(False)
    # axes[1].spines["top"].set_visible(False)
    # axes[1].spines["right"].set_visible(False)
    # #plt.savefig("LGN_spikes.svg")
    # plt.tight_layout()
    # plt.savefig("/home/samknu/MyRepos/MasterProject/Thesis/Figurer/LGN_spikes.pdf")
    # #plt.show()
    # exit("hore")
    # # ########################################################################
    # # REMEMBER the poprate is in units of kHz, so to get rate per neuron we need to multiply by 1000 (to get Hz) and then divide by population size
    # #print("mean poprate", np.mean(population_rates[0]) * 1000 /10000 )
    # #print("mean poprate", np.mean(population_rates[1]) * 1000 /2500 )
    # ########################################################################


t_stop = time.time() - t_start
print(f"sims_per_job = {n_total_sims/n_jobs}" )
print(f"Run time = {t_stop/(60**2)} h")
print(f"Run time = {t_stop/(60)} min")