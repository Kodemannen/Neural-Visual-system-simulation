"""
                Simulating the spiking activity of the point neuron network
"""
import numpy as np
import nest
import time
import os
import matplotlib.pyplot as plt
from parameters import ParameterSet
import h5py
from set_parameters import Set_parameters
import sys



def Run_simulation(rate_times, poisson_rates, network_parameters, simulation_index=0, class_label=""):
    #np.random.seed(0)
    #############################################
    # Extracting network/simulation parameters: #
    #############################################
    #simulation_index = 109
    PS = network_parameters
    spike_dir = PS.nest_output_path
    threads = PS.threads    # number of parallel threads

    NE = PS.NE
    NI = PS.NI
    N_LGN = PS.N_LGN

    CE = PS.CE
    CI = PS.CI
    C_LGN = PS.C_LGN
    #C_background = PS.C_background

    J_EX  = PS.J_EX
    J_IN  = PS.J_IN
    J_LGN = J_EX
    J_background = J_EX
    g  = PS.g            # ratio inhibitory weight / excitatory weight

    dt = PS.dt
    delay = PS.delay     # synaptic delay
    background_rate = PS.background_rate


    # print(background_rate/(np.mean(poisson_rates)*C_LGN))

    simtime = PS.simtime
    J_background = PS.J_background

    fixed_connectome = PS.fixed_connectome

    neuron_params = {"C_m" : PS.CMem,
                    "tau_m": PS.tauMem,
                    "t_ref": PS.t_ref,
                    "E_L"  : PS.E_L,
                    "V_reset" : PS.V_reset,
                    "V_m"  : PS.V_m,    # Current potential
                    "V_th" : PS.theta,}

    label = PS["label"]
    store_spikes = True

    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": dt,
                          "print_time": True,
                          "overwrite_files": True,
                          "grng_seed": simulation_index+threads,
                          #"rng_seeds" : (simulation_index,),
                          "rng_seeds" : range(simulation_index*10+threads+1, simulation_index*10+2*threads+1),
                          "total_num_virtual_procs": threads,
                          })

    nest.SetDefaults("iaf_psc_delta", params=neuron_params)
    nest.CopyModel("static_synapse", "excitatory", {"weight": J_EX, "delay": delay})
    nest.CopyModel("static_synapse", "inhibitory", {"weight": J_IN, "delay": delay})

    ###############################
    # Creating nodes and devices: #
    ###############################
    nodes_EX = nest.Create("iaf_psc_delta", NE)     # excitatory neurons
    nodes_IN = nest.Create("iaf_psc_delta", NI)     # inhibitory neurons
    nodes_LGN = nest.Create("parrot_neuron", N_LGN) # LGN neurons for recording their synapse activity

    # Initializing nodes to be randomly distributed around the threshold potential
    nest.SetStatus(nodes_EX, "V_m", np.random.rand(NE)*PS.theta )
    nest.SetStatus(nodes_IN, "V_m", np.random.rand(NI)*PS.theta )
    ## SET INITIAL V_m STATUS HERE!

    # Same connection rules for all:
    conn_spec_EX  = {"rule": "fixed_indegree", "indegree" : CE}
    conn_spec_IN  = {"rule": "fixed_indegree", "indegree" : CI}
    conn_spec_LGN  = {"rule": "fixed_indegree", "indegree" : C_LGN}
    #conn_spec_background  = {"rule": "fixed_indegree", "indegree" : C_background}
    conn_spec_background  = {"rule": "all_to_all"}


    # connection_specifications_LGN_EX = {"rule": "fixed_indegree", "indegree" : CE_LGN}
    # connection_specifications_LGN_IN = {"rule": "fixed_indegree", "indegree" : CI_LGN}


    # syn_spec_EX = {"weight": J_EX, 'delay': delay}
    # syn_spec_IN = {"weight": J_IN, 'delay': delay}
    # syn_spec_LGN = {"weight": J_LGN, "delay" : delay}
    # syn_spec_background = {"weight": J_background}

    
    LGN_output = nest.Create("inhomogeneous_poisson_generator")     # output from LGN
    nest.SetStatus(LGN_output, {"rate_times": rate_times, "rate_values": poisson_rates})

    background = nest.Create("poisson_generator")                   # background activity (rest of the brain)
    nest.SetStatus(background, {"rate" : background_rate})

    spike_detector_EX = nest.Create("spike_detector",  params={"to_file": store_spikes,
                                    "label" : spike_dir + "/" + label + "-EX-"+str(simulation_index)+"-"+str(class_label),
                                    "withtime": True,
                                    "withgid": True ,
                                    "use_gid_in_filename": False})
    spike_detector_IN = nest.Create("spike_detector",  params={"to_file": store_spikes,
                                    "label" : spike_dir + "/" + label + "-IN-"+str(simulation_index)+"-"+str(class_label),
                                    "withtime": True,
                                    "withgid": True,
                                    "use_gid_in_filename": False  })
    spike_detector_LGN = nest.Create("spike_detector", params={"to_file": store_spikes,
                                    "label" : spike_dir + "/" + label + "-LGN-"+str(simulation_index)+"-"+str(class_label),
                                    "withtime": True,
                                    "withgid": True,
                                    "use_gid_in_filename": False  })

    volt_check = nest.Create("multimeter")
    nest.SetStatus(volt_check, {"withtime": True, "record_from":["V_m"]})
    
    #######################
    # Connecting network: #
    #######################
    if fixed_connectome:
        nest.SetKernelStatus({
                            "grng_seed": 0,
                            "rng_seeds" : range(1, threads+1),
                            })
        
    nest.Connect(LGN_output, nodes_LGN)     # connecting to the parrot neurons so we can record the LGN synapse activity

    nest.Connect(nodes_LGN, nodes_EX + nodes_IN, conn_spec=conn_spec_LGN, syn_spec="excitatory")
    nest.Connect(background, nodes_EX + nodes_IN, conn_spec=conn_spec_background, syn_spec="excitatory")

    nest.Connect(nodes_EX, nodes_IN + nodes_EX, conn_spec=conn_spec_EX, syn_spec="excitatory")
    nest.Connect(nodes_IN, nodes_EX + nodes_IN, conn_spec=conn_spec_IN, syn_spec="inhibitory")

    # nest.Connect(LGN_output, nodes_ex+nodes_in, conn_spec=connection_specifications, syn_spec=syn_spec_LGN)
    # nest.Connect(background, nodes_ex+nodes_in, conn_spec=connection_specifications, syn_spec=syn_spec_background)
    nest.Connect(nodes_EX,  spike_detector_EX)
    nest.Connect(nodes_IN,  spike_detector_IN)
    nest.Connect(nodes_LGN, spike_detector_LGN)

    #nest.Connect(volt_check, nodes_EX[0:1])


    ######################
    # Running simulation #
    ######################
    if fixed_connectome:    # Resetting seed:
        nest.SetKernelStatus({"grng_seed": simulation_index+threads,
                            #"rng_seeds" : (simulation_index,),
                            "rng_seeds" : range(simulation_index*10+threads+1, simulation_index*10+2*threads+1),
                            })

    nest.Simulate(simtime)

    events_EX   = nest.GetStatus(spike_detector_EX)[0]["events"]
    events_IN   = nest.GetStatus(spike_detector_IN)[0]["events"]
    events_LGN  = nest.GetStatus(spike_detector_LGN)[0]["events"]


    voltages = nest.GetStatus(volt_check)[0]
    volt = voltages["events"]["V_m"]
    times = voltages["events"]["times"]


    #plt.plot(times, volt)
    #plt.close()
    #plt.show()


    # # ####################
    # # # Plotting spikes: #
    # # ####################
    # ax1 = plt.axes()
    # times_EX,senders_EX = events_EX["times"], events_EX["senders"]
    # times_IN,senders_IN = events_IN["times"], events_IN["senders"]
    # ax1.plot(times_EX, senders_EX, "b.", label="Excitatory neurons")
    # ax1.plot(times_IN, senders_IN, "r.", label="Inhibitory neurons")
    # ax1.legend(loc=4)
    # ax1.set_xlabel("Time (ms)")
    # ax1.set_ylabel("Neuron ID")
    # plt.savefig("spikes%i.png" % simulation_index)
    # plt.close()
    #del ax1
    #plt.show()
    return events_EX, events_IN, events_LGN





# if __name__ == "__main__":
#
#     ########################################################
#     # Update and import the network/simulation parameters: #
#     ########################################################
#     current_working_dir = os.getcwd()
#     parameters_path = "test_params"             # path to file with the parameters
#
#     Set_parameters(parameters_path)
#     network_parameters = ParameterSet(parameters_path)  # importing
#     PS = network_parameters
#     ########################################################################
#     # Creating the kernel for LFP approximation by population firing rate: #
#     ########################################################################
#     create_kernel = False
#     if create_kernel:
#         Make_kernels(network_parameters)
#         Plot_kernels(network_parameters)    # saving to file
#
#     #poisson_rates = [0., 0., 0.]           # [rate0, rate1, ..]
#     #rate_times = [10., 1000., 1500.0]       # [rate0_stop_time, rate1_stop_time, ..]
#     #Create_LFP_approximation_kernel_spikes(dir="data/", network_parameters=network_parameters)
#
#     #events = Run_simulation(rate_times, poisson_rates, network_parameters, simulation_index=1)
#     #LFP = Calculate_LFP(events, network_parameters)
#
#     #Plot_LFP(LFP, 0, "LFP.png")
#
#     ######################
#     # Sinusioidal input: #
#     A = 5   # Amplitude of the sinusioid
#     b = 15  # mean of sinusoid
#     t = np.around(np.linspace(10, PS.simtime, 50), decimals=1)
#
#     #t[1] = 109.11
#     #print(t)
#     #exit("ballemos")
#     Ns = np.array([3, 7, 9, 12])  # number of waves per simtime
#     for i in range(len(Ns)):
#         freq = 2*np.pi*Ns[i]
#         signal = A*np.sin(freq*t) + b
#
#         events = Run_simulation(t, signal, PS, simulation_index=i)
#         LFP = Calculate_LFP(events, PS)
#         Plot_LFP(LFP, 0, "LFP%i.png" % i)
