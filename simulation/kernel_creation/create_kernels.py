import numpy as np
import os
from parameters import ParameterSet
import h5py
from .create_LFP_from_simultaneous_firings import Create_LFP_from_simultaneous_firings
import matplotlib.pyplot as plt

def Create_fake_spikes(network_parameters):
    """
    Creates a "fake" Nest spike file for each of the 3 populations where all neurons
    in each population fire simultaneously. At t=100 ms for the excitatory population,
    t=300 ms for the inhibitory population, and t=500 ms for the LGN population.
    """

    PS = network_parameters
    dir = PS.fake_spikes_path

    #########################################
    # CLEARING FOLDER (deleting old files): #
    #########################################
    print(list(os.walk(dir)))
    files = list(os.walk(dir))[0][2]
    
    for name in files:
        os.remove(dir+ "/" + name)


    #########################
    # Generating the files: #
    #########################

    NE = PS.NE
    NI = PS.NI
    N_LGN = PS.N_LGN

    #N_LGN = round(PS.N_neurons*PS.epsilon)
    label = network_parameters["label"]
    ex_file = open(dir + "/" + label + "-py-" + "EX-%.i-0.gdf" % NE , "w")
    ex_time = "100.000"     # time when we pretend all the excitatories spike

    for i in range(1,NE+1):
        ex_file.write(str(i) + " " + ex_time + "\r\n" )
    ex_file.close()


    in_file = open(dir + "/" + label + "-py-" + "IN-%.i-0.gdf" % NI, "w")
    in_time = "300.000"     # time when we pretend all the inhibitories spike

    for i in range(NE+1, NE + NI+1):
        in_file.write(str(i) + " " + in_time + "\r\n" )
    in_file.close()

    ###################
    # The LGN spikes: #
    ###################
    in_file = open(dir + "/" + label + "-py-" + "LGN-%.i-0.gdf" % NI, "w")
    in_time = "500.000"     # time when we pretend all the LGN neurons spike

    for i in range(NE+1 + NI, NE + NI + N_LGN + 1):
        in_file.write(str(i) + " " + in_time + "\r\n" )
    in_file.close()

    return 0


def Create_kernels(network_parameters):
    """
    Creating the kernels used for mapping population firing rates to
    LFP signals.
    The kernels have units mV.
    """

    Create_fake_spikes(network_parameters)
    Create_LFP_from_simultaneous_firings(network_parameters)

    PS = network_parameters     # just for faster typing
    data_folder = PS.hybrid_output_path + "/populations/"
    kernel_path = PS.kernel_path
    
    with h5py.File(data_folder + "EX_population_LFP.h5", "r") as file:
        EX_LFP = file["data"][()]
        #print(EX_LFP.shape)
    with h5py.File(data_folder + "IN_population_LFP.h5", "r") as file:
        IN_LFP = file["data"][()]
        #print(IN_LFP.shape)

    ############################################################################
    # Scaling the kernel so it represents the ratio  of LFP per firing neuron: #
    ############################################################################
    
    EX_kernel = (EX_LFP[:,0:199] + IN_LFP[:,0:199])/PS.NE
    IN_kernel = (EX_LFP[:,200:399] + IN_LFP[:,200:399])/PS.NI
    LGN_kernel = (EX_LFP[:,400:599] + IN_LFP[:,400:599])/PS.N_LGN

    #########################
    # Save kernels to file: #
    #########################
    kernel_file = h5py.File(kernel_path, "w")
    kernel_file.create_dataset("EX", data=EX_kernel)
    kernel_file.create_dataset("IN", data=IN_kernel)
    kernel_file.create_dataset("LGN", data=LGN_kernel)
    kernel_file.close()
