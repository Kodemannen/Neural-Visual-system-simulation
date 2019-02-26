import numpy as np
import nest
import time
import os
import matplotlib.pyplot as plt
from parameters import ParameterSet
import h5py
from set_parameters import Set_parameters
import sys


def Calculate_LFP(events, network_parameters):
    PS = network_parameters

    ####################
    # Fetching kernel: #
    ####################
    # Kernels are in mV:
    kernel_path = PS.kernel_path
    with h5py.File(kernel_path, "r") as file:
        EX_kernel = file["EX"][:]     
        IN_kernel = file["IN"][:]
        LGN_kernel = file["LGN"][:]

    #########################################
    # Binning and calculating firing rates: #
    #########################################
    events_EX, events_IN, events_LGN = events
    times_EX=events_EX["times"]
    times_IN=events_IN["times"]
    times_LGN=events_LGN["times"]

    bins = np.arange(0, PS.simtime + 2, step=1) - 0.5
    rates_EX = np.histogram(times_EX, bins)[0]      # population firing rates
    rates_IN = np.histogram(times_IN, bins)[0]
    rates_LGN = np.histogram(times_LGN, bins)[0]
    # histogram because each bin will be the number of neurons that fire during the the timestep that
    # corresponds to that bin

    
    LFP = np.zeros(shape=(PS.n_channels, len(rates_EX)))

    for i in range(PS.n_channels):
        LFP[i] =   np.convolve(rates_EX, EX_kernel[i], mode="same") \
                 + np.convolve(rates_IN, IN_kernel[i], mode="same") \
                 #+ np.convolve(rates_LGN, LGN_kernel[i], mode="same") # unit mV
    return LFP, [rates_EX, rates_IN, rates_LGN, bins] 
