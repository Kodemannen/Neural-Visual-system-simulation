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

    bins = np.arange(PS.simtime + 2, ) - 0.5
    rates_EX = np.histogram(times_EX, bins)[0]
    rates_IN = np.histogram(times_IN, bins)[0]
    rates_LGN = np.histogram(times_LGN, bins)[0]

    LFP = np.zeros(shape=(PS.n_channels, len(rates_EX)))
    print(rates_EX.shape)
    print(EX_kernel.shape)
    for i in range(PS.n_channels):
        LFP[i] =   np.convolve(rates_EX, EX_kernel[i], mode="same") \
                 + np.convolve(rates_IN, IN_kernel[i], mode="same") \
                 + np.convolve(rates_LGN, LGN_kernel[i], mode="same")
    return LFP # unit mV
