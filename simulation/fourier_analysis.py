"""
                Power spectrum analysis (using Welch's method) of LFPs created
                with sinusoidal inputs from "LGN"
"""
import numpy as np
import parameters as ps
import matplotlib.pyplot as plt
import time
from scipy import signal
import sys
#from ..Simulation import set_parameters
from set_parameters import Set_parameters
##################################
# Getting simulation parameters: #
##################################
params_path = "params"
Set_parameters(params_path)     # updating parameters file
network_parameters = ps.ParameterSet(params_path)
LFP_path = network_parameters.LFP_path

LFP_example = np.load(LFP_path + f"LFP-{20}-{4}.npy")[0]
f, powers = signal.welch(LFP_example, network_parameters.dt*1000, nperseg=800)  # *1000 converts ms to s
powers_placeholder = np.zeros(shape=powers.shape)

# LFP1 = np.load(LFP_path + f"LFP-{700}-{4}.npy")[0]         # 4 Hz
# LFP2 = np.load(LFP_path + f"LFP-{3000}-{16}.npy")[0]      # 16 Hz
#
# f, powers1 = signal.welch(LFP1, fs=network_parameters.dt*1000, nperseg=300)
# f, powers2 = signal.welch(LFP2, fs=network_parameters.dt*1000, nperseg=300)
#
# #plt.plot(f,powers)
# plt.semilogy(f, powers1)
# plt.semilogy(f, powers2)
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD [V**2/Hz]')
# plt.show()


freqs = [4,8,12,16] # Hz
sim_index = 0



# for j in range(4):     #4
#     power_spectra_avg = powers_placeholder.copy()
#     for i in range(1000): #1000
#         LFP = np.load(LFP_path + f"LFP-{sim_index}-{freqs[j]}.npy")[0]  # Only channel 1
#
#         f, powers = signal.welch(LFP, fs=network_parameters.dt*1000*10, nperseg=800)    # *1000 to convert from ms to s
#                                                                                         # *10 because the firing rates
#         power_spectra_avg += powers                                                     # are sampled at 10*dt resolution
#
#         sim_index += 1
#
#     power_spectra_avg /= 1000.   # averaging
#     fig = plt.semilogy(f, power_spectra_avg, label=f"Freq={freqs[j]} Hz")
#     #fig = plt.plot(f, power_spectra_avg, label=f"Freq={freqs[j]} Hz")
#
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD [V**2/Hz]')
# plt.legend()
# plt.savefig("semliogy")
#
# plt.show()


sim_index = 0
for j in range(4):     #4
    power_spectra_avg = powers_placeholder.copy()
    for i in range(25): #1000
        LFP = np.load(LFP_path + f"LFP-{sim_index}-{freqs[j]}.npy")[0]  # Only channel 1

        f, powers = signal.welch(LFP, fs=network_parameters.dt*1000*10, nperseg=800)    # *1000 to convert from ms to s
                                                                                        # *10 because the firing rates
        power_spectra_avg += powers                                                     # are sampled at 10*dt resolution

        sim_index += 1

    power_spectra_avg /= 1000.   # averaging
    #fig = plt.semilogy(f, power_spectra_avg, label=f"Freq={freqs[j]} Hz")
    fig = plt.plot(f, power_spectra_avg, label=f"Freq={freqs[j]} Hz")


plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
#plt.xlim([0, 100])
plt.semilogy()
plt.semilogx()
plt.legend()
plt.savefig("plot")


plt.show()



#plt.savefig("plot")

#plt.show()

    #all_power_spectra_avg.append(power_spectra_avg)
