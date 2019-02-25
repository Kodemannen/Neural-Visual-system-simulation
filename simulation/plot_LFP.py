import numpy as np
import nest
import time
import os
import matplotlib.pyplot as plt
from parameters import ParameterSet
import h5py
from set_parameters import Set_parameters
import sys




# def Plot_LFP(LFP, network_parameters, sim_index, class_label, ax=0):
#     if ax == 0:
#         ax = plt.axes()
#         single_plot = True
#     else:
#         single_plot = False

#     PS = network_parameters
#     filename = PS.LFP_plot_path + f"LFP-{sim_index}-{class_label}.png"
#     time = np.linspace(0, PS.simtime, LFP.shape[1])

#     scalebar=True
#     k = np.shape(LFP)[0]

#     unit = 1    # mV
#     space = 1.2

#     ##############
#     # Normalize: #
#     ##############
#     scale = np.max( [np.max(abs(LFP[i])) for i in range(k)] )


#     for i in range(k):
#         ax.plot(LFP[i] + space*(k-i), color = "black")


#     ####################
#     # Adding scalebar: #
#     ####################
#     if scalebar:
#         posx = 1021 # ms
#         posy = 2.*space
#         barlength = 0.25  # mV

#         ax.plot([posx,posx],[posy, posy+barlength], color="k", linewidth=3)
#         ax.text(posx*1.01,  posy+barlength/2*0.7, " %s mV" % barlength)

#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)

#     yticks = np.arange(space, (k+1)*space, space)
#     ax.yaxis.set_ticks(ticks=yticks)#, labels = np.flip(["Ch. %s" %i for i in range(1,n_channels+1)]))
#     ax.yaxis.set_ticklabels(np.flip(["Ch. %s" %i for i in range(1,k+1)]))

#     ax.xaxis.set_ticks_position('bottom')
#     ax.yaxis.set_ticks_position('left')
#     ax.set_xlabel("Time (ms)")

#     ax.set_title("LFP")
#     #ax.legend(loc=4, prop={"size": 12})
    
#     if single_plot:
#         plt.savefig(filename+".pdf") ### remove
#         #plt.close()
#         #plt.show()
#     return ax




def Plot_LFP(LFP,ax=0, letter=None, scalebar=True, channels_on=True,title_fontsize = None):

    if ax == 0:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    simtime = 1001 # ms
    time = np.linspace(0, simtime, LFP.shape[1])

    k = np.shape(LFP)[0]

    ##############
    # Normalize: #
    ##############
    scale = np.max( [np.max(abs(LFP[i])) for i in range(k)] )
    scale = 1
    #LFP /= scale

    space = 0.5 # mV
    for i in range(k):
        ax.plot(LFP[i] - np.mean(LFP[i]) + space*(k-i)- space/2, color = "k", linewidth=1)

    #sprint(scale)
    ####################
    # Adding scalebar: #
    ####################
    if scalebar:
        posx = 1021  # ms
        posy = 2*space - space/2 #- np.mean(LFP[4])
        barlength = .25  # mV

        ax.plot([posx,posx],[posy-barlength/2, posy+barlength/2], color="k", linewidth=3)
        ax.text(posx*1.01,posy, 
                " %smV" % barlength, fontsize=8, va="center")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if channels_on:
        yticks = np.arange(space, (k+1)*space, space) - space/2
        ax.yaxis.set_ticks(ticks=yticks)#, labels = np.flip(["Ch. %s" %i for i in range(1,n_channels+1)]))
        ax.yaxis.set_ticklabels(np.flip(["Ch. %s" %i for i in range(1,k+1)]))

        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    ax.set_xlabel("t (ms)")
    if letter != None:
        ax.text(-100, space*6.5, letter, fontsize=12)
    ax.set_title("LFP", fontsize=title_fontsize)
    #ax.legend(loc=4, prop={"size": 12})
    return ax