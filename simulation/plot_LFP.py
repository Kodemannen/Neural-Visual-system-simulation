import numpy as np
import nest
import time
import os
import matplotlib.pyplot as plt
from parameters import ParameterSet
import h5py
from set_parameters import Set_parameters
import sys



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
        ax.plot(LFP[i] - np.mean(LFP[i]) - np.mean(LFP[-1]) + space*(k-i)- space/2, color = "k", linewidth=1)

    #sprint(scale)
    ####################
    # Adding scalebar: #
    ####################
    if scalebar:
        posx = 1021  # ms
        posy = 2*space - space/2  - np.mean(LFP[-1]) #- np.mean(LFP[4])
        barlength = .25  # mV

        ax.plot([posx,posx],[posy-barlength/2, posy+barlength/2], color="k", linewidth=3)
        ax.text(posx*1.01,posy, 
                " %smV" % barlength, fontsize=8, va="center")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if channels_on:
        yticks = np.arange(space, (k+1)*space, space) - space/2  - np.mean(LFP[-1])
        ax.yaxis.set_ticks(ticks=yticks)#, labels = np.flip(["Ch. %s" %i for i in range(1,n_channels+1)]))
        ax.yaxis.set_ticklabels(np.flip(["Ch. %s" %i for i in range(1,k+1)]))

        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    ax.set_ylim([0 - np.mean(LFP[-1]), 6*space - np.mean(LFP[-1])])
    ax.set_xlabel("t (ms)")
    if letter != None:
        ax.text(-100, space*6.5, letter, fontsize=12)
    ax.set_title("LFP", fontsize=title_fontsize)
    #ax.legend(loc=4, prop={"size": 12})
    return ax