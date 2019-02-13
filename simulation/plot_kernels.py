"""
            Plots the 3 kernels used for mapping population firing rates to LFP 
"""
import numpy as np
import os
from parameters import ParameterSet
import h5py
import matplotlib.pyplot as plt


def Plot_kernels(network_parameters, ax=0):
    if ax == 0:
        ax = plt.axes()
        single_plot = True
    else:
        single_plot = False

    scalebar=True
    PS = network_parameters
    n_channels = PS.n_channels
    kernel_path = PS.kernel_path
    #kernel_path = "/home/samknu/MyRepos/Brunel-with-optical-input/simulation/../output/out4/kernels.h5"
    print(PS.kernel_path)
    with h5py.File(kernel_path, "r") as file:
        # * 1000 to plot as uV instead of mV
        EX_kernel = file["EX"][:]   *1000
        IN_kernel = file["IN"][:]   *1000
        LGN_kernel = file["LGN"][:] *1000


    space = 10      # uV 

    time = np.linspace(0,200,201)

    ##############
    # Normalize: #    # print(diffch5)
    # #IN_kernel[4] *= 0
    # print("scale=", scale, "uV")
    # EX_kernel /= scale
    # IN_kernel /= scale
    # LGN_kernel /= scale
    ##############
    #scale = np.max([np.max(abs(EX_kernel)), np.max(abs(IN_kernel)), np.max(abs(LGN_kernel))])
    diffch5 = np.max(IN_kernel[4])-np.min(IN_kernel[4])
    #space = diffch5
    space = diffch5*1.5
    print(diffch5)
    # #IN_kernel[4] *= 0
    # print("scale=", scale, "uV")
    # EX_kernel /= scale
    # IN_kernel /= scale
    # LGN_kernel /= scale
    #print(([np.min((EX_kernel)), np.min((IN_kernel)), np.min((LGN_kernel))]))

    for i in range(n_channels):
        ax.plot(EX_kernel[i]  + space*(n_channels-i), color="#f58231", label="EX" if i==1 else None)
        ax.plot(IN_kernel[i]  + space*(n_channels-i),  color="#4363d8", label="IN" if i==1 else None)
        ax.plot(LGN_kernel[i] + space*(n_channels-i), color="k", label="LGN" if i==1 else None)

    ####################
    # Adding scalebar: #
    ####################
    if scalebar:
        posx = time[-1]
        posy = 3*space
        barlength = 0.1   # uV
        #print(barlength*scale, "uV")
        #line=barlength
        line = barlength#/scale # barlength uV

        ax.plot([posx,posx],[posy, posy+line], color="k", linewidth=3)
        ax.text(posx*1.01, posy+line/2*0.8, "$%s \mu V$" % barlength)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    yticks = np.arange(space, (n_channels+1)*space, space)

    ax.yaxis.set_ticks(ticks=yticks)#, labels = np.flip(["Ch. %s" %i for i in range(1,n_channels+1)]))
    ax.yaxis.set_ticklabels(np.flip(["Ch. %s" %i for i in range(1,n_channels+1)]))

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xlabel("Time (ms)")

    ax.set_title("Kernels")
    ax.legend(loc=4, prop={"size": 12})
    plt.tight_layout()
    if single_plot:

        plt.savefig(PS.kernel_plot) ### remove
        #plt.savefig("asdsdd")
        plt.show()
        plt.close()
# if __name__=="__main__":

#     from parameters import ParameterSet
#     network_parameters = ParameterSet("params")
#     Plot_kernels(network_parameters)