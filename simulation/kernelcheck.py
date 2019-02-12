
import h5py
import numpy as np
import matplotlib.pyplot as plt
#kernel_path = PS.kernel_path
#data_folder = PS.hybrid_output_path + "/populations/"

kernel_path = "/home/samknu/MyRepos/MasterProject/Results_and_analysis/data/Output folder after creating kernel with order=2500/sim_output/kernels2.h5"
data_folder = "/home/samknu/MyRepos/MasterProject/Results_and_analysis/data/Output folder after creating kernel with order=2500/sim_output/hybridLFPy_output/populations/"


with h5py.File(data_folder + "EX_population_LFP.h5", "r") as file:
    EX_LFP = file["data"][()]
    #print(EX_LFP.shape)
with h5py.File(data_folder + "IN_population_LFP.h5", "r") as file:
    IN_LFP = file["data"][()]
    #print(IN_LFP.shape)

############################################################################
# Scaling the kernel so it represents the ratio  of LFP per firing neuron: #
############################################################################


EX_kernel = (EX_LFP[:,0:200] + IN_LFP[:,0:200])/10000           *1e6
IN_kernel = (EX_LFP[:,200:400] + IN_LFP[:,200:400])/2500        *1e6
LGN_kernel = (EX_LFP[:,400:600] + IN_LFP[:,400:600])/1000       *1e6
plt.plot(IN_kernel[4], label="IN")
plt.plot(EX_kernel[4], label="EX")
plt.plot(LGN_kernel[4],"k", label="LGN")
plt.legend()
plt.show()

# #########################
# # Save kernels to file: #
# #########################
# kernel_file = h5py.File(kernel_path, "w")
# kernel_file.create_dataset("EX", data=EX_kernel)
# kernel_file.create_dataset("IN", data=IN_kernel)
# kernel_file.create_dataset("LGN", data=LGN_kernel)
# kernel_file.close()
