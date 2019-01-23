import pylgn
import quantities as pq
import pylgn.kernels.spatial as spl
import pylgn.kernels.temporal as tpl
from scipy.misc import imread 
import numpy as np



######################## Getting the paths to the input images ########################
import os
from itertools import permutations
current_directory = os.getcwd()
stimulus_image_directory = current_directory + "/stimulus_images/" 
dir_info = list(os.walk(stimulus_image_directory))
stimulus_image_paths =  sorted([stimulus_image_directory + dir_info[0][2][i] \
                        for i in range(len(dir_info[0][2])) if not dir_info[0][2][i][0] == "."])
# The if test above makes sure it ommits hidden files
N = len(stimulus_image_paths)
image_indices = list(range(N))
print(stimulus_image_paths)
exit("bal")

def Get_image_sequence(permutation):
    """
    Function for getting the paths to the stimuli images in a given permutation.

    Arguments:
    ----------
    permutation : list / numpy array (1d)
        The order of the input images 

    Returns:
    --------
    sequence : list 
        Contains the paths to the images
    """
    sequence = [stimulus_image_paths[i] for i in permutation]
    return sequence





######################## Simulation parameters: ########################

# Stimuli parameters:
image_duration = 100*pq.ms      # duration of each individual stimulus image
delay = 0*pq.ms                 # delay between images
total_simulation_time = N*image_duration + N*delay


# Network resolution parameters:
nt=8        # 2**nt is the number of time steps in the simulation
nr=8
dt=1*pq.ms
dr=0.1*pq.deg

# Ganglion DOG parameters:
A_g = 1
a_g = 0.62*pq.deg
B_g = 0.85
b_g = 1.26*pq.deg

# Ganglion weights:
weights = 0.81 

# Relay DOG parameters:
A_r = 1
a_r = 0.62*pq.deg
B_r = 0.85
b_r = 1.26*pq.deg


######################## Setting up pyLGN network ########################
network = pylgn.Network()
integrator = network.create_integrator(nt, nr, dt, dr)

# Ganglion Kernels
Wg_r = spl.create_dog_ft(A_g, a_g, B_g, b_g)
#Wg_t = tpl.create_biphasic_ft()
Wg_t = tpl.create_delta_ft()

# Relay kernels
Wr_r = spl.create_dog_ft(A_r, a_r, B_r, b_r)
#Wr_t = tpl.create_biphasic_ft()
Wr_t = tpl.create_delta_ft()

ganglion = network.create_ganglion_cell()
relay = network.create_relay_cell()

network.connect(ganglion, relay, kernel=(Wr_r, Wr_t), weight=weights)


stimulus = pylgn.stimulus.create_natural_image(filenames=stimulus_image_paths,
                                               delay=delay,
                                               duration=image_duration)
network.set_stimulus(stimulus, compute_fft=True)

network.compute_response(relay)

# relay.center_response contains the relay fire rates for each timestep in the simulation


# pylgn.plot.animate_cube(relay.response,
#                         title = "asd",
#                         dt=integrator.dt.rescale("ms"))