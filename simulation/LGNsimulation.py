import pylgn
import quantities as pq
import pylgn.kernels.spatial as spl
import pylgn.kernels.temporal as tpl

import numpy as np
import quantities as qp

######################## Getting the paths to the input images ########################
import os


current_directory = os.getcwd()
stimulus_image_directory = current_directory + "/stimulus_images/" 
dir_info = list(os.walk(stimulus_image_directory))
stimulus_image_paths =  sorted([stimulus_image_directory + dir_info[0][2][i] \
                        for i in range(len(dir_info[0][2])) if not dir_info[0][2][i][0] == "."])
# The if test above makes sure it ommits hidden files
N = len(stimulus_image_paths)
image_indices = list(range(N))

def Get_LGN_signal(permutation, amplitude):

    sequence = [stimulus_image_paths[-1]]
    for index in permutation:
        sequence.append(stimulus_image_paths[index])
    sequence.append(stimulus_image_paths[-1])

    
    ######################## Simulation parameters: ########################

    # Network resolution parameters:
    nt=12       # 2**nt is the number of time steps in the simulation
    nr=5
    dt=1*pq.ms
    dr=0.1*pq.deg


    # Stimuli parameters:
    image_duration = 250 * pq.ms      # duration of each individual stimulus image
    delay = 0*pq.ms                 # delay between images (?)
    total_simulation_time = N*image_duration + N*delay


    # Ganglion DOG parameters:
    A_g = 1             # center strength
    a_g = 0.62*pq.deg   # center width
    B_g = 0.85          # surround strength
    b_g = 1.26*pq.deg   # surround width

    # Ganglion weights:
    weights = 1 

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
    Wg_t = tpl.create_biphasic_ft()
    #Wg_t = tpl.create_delta_ft()

    # Relay kernels
    Wr_r = spl.create_dog_ft(A_r, a_r, B_r, b_r)
    Wr_t = tpl.create_biphasic_ft()
    #Wr_t = tpl.create_delta_ft()

    ganglion = network.create_ganglion_cell()
    relay = network.create_relay_cell()

    network.connect(ganglion, relay, kernel=(Wr_r, Wr_t), weight=weights)
    
    stimulus = pylgn.stimulus.create_natural_image(filenames=sequence,
                                                delay=delay,
                                                duration=image_duration)
    
    network.set_stimulus(stimulus, compute_fft=True)
    
    network.compute_response(relay)
    signal = relay.center_response

    signal = signal[:250*12] # assuming duration of 250 ms, dt=1 ms
    
    # shifting and normalizing for nr=7
    #signal += 126/qp.s
    #signal *= np.sqrt(1.5/554)  # found the variance of the pyLGN signal to be 554 empirically. 1.5 is the variance we want the signal to have, as that corresponds to the variance of the sinusoid signal
    # This gives it a mean of 6.846617063994459 1/s
    

    # shifting and normalizing for nr = 5
    signal += 140/qp.s
    signal *= np.sqrt(amplitude/2/740.4)
    mean = np.mean(signal)
    
    # This gives it a mean of 6.512339651169235

    # shifting signal and "normalizing, sort of"
    #signal = signal + 85 / qp.s
    #signal = signal / 170

    # want to get an equal variance to a sinusoid of amplitude 3 that we used earlier.
    # a sine has variance 0.5 so variance was 0.5*3 = 1.5
    # we thus need to multiply by sqrt(1.5)

    #signal *= np.sqrt(1.5)
    

    # mean is now 0.49  Hz
    # pylgn.plot.animate_cube(relay.response,
    #                         title = "asd",
    #                         dt=integrator.dt.rescale("ms"))


    return signal*(signal>0), mean


def LGN_classification_test():
    N = 10000
    data = np.zeros((10,N,250))
    #np.random.seed(0)
    for i in range(N):
        seq = np.random.choice(10, size=10, replace=False)
        #seq = np.arange(10)

        signal = Get_LGN_signal(seq, amplitude=6)
        #plt.plot(signal, "black")
        splitted = np.split(signal,12)
        
        for j in range(10):
            data[j,i] = splitted[seq[j]+1]
            #plt.plot(np.arange((seq[j]+1)*250,(seq[j]+2)*250), data[j,i])
            #plt.plot(np.arange((j+1)*250,(j+2)*250), data[j,i])
        #plt.show()
        #exit("asd")

        # plt.plot(signal)
        # for p in range(10):
        #     plt.plot(np.arange(250,250*11), data[:,0,:].reshape(-1), linestyle="--", color="green")
        # plt.show()
        # exit("hore")
        print(i)
    
    class_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']


    np.save("/home/samknu/data/pyLGN.npy", data)
    # for j in classes:
    #     mean = np.mean(data[j], axis=0)

    #     plt.plot(mean, color=class_colors[j], label=str(j))

    # plt.legend()
    # plt.show()
    

if __name__=="__main__":
    from scipy.misc import imread 
    import matplotlib.pyplot as plt
    #np.random.seed(1)
    # for i in range(1):
    #     rate = Get_LGN_signal(np.random.choice(10, size=10, replace=False))
    #     print(np.shape(rate))

    # nr2=Get_LGN_signal(np.random.choice(10, size=10, replace=False))
    LGN_classification_test()


    # print(np.mean(nr1))
    # print(np.mean(nr2))

    # print(np.std(nr1))
    # print(np.std(nr2))

    # plt.plot(nr1)
    # plt.plot(nr2)
    # plt.show()