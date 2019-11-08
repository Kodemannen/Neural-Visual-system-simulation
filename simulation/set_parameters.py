"""
                    Parameters for network simulation
                    Must be in the same folder as main.
"""
import os
import sys
import parameters as ps
import numpy as np
import datetime

time = datetime.datetime.now()
timestamp = time.strftime("%c")#+"/"+time.strftime("%X")

def Set_parameters(abelrun):

    ###############################################################
    # Deciding if kernel will be created or if its already there: #
    ###############################################################
    create_kernel = True       # whether to create kernel or not
    
    ###########################################
    # Setting paths (sim_dir = absolute path) #
    ###########################################
    if abelrun:
        # Running on Abel: 
        sim_dir = os.path.join(os.getcwd(),os.path.dirname(os.path.relpath(__file__)))
        if sim_dir[-1] == "/":  
            sim_dir = sim_dir[:-1]  # removing "/" at the end
        sim_output_dir ="/work/users/samuelkk/output/out"     # work dir on Abel
    else:
        # Running on local comp: 
        sim_dir = os.path.join(os.getcwd(),os.path.dirname(os.path.relpath(__file__)))
        if sim_dir[-1] == "/":  
            sim_dir = sim_dir[:-1]  # removing "/" at the end
        sim_output_dir = sim_dir + "/../output/out"            # on my computer

    
    ##########################################################
    # Adding index to output/sim folder if it already exist: #
    ##########################################################
    original_name = sim_output_dir
    index=0
    if os.path.isdir(sim_output_dir):
        while os.path.isdir(sim_output_dir)==True:
            sim_output_dir = original_name + str(index)
            index+=1

    PS = ps.ParameterSet(dict(
        #################
        # Folder paths: #
        #################
        sim_dir = sim_dir,
        sim_output_dir = sim_output_dir,
        
        pyLGNimgs = sim_output_dir + "/pyLGNimgs",
        heatmap_matrices = sim_output_dir + "/heatmap_matrices",
        hybrid_output_path = sim_output_dir + "/hybridLFPy_output",   # folder where the hybridLFPy stuff is stored
        nest_output_path= sim_output_dir + "/nest_output",
        population_rates_path=sim_output_dir + "/population_rates",
        LFP_path = sim_output_dir + "/LFP_files",                # where the LFP signals are kept
        #LFP_plot_path = sim_output_dir + "/LFP_plots/",
        
        fake_spikes_path = sim_output_dir + "/kernel_creation/fake_spikes",      # folder where the fake spikes are stored
    ))
    
    #####################
    # Creating folders: #
    #####################

    os.makedirs(PS.sim_output_dir)
    for key in PS:
        if not os.path.isdir(PS[key]):
            os.makedirs(PS[key])


    PS.update(dict(
        create_kernel = create_kernel,

        ###############
        # File paths: #
        ###############
        #kernel_path = sim_output_dir + "/kernels.h5",        # meaning this folder
        #kernel_plot = sim_output_dir + "/kernels.png",

        kernel_path = sim_output_dir + "/kernels.h5"  if create_kernel else sim_dir + "/kernels.h5" ,        # meaning this folder
        kernel_plot = sim_output_dir + "/kernels.pdf",# if create_kernel else sim_dir + "/kernels.png",
        params_path = sim_output_dir + "/params",



        ###########################
        # Independent parameters: #
        ###########################
        ps_id = "cortex",
        #LFP_kernel_path = '/home/samknu/MyRepos/MasterProject/test_simulation/data/kernels/test_kernel.h5',
        predict_LFP = True,
        label ="spikes",    # label for the spike files
        save_spikes=False,
        dt=0.1,             # (ms) Simulation time resolution 

        #simtime = 1001.,    # Simulation time in ms
        #simtime = 250.*12,
        simtime = 100,
        create_kernel_simtime = 600,    # simtime for when creating kernel

        #nest_seed=int(time.time()), # base for seeds, will be updated for each individual parameterset
        #numpy_seed_nest=int(time.time()/2),
        threads=8,          # number of parallel threads in Nest
        nest_seed = 1,
        #hybrid_seed=int(time.time()/3),
        #numpy_seed_hybrid=int(time.time()/4),
        hybrid_seed = 0,
        numpy_seed_hybrid = 1,


        ##################################
        # Neuron and synapse parameters: #
        ##################################
        J_EX = .1,     # excitatory weight, unit: nS ?
        g=5.2,#*1.1,         # ratio inhibitory weight/excitatory weight (before: 5.0)
        eta=0.0,        # external rate relative to threshold rate
        mean_eta=1.1,    # effective eta
        #ean_eta=0.85,
        #background_rate = 10.0,  # poissonian background rate

        epsilon=0.1,    # connection probability
        CMem=1.0,       # capacitance of membrane in in pF      (specific capacitance?)
        theta=20.0,     # membrane threshold potential in mV
        V_reset=10.0,   # reset potential of membrane in mV
        E_L=0.0,      # resting membrane potential in mV
        V_m=0.0,      # membrane potential in mV

        tauSyn = 5.0,   # synaptic time constant for alpha synapses in HybridLFPy (ms)
        tauMem=20.0,    # time constant of membrane potential in ms
        delay=1.5,      # synaptic delay
        t_ref=2.0,      # refractory period
        sigma_ex=0.3,   # width of Gaussian profile of excitatory connections
        sigma_in=0.3,   # sigma in mm

        n_channels = 6, # number of LFP recording channels
        fixed_connectome=False, # use fixed connectome from file

        order=25     # network scaling factor
    ).items())

    ################################################
    # Params that are dependent on the ones above: #
    ################################################
    PS.update(dict(
        NE = 4 * PS.order, # number of excitatory neurons
        NI = 1 * PS.order,  # number of inhibitory neurons
        N_LGN = round(1/2.5 * PS.order),   # 1000
        #J_EX = PS.J_EX,
        J_IN = -PS.g*PS.J_EX,
        J_LGN = PS.J_EX,
        J_background = PS.J_EX
    ).items())

    PS.update(dict(
        N_neurons = PS.NE + PS.NI, # total number of neurons
        CE = round(PS.NE * PS.epsilon),  # number of excitatory synapses per neuron # C_EX
        CI = round(PS.NI * PS.epsilon),  # number of inhibitory synapses per neuron # C_IN
        C_LGN = round(PS.N_LGN * PS.epsilon), # number of LGN synapses onto an EX or IN neuron

    ).items())

    PS.update(dict(
        C_tot = PS.CE + PS.CI, #+ PS.C_LGN + PS.C_background ??,  # total number of synapses per neuron
        #C_background = PS.CE,
        C_background = 1.5,

        threshold_rate = PS.theta/(PS.J_EX*PS.tauMem),  #kHz
    ).items())

    PS.update(dict(
        background_rate = PS.eta*PS.threshold_rate*1000,    # Hz
    ).items())  # nest uses Hz while threshold_rate is in kHz since tauMem is in ms
        # to 15 kHz = 15 000 Hz, etc. so we have to *1000

    ##########################
    # hybridLFPy parameters: #
    ##########################
    #population (and cell type) specific parameters
    PS.update(dict(
        #no cell type specificity within each E-I population
        #hence X == x and Y == X
        X = ["EX", "IN", "LGN"],    # Presynaptic
        Y = ["EX", "IN"],           # Postsynaptic

        #population-specific LFPy.Cell parameters
        cellParams = dict(
            #excitory cells:
            EX = dict(
                morphology = sim_dir + '/kernel_creation/morphologies/stretched/L4E_53rpy1_cut.hoc',
                v_init = PS.E_L,
                cm = 1.0,
                Ra = 150,

                nsegs_method = 'lambda_f',
                lambda_f = 100,

                passive = True,
                passive_parameters = {"g_pas": 1/(PS.tauMem*1E+3 / 1.0),
                                      "e_pas": PS.E_L},
                dt = PS.dt,
                tstart = 0,
                tstop = PS.simtime,
                verbose = False,

                # Deprecated:
                #rm = PS.tauMem * 1E3 / 1.0, #assume cm=1.
                #g_pas = 1/(PS.tauMem*1E+3 / 1.0),
                #e_pas = PS.E_L,

            ),
            #inhibitory cells
            IN = dict(
                morphology = sim_dir + '/kernel_creation/morphologies/stretched/L4E_j7_L4stellate.hoc',
                v_init = PS.E_L,
                cm = 1.0,
                Ra = 150,

                nsegs_method = 'lambda_f',
                lambda_f = 100,

                passive = True,
                passive_parameters = {"g_pas": 1/(PS.tauMem*1E+3 / 1.0),
                                      "e_pas": PS.E_L},

                dt = PS.dt,
                tstart = 0,
                tstop = PS.simtime,
                verbose = False,

                # Deprecated:
                #g_pas = 1/(PS.tauMem*1E+3 / 1.0),
                #e_pas = PS.E_L,
            ),
        ),

        #assuming excitatory cells are pyramidal
        rand_rot_axis = dict(
            EX = ['z'],
            IN = ['x', 'y', 'z'],
        ),

        #kwargs passed to LFPy.Cell.simulate()
        simulationParams = dict(),

        #set up parameters corresponding to cylindrical model populations
        populationParams = dict(
            EX = dict(
                number = PS.NE,
                radius = np.sqrt(1000**2 / np.pi),
                z_min = -450,
                z_max = -350,
                min_cell_interdist = 1.,
                ),
            IN = dict(
                number = PS.NI,
                radius = np.sqrt(1000**2 / np.pi),
                z_min = -450,
                z_max = -350,
                min_cell_interdist = 1.,
                ),
        ),

        #set the boundaries between the "upper" and "lower" layer
        layerBoundaries = [[0., -300],
                           [-300, -500]],

        #set the geometry of the virtual recording device
        electrodeParams = dict(
                #contact locations:
                x = [0]*6,
                y = [0]*6,
                z = [x*-100. for x in range(6)],
                #extracellular conductivity:
                sigma = 0.3,
                #contact surface normals, radius, n-point averaging
                N = [[1, 0, 0]]*6,
                r = 5,
                n = 20,
                seedvalue = None,
                #dendrite line sources, soma as sphere source (Linden2014)
                method = 'soma_as_point',
                #no somas within the constraints of the "electrode shank":
                r_z = [[-1E199, -600, -550, 1E99],[0, 0, 10, 10]],
        ),

        #runtime, cell-specific attributes and output that will be stored
        savelist = [
            'somapos',
            'x',
            'y',
            'z',
            'LFP'
        ],
        pp_savelist = ['LFP'],
        plots=False,
        #flag for switching on calculation of CSD
        calculateCSD = False,

        #time resolution of saved signals
        dt_output = 1.
    ).items())



    ########################################################
    # Creating matrix with synapse weights for hybridLFPy: #
    ########################################################
    K = 1/(np.e*PS.tauSyn**2)   # for scaling since hybridLFPy uses
    # alpha synapses while NEST uses delta
    PS.update(dict(J_yX = dict(
                     EX = [PS['J_EX']*K, PS['J_IN']*K, PS["J_LGN"]*K],
                     IN = [PS['J_EX']*K, PS['J_IN']*K, PS["J_LGN"]*K],
                     )).items())
    


    #for each population, define layer- and population-specific connectivity
    #parameters
    PS.update(dict(
        #number of connections from each presynaptic population onto each
        #layer per postsynaptic population, preserving overall indegree

        ######################
        # IKKE FERDIG!!! #####

        # SPECIFIES TWO LAYERS:

        k_yXL = dict(   # k = indegree
            # NUMBERS IN EACH column must sum to CE and CI etc.
            ## TWO LAYERS:
            EX = [[round(PS.CE*0.5), 0,     round(PS.C_LGN*0.5)],    # FINF CI_LGN og CE_XLGN
                  [round(PS.CE*0.5), PS.CI, round(PS.C_LGN*0.5)]],   # are the parameters that decide how many
            IN = [[0,     0,     round(PS.C_LGN*0.5)],               # lgn neurons a cortex neuron gets signals from
                  [PS.CE, PS.CI, round(PS.C_LGN*0.5)]],
        ),

        #set up synapse parameters as derived from the network
        synParams = dict(
            EX = dict(
                section = ['apic', 'dend'], # excitatory synapses are
                # only connected to apic and dendrite sections of neurons
                syntype = 'AlphaISyn'
            ),
            IN = dict(
                section = ['dend', 'soma'],
                syntype = 'AlphaISyn'
            ),
            LGN = dict(
                section = ['apic', 'dend'],  # kobler til apical-dendritter og basal-dendritter
                syntype = "AlphaISyn"
            ),
        ),

        #set up table of synapse time constants from each presynaptic populations
        tau_yX = dict(
            EX = [PS.tauSyn, PS.tauSyn, PS.tauSyn],
            IN = [PS.tauSyn, PS.tauSyn, PS.tauSyn],
        ),
        #set up delays, here using fixed delays of network
        synDelayLoc = dict(
            EX = [PS.delay, PS.delay, PS.delay],
            IN = [PS.delay, PS.delay, PS.delay],
        ),
        #no distribution of delays
        synDelayScale = dict(
            EX = [None, None, None],
            IN = [None, None, None],
        ),
    ).items())

    #putative mappting between population type and cell type specificity,
    #but here all presynaptic senders are also postsynaptic targets
    PS.update(dict(
        mapping_Yy = list(zip(PS.X, PS.X))      # DENNE??
    ).items())

    PS.save(PS.params_path)
    print(PS.params_path)
    return PS


if __name__=="__main__":
    abelrun = sys.argv[1].lower() == "abel"
    Set_parameters(abelrun)
