import os
import numpy as np
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from time import time
#import nest #not used, but load order determine if network is run in parallel
from hybridLFPy import PostProcess, Population, CachedNetwork, setup_file_dest
from parameters import ParameterSet
import h5py
import neuron
from mpi4py import MPI
import sys
from .example_plotting import *
#exit("ballemos")

def Create_LFP_from_simultaneous_firings(network_parameters):
    #!/usr/bin/env python
    '''
    Uses hybridLFPy to generate the LFP from a point neuron network consisting
    of 3 populations where all the neurons in each population fires simultaneouslyself.
    I.e. all excitatory neurons fire at t=100 ms, inhibitory neurons at t=300 ms, and
    all LGN neurons at t=500 ms.

    The LFP from this activity can then be used as a kernel for mapping population
    firing rates to an LFP approximation.

    Adapted from example_brunel.py
    '''

    ########## matplotlib settings #################################################
    plt.close('all')
    plt.rcParams.update({'figure.figsize': [10.0, 8.0]})


    ################# Initialization of MPI stuff ##################################
    COMM = MPI.COMM_WORLD
    SIZE = COMM.Get_size()
    RANK = COMM.Get_rank()
    print("MY RANK IS: ",RANK)

    #if True, execute full model. If False, do only the plotting. Simulation results
    #must exist.
    properrun = True

    #parameter_set_file = sys.argv[-1]

    PS = network_parameters
    spike_path = PS.fake_spikes_path

    #set some seed values
    SEED = PS.numpy_seed_hybrid
    SIMULATIONSEED = PS.hybrid_seed
    np.random.seed(int(SEED))


    cell_path = os.path.join(PS.hybrid_output_path, 'cells')
    population_path = os.path.join(PS.hybrid_output_path, 'populations')
    figure_path = os.path.join(PS.hybrid_output_path, 'figures')

    # cell_path = PS.hybrid_output_path + "cells"
    # population_path = PS.hybrid_output_path + "populations"
    # figures = PS.hybrid_output_path + "figures"

    #check if mod file for synapse model specified in alphaisyn.mod is loaded
    if not hasattr(neuron.h, 'AlphaISyn'):
        if RANK == 0:
            os.system('nrnivmodl')
        COMM.Barrier()
        neuron.load_mechanisms('.')
    if RANK == 0:
        if not os.path.isdir(cell_path):
            os.mkdir(cell_path)
        if not os.path.isdir(population_path):
            os.mkdir(population_path)
        if not os.path.isdir(figure_path):
            os.mkdir(figure_path)


    ################################################################################
    # MAIN simulation procedure                                                    #
    ################################################################################

    #tic toc
    tic = time()




    #Create an object representation containing the spiking activity of the network
    #simulation output that uses sqlite3. Again, kwargs are derived from the brunel
    #network instance.
    networkSim = CachedNetwork(
        simtime = PS.create_kernel_simtime,
        dt = PS.dt,
        #spike_output_path = PS.spike_output_path,
        spike_output_path = PS.fake_spikes_path, #
        label = PS['label'],
        ext = 'gdf',
        GIDs = {'EX'  : [1, PS.NE],
                'IN'  : [PS.NE+1, PS.NI],
                "LGN" : [PS.NE + PS.NI + 1, PS.N_LGN]},  # list==[first GID, Population size]

        #X = ['EX', 'IN', "LGN"],
        X = PS.X,
        cmap='rainbow_r',
    )

    ####### Set up populations #####################################################

    if properrun:
        #iterate over each cell type, and create populationulation object
        for i, Y in enumerate(PS.Y):
            # print(type(dict(PS.cellParams[Y]["passive_parameters"])))

            PS.cellParams[Y]["passive_parameters"] = dict(PS.cellParams[Y]["passive_parameters"])
            #exit("ballefrans")
            pop = Population(
                    cellParams = PS.cellParams[Y],
                    rand_rot_axis = PS.rand_rot_axis[Y],
                    simulationParams = PS.simulationParams,
                    populationParams = PS.populationParams[Y],
                    y = Y,
                    layerBoundaries = PS.layerBoundaries,
                    electrodeParams = PS.electrodeParams,
                    savelist = PS.savelist,
                    #savefolder = PS.hybrid_output_path,
                    savefolder = PS.hybrid_output_path,
                    calculateCSD = PS.calculateCSD,
                    dt_output = PS.dt_output,
                    POPULATIONSEED = SIMULATIONSEED + i,
                    X = PS.X,
                    networkSim = networkSim,
                    k_yXL = PS.k_yXL[Y],
                    synParams = PS.synParams[Y],
                    synDelayLoc = PS.synDelayLoc[Y],
                    synDelayScale = PS.synDelayScale[Y],
                    J_yX = PS.J_yX[Y],
                    tau_yX = PS.tau_yX[Y],
                )

            print(i, Y)

            #run population simulation and collect the data
            pop.run()
            pop.collect_data()

            #object no longer needed
            del pop


    ####### Postprocess the simulation output ######################################

    #reset seed, but output should be deterministic from now on
    np.random.seed(SIMULATIONSEED)

    if properrun:
        #do some postprocessing on the collected data, i.e., superposition
        #of population LFPs, CSDs etc
        postproc = PostProcess(y = PS.Y,      # postsynaptic
                               dt_output = PS.dt_output,
                               savefolder = PS.hybrid_output_path,
                               mapping_Yy = PS.mapping_Yy,
                               savelist = PS.pp_savelist,
                               cells_subfolder = os.path.split(cell_path)[-1],
                               populations_subfolder = os.path.split(population_path)[-1],
                               figures_subfolder = os.path.split(figure_path)[-1],
                               compound_file = '{}{}sum.h5'.format(PS.ps_id, '{}')
                               )

        #run through the procedure
        postproc.run()

        #create tar-archive with output for plotting, ssh-ing etc.
        # postproc.create_tar_archive()


    #if RANK == 0:
    #    os.system('mv {} {}'.format(os.path.join(PS.hybrid_output_path, '{}{}sum.h5'.format(PS.ps_id, 'LFP')), PS.pp_savefolder))

    COMM.Barrier()

    #tic toc
    print('Execution time: %.3f seconds' %  (time() - tic))



    ################################################################################
    # Create set of plots from simulation output
    ################################################################################

    #import some plotter functions
    # from example_plotting import *

    #turn off interactive plotting
    plt.ioff()

    if RANK == 0 and PS.plots is True:
        #create network raster plot
         fig = networkSim.raster_plots(xlim=(500, 1000), markersize=2.)
         fig.savefig(os.path.join(figure_path, 'network.pdf'), dpi=300)


         #plot cell locations
         fig, ax = plt.subplots(1,1, figsize=(5,8))
         plot_population(ax, PS.populationParams, PS.electrodeParams,
                         PS.layerBoundaries,
                         X=['EX', 'IN'], markers=['^', 'o'], colors=['r', 'b'],
                         layers = ['upper', 'lower'],
                         isometricangle=np.pi/12, aspect='equal')
         fig.savefig(os.path.join(figure_path, 'layers.pdf'), dpi=300)


         #plot cell locations
         fig, ax = plt.subplots(1,1, figsize=(5,8))
         plot_population(ax, PS.populationParams, PS.electrodeParams,
                         PS.layerBoundaries,

                         X=['EX', 'IN'], markers=
    ['^', 'o'], colors=['r', 'b'],
                         layers = ['upper', 'lower'],
                         isometricangle=np.pi/12,
     aspect='equal')
         plot_soma_locations(ax, X=['EX', 'IN'],
                             populations_path=population_path,
                             markers=['^', 'o'], colors=['r', 'b'],
                             isometricangle=np.pi/12, )
         fig.savefig(os.path.join(figure_path, 'soma_locations.pdf'), dpi=300)


         #plot morphologies in their respective locations
         fig, ax = plt.subplots(1,1, figsize=(5,8))
         plot_population(ax, PS.populationParams, PS.electrodeParams,
                         PS.layerBoundaries,
                         X=['EX', 'IN'], markers=['^', 'o'], colors=['r', 'b'],
                         layers = ['upper', 'lower'],
                         aspect='equal')
         plot_morphologies(ax, X=['EX', 'IN'], markers=['^', 'o'], colors=['r', 'b'],
                         isometricangle=np.pi/12,
                         populations_path=population_path,
                         cellParams=PS.cellParams)
         fig.savefig(os.path.join(figure_path, 'populations.pdf'), dpi=300)


         #plot morphologies in their respective locations
         fig, ax = plt.subplots(1,1, figsize=(5,8))
         plot_population(ax, PS.populationParams, PS.electrodeParams,
                         PS.layerBoundaries,
                         X=['EX', 'IN'], markers=['^', 'o'], colors=['r', 'b'],
                         layers = ['upper', 'lower'],
                         aspect='equal')
         plot_individual_morphologies(ax, X=['EX', 'IN'], markers=['^', 'o'],
                                      colors=['r', 'b'],
                                      isometricangle=np.pi/12,
                                      cellParams=PS.cellParams,
                                      populationParams=PS.populationParams)
         fig.savefig(os.path.join(figure_path, 'cell_models.pdf'), dpi=300)


         #plot EX morphologies in their respective locations
         fig, ax = plt.subplots(1,1, figsize=(5,8))
         plot_population(ax, PS.populationParams, PS.electrodeParams,
                         PS.layerBoundaries,
                         X=['EX'], markers=['^'], colors=['r'],
                         layers = ['upper', 'lower'],
                         aspect='equal')
         plot_morphologies(ax, X=['EX'], markers=['^'], colors=['r'],
                         isometricangle=np.pi/12,
                         populations_path=population_path,
                         cellParams=PS.cellParams)
         fig.savefig(os.path.join(figure_path, 'EX_population.pdf'), dpi=300)


         #plot IN morphologies in their respective locations
         fig, ax = plt.subplots(1,1, figsize=(5,8))
         plot_population(ax, PS.populationParams, PS.electrodeParams,
                         PS.layerBoundaries,
                         X=['IN'], markers=['o'], colors=['b'],
                         layers = ['upper', 'lower'],
                         isometricangle=np.pi/12, aspect='equal')
         plot_morphologies(ax, X=['IN'], markers=['o'], colors=['b'],
                         isometricangle=np.pi/12,
                         populations_path=population_path,
                         cellParams=PS.cellParams)
         fig.savefig(os.path.join(figure_path, 'IN_population.pdf'), dpi=300)
