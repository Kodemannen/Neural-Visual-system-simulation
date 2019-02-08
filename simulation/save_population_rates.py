import numpy as np

def Save_population_rates(poprates, network_parameters, sim_index, class_label):
    """
    Saves Population_rates signal to file
    """
    dir = network_parameters.population_rates_path
    np.save(dir+"/poprates-{}-{}.npy".format(sim_index, class_label), poprates)
    return None


