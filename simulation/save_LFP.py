import numpy as np

def Save_LFP(LFP, network_parameters, sim_index, class_label):
    """
    Saves LFP signal to file
    """
    dir = network_parameters.LFP_path
    np.save(dir+"/LFP-{}-{}".format(sim_index, class_label), LFP)
    return None
