n_jobs = 7
n_sims_per_state = 10
n_states = 5
n_total_sims = n_sims_per_state*n_states

inds = []
for i in range(n_jobs):

    rank = i

    import numpy as np
    sim_indices = np.arange(rank, n_total_sims, step=n_jobs)
    for ind in sim_indices:
        inds.append(ind)
    #print(sim_indices)
    print(len(sim_indices))
print(np.sort(inds))


    