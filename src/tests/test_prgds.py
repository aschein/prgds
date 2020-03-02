import sys
from path import Path
sys.path.append(Path(__file__).parent.parent)
from apf.models.prgds import PRGDS

import numpy as np
import numpy.random as rn
import scipy.stats as st

def main():
    """Main method."""
    data_shp = (4, 5, 3)
    core_shp = (3,)
    binary = 0
    eps = 1.0
    time_mode = 1
    stationary = 0
    eps_theta = 0.0
    eps_nu = 0.0
    n_threads = 1
    mask_p = 0.15
    mtx_is_dirichlet = [0]
    block_sample_Theta_and_H = 1
    block_sample_nu_and_g = 1

    seed = rn.randint(10000)
    # seed = 1586
    rn.seed(seed)
    print('seed: %d' % seed)

    model = PRGDS(data_shp=data_shp,
                  core_shp=core_shp,
                  time_mode=time_mode,
                  eps=eps,
                  binary=binary,
                  stationary=stationary,
                  eps_theta=eps_theta,
                  eps_nu=eps_nu,
                  mtx_is_dirichlet=mtx_is_dirichlet,
                  block_sample_Theta_and_H=block_sample_Theta_and_H,
                  block_sample_nu_and_g=block_sample_nu_and_g,
                  seed=seed,
                  n_threads=n_threads)

    Y = np.zeros(data_shp, dtype=np.int32)
    mask = None
    if mask_p > 0:
        mask = rn.binomial(1, mask_p, size=data_shp)
        mask = np.rollaxis(mask, time_mode, 0)
        mask[1] = 1  # entirely missing time column
        mask = np.rollaxis(mask, 0, time_mode+1)
        percent_missing = np.ceil(100 * mask.sum() / float(mask.size))
        print('%d%% missing' % percent_missing)

    data = np.ma.array(Y, mask=mask)
    model._initialize_data(data)

    def get_schedule_func(burnin=0, update_every=1):
        return lambda x: x >= burnin and x % update_every == 0

    schedule = {'gam': get_schedule_func(0, 1),
                'g_K': get_schedule_func(0, 1),
                'beta': get_schedule_func(0, 1),
                'nu_K': get_schedule_func(0, 1),
                'Pi_KK': get_schedule_func(0, 1),
                'tau': get_schedule_func(0, 1),
                'b_T': get_schedule_func(0, 1),  # failing *slightly* in non-stationary mode
                'Theta_TK': get_schedule_func(0, 1),
                'H_TKK': get_schedule_func(0, 1),
                'delta_T': get_schedule_func(0, 1),  # failing *slightly* in non-stationary mode
                'b_M': get_schedule_func(0, 1),
                'mtx_MKD': get_schedule_func(0, 1),
                'core_Q': get_schedule_func(0, 1),
                'Y_MKD': get_schedule_func(0, 1),
                'Y_Q': get_schedule_func(0, 1)}

    def get_matrices(mtx_MKD):
        core_dims_M = core_shp
        if len(core_shp) < len(data_shp):
            core_dims_M = core_shp * len(data_shp)
        matrices = []
        for m, (Km, Dm) in enumerate(zip(core_dims_M, data_shp)):
            if m != time_mode:
                matrices.append(mtx_MKD[m, :Km, :Dm])
        return matrices

    def mtx_MKD_min_entropy(mtx_MKD):
        return np.min([np.min(st.entropy(X.T)) for X in get_matrices(mtx_MKD)])

    def mtx_MKD_max_entropy(mtx_MKD):
        return np.max([np.max(st.entropy(X.T)) for X in get_matrices(mtx_MKD)])

    def mtx_MKD_mean_entropy(mtx_MKD):
        return np.mean([np.mean(st.entropy(X.T)) for X in get_matrices(mtx_MKD)])

    def mtx_MKD_var_entropy(mtx_MKD):
        return np.mean([np.var(st.entropy(X.T)) for X in get_matrices(mtx_MKD)])


    var_funcs = {}
    if len(mtx_is_dirichlet) > 0:
        var_funcs['mtx_MKD'] = {'Entropy min': mtx_MKD_min_entropy,
                                'Entropy max': mtx_MKD_max_entropy,
                                'Entropy mean': mtx_MKD_mean_entropy,
                                'Entropy var': mtx_MKD_var_entropy}

    var_funcs['Pi_KK'] = {'Entropy min': lambda x: np.min(st.entropy(x)),
                          'Entropy max': lambda x: np.max(st.entropy(x)),
                          'Entropy mean': lambda x: np.mean(st.entropy(x)),
                          'Entropy var': lambda x: np.var(st.entropy(x))}

    if stationary: 
        var_funcs['delta_T'] = {'delta': lambda x: x[0]}
        var_funcs['b_T'] = {'b': lambda x: x[0]}

    model.alt_geweke(20000, var_funcs=var_funcs, schedule=schedule)


if __name__ == '__main__':
    main()
