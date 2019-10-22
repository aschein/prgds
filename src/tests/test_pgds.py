import sys
from path import Path
sys.path.append(Path(__file__).parent.parent)
from apf.models.pgds import PGDS

import numpy as np
import numpy.random as rn
import scipy.stats as st


def main():
    """Main method."""
    data_shp = (4, 3, 3)
    core_shp = (2, 2, 2)
    binary = 0
    eps = 3.
    time_mode = 0
    stationary = 0
    tau = 0.9
    gam = 15
    n_threads = 1
    mask_p = 0.25

    seed = rn.randint(10000)
    rn.seed(seed)
    print('seed: %d' % seed)

    model = PGDS(data_shp=data_shp,
                 core_shp=core_shp,
                 time_mode=time_mode,
                 eps=eps,
                 binary=binary,
                 stationary=stationary,
                 tau=tau,
                 gam=gam,
                 seed=seed,
                 n_threads=n_threads)

    Y = np.zeros(data_shp, dtype=np.int32)
    mask = rn.binomial(1, mask_p, size=data_shp)
    data = np.ma.array(Y, mask=mask)
    model._initialize_data(data)

    def get_schedule_func(burnin=0, update_every=1):
        return lambda x: x >= burnin and x % update_every == 0

    schedule = {'beta': get_schedule_func(0, 1),
                'nu_K': get_schedule_func(0, 1),
                'xi_K': get_schedule_func(0, 1),
                'Pi_KK': get_schedule_func(0, 1),
                'b_T': get_schedule_func(0, 1),
                'Theta_TK': get_schedule_func(0, 1),
                'delta_T': get_schedule_func(0, 1),
                'mtx_MKD': get_schedule_func(0, 1),
                'core_Q': get_schedule_func(0, 1),
                'Y_MKD': get_schedule_func(0, 1),
                'Y_Q': get_schedule_func(0, 1),
                'L_TKK': get_schedule_func(0, 1)}

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

    def get_core_KQ_(core_Q):
        core = np.reshape(core_Q, core_shp)
        core = np.rollaxis(core, axis=time_mode)
        core_KQ_ = core.reshape((core.shape[0], -1))
        assert np.allclose(core_KQ_.sum(axis=1), 1)
        return core_KQ_

    var_funcs = {}
    var_funcs['mtx_MKD'] = {'Entropy min': mtx_MKD_min_entropy,
                           'Entropy max': mtx_MKD_max_entropy,
                           'Entropy mean': mtx_MKD_mean_entropy,
                           'Entropy var': mtx_MKD_var_entropy}

    var_funcs['core_Q'] = {'Entropy min': lambda x: np.min(st.entropy(get_core_KQ_(x).T)),
                           'Entropy max': lambda x: np.max(st.entropy(get_core_KQ_(x).T)),
                           'Entropy mean': lambda x: np.mean(st.entropy(get_core_KQ_(x).T)),
                           'Entropy var': lambda x: np.var(st.entropy(get_core_KQ_(x).T))}

    var_funcs['Pi_KK'] = {'Entropy min': lambda x: np.min(st.entropy(x.T)),
                          'Entropy max': lambda x: np.max(st.entropy(x.T)),
                          'Entropy mean': lambda x: np.mean(st.entropy(x.T)),
                          'Entropy var': lambda x: np.var(st.entropy(x.T))}

    var_funcs['xi_K'] = {'xi': lambda x: x[0]}                      
    if stationary: 
        var_funcs['delta_T'] = {'delta': lambda x: x[0]}
        var_funcs['b_T'] = {'b': lambda x: x[0]}

    model.alt_geweke(20000, var_funcs=var_funcs, schedule=schedule)


if __name__ == '__main__':
    main()
