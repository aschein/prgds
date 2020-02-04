import sys
from path import Path
sys.path.append(Path(__file__).parent.parent)
from apf.base.apf import APF

import numpy as np
import numpy.random as rn
import scipy.stats as st


def main():
    """Main method."""
    data_shp = (5,4,4)
    core_shp = (3,)
    eps = 0.75
    n_threads = 3
    binary = 0
    mtx_is_dirichlet = [1]
    # mtx_is_dirichlet = list(np.arange(len(data_shp)))
    # mtx_is_dirichlet = [0, 2]
    mask_p = 0.25

    seed = rn.randint(10000)
    rn.seed(seed)
    print('seed: %d' % seed)

    model = APF(data_shp=data_shp,
                core_shp=core_shp,
                eps=eps,
                binary=binary,
                mtx_is_dirichlet=mtx_is_dirichlet,
                seed=seed,
                n_threads=n_threads)

    Y = np.zeros(data_shp, dtype=np.int32)
    mask = rn.binomial(1, mask_p, size=data_shp)
    data = np.ma.array(Y, mask=mask)
    model._initialize_data(data)
    if mask.sum() > 0:
        assert np.allclose(mask, model.get_dense_mask())
    else:
        assert model.get_dense_mask() is None

    def get_schedule_func(burnin=0, update_every=1):
        return lambda x: x >= burnin and x % update_every == 0

    schedule = {'core_Q': get_schedule_func(0, 1),
                'b_M': get_schedule_func(0, 1),
                'mtx_MKD': get_schedule_func(0, 1),
                'Y_MKD': get_schedule_func(0, 1),
                'Y_Q': get_schedule_func(0, 1)}

    def mtx_MKD_min_entropy(mtx_MKD):
        return np.min([np.min(st.entropy(mtx_MKD[m, :Km, :Dm])) \
                for m, (Km, Dm) in enumerate(zip(core_shp, data_shp))])

    def mtx_MKD_max_entropy(mtx_MKD):
        return np.max([np.max(st.entropy(mtx_MKD[m, :Km, :Dm])) \
                for m, (Km, Dm) in enumerate(zip(core_shp, data_shp))])

    def mtx_MKD_mean_entropy(mtx_MKD):
        return np.mean([np.mean(st.entropy(mtx_MKD[m, :Km, :Dm])) \
                for m, (Km, Dm) in enumerate(zip(core_shp, data_shp))])

    def mtx_MKD_var_entropy(mtx_MKD):
        return np.mean([np.var(st.entropy(mtx_MKD[m, :Km, :Dm])) \
                for m, (Km, Dm) in enumerate(zip(core_shp, data_shp))])

    entropy_funcs = {'Entropy min': mtx_MKD_min_entropy,
                     'Entropy max': mtx_MKD_max_entropy,
                     'Entropy mean': mtx_MKD_mean_entropy,
                     'Entropy var': mtx_MKD_var_entropy}

    var_funcs = {}
    if len(mtx_is_dirichlet) > 0:
        var_funcs['mtx_MKD'] = entropy_funcs

    model.schein(5000, var_funcs=var_funcs, schedule=schedule)


if __name__ == '__main__':
    main()
