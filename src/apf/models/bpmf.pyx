# cython: boundscheck = False
# cython: initializedcheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: language_level = 3

import sys
import numpy as np
import numpy.random as rn
cimport numpy as np

from cython.parallel import parallel, prange

from apf.base.apf cimport APF
from apf.base.mcmc_model_parallel import exit_if

cdef class BPMF(APF):
    """Bayesian Poisson Matrix Factorization

    - Gibbs sampling
    - Allows for missing data
    - Allows for binary data

    Count likelihood:
        y_ij ~ Pois(sum_k lambda_k theta_ik phi_jk)

    Binary likelihood:
        m_ij ~ Pois(sum_k lambda_k theta_ik phi_jk)
        y_ij = 1 if m_ij > 0 else 0

    Priors: 
        lambda_k ~ Gamma(eps, eps)
        theta_ik ~ Gamma(eps, eps * b)
        phi_k ~ Dirichlet(eps...eps)
        b ~ Gamma(eps, eps)
    """

    cdef:
        int n_samps, n_feats, n_comps
        double b
        double[::1] Lambda_K
        double[:,::1] Theta_IK, Phi_KJ
        long[:,::1] Y_PK

    def __init__(self, int n_samps, int n_feats, int n_comps, int binary, double eps=0.1,
                 object seed=None, object n_threads=None):

        super().__init__(data_shp=(n_samps, n_feats),
                         core_shp=(n_comps,),
                         eps=eps,
                         binary=binary,
                         mtx_is_dirichlet=[1],
                         seed=seed,
                         n_threads=n_threads)
        
        self.b = 1.
        self.Lambda_K = np.ones(n_comps)
        self.Theta_IK = np.zeros((n_samps, n_comps))
        self.Phi_KJ = np.zeros((n_comps, n_feats))
        self.Y_PK = np.zeros((0, n_comps), dtype=np.int)

    cdef list _get_variables(self):
        """
        Return variable names, values, and sampling methods for testing.

        MUST BE IN TOPOLOGICAL ORDER!
        """
        variables = [('b', self.b, self._update_b),
                     ('Lambda_K', self.Lambda_K, self._update_Lambda_K),
                     ('Theta_IK', self.Theta_IK, self._update_Theta_IK),
                     ('Phi_KJ', self.Phi_KJ, self._update_Phi_KJ),
                     ('Y_PK', self.Y_PK, self._update_Y_PK)]

        return variables
    
    cdef void _update_Y_PK(self, int update_mode):
        self._update_Y_PQ(update_mode=update_mode)
        
    cdef void _update_Theta_IK(self, int update_mode):
        self._update_mtx_m_KD(0, update_mode=update_mode)
        self.Theta_IK = np.array(self.mtx_MKD[0])[:, :self.n_samps].T

    cdef void _update_Phi_KJ(self, int update_mode):
        self._update_mtx_m_KD(1, update_mode=update_mode)
        self.Phi_KJ = np.array(self.mtx_MKD[1])[:, :self.n_feats]

    cdef void _update_Lambda_K(self, int update_mode):
        self._update_core_Q(update_mode=update_mode)
        self.Lambda_K[:] = self.core_Q

    cdef void _update_b(self, int update_mode):
        self._update_b_m(0, update_mode=update_mode)
        self.b = self.b_M[0]

    def set_state(self, state):
        for key, val, _ in self._get_variables():
            if key in state.keys():
                state_val = state[key]
                if key == 'b':
                    self.b = state_val
                    self.b_M[:] = 1
                    self.b_M[0] = self.b
                else:
                    assert val.shape == state_val.shape
                    for idx in np.ndindex(val.shape):
                        val[idx] = state_val[idx]
        self._update_cache()
    
    cdef void _update_cache(self):
        cdef:
            np.npy_intp k, p, i, j
        
        self.mtx_MKD[:] = 0
        for k in range(self.n_comps):
            self.core_Q[k] = self.Lambda_K[k]
            for p in range(self.Y_PQ.shape[0]):
                self.Y_PQ[p, k] = self.Y_PK[p, k]
            for i in range(self.n_samps):
                self.mtx_MKD[0, k, i] = self.Theta_IK[i, k]
            for j in range(self.n_feats):
                self.mtx_MKD[1, k, j] = self.Phi_KJ[k, j]