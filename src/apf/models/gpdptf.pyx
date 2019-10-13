# cython: boundscheck = False
# cython: initializedcheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: language_level = 3

import sys
import numpy as np
import numpy.random as rn
cimport numpy as np
from libc.math cimport log1p

from cython.parallel import parallel, prange

from apf.base.apf cimport APF
from apf.base.sample cimport _sample_gamma, _sample_dirichlet, _sample_crt

cdef extern from "gsl/gsl_rng.h" nogil:
    ctypedef struct gsl_rng:
        pass

cdef class GPDPTF(APF):

    cdef:
        int time_mode, n_timesteps, stationary
        double gam, beta, tau
        double[::1] delta_T, b_T
        double[:,::1] Theta_TK, L_zeta_TK
        int[:,::1] L_TK

    def __init__(self, tuple data_shp, tuple core_shp, int time_mode=0,
                 int stationary=0, double tau=1.0, double eps=0.1, int binary=0, 
                 list mtx_is_dirichlet=[], object seed=None, object n_threads=None):

        super(GPDPTF, self).__init__(data_shp=data_shp,
                                     core_shp=core_shp,
                                     eps=eps,
                                     binary=binary,
                                     mtx_is_dirichlet=mtx_is_dirichlet,
                                     seed=seed,
                                     n_threads=n_threads)

        # Marginalization of missing data is not available to gamma variables
        # that have a hierarchical gamma prior on their shape.
        # The augment-and-conquer strategy for inferring their shape parameters
        # involves marginalizing them which is only possible if all data are 
        # conditioned on (either observed or imputed). This model employs the
        # augment-and-conquer strategy for both Theta_TK and core_Q.
        self.impute_Y_M[self.time_mode] = 1
        self.impute_Y_Q = 1

        # Params
        self.time_mode = self.param_list['time_mode'] = time_mode
        self.stationary = self.param_list['stationary'] = stationary
        self.tau = self.param_list['tau'] = tau

        self.n_timesteps = T = data_shp[time_mode]
        K = self.core_dims_M[time_mode]

        # State variables
        self.gam = 1.
        self.beta = 1.
        self.delta_T = np.ones(T)
        self.Theta_TK = np.ones((T, K))
        self.b_T = np.ones(T)
 
        # Auxiliary variables and data structures
        self.L_TK = np.zeros((T, K), dtype=np.int32)
        self.L_zeta_TK = np.zeros((T, K))

    cdef list _get_variables(self):
        """
        Return variable names, values, and sampling methods for testing.

        MUST BE IN TOPOLOGICAL ORDER!
        """
        variables = [
                     ('gam', self.gam, self._update_gam),
                     ('beta', self.beta, self._update_beta),
                     ('core_Q', self.core_Q, self._update_core_Q),
                     ('b_T', self.b_T, self._update_b_T),
                     ('Theta_TK', self.Theta_TK, self._update_Theta_TK),
                     ('delta_T', self.delta_T, self._update_delta_T),
                     ('b_M', self.b_M, self._update_b_M),
                     ('mtx_MKD', self.mtx_MKD, self._update_mtx_MKD),
                     ('Y_MKD', self.Y_MKD, self._update_Y_PQ),
                     ('Y_Q', self.Y_Q, self._dummy_update)]
        return variables

    def set_state(self, state):
        for key, var, _ in self._get_variables():
            if key in state.keys():
                state_var = state[key]
                if key == 'tau':
                    self.tau = state_var
                elif key == 'gam':
                    self.gam = state_var
                elif key == 'beta':
                    self.beta = state_var
                else:
                    assert var.shape == state_var.shape
                    for idx in np.ndindex(var.shape):
                        var[idx] = state_var[idx]
        self._compute_mtx_KT()
        self._update_cache()

    cdef double[:,::1] _forecast_Theta_TK(self, int n_timesteps) nogil:
        pass

    cdef double[::1] _forecast_delta_T(self, int n_timesteps) nogil:
        pass

    def forecast(self, n_timesteps=1, subs=()):
        if self.stationary:
            delta_T = np.repeat(self.delta_T[0], repeats=n_timesteps)
        else:
            delta_T = self._forecast_delta_T(n_timesteps)
        Theta_TK = self._forecast_Theta_TK(n_timesteps)
        return self.reconstruct(subs=subs, state={'delta_T': delta_T, 'Theta_TK': Theta_TK})

    cdef void _update_b_M(self, int update_mode):
        cdef:
             np.npy_intp m

        for m in range(self.n_modes):
            if (m not in self.mtx_is_dirichlet) and (m != self.time_mode):
                self._update_b_m(m, update_mode)

    cdef void _update_mtx_MKD(self, int update_mode):
        cdef: 
            np.npy_intp m

        for m in range(self.n_modes):
            if m != self.time_mode:
                self._update_mtx_m_KD(m, update_mode)

    cdef void _compute_mtx_KT(self) nogil:
        cdef:
            np.npy_intp T, K, k, t
            double[::1] mtx_K
            double[:,::1] mtx_KT
        
        T, K = self.Theta_TK.shape[:2]
        mtx_K = self.mtx_MK[self.time_mode]; mtx_K[:] = 0
        mtx_KT = self.mtx_MKD[self.time_mode]; mtx_KT[:] = 0
        for k in prange(K, schedule='static', nogil=True):
            for t in range(T):
                mtx_KT[k, t] = self.delta_T[t] * self.Theta_TK[t, k]
                mtx_K[k] += mtx_KT[k, t]

    cdef void _update_delta_T(self, int update_mode):
        cdef:
            np.npy_intp K, T, t
            double prior_shp, prior_rte, shp, rte, shp_t, rte_t
            double[:,::1] zeta_TK
            double[::1] zeta_T
            long[::1] Y_T
            gsl_rng * rng

        if update_mode == self._INITIALIZE_MODE:
            self.delta_T[:] = 1
        else:
            if update_mode == self._INFER_MODE:
                # Note that ONLY imputation of Y_T is currently supported. 
                T, K = self.Theta_TK.shape[0], self.Theta_TK.shape[1]
                Y_T = np.sum(self.Y_MKD[self.time_mode, :K, :T], axis=0, dtype=np.int)
                zeta_TK = self._compute_zeta_m_DK(self.time_mode)
                zeta_T = np.sum(np.multiply(self.Theta_TK, zeta_TK), axis=1)
            
            prior_shp = prior_rte = self.eps
            if self.stationary:
                shp, rte = prior_shp, prior_rte
                if update_mode == self._INFER_MODE:
                    shp += np.sum(Y_T) 
                    rte += np.sum(zeta_T)
                self.delta_T[:] = _sample_gamma(self.rng, shp, 1./rte)
            
            else:
                for t in prange(self.n_timesteps, schedule='static', nogil=True):
                    rng = self.rngs[self._get_thread()]
                    shp_t, rte_t = prior_shp, prior_rte
                    if update_mode == self._INFER_MODE:
                        shp_t = shp_t + Y_T[t]
                        rte_t = rte_t + zeta_T[t]
                    self.delta_T[t] = _sample_gamma(rng, shp_t, 1./rte_t)

        self._compute_mtx_KT()

    cdef void _update_Theta_TK(self, int update_mode):
        """
        Theta_TK: Begins at t=1 and ends at t=T  (self.n_timesteps, K)
        b_T:      Begins at t=1 and ends at t=T  (self.n_timesteps,)
        delta_T:  Begins at t=1 and ends at t=T  (self.n_timesteps, K)
        Y_KT:     Begins at t=1 and ends at t=T  (K, self.n_timesteps)
        L_TK:     Begins at t=2 and ends at t=T+1  (self.n_timesteps, K)
        zeta_TK:  Begins at t=2 and ends at t=T+1  (self.n_timesteps, K)
        """

        cdef:
            np.npy_intp T, K, k, t
            int m_tk
            double eps, tau, m_zeta, shp_tk, rte_tk
            double[:,::1] Y_zeta_TK
            long[:,::1] Y_KT
            gsl_rng * rng

        if update_mode == self._INITIALIZE_MODE:
            self.Theta_TK[:] = 1.

        else:
            T, K = self.Theta_TK.shape[0], self.Theta_TK.shape[1]
            
            if update_mode == self._INFER_MODE:
                Y_KT = self.Y_MKD[self.time_mode, :K, :T]
                Y_zeta_TK = np.einsum('t,tk->tk', self.delta_T, 
                                      self._compute_zeta_m_DK(self.time_mode))

            eps, tau = self.eps, self.tau

            for k in prange(K, schedule='static', nogil=True):
                rng = self.rngs[self._get_thread()]
                
                if update_mode == self._INFER_MODE:
                    # Perform backward pass
                    for t in range(T-2, -1, -1):
                        m_zeta = self.L_zeta_TK[t+1, k] + Y_zeta_TK[t+1, k]
                        self.L_zeta_TK[t, k] = tau * log1p(m_zeta / (tau * self.b_T[t+1]))

                        m_tk = Y_KT[k, t+1] + self.L_TK[t+1, k]
                        self.L_TK[t, k] = _sample_crt(rng, m_tk, tau * self.Theta_TK[t, k])

                for t in range(T):
                    shp_tk = tau * eps if t == 0 else tau * self.Theta_TK[t-1, k]
                    rte_tk = tau * self.b_T[t]

                    if update_mode == self._INFER_MODE:
                        shp_tk = shp_tk + Y_KT[k, t] + self.L_TK[t, k]
                        rte_tk = rte_tk + Y_zeta_TK[t, k] + self.L_zeta_TK[t, k]

                    self.Theta_TK[t, k] = _sample_gamma(rng, shp_tk, 1./rte_tk)
        self._compute_mtx_KT()

    cdef void _update_b_T(self, int update_mode):
        cdef:
            np.npy_intp t, K
            double eps, tau, shp, rte, shp_t, rte_t, theta_
            double[::1] Theta_T
            gsl_rng * rng

        if update_mode == self._INITIALIZE_MODE:
            self.b_T[:] = 1.

        else:
            eps, tau = self.eps, self.tau
            K = self.Theta_TK.shape[1]
            
            if update_mode == self._INFER_MODE:
                Theta_T = np.sum(self.Theta_TK, axis=1)
                theta_ = np.sum(Theta_T)

            if self.stationary:
                shp = rte = eps
                if update_mode == self._INFER_MODE:
                    shp = eps + (tau * eps * K) + tau * (theta_ - Theta_T[self.n_timesteps-1])
                    rte = eps + tau * theta_
                self.b_T[:] = _sample_gamma(self.rng, shp, 1. / rte)
            
            else:
                for t in prange(self.n_timesteps, schedule='static', nogil=True):
                    rng = self.rngs[self._get_thread()]
                    shp_t = rte_t = eps
                    if update_mode == self._INFER_MODE:
                        shp_t = eps + tau * eps * K if t == 0 else eps + tau * Theta_T[t-1]
                        rte_t = eps + tau * Theta_T[t]
                    self.b_T[t] = _sample_gamma(rng, shp_t, 1. / rte_t)

    @property
    def core_Q_prior(self):
        """
        Returns the prior shape and rate parameter for the core elements.

        This class inherits _update_core_Q from apf.pyx and simply places a
        different hyper-prior over the core elements.
        """
        return self.gam / self.n_classes, self.beta

    cdef void _update_beta(self, int update_mode):
        cdef: 
            double shp, rte

        if update_mode == self._INITIALIZE_MODE:
            self.beta = 1.
        else:
            shp = rte = 10.
            if update_mode == self._INFER_MODE:
                shp += self.gam
                rte += np.sum(self.core_Q)
            self.beta = _sample_gamma(self.rng, shp, 1./rte)

    cdef void _update_gam(self, int update_mode):
        cdef: 
            np.npy_intp q, k, m, Km, Dm
            double shp, rte, gam_q
            double[::1] zeta_Q
            gsl_rng * rng

        shp = rte = self.eps 
        if update_mode == self._INFER_MODE:
            gam_q = self.gam / self.n_classes
            zeta_Q = self._compute_zeta_Q()

            for q in prange(self.n_classes, schedule='static', nogil=True):
                rng = self.rngs[self._get_thread()]
                rte += log1p(zeta_Q[q] / self.beta) / self.n_classes
                shp += _sample_crt(rng, self.Y_Q[q], gam_q)
        self.gam = _sample_gamma(self.rng, shp, 1./rte)
