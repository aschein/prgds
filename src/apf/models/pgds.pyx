# cython: boundscheck = False
# cython: initializedcheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: language_level = 3

import sys
import numpy as np
import numpy.random as rn
import tensorly as tl
cimport numpy as np
from libc.math cimport log1p

from cython.parallel import parallel, prange

from apf.base.apf cimport APF
from apf.base.sample cimport _sample_gamma, _sample_dirichlet, _sample_crt, _sample_lnbeta
from apf.base.cyutils cimport _sum_double_vec, _dot_vec
from apf.base.mcmc_model_parallel import exit_if

cdef extern from "gsl/gsl_rng.h" nogil:
    ctypedef struct gsl_rng:
        pass

cdef extern from "gsl/gsl_randist.h" nogil:
    void gsl_ran_multinomial(gsl_rng * r,
                             size_t K,
                             unsigned int N,
                             const double p[],
                             unsigned int n[])

cdef class PGDS(APF):

    cdef:
        int time_mode, n_timesteps, stationary
        double gam, beta, tau
        double[::1] nu_K, xi_K, delta_T, b_T, L_zeta_T, lnq_K
        double[:,::1] Theta_TK, shp_KK, Pi_KK
        int[:,:,::1] L_TKK
        int[:,::1] H_KK, L_TK, L_KK

    def __init__(self, tuple data_shp, tuple core_shp, double gam=10., 
                 int time_mode=0, int stationary=0, double tau=1.0, 
                 double eps=0.1, int binary=0, 
                 object seed=None, object n_threads=None):

        # All factor matrices must be Dirichlet, except Theta_TK
        mtx_is_dirichlet = list(range(len(data_shp)))
        mtx_is_dirichlet.remove(time_mode)

        super(PGDS, self).__init__(data_shp=data_shp,
                                   core_shp=core_shp,
                                   eps=eps,
                                   binary=binary,
                                   mtx_is_dirichlet=mtx_is_dirichlet,
                                   seed=seed,
                                   n_threads=n_threads)
        
        # Missing data cannot be marginalized out in this model.
        self.impute_Y_Q = 1
        self.impute_Y_M[:] = 1

        # Make sure core elements are initialized to zero.
        # In Tucker decomposition, they will be updated, but in 
        # CP-decomposition they are 1, by definition.
        self.core_Q[:] = 1.

        # Params
        self.time_mode = self.param_list['time_mode'] = time_mode
        self.stationary = self.param_list['stationary'] = stationary
        self.tau = self.param_list['tau'] = tau
        self.gam = self.param_list['gam'] = gam

        self.n_timesteps = T = data_shp[time_mode]
        K = self.core_dims_M[time_mode]

        # State variables
        self.beta = 1.
        self.nu_K = np.ones(K)
        self.xi_K = np.ones(K)
        self.Pi_KK = np.ones((K, K))
        self.b_T = np.ones(T)
        self.Theta_TK = np.ones((T, K))
        self.delta_T = np.ones(T)
 
        # Auxiliary variables and data structures
        self.L_TKK = np.zeros((T, K, K), dtype=np.int32)
        self.L_KK = np.zeros((K, K), dtype=np.int32)
        self.L_TK = np.zeros((T, K), dtype=np.int32)
        self.L_zeta_T = np.zeros(T)
        self.shp_KK = np.zeros((K, K))
        self.H_KK = np.zeros((K, K), dtype=np.int32)
        self.lnq_K = np.zeros(K)

    cdef list _get_variables(self):
        """
        Return variable names, values, and sampling methods for testing.

        MUST BE IN TOPOLOGICAL ORDER!
        """
        variables = [('beta', self.beta, self._update_beta),
                     ('nu_K', self.nu_K, self._update_nu_K_and_xi_K),
                     ('xi_K', self.xi_K[0], self._dummy_update),
                     ('Pi_KK', self.Pi_KK, self._update_Pi_KK),
                     ('b_T', self.b_T, self._update_b_T),
                     ('Theta_TK', self.Theta_TK, self._update_Theta_TK),
                     ('delta_T', self.delta_T, self._update_delta_T),
                     ('mtx_MKD', self.mtx_MKD, self._update_mtx_MKD),
                     ('core_Q', self.core_Q, self._update_core_Q),
                     ('Y_MKD', self.Y_MKD, self._update_Y_PQ),
                     ('Y_Q', self.Y_Q, self._dummy_update),
                     ('L_TKK', self.L_TKK, self._update_L_TKK)]
        return variables

    def set_state(self, state):
        for key, val, _ in self._get_variables():
            if key in state.keys():
                state_val = state[key]
                if key == 'beta':
                    self.beta = state_val
                elif key == 'xi_K':
                    self.xi_K[:] = state_val
                else:
                    assert val.shape == state_val.shape
                    for idx in np.ndindex(val.shape):
                        val[idx] = state_val[idx]
        self._compute_mtx_KT()
        self._update_cache()

    cdef void _initialize_state(self, dict state={}):
        """
        Initialize internal state.
        """
        for key, val, update_func in self._get_variables():
            if key in state.keys():
                state_val = state[key]
                if key == 'beta':
                    self.beta = state_val
                elif key == 'xi_K':
                    self.xi_K[:] = state_val
                else:
                    if np.isscalar(state_val):
                        assert NotImplementedError
                    assert val.shape == state_val.shape
                    for idx in np.ndindex(val.shape):
                        val[idx] = state_val[idx]
            else:
                output = update_func(self, update_mode=self._INITIALIZE_MODE)
                exit_if(output, 'updating %s' % key)
        self._compute_mtx_KT()
        self._update_cache()

    def generate_state(self):
        self._generate_state()
        self._generate_data()
        return dict(self.get_state())

    cdef void _generate_state(self):
        """
        Generate internal state.
        """
        for key, _, update_func in self._get_variables():
            if key not in ['Y_MKD', 'Y_Q', 'L_TKK']:
                update_func(self, update_mode=self._GENERATE_MODE)

    cdef void _generate_data(self):
        self._update_Y_PQ(update_mode=self._GENERATE_MODE)
        self._update_L_TKK(update_mode=self._GENERATE_MODE)

    cdef double[:,::1] _forecast_Theta_TK(self, int n_timesteps):
        
        cdef:
            np.npy_intp K, t, k
            double rte_t, shp_tk
            double[::1] b_forecast_T
            double[:,::1] Theta_forecast_TK 

        assert self.stationary  # if not stationary b_forecast_T is time-dependent
        b_forecast_T = np.repeat(self.b_T[0], repeats=n_timesteps)

        K = self.Theta_TK.shape[1]
        Theta_forecast_TK = np.zeros((n_timesteps, K))

        for t in range(n_timesteps):
            rte_t = self.tau * b_forecast_T[t]
            for k in prange(K, schedule='static', nogil=True):
                rng = self.rngs[self._get_thread()]
                if t == 0:
                    shp_tk = self.tau * _dot_vec(self.Theta_TK[self.n_timesteps-1], self.Pi_KK[:, k])
                else:
                    shp_tk = self.tau * _dot_vec(Theta_forecast_TK[t-1], self.Pi_KK[:, k])
                # Theta_forecast_TK[t, k] = _sample_gamma(rng, shp_tk, 1./rte_t)
                Theta_forecast_TK[t, k] = shp_tk / rte_t
        return Theta_forecast_TK

    def forecast_Theta_TK(self, n_timesteps=1):
        return np.array(self._forecast_Theta_TK(n_timesteps))

    def forecast(self, n_timesteps=1, n_samples=1, subs=()):
        if not self.stationary:
            raise NotImplementedError('Forecasting in non-stationary model not available.')
        delta_T = np.repeat(self.delta_T[0], repeats=n_timesteps)
        if n_samples == 1:
            mtx = self._forecast_Theta_TK(n_timesteps) * delta_T[:, np.newaxis]
            return self.decode(mtx=mtx, mode=self.time_mode, subs=subs)
        else:
            return np.array([self.forecast(n_timesteps=n_timesteps, n_samples=1, subs=subs) for _ in range(n_samples)])

    cdef void _update_b_M(self, int update_mode):
        """There are no gamma-distributed factors in this model.
    
        This overwrites the method from apf.pyx that updates the 
        rate hyperprior for any gamma-distributed factors.
        """
        pass

    cdef void _update_mtx_MKD(self, int update_mode):
        cdef: 
            np.npy_intp m

        for m in range(self.n_modes):
            if m != self.time_mode:
                self._update_mtx_m_KD(m, update_mode)

    cdef void _compute_mtx_KT(self) nogil:
        cdef:
            int T, K, k, t
            double[::1] mtx_K
            double[:,::1] mtx_KT
        
        T, K = self.Theta_TK.shape[:2]
        mtx_K = self.mtx_MK[self.time_mode]; mtx_K[:] = 0
        mtx_KT = self.mtx_MKD[self.time_mode]; mtx_KT[:] = 0
        for k in prange(K, schedule='static', nogil=True):
            for t in range(T):
                mtx_KT[k, t] = self.delta_T[t] * self.Theta_TK[t, k]
                mtx_K[k] += mtx_KT[k, t]

    @property
    def core_Q_prior(self):
        """
        Returns the prior shape parameter for the core elements.

        This is only called when the model is a Tucker decomposition.
        
        The core elements indexed by a given dimension of the time mode
        must sum to one. In the CP-decmoposition case, there is only one 
        core element per class; in this case, the core is stored as only 
        the super-diagonal of the core tensor, and all of those entries 
        are equal to 1.

        This class inherits from apf.pyx which implements a gamma
        distributed core tensor. Overwriting this property is mostly
        for tidyness: the old property returns a shape and rate for
        the gamma prior while this model imposes a Dirichlet prior.
        """
        shp = np.ones(self.core_shp) * self.eps
        return tl.unfold(shp, self.time_mode)

    cdef void _update_core_Q(self, int update_mode):
        cdef:
            np.npy_intp K, k, tm
            double[:,::1] shp_KQ_, core_KQ_
            gsl_rng * rng

        if len(self.core_shp) == 1:
            self.core_Q[:] = 1.
        else:
            shp_KQ_ = self.core_Q_prior
            core_KQ_ = np.zeros_like(shp_KQ_)

            tm = self.time_mode
            if update_mode == self._INFER_MODE:
                Y_KQ_ = tl.unfold(np.reshape(self.Y_Q, self.core_shp), tm)
                shp_KQ_ = np.add(shp_KQ_, Y_KQ_)

            K = core_KQ_.shape[0]
            for k in prange(K, schedule='static', nogil=True):
                rng = self.rngs[self._get_thread()]
                _sample_dirichlet(rng, shp_KQ_[k], core_KQ_[k])

            self.core_Q = tl.fold(core_KQ_, tm, self.core_shp).ravel()

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
    
                # In any future models that include per-mode rates that are not 1 should
                # replace this line with the two below it.
                zeta_T = np.sum(self.Theta_TK, axis=1)
                # zeta_TK = self._compute_zeta_m_DK(self.time_mode)
                # zeta_T = np.sum(np.multiply(self.Theta_TK, zeta_TK), axis=1)

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
    
    cdef void _update_b_T(self, int update_mode):
        cdef:
            np.npy_intp t, K
            double eps, tau, shp, rte, shp_t, rte_t, nu_
            double[::1] Theta_T, shp_T
            gsl_rng * rng

        eps = self.eps

        if update_mode == self._INITIALIZE_MODE:
            self.b_T[:] = 1.

        elif update_mode == self._GENERATE_MODE:
            if self.stationary:
                self.b_T[:] = _sample_gamma(self.rng, eps, 1./eps)
            else:
                for t in prange(self.n_timesteps, schedule='static', nogil=True):
                    self.b_T[t] = _sample_gamma(self.rng, eps, 1./eps)

        elif update_mode == self._INFER_MODE:
            tau = self.tau
            Theta_T = np.sum(self.Theta_TK, axis=1)
            shp_T = tau * np.sum(np.dot(self.Theta_TK, self.Pi_KK), axis=1)
            nu_ = np.sum(self.nu_K)

            if self.stationary:
                shp = eps + tau * nu_ + np.sum(shp_T[:self.n_timesteps-1])
                rte = eps + tau * np.sum(Theta_T)
                self.b_T[:] = _sample_gamma(self.rng, shp, 1. / rte)
            
            else:
                for t in prange(self.n_timesteps, schedule='static', nogil=True):
                    rng = self.rngs[self._get_thread()]
                    shp_t = eps + tau * nu_ if t == 0 else eps + shp_T[t-1]
                    rte_t = eps + tau * Theta_T[t]
                    self.b_T[t] = _sample_gamma(rng, shp_t, 1. / rte_t)

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
            np.npy_intp K, k, t
            double shp_tk, rte_t, rte_tk
            double[::1] Y_zeta_T
            gsl_rng * rng

        K = self.Theta_TK.shape[1]

        if update_mode == self._INITIALIZE_MODE:
            self.Theta_TK[:] = 1.

        else:
            if update_mode == self._INFER_MODE:
                self._compute_L_zeta_T()
                Y_zeta_T = self.delta_T

            with nogil:
                for t in range(self.n_timesteps):
                    
                    rte_t = self.tau * self.b_T[t]
                    
                    if update_mode == self._INFER_MODE:
                        rte_t = rte_t + Y_zeta_T[t] + self.L_zeta_T[t]
                    
                    for k in prange(K, schedule='static'):
                        rng = self.rngs[self._get_thread()]

                        if t == 0:
                            shp_tk = self.tau * self.nu_K[k]
                        else:
                            shp_tk = self.tau * _dot_vec(self.Theta_TK[t-1], self.Pi_KK[:, k])
                        
                        if update_mode == self._INFER_MODE:
                            shp_tk = shp_tk + self.Y_MKD[self.time_mode, k, t] + self.L_TK[t, k]
                        
                        self.Theta_TK[t, k] = _sample_gamma(rng, shp_tk, 1./rte_t)

        self._compute_mtx_KT()

    cdef void _compute_L_zeta_T(self) nogil:
        cdef:
            np.npy_intp t
            double m_zeta, tau
            double[::1] Y_zeta_T

        tau = self.tau
        Y_zeta_T = self.delta_T
        self.L_zeta_T[self.n_timesteps-1] = 0
        for t in range(self.n_timesteps-2, -1, -1):
            m_zeta = self.L_zeta_T[t+1] + Y_zeta_T[t+1]
            self.L_zeta_T[t] = tau * log1p(m_zeta / (tau * self.b_T[t+1]))

    cdef void _update_L_TKK(self, int update_mode):

        cdef:
            np.npy_intp K, t, k, k1, k2, tid
            int l_tk, m_tk
            double p_tk
            long[:,::1] Y_KT
            gsl_rng * rng

        K = self.core_dims_M[self.time_mode]    
        Y_KT = self.Y_MKD[self.time_mode, :K, :self.n_timesteps]

        self.L_TK[:] = 0
        self.L_KK[:] = 0
        self.L_TKK[:] = 0
        
        for t in range(self.n_timesteps-2, -1, -1):
            for k in prange(K, schedule='static', nogil=True):
                tid = self._get_thread()
                rng = self.rngs[tid]
                
                for k2 in range(K):
                    self.L_TK[t+1, k] += self.L_TKK[t+1, k, k2]

                m_tk = Y_KT[k, t+1] + self.L_TK[t+1, k]

                if m_tk > 0:
                    for k1 in range(K):
                        self.P_XMQ[tid, 0, k1] = self.Theta_TK[t, k1] * self.Pi_KK[k1, k]
                    p_tk = _sum_double_vec(self.P_XMQ[tid, 0, :K])

                    l_tk = _sample_crt(rng, m_tk, self.tau * p_tk)
                    if l_tk > 0:
                        gsl_ran_multinomial(rng,
                                            K, 
                                            l_tk, 
                                            &self.P_XMQ[tid, 0, 0], 
                                            &self.N_XMQ[tid, 0, 0])
                        for k1 in range(K):
                            self.L_TKK[t, k1, k] = self.N_XMQ[tid, 0, k1]

        self.L_KK = np.sum(self.L_TKK, axis=0, dtype=np.int32)
        self.L_TK = np.sum(self.L_TKK, axis=2, dtype=np.int32)  # only necessary for t=0

    cdef void _update_Pi_KK(self, int update_mode):

        cdef:
            np.npy_intp K, k, k2
            gsl_rng * rng

        K = self.core_dims_M[self.time_mode]
        for k in prange(K, schedule='static', nogil=True):
            rng = self.rngs[self._get_thread()]

            self.shp_KK[k, k] = self.xi_K[k] * self.nu_K[k]
            if update_mode == self._INFER_MODE:
                self.shp_KK[k, k] += self.L_KK[k, k]

            for k2 in range(K):
                if k != k2:
                    self.shp_KK[k, k2] = self.nu_K[k] * self.nu_K[k2]
                    if update_mode == self._INFER_MODE:
                        self.shp_KK[k, k2] += self.L_KK[k, k2]

            _sample_dirichlet(rng, self.shp_KK[k], self.Pi_KK[k])

    cdef void _update_nu_K_and_xi_K(self, int update_mode):
        cdef:
            np.npy_intp K, k, k2
            int m_k1, l_k1
            double gam_k, beta, eps, tau, m_zeta, zeta_1, nu_, nu_k
            double shp_k, rte_k, shp, rte, y_zeta_0
            gsl_rng * rng

        K = self.core_dims_M[self.time_mode]
        gam_k = self.gam / K
        beta = self.beta
        eps = self.eps
        rng = self.rng

        if update_mode == self._GENERATE_MODE:
            with nogil:
                self.xi_K[:] = _sample_gamma(rng, eps, 1./eps)
                for k in range(K):
                    self.nu_K[k] = _sample_gamma(rng, gam_k, 1./beta)
            
        elif update_mode == self._INITIALIZE_MODE:
            self.xi_K[:] = 1.
            self.nu_K[:] = 1.

        elif update_mode == self._INFER_MODE:
            self._compute_L_zeta_T()
            self._update_H_KK_and_lnq_K()

            with nogil:
                shp = rte = eps
                for k in range(K):
                    shp += self.H_KK[k, k]
                    rte -= self.nu_K[k] * self.lnq_K[k]
                self.xi_K[:] = _sample_gamma(rng, shp, 1./rte)

                tau = self.tau
                y_zeta_0 = self.delta_T[0]
                m_zeta = self.L_zeta_T[0] + y_zeta_0
                zeta_1 = tau * log1p(m_zeta / (tau * self.b_T[0]))
                nu_ = _sum_double_vec(self.nu_K)

                for k in range(K):
                    nu_k = self.nu_K[k]
                    nu_ -= nu_k

                    shp_k = gam_k
                    rte_k = beta

                    m_k1 = self.L_TK[0, k] + self.Y_MKD[self.time_mode, k, 0]
                    l_k1 = _sample_crt(rng, m_k1, tau * nu_k)
                    
                    shp_k += l_k1
                    rte_k += zeta_1

                    shp_k += self.H_KK[k, k]
                    rte_k -= (self.xi_K[k] + nu_) * self.lnq_K[k]
                    for k2 in range(K):
                        if k2 != k:
                            shp_k += self.H_KK[k, k2] + self.H_KK[k2, k]
                            rte_k -= self.nu_K[k2] * self.lnq_K[k2]

                    self.nu_K[k] = _sample_gamma(rng, shp_k, 1./rte_k)
                    nu_ += self.nu_K[k]

    cdef void _update_H_KK_and_lnq_K(self):
        cdef:
            np.npy_intp K, k, k2
            double nu_, nu_k, xi_k, tmp
            int[::1] L_K
            gsl_rng * rng

        K = self.core_dims_M[self.time_mode]
        nu_ = _sum_double_vec(self.nu_K)
        L_K = np.sum(self.L_KK, axis=1, dtype=np.int32)
        rng = self.rng

        self.lnq_K[:] = 0
        for k in range(K):
            nu_k = self.nu_K[k]
            xi_k = self.xi_K[k]

            if L_K[k] > 0:
                tmp = (xi_k  + nu_ - nu_k)
                self.lnq_K[k] = _sample_lnbeta(rng, nu_k * tmp, L_K[k])
                assert np.isfinite(self.lnq_K[k]) and self.lnq_K[k] <= 0

            self.H_KK[k, k] = _sample_crt(rng, self.L_KK[k, k], nu_k * xi_k)
            assert self.H_KK[k, k] >= 0

            for k2 in range(K):
                if k2 != k:
                    self.H_KK[k, k2] = _sample_crt(rng, self.L_KK[k, k2], nu_k * self.nu_K[k2])
                    assert self.H_KK[k, k2] >= 0

    cdef void _update_beta(self, int update_mode) nogil:
        cdef: 
            double shp, rte

        if update_mode == self._INITIALIZE_MODE:
            self.beta = 1.
        else:
            shp = rte = self.eps
            if update_mode == self._INFER_MODE:
                shp += self.gam
                rte += _sum_double_vec(self.nu_K)
            self.beta = _sample_gamma(self.rng, shp, 1. / rte)
