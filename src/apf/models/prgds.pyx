# cython: boundscheck = False
# cython: initializedcheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: language_level = 3

import sys
import numpy as np
import numpy.random as rn
cimport numpy as np
from libc.math cimport log1p, sqrt

from cython.parallel import parallel, prange

from apf.base.apf cimport APF
from apf.base.sample cimport _sample_gamma, _sample_dirichlet, _sample_crt, _sample_lnbeta, _sample_poisson
from apf.base.bessel cimport _sample as _sample_bessel
from apf.base.conf_hypergeom cimport _sample as _sample_conf_hypergeom
from apf.base.sbch cimport _sample as _sample_sbch
from apf.base.cyutils cimport _sum_double_vec, _sum_int_vec, _dot_vec
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

cdef class PRGDS(APF):

    cdef:
        int time_mode, n_timesteps, stationary
        int block_sample_Theta_and_H, block_sample_nu_and_g
        double gam, beta, tau, eps_theta, eps_nu
        double[::1] nu_K, delta_T, b_T
        double[:,::1] Theta_TK, Pi_KK
        int[:,:,::1] H_TKK
        int[::1] g_K, n_nonblock_X

    def __init__(self, tuple data_shp, tuple core_shp, double eps=0.1, int binary=0,
                 list mtx_is_dirichlet=[], object seed=None, object n_threads=None,
                 int time_mode=0, int stationary=0, double eps_theta=0.0, double eps_nu=0.0,
                 block_sample_Theta_and_H=True, block_sample_nu_and_g=True):

        if time_mode in mtx_is_dirichlet:
            raise ValueError('Time-mode matrix cannot be Dirichlet.')

        assert (eps_theta == 0) or (eps_theta >= 1e-10)

        super().__init__(data_shp=data_shp,
                         core_shp=core_shp,
                         eps=eps,
                         binary=binary,
                         mtx_is_dirichlet=mtx_is_dirichlet,
                         seed=seed,
                         n_threads=n_threads)
        self.core_Q[:] = 1.

        # Params
        self.time_mode = self.param_list['time_mode'] = time_mode
        self.stationary = self.param_list['stationary'] = stationary
        self.eps_theta = self.param_list['eps_theta'] = eps_theta
        self.eps_nu = self.param_list['eps_nu'] = eps_nu
        self.block_sample_Theta_and_H = self.param_list['block_sample_Theta_and_H'] = block_sample_Theta_and_H
        self.block_sample_nu_and_g = self.param_list['block_sample_nu_and_g'] = block_sample_nu_and_g
        self.n_timesteps = T = data_shp[time_mode]
        K = self.core_dims_M[time_mode]

        # State variables
        self.gam = 1.
        self.g_K = np.ones(K, dtype=np.int32)
        self.nu_K = np.ones(K)

        self.tau = 1.
        self.b_T = np.ones(T)
        self.delta_T = np.ones(T)
        self.Pi_KK = np.ones((K, K))
        self.Theta_TK = np.ones((T, K))
        self.H_TKK = np.ones((T, K, K), dtype=np.int32)

        # Auxiliary structures
        self.n_nonblock_X = np.zeros(self.n_threads, dtype=np.int32)

    @property
    def percent_nonblock_samples(self):
        n_nonblock = np.sum(self.n_nonblock_X)
        T, K = self.Theta_TK.shape[:2]
        return 100 * n_nonblock / float(T * K)

    cdef list _get_variables(self):
        """
        Return variable names, values, and sampling methods for testing.

        MUST BE IN TOPOLOGICAL ORDER!
        """
        variables = [('gam', self.gam, self._update_gam),
                     ('g_K', self.g_K, self._update_g_K),
                     ('beta', self.beta, self._update_beta),
                     ('nu_K', self.nu_K, self._update_nu_K),
                     ('Pi_KK', self.Pi_KK, self._update_Pi_KK),
                     ('tau', self.tau, self._update_tau),
                     ('b_T', self.b_T, self._update_b_T)]

        if self.block_sample_Theta_and_H:
            variables += [('H_TKK', self.H_TKK, self._update_Theta_TK_and_H_TKK),
                          ('Theta_TK', self.Theta_TK, self._dummy_update)]
        else:
            variables += [('H_TKK', self.H_TKK, self._update_H_TKK),
                          ('Theta_TK', self.Theta_TK, self._update_Theta_TK)]

        variables += [('delta_T', self.delta_T, self._update_delta_T),
                      ('b_M', self.b_M, self._update_b_M),
                      ('mtx_MKD', self.mtx_MKD, self._update_mtx_MKD),
                      ('core_Q', self.core_Q, self._update_core_Q),
                      ('Y_MKD', self.Y_MKD, self._update_Y_PQ),
                      ('Y_Q', self.Y_Q, self._dummy_update)]
        return variables

    def get_default_schedule(self):
        return {'gam': lambda x: x > 100,
                'beta': lambda x: x > 2,
                'g_K': lambda x: x > 1,
                'nu_K': lambda x: x > 1}

    def set_state(self, state):
        for key, val, _ in self._get_variables():
            if key in state.keys():
                state_val = state[key]
                if key == 'tau':
                    self.tau = state_val
                elif key == 'gam':
                    self.gam = state_val
                elif key == 'beta':
                    self.beta = state_val
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
                if key == 'tau':
                    self.tau = state_val
                elif key == 'gam':
                    self.gam = state_val
                elif key == 'beta':
                    self.beta = state_val
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

    cdef void _update_core_Q(self, int update_mode):
        """
        Overwrite APF's update_core_Q method to ensure that
        the core_Q data structure is always all ones.
        """
        self.core_Q[:] = 1.

    cdef void _update_gam(self, int update_mode):
        cdef:
            double shp, rte

        shp = rte = 10.

        if update_mode == self._INFER_MODE:
            shp += _sum_int_vec(self.g_K)
            rte += + 1.

        self.gam = _sample_gamma(self.rng, shp, 1./rte)

    cdef int _update_g_K(self, int update_mode):
        cdef:
            np.npy_intp T, K, k, tm
            double mu_k, ordr, tmp, crdt_k, m_zeta_k, r_k
            int m_k, g_k
            long[::1] Y_K
            double[::1] Y_zeta_K
            gsl_rng * rng

        T, K = self.Theta_TK.shape[:2]

        if update_mode == self._INITIALIZE_MODE:
            for k in prange(K, schedule='static', nogil=True):
                rng = self.rngs[self._get_thread()]
                self.g_K[k] = _sample_poisson(rng, 0.75) 

        elif update_mode == self._GENERATE_MODE:
            mu_k = self.gam / K
            for k in prange(K, schedule='static', nogil=True):
                rng = self.rngs[self._get_thread()]
                self.g_K[k] = _sample_poisson(rng, mu_k)

        elif update_mode == self._INFER_MODE:
            
            if not self.block_sample_nu_and_g:
                ordr = self.eps_nu / K - 1.
                tmp = self.beta * self.gam / K
                for k in prange(K, schedule='static', nogil=True):
                    rng = self.rngs[self._get_thread()]
                    crdt_k = 2 * sqrt(self.nu_K[k] * tmp)
                    self.g_K[k] = _sample_bessel(rng, ordr, crdt_k)
                    
                    if self.g_K[k] < 0:
                        with gil:
                            raise RuntimeError('Negative value returned by Bessel:\n \
                                               %f: Order\n \
                                               %f: Coordinate' % (ordr, crdt_k))
            else:
                tm = self.time_mode
                Y_K = np.sum(self.Y_MKD[tm, :K, :T], axis=1, dtype=long)
                Y_zeta_K = np.einsum('t,tk,tk->k', self.delta_T, 
                                     self._compute_zeta_m_DK(tm),
                                     self.Theta_TK)

                mu_k = self.gam / K
                for k in prange(K, schedule='static', nogil=True):
                    rng = self.rngs[self._get_thread()]
                    m_k = Y_K[k] + _sum_int_vec(self.H_TKK[0, k])
                    m_zeta_k = Y_zeta_K[k] + self.tau
                    r_k = mu_k * self.beta / (m_zeta_k + self.beta)
                    if r_k == 0:
                        r_k = 1e-300

                    if m_k == 0:
                        g_k = _sample_poisson(rng, r_k)
                    elif self.eps_nu == 0:
                        g_k = _sample_sbch(rng, m_k, r_k)
                    else:
                        g_k = _sample_conf_hypergeom(rng, m_k, self.eps_nu, r_k)

                    if g_k < 0:
                        with gil:
                            raise RuntimeError('Negative value returned by Conf Hypergoem:\n \
                                               %f: Population\n \
                                               %f: Rate' % (m_k, r_k))
        return 1

    cdef void _update_beta(self, int update_mode):
        cdef:
            double shp, rte

        shp = rte = 10.
        
        if update_mode == self._INFER_MODE:
            shp += self.eps_nu + np.sum(self.g_K)
            rte += np.sum(self.nu_K)

        self.beta = _sample_gamma(self.rng, shp, 1./rte)

    cdef void _update_nu_K(self, int update_mode):
        """
        nu_k ~ Gamma(eps_nu + g_k, beta)
        y_k ~ Pois(nu_k zeta_k)
        h_k ~ Pois(nu_k * tau)

        zeta_k = sum_t delta_t theta_tk sum_

        (nu_k|-) 
        """
        cdef:
            np.npy_intp T, K, k, tm
            double shp, rte, shp_k, rte_k
            double[::1] Y_zeta_K
            long[::1] Y_K
            gsl_rng * rng

        T, K = self.Theta_TK.shape[:2]

        if update_mode == self._INITIALIZE_MODE:
            self.nu_K[:] = 1. / K 

        elif update_mode == self._GENERATE_MODE:
            shp = self.eps_nu / float(K)
            rte = self.beta
            
            for k in prange(K, schedule='static', nogil=True):
                rng = self.rngs[self._get_thread()]
                shp_k = shp + self.g_K[k]
                if shp_k == 0:
                    self.nu_K[k] = 0
                else:
                    self.nu_K[k] = _sample_gamma(rng, shp_k, 1./rte)

        elif update_mode == self._INFER_MODE:
            shp = self.eps_nu / float(K)
            rte = self.beta

            tm = self.time_mode
            Y_K = np.sum(self.Y_MKD[tm, :K, :T], axis=1, dtype=int)
            Y_zeta_K = np.einsum('t,tk,tk->k', self.delta_T, 
                                 self._compute_zeta_m_DK(tm),
                                 self.Theta_TK)

            for k in prange(K, schedule='static', nogil=True):
                rng = self.rngs[self._get_thread()]
                shp_k = shp + self.g_K[k]
                if shp_k == 0:
                    self.nu_K[k] = 0
                else:
                    shp_k = shp_k + Y_K[k] + _sum_int_vec(self.H_TKK[0, k])
                    rte_k = rte + Y_zeta_K[k] + self.tau
                    self.nu_K[k] = _sample_gamma(rng, shp_k, 1./rte_k)

        self._compute_mtx_KT()

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
                mtx_KT[k, t] = self.delta_T[t] * self.nu_K[k] * self.Theta_TK[t, k]
                mtx_K[k] += mtx_KT[k, t]

    cdef void _update_delta_T(self, int update_mode):
        cdef:
            np.npy_intp K, T, t, tm
            double prior_shp, prior_rte, shp, rte, shp_t, rte_t
            double[::1] mtx_T, Y_zeta_T
            long[::1] Y_T
            gsl_rng * rng

        if update_mode == self._INITIALIZE_MODE:
            self.delta_T[:] = 1
        else:
            if update_mode == self._INFER_MODE:
                tm = self.time_mode
                T, K = self.Theta_TK.shape[:2]
                Y_T = np.sum(self.Y_MKD[tm, :K, :T], axis=0, dtype=np.int)
                Y_zeta_T = np.einsum('tk,tk,k->t', self._compute_zeta_m_DK(tm), 
                                                   self.Theta_TK,
                                                   self.nu_K)

            prior_shp = prior_rte = self.eps
            if self.stationary:
                shp, rte = prior_shp, prior_rte
                if update_mode == self._INFER_MODE:
                    shp += np.sum(Y_T) 
                    rte += np.sum(Y_zeta_T)
                self.delta_T[:] = _sample_gamma(self.rng, shp, 1./rte)
            
            else:
                for t in prange(self.n_timesteps, schedule='static', nogil=True):
                    rng = self.rngs[self._get_thread()]
                    shp_t, rte_t = prior_shp, prior_rte
                    if update_mode == self._INFER_MODE:
                        shp_t = shp_t + Y_T[t]
                        rte_t = rte_t + Y_zeta_T[t]
                    self.delta_T[t] = _sample_gamma(rng, shp_t, 1./rte_t)

        self._compute_mtx_KT()

    def forecast(self, n_timesteps=1, n_samples=1, subs=()):
        
        if not self.stationary:
            raise NotImplementedError('Forecasting in non-stationary model not available.')
        
        delta_T = np.repeat(self.delta_T[0], repeats=n_timesteps)
        
        if n_samples == 1:
            mtx = self._forecast_mean_Theta_TK(n_timesteps) * delta_T[:, np.newaxis] * np.array(self.nu_K)
            return self.decode(mtx=mtx, mode=self.time_mode, subs=subs)
        else:
            return np.array([self.forecast(n_timesteps=n_timesteps, n_samples=1, subs=subs) for _ in range(n_samples)])

    cdef double[:,::1] _forecast_mean_Theta_TK(self, n_timesteps=1):
        cdef:
            np.npy_intp T, K, t
            double tau, b, eps
        
        assert self.stationary  # if not stationary b_forecast_T is time-dependent

        T, K = self.Theta_TK.shape[:2]
        tau = self.tau
        b = self.b_T[0]
        eps = self.eps_theta

        bias_K = np.zeros(K)
        Lam_KK = np.multiply(self.Pi_KK, 1./b)
        Theta_forecast_TK = np.zeros((n_timesteps, K)) 

        for t in range(n_timesteps):
            bias_K += np.linalg.matrix_power(Lam_KK, t).sum(axis=1) * eps / (b * tau)
            Theta_forecast_TK[t] = bias_K + np.dot(np.linalg.matrix_power(Lam_KK, t+1), self.Theta_TK[T-1])

        return Theta_forecast_TK

    def test_forecast_mean_Theta_TK(self):
        foo = self._forecast_Theta_TK(5, sample=0)
        bar = self._forecast_mean_Theta_TK(5)
        assert np.allclose(foo, bar)

    def forecast_Theta_TK(self, n_timesteps=1, sample=False):
        if sample:
            return np.array(self._forecast_Theta_TK(n_timesteps, sample=1))
        else:
            return np.array(self._forecast_mean_Theta_TK(n_timesteps))

    cdef double[:,::1] _forecast_Theta_TK(self, int n_timesteps, int sample=1):
        cdef:
            np.npy_intp K, t, k
            int h_tk
            double rte_t, mu_tk
            double[::1] b_forecast_T
            double[:,::1] Theta_forecast_TK 

        assert self.stationary  # if not stationary b_forecast_T is time-dependent
        b_forecast_T = np.repeat(self.b_T[0], repeats=n_timesteps)

        K = self.Theta_TK.shape[1]
        Theta_forecast_TK = np.zeros((n_timesteps, K)) 

        for t in range(n_timesteps):
            rte_t = self.tau * b_forecast_T[t]
            for k in range(K):
                if t == 0:
                    mu_tk = self.tau * _dot_vec(self.Pi_KK[k], self.Theta_TK[self.n_timesteps-1])
                else:
                    mu_tk = self.tau * _dot_vec(self.Pi_KK[k], Theta_forecast_TK[t-1])
                
                if sample:
                    h_tk = _sample_poisson(self.rng, mu_tk)
                    if not h_tk >= 0:
                        raise ValueError('Lambda values too large (>2e9).')
                    if h_tk == 0 and self.eps_theta == 0:
                        Theta_forecast_TK[t, k] = 0
                    else:
                        Theta_forecast_TK[t, k] = _sample_gamma(self.rng, self.eps_theta + h_tk, 1./rte_t)
                else:
                    Theta_forecast_TK[t, k] = (self.eps_theta + mu_tk) / rte_t

        return Theta_forecast_TK

    cdef int _update_Theta_TK_and_H_TKK(self, int update_mode):
        cdef:
            np.npy_intp T, K, t, k, k1, k2, tid, tm
            int n_nonblock, m_tk, h_tk
            double mu_tkk, mu_tk, shp_tk, rte_t, rte_tk, m_zeta_tk, r_tk
            double[::1] Pi_K
            double[:,::1] Y_zeta_TK
            long[:,::1] Y_KT
            gsl_rng * rng

        T, K = self.Theta_TK.shape[:2]

        if update_mode == self._INITIALIZE_MODE:
            self._initialize_H_TKK()
            self._initialize_Theta_TK()

        elif update_mode == self._GENERATE_MODE:
            """In generate mode, the generation of theta_TK and H_TK must be
            interleaved. Currently, when generate_state is called, this method
            will be called twice."""
            for t in range(T):
                rte_t = self.tau * self.b_T[t]
                for k1 in prange(K, schedule='static', nogil=True):
                    rng = self.rngs[self._get_thread()]

                    mu_tkk = self.nu_K[k1] / K
                    for k2 in range(K):
                        if t > 0:
                            mu_tkk = self.Pi_KK[k1, k2] * self.Theta_TK[t-1, k2]
                        self.H_TKK[t, k1, k2] = _sample_poisson(rng, self.tau * mu_tkk)

                    shp_tk = self.eps_theta + _sum_int_vec(self.H_TKK[t, k1])
                    if shp_tk > 0:
                        self.Theta_TK[t, k1] = _sample_gamma(rng, shp_tk, 1./rte_t)
                    else:
                        self.Theta_TK[t, k1] = 0

                if (np.array(self.H_TKK[t]) < 0).any():
                    raise ValueError('Lambda values too large (>2e9).')

        elif update_mode == self._INFER_MODE:
            tm = self.time_mode
            Pi_K = np.sum(self.Pi_KK, axis=0)
            Y_KT = self.Y_MKD[tm, :K, :T]
            Y_zeta_TK = np.einsum('tk,t,k->tk', self._compute_zeta_m_DK(tm), 
                                                self.delta_T,
                                                self.nu_K)
            self.n_nonblock_X[:] = 0
            for t in range(T):
                rte_t = self.tau * self.b_T[t]
                for k in prange(K, schedule='static', nogil=True):
                    tid = self._get_thread(); rng = self.rngs[tid]

                    # SAMPLE H_t
                    if t == 0:
                        mu_tk = self.tau * self.nu_K[k]
                        self.P_XMQ[tid, tm, :K] = 1. / K
                    else:
                        for k2 in range(K):
                            self.P_XMQ[tid, tm, k2] = self.Pi_KK[k, k2] * self.Theta_TK[t-1, k2]
                        mu_tk = self.tau * _sum_double_vec(self.P_XMQ[tid, tm, :K])

                    m_tk = Y_KT[k, t]
                    m_zeta_tk = Y_zeta_TK[t, k]
                    if t < T-1:
                        m_tk = m_tk + _sum_int_vec(self.H_TKK[t+1, :, k])
                        m_zeta_tk = m_zeta_tk + self.tau * Pi_K[k]
                    r_tk = mu_tk * (rte_t / (m_zeta_tk + rte_t))
                    if r_tk == 0:
                        r_tk = 1e-300

                    # Sample from (size-based) confluent hypergeometric
                    if m_tk == 0:
                        h_tk = _sample_poisson(rng, r_tk)
                    elif self.eps_theta == 0:
                        h_tk = _sample_sbch(rng, m_tk, r_tk)
                    else:
                        h_tk = _sample_conf_hypergeom(rng, m_tk, self.eps_theta, r_tk)

                    # IF sampling from the confluent hypergeometric fails...
                    # ...sample from the Bessel (conditioned on future theta)
                    if h_tk < 0:
                        shp_tk = mu_tk * rte_t * self.Theta_TK[t, k]
                        if shp_tk > 0:
                            h_tk = _sample_bessel(rng, self.eps_theta-1, 2 * sqrt(shp_tk))
                        else:
                            h_tk = 0
                        # Record how many times we cannot sample h,theta as a block
                        self.n_nonblock_X[tid] += 1
                    
                    if h_tk < 0:
                        with gil:
                            raise ValueError('h_tk < 0')

                    # Thin using Multinomial
                    gsl_ran_multinomial(rng, K, h_tk, &self.P_XMQ[tid, tm, 0], &self.N_XMQ[tid, tm, 0])
                    for k2 in range(K):
                        self.H_TKK[t, k, k2] = self.N_XMQ[tid, tm, k2]

                    # SAMPLE THETA_t
                    shp_tk = self.eps_theta + h_tk
                    if shp_tk == 0:
                        self.Theta_TK[t, k] = 0
                    else:
                        shp_tk = shp_tk + Y_KT[k, t]
                        rte_tk = rte_t + Y_zeta_TK[t, k]
                        if t < T-1:
                            shp_tk = shp_tk + _sum_int_vec(self.H_TKK[t+1, :, k])
                            rte_tk = rte_tk + self.tau * Pi_K[k]
                        self.Theta_TK[t, k] = _sample_gamma(rng, shp_tk, 1./rte_tk)

        self._compute_mtx_KT()
        return 1

    cdef void _initialize_H_TKK(self):
        cdef:
            np.npy_intp tm, T, K, t, k, k2, tid
            double mu_tkk 
            gsl_rng * rng

        tm = self.time_mode
        T = self.data_dims_M[tm]
        K = self.core_dims_M[tm]

        for t in prange(T, schedule='static', nogil=True):
            tid = self._get_thread(); rng = self.rngs[tid]
            for k in range(K):
                for k2 in range(K):
                    mu_tkk = 0.5 if k != k2 else 2.
                    self.H_TKK[t, k, k2] = _sample_poisson(rng, mu_tkk)

    cdef void _initialize_Theta_TK(self):
        cdef:
            np.npy_intp tm, T, K, t, k
            gsl_rng * rng

        tm = self.time_mode
        T = self.data_dims_M[tm]
        K = self.core_dims_M[tm]

        for k in prange(K, schedule='static', nogil=True):
            rng = self.rngs[self._get_thread()]
            for t in range(T):
                self.Theta_TK[t, k] = 0.1 * _sample_gamma(rng, 1., 1.)

        self._compute_mtx_KT()

    cdef int _update_H_TKK(self, int update_mode):
        cdef:
            np.npy_intp tm, T, K, tid, t, k, k2
            int h_tk
            double rte_t, shp_tk, mu_tk
            gsl_rng * rng
                
        tm = self.time_mode
        T = self.data_dims_M[tm]
        K = self.core_dims_M[tm]

        self.H_TKK[:] = 0
        
        if update_mode == self._GENERATE_MODE:
            self._update_Theta_TK_and_H_TKK(update_mode=self._GENERATE_MODE)

        elif update_mode == self._INITIALIZE_MODE:
            self._initialize_H_TKK()

        elif update_mode == self._INFER_MODE:
            for t in prange(T, schedule='static', nogil=True):
                tid = self._get_thread(); rng = self.rngs[tid]
                rte_t = self.tau * self.b_T[t]
                for k in range(K):
                    if t == 0:
                        mu_tk = self.tau * self.nu_K[k]
                        self.P_XMQ[tid, tm, :K] = 1. / K
                    else:
                        for k2 in range(K):
                            self.P_XMQ[tid, tm, k2] = self.Pi_KK[k, k2] * self.Theta_TK[t-1, k2]
                        mu_tk = self.tau * _sum_double_vec(self.P_XMQ[tid, tm, :K])
                    shp_tk = mu_tk * rte_t * self.Theta_TK[t, k]
                    if shp_tk > 0:
                        h_tk = _sample_bessel(rng, self.eps_theta-1, 2 * sqrt(shp_tk))
                        if h_tk < 0:
                            with gil:
                                raise ValueError('h_tk < 0')
                        if h_tk > 0:
                            gsl_ran_multinomial(rng, K, h_tk, &self.P_XMQ[tid, tm, 0], &self.N_XMQ[tid, tm, 0])
                            for k2 in range(K):
                                self.H_TKK[t, k, k2] = self.N_XMQ[tid, tm, k2]
        return 1

    cdef void _update_Theta_TK(self, int update_mode):
        """This method re-samples all theta_tk conditioned on all h_tk.

        This method is called in the regime where we alternate between
        sampling Theta ~ P(Theta | H, -) and H ~ P(H | Theta, -); the only 
        difference w/r/t to the Theta sampling is the order in which the 
        Theta's are sampled. In
        """
        cdef: 
            np.npy_intp tm, T, K, k, t
            int h_tk
            double rte_t, shp_tk, rte_tk, s
            double[:,::1] Y_zeta_TK
            long[:,::1] Y_KT
            gsl_rng * rng
        
        tm = self.time_mode
        T = self.data_dims_M[tm]
        K = self.core_dims_M[tm]

        if update_mode == self._GENERATE_MODE:
            self._update_Theta_TK_and_H_TKK(update_mode=self._GENERATE_MODE)

        elif update_mode == self._INITIALIZE_MODE:
            self._initialize_Theta_TK()

        elif update_mode == self._INFER_MODE:
            Y_KT = self.Y_MKD[tm, :K, :T]
            Y_zeta_TK = np.einsum('tk,t,k->tk', self._compute_zeta_m_DK(tm), 
                                                self.delta_T,
                                                self.nu_K)

            for t in prange(T, schedule='static', nogil=True):
                rng = self.rngs[self._get_thread()]

                rte_t = self.tau * self.b_T[t]

                for k in range(K):
                    shp_tk = self.eps_theta + _sum_int_vec(self.H_TKK[t, k])
                    if shp_tk == 0:
                        self.Theta_TK[t, k] = 0
                    else:
                        shp_tk = shp_tk + Y_KT[k, t]
                        rte_tk = rte_t + Y_zeta_TK[t, k]

                        if t < T-1:
                            shp_tk = shp_tk + _sum_int_vec(self.H_TKK[t+1, :, k])
                            rte_tk = rte_tk + self.tau * _sum_double_vec(self.Pi_KK[:, k])

                        self.Theta_TK[t, k] = _sample_gamma(rng, shp_tk, 1./rte_tk)

            self._compute_mtx_KT()

    cdef void _update_tau(self, int update_mode):
        cdef: 
            np.npy_intp T, K
            long h_
            double shp, rte
        
        if update_mode == self._INITIALIZE_MODE:
            self.tau = 1.
        else:
            shp = rte = 10
            if update_mode == self._INFER_MODE:
                h_ = np.sum(self.H_TKK)
                # Add all H_TKK
                shp += h_
                # Add all the corresponding poisson rates (excluding tau) 
                rte += np.sum(self.nu_K)
                # The last Theta_TK's don't lead to another H_TKK
                rte += np.dot(np.sum(self.Theta_TK[:self.n_timesteps-1], axis=0),
                              np.sum(self.Pi_KK, axis=0))

                # Add the sum of Theta_TK times their prior rate parameters (excluding tau)
                rte += np.dot(np.sum(self.Theta_TK, axis=1), self.b_T)
                # Add the sum of the prior shape parameters for Theta_TK
                T, K = self.Theta_TK.shape[:2]
                shp += (self.eps_theta * T * K) + h_
            self.tau = _sample_gamma(self.rng, shp, 1./rte)

    cdef void _update_b_T(self, int update_mode):
        cdef:
            np.npy_intp T, K, t
            double prior_shp, prior_rte, shp, rte, shp_t, rte_t
            int[::1] H_T
            double[::1] Theta_T
            gsl_rng * rng

        if update_mode == self._INITIALIZE_MODE:
            self.b_T[:] = 1.
        else:
            T, K = self.Theta_TK.shape[:2]
            prior_shp = prior_rte = 10.

            if update_mode == self._INFER_MODE:
                H_T = np.sum(self.H_TKK, axis=(1, 2), dtype=np.int32)
                Theta_T = np.sum(self.Theta_TK, axis=1)
            
            if self.stationary:
                shp, rte = prior_shp, prior_rte
                if update_mode == self._INFER_MODE:
                    shp += self.eps_theta * T * K + np.sum(H_T)
                    rte += self.tau * np.sum(Theta_T)
                self.b_T[:] = _sample_gamma(self.rng, shp, 1./rte)
            else:
                for t in prange(T, schedule='static', nogil=True):
                    rng = self.rngs[self._get_thread()]
                    shp_t, rte_t = prior_shp, prior_rte
                    if update_mode == self._INFER_MODE:
                        shp_t = shp_t + self.eps_theta * K + H_T[t]
                        rte_t = rte_t + self.tau * Theta_T[t]
                        self.b_T[t] = _sample_gamma(rng, shp_t, 1./rte_t)

    cdef void _update_Pi_KK(self, int update_mode):
        cdef:
            np.npy_intp K, k1, k2
            double shp_kk, rte_kk, pi_k, shp_k
            int[:,::1] H_KK
            double[::1] H_zeta_K
            double[::1,:] shp_KK
            double[::1,:] Pi_KK
            gsl_rng * rng

        K = self.core_dims_M[self.time_mode]
        
        self.Pi_KK[:] = 0

        shp_KK = self.eps * np.ones((K, K), order='F')
        if update_mode == self._INFER_MODE:
            H_KK = np.sum(self.H_TKK[1:], axis=0, dtype=np.int32)
            shp_KK = np.asarray(np.add(shp_KK, H_KK), order='F')

        Pi_KK = np.zeros((K, K), order='F')
        for k2 in prange(K, schedule='static', nogil=True):
            rng = self.rngs[self._get_thread()]
            _sample_dirichlet(rng, shp_KK[:, k2], Pi_KK[:, k2])
        self.Pi_KK = np.ascontiguousarray(Pi_KK)

