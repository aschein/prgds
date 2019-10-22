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
from copy import deepcopy

from cython.parallel import parallel, prange

from apf.base.mcmc_model_parallel cimport MCMCModel
from apf.base.sample cimport _sample_gamma, _sample_dirichlet, _sample_trunc_poisson, _sample_poisson
from apf.base.allocate cimport _compute_prob, _allocate
from apf.base.cyutils cimport _sum_double_vec
from apf.base.utils import uttut, uttkrp, sp_uttkrp


cdef extern from "gsl/gsl_rng.h" nogil:
    ctypedef struct gsl_rng:
        pass


cdef class APF(MCMCModel):

    def __init__(self, tuple data_shp, tuple core_shp, double eps=0.1, 
                 int binary=0, list mtx_is_dirichlet=[],
                 object seed=None, object n_threads=None):

        super(APF, self).__init__(seed=seed, n_threads=n_threads)

        # Params
        self.data_shp = self.param_list['data_shp'] = data_shp
        self.core_shp = self.param_list['core_shp'] = core_shp
        self.eps = self.param_list['eps'] = eps
        self.binary = self.param_list['binary'] = binary
        self.mtx_is_dirichlet = self.param_list['mtx_is_dirichlet'] = mtx_is_dirichlet

        self.n_modes = M = len(self.data_shp)
        self.max_data_dim = D = max(self.data_shp)
        self.max_core_dim = K = max(self.core_shp)
        self.n_classes = Q = np.prod(self.core_shp)
        
        self.is_tucker = int(len(self.core_shp) > 1)
        if self.is_tucker:
            assert len(self.core_shp) == len(self.data_shp)
            self.subs_QM = np.array(list(np.ndindex(core_shp)), dtype=np.int32)
            self.core_dims_M = np.array(core_shp, dtype=np.int32)
        else:
            self.core_dims_M = np.repeat(Q, repeats=M).astype(np.int32)
        self.data_dims_M = np.array(data_shp, dtype=np.int32)

        assert all([m in range(M) for m in self.mtx_is_dirichlet])
        
        impute_Y_M = np.zeros(M, dtype=np.int32)
        impute_Y_M[self.mtx_is_dirichlet] = 1
        self.impute_Y_M = impute_Y_M
        self.impute_Y_Q = 0
        self.impute_after = 0

        # State variables
        self.b_M = np.ones(M)
        self.mtx_MKD = np.zeros((M, K, D))
        self.core_Q = np.ones(Q)
        self.Y_MKD = np.zeros((M, K, D), dtype=np.int)
        self.Y_Q = np.zeros(Q, dtype=np.int)

        # Cache and auxiliary data structures
        X = self.n_threads
        self.Y_XQ = np.zeros((X, Q), dtype=np.int)
        self.Y_XMKD = np.zeros((X, M, K, D), dtype=np.int)
        self.N_XMQ = np.zeros((X, M, Q), dtype=np.uint32)
        self.P_XMQ = np.zeros((X, M, Q))
        self.shp_MKD = np.zeros((M, K, D))
        self.mtx_MK = np.zeros((M, K))

        # Copy of the data 
        self.n_nonzero = 0    # placeholders
        self.nonzero_data_P = np.zeros(0, dtype=np.int)
        self.nonzero_subs_PM = np.zeros((0, M), dtype=np.int32)

        self.n_missing = 0
        self.missing_data_P = np.zeros(0, dtype=np.int)
        self.missing_subs_PM = np.zeros((0, M), dtype=np.int32)

        # Whether *all* data entries indexed by a given mode dimension are missing
        self.all_missing_MD = np.zeros((M, D), dtype=np.int32)
        # Whether *any* data entries indexed by a given mode dimension are missing
        self.any_missing_MD = np.zeros((M, D), dtype=np.int32)
        # Whether *any-but-NOT-all* data entries indexed by a given mode dimension are missing
        self.any_not_all_missing_MD = np.zeros((M, D), dtype=np.int32)
        self.mask = None
        self.inv_mask = None

    cdef list _get_variables(self):
        """
        Return variable names, values, and sampling methods for testing.

        MUST BE IN TOPOLOGICAL ORDER!
        """
        variables = [('core_Q', self.core_Q, self._update_core_Q),
                     ('b_M', self.b_M, self._update_b_M),
                     ('mtx_MKD', self.mtx_MKD, self._update_mtx_MKD),
                     ('Y_MKD', self.Y_MKD, self._update_Y_PQ),
                     ('Y_Q', self.Y_Q, self._dummy_update)]

        return variables

    cdef void _dummy_update(self, int update_mode) nogil:
        pass

    def _initialize_data(self, data):
        cdef:
            np.npy_intp m, d

        if not isinstance(data, np.ma.core.MaskedArray):
            data = np.ma.array(data, mask=None)

        assert data.shape == self.data_shp
        assert (data >= 0).all() is np.ma.masked or True
        if self.binary:
            assert (data <= 1).all() is np.ma.masked or True

        missing_subs = np.where(data.mask)
        self.n_missing = missing_subs[0].shape[0]
        self.missing_data_P = np.zeros(self.n_missing, dtype=int)
        if self.n_missing > 0:
            self.missing_subs_PM = np.array(missing_subs, dtype=np.int32, order='F').T
            all_missing_MD = np.zeros_like(self.all_missing_MD)
            any_missing_MD = np.zeros_like(self.any_missing_MD)
            for m in range(self.n_modes):
                modes = tuple([m_ for m_ in range(self.n_modes) if m_ != m])
                all_missing_MD[m, :self.data_shp[m]] = np.all(data.mask, axis=modes)
                any_missing_MD[m, :self.data_shp[m]] = np.any(data.mask, axis=modes)
            self.all_missing_MD, self.any_missing_MD = all_missing_MD, any_missing_MD
            self.any_not_all_missing_MD = (1-all_missing_MD) * any_missing_MD
        else:
            self.missing_subs_PM = np.zeros((0, self.n_modes), dtype=np.int32)
            self.all_missing_MD[:] = 0
            self.any_missing_MD[:] = 0
            self.any_not_all_missing_MD[:] = 0

        filled_data = data.astype(np.int).filled(fill_value=0)
        
        nonzero_subs = filled_data.nonzero()
        self.n_nonzero = nonzero_subs[0].shape[0]
        self.nonzero_data_P = filled_data[nonzero_subs]
        if self.n_nonzero > 0:
            self.nonzero_subs_PM = np.array(nonzero_subs, dtype=np.int32, order='F').T
        else:
            self.nonzero_subs_PM = np.zeros((0, self.n_modes), dtype=np.int32)

    def fit(self, data, n_itns=1000, initialize=True, verbose=1,
            impute_after=0, schedule={}, fix_state={}, init_state={}):

        self._initialize_data(data)

        schedule = deepcopy(schedule)
        init_state = deepcopy(init_state)
        for k in fix_state.keys():
            schedule[k] = lambda x: False
            if k in init_state.keys() and verbose:
                print('WARNING: Variable appears in fix_state and init_state.')
            init_state[k] = fix_state[k]  # fix_state takes priority!

        if initialize:
            if verbose:
                print('\nINITIALIZING...\n')
            self._initialize_state(init_state)
        
        elif init_state:
            if verbose:
                print('\n Setting given states...\n')
            self.set_state(init_state)

        self.impute_after = impute_after
        if verbose:
            print('\nSTARTING INFERENCE...\n')
        self._update(n_itns=n_itns, verbose=int(verbose), schedule=schedule)

    def get_matrices(self, transpose=False):
        cdef: 
            np.npy_intp m, K_m, D_m
        
        for m, (K_m, D_m) in enumerate(zip(self.core_dims_M, self.data_dims_M)):
            mtx = self.mtx_MKD[m, :K_m, :D_m]
            yield np.transpose(mtx) if transpose else np.array(mtx)

    def set_matrices(self, matrices=[], modes=None, transpose=False):
        cdef: 
            np.npy_intp m, K_m, D_m, k, d

        if modes is None:
            modes = range(len(matrices))

        for m, mtx in zip(modes, matrices):
            if transpose:
                mtx = np.transpose(mtx)

            K_m, D_m = self.core_dims_M[m], self.data_dims_M[m]
            assert mtx.shape == (K_m, D_m)

            for k, d in np.ndindex(K_m, D_m):
                self.mtx_MKD[m, k, d] = mtx[k, d]
            self.mtx_MKD[m, K_m:, D_m:] = 0

    def reconstruct(self, subs=()):
        mtxs = list(self.get_matrices(transpose=True))
        if self.is_tucker:
            core = np.reshape(self.core_Q, self.core_shp)
            return tl.tucker_to_tensor(core, mtxs)[subs]
        else:
            return tl.kruskal_to_tensor(mtxs, weights=self.core_Q)[subs]

    def decode(self, mtx, mode, subs=()):
        mtxs = list(self.get_matrices(transpose=True))
        mtxs[mode] = mtx
        if self.is_tucker:
            core = np.reshape(self.core_Q, self.core_shp)
            return tl.tucker_to_tensor(core, mtxs)[subs]
        else:
            return tl.kruskal_to_tensor(mtxs, weights=self.core_Q)[subs]

    def get_dense_mask(self, missing_val_is=1):
        if self.n_missing == 0:
            return None

        if missing_val_is == 1:
            if self.mask is None:
                subs = tuple([self.missing_subs_PM[:, m] for m in range(self.n_modes)])
                self.mask = np.zeros(self.data_shp, dtype=np.int32)
                self.mask[subs] = 1
            return self.mask
        
        elif missing_val_is == 0:
            if self.inv_mask is None:
                subs = tuple([self.missing_subs_PM[:, m] for m in range(self.n_modes)])
                self.inv_mask = np.ones(self.data_shp, dtype=np.int32)
                self.inv_mask[subs] = 0
            return self.inv_mask

        else:
            raise ValueError('missing val must be 0 or 1')

    cdef void _generate_state(self):
        """
        Generate internal state.
        """
        for key, _, update_func in self._get_variables():
            if key not in ['Y_MKD', 'Y_Q']:
                update_func(self, update_mode=self._GENERATE_MODE)

    cdef void _generate_data(self):
        self._update_Y_PQ(update_mode=self._GENERATE_MODE)

    cdef void _update_Y_PQ(self, int update_mode):
        cdef:
            np.npy_intp p, m
            int tid, any_impute_vars
            double mu_p

        if update_mode == self._GENERATE_MODE:
            # Generate data as if came in the form of a tensor
            # The point of this is to make sure _initialize_data works
            # This works for testing even when binary=True
            data = np.ma.MaskedArray(rn.poisson(self.reconstruct()),
                                     mask=self.get_dense_mask())
            self._initialize_data(data)

        self.Y_XQ[:] = 0
        self.Y_XMKD[:] = 0

        any_impute_vars = int(any(self.impute_Y_M) or self.impute_Y_Q)
        if any_impute_vars and self._total_itns >= self.impute_after:
            # This is the imputation step; we still run this in GENERATE_MODE
            # because the code that calls _initialize_data doesnt save the masked vals.

            # We always impute missing vals BEFORE thinning observed nonzero ones because 
            # we first zero out the latent sources for modes with gamma-distributed factors
            # (which can update with the missing data marginalized out instead of imputed).

            for p in prange(self.n_missing, schedule='static', nogil=True):
                tid = self._get_thread()

                _compute_prob(self.missing_subs_PM[p],
                              self.core_dims_M,
                              self.core_Q,
                              self.mtx_MKD,
                              self.P_XMQ[tid])

                mu_p = _sum_double_vec(self.P_XMQ[tid, 0, :self.core_dims_M[0]])
                self.missing_data_P[p] = _sample_poisson(self.rngs[tid], mu_p)
                
                if self.missing_data_P[p] > 0:
                    self.N_XMQ[tid] = 0
                    
                    _allocate(self.missing_data_P[p], 
                              self.missing_subs_PM[p],
                              self.core_dims_M,
                              self.Y_XMKD[tid],
                              self.Y_XQ[tid],
                              self.P_XMQ[tid], 
                              self.N_XMQ[tid],
                              self.rngs[tid])

            if (np.array(self.missing_data_P) < 0).any():
                raise ValueError('Lambda values too large (>2e9).')
            # print(np.sum(self.missing_data_P), np.sum(self.Y_XQ))
            assert np.sum(self.missing_data_P) == np.sum(self.Y_XQ)

            # Delete any imputed sources for variables that marginalize them out.
            if not self.impute_Y_Q:
                self.Y_XQ[:] = 0

            for m in range(self.n_modes):
                if not self.impute_Y_M[m]:
                    for tid in range(self.n_threads):
                        # This inner loop is necesssary because the following assignment fails:
                        # 
                        #                   self.Y_XMKD[:, m] = 0  # doesn't work!
                        # 
                        #                   assert np.all(np.array(self.Y_XMKD[:, m]) == 0)  # fails!
                        # 
                        # For some reason, this does NOT zero out all the entries.
                        # Might have something to do with this being declared a C-continuous memview.
                        self.Y_XMKD[tid, m] = 0

        # This is where we thin the observed nonzeros. =For the binary model, this step 
        # also imputes missing count values at the observed nonzero entries.  

        # There's no reason to call this during GENERATE_MODE since we dont binarize
        # the generated count tensor in the above code that calls _initialize_data.
        for p in prange(self.n_nonzero, schedule='static', nogil=True):
            tid = self._get_thread()

            _compute_prob(self.nonzero_subs_PM[p],
                          self.core_dims_M,
                          self.core_Q,
                          self.mtx_MKD,
                          self.P_XMQ[tid])

            if (update_mode != self._GENERATE_MODE) and self.binary:
                mu_p = _sum_double_vec(self.P_XMQ[tid, 0, :self.core_dims_M[0]])
                self.nonzero_data_P[p] = _sample_trunc_poisson(self.rngs[tid], mu_p)

            if self.nonzero_data_P[p] > 0:
                self.N_XMQ[tid] = 0
                
                _allocate(self.nonzero_data_P[p], 
                          self.nonzero_subs_PM[p],
                          self.core_dims_M,
                          self.Y_XMKD[tid],
                          self.Y_XQ[tid],
                          self.P_XMQ[tid], 
                          self.N_XMQ[tid],
                          self.rngs[tid])

        # This is where we reduce thread-local arrays into single ones.
        self.Y_MKD = np.sum(self.Y_XMKD, axis=0, dtype=np.int)
        self.Y_Q = np.sum(self.Y_XQ, axis=0, dtype=np.int)

    cdef void _update_mtx_MKD(self, int update_mode):
        cdef: 
            np.npy_intp m

        for m in range(self.n_modes):
            # Padding with zeros and updating the cached sum mtx_MK
            # is handled within update_mtx_m. Make sure to do those
            # steps in any alternative method.
            self._update_mtx_m_KD(m, update_mode)
            # self.mtx_MKD[m, :, self.data_shp[m]:] = 0
        # self.mtx_MK = np.sum(self.mtx_MKD, axis=2)

    cdef void _update_mtx_m_KD(self, int mode, int update_mode):
        cdef: 
            np.npy_intp D_m, K_m, k, d, 
            double shp_kd, rte_kd, s
            gsl_rng * rng
            double[:,::1] zeta_DK
        
        D_m = self.data_dims_M[mode]
        K_m = self.core_dims_M[mode]
        self.mtx_MKD[mode] = 0

        if mode in self.mtx_is_dirichlet:
            for k in prange(K_m, schedule='static', nogil=True):
                rng = self.rngs[self._get_thread()]
                
                for d in range(D_m):
                    self.shp_MKD[mode, k, d] = self.eps
                    
                    if update_mode == self._INFER_MODE:
                        self.shp_MKD[mode, k, d] += self.Y_MKD[mode, k, d]

                _sample_dirichlet(rng, 
                                  self.shp_MKD[mode, k, :D_m], 
                                  self.mtx_MKD[mode, k, :D_m])

            assert np.all(np.array(self.mtx_MKD) >= 0)
            self.mtx_MK[mode] = 1
        
        else:
            self.mtx_MK[mode] = 0
            if update_mode == self._INITIALIZE_MODE:
                s = 1  # smoothness parameter
                for k in prange(K_m, schedule='static', nogil=True):
                    rng = self.rngs[self._get_thread()]
                    for d in range(D_m):
                        self.mtx_MKD[mode, k, d] = s * _sample_gamma(rng, s, 1./s)
                        self.mtx_MK[mode, k] += self.mtx_MKD[mode, k, d]
            else:    
                if update_mode == self._INFER_MODE:
                    zeta_DK = self._compute_zeta_m_DK(mode)

                for k in prange(K_m, schedule='static', nogil=True):
                    rng = self.rngs[self._get_thread()]

                    for d in range(D_m):
                        shp_kd = self.eps
                        rte_kd = self.eps * self.b_M[mode]
                        
                        if update_mode == self._INFER_MODE:
                            shp_kd = shp_kd + self.Y_MKD[mode, k, d]
                            rte_kd = rte_kd + zeta_DK[d, k]

                        self.mtx_MKD[mode, k, d] = _sample_gamma(rng, shp_kd, 1./rte_kd)
                        self.mtx_MK[mode, k] += self.mtx_MKD[mode, k, d]

    cdef double[:,::1] _compute_zeta_m_DK(self, int mode):

        cdef:
            np.npy_intp m, D, K
            list modes, vects, mtxs
            tuple subs
            np.ndarray core, mask, tmp
            np.ndarray[double, ndim=1] zeta_K, vals
            np.ndarray[double, ndim=2] zeta_DK, correction_DK
            np.ndarray[np.npy_intp, ndim=1] all_missing_subs_D
            np.ndarray[np.npy_intp, ndim=1] any_not_all_missing_subs_D
        
        # First compute zeta as if there are no missing entries
        modes = [m for m in range(self.n_modes) if m != mode]
        if self.is_tucker:
            # This implements a multi-mode dot. 
            # tl.tenalg.multi_mode_dot and tl.tucker_to_vec dont work.
            # They seem to be broken for more than 3 modes.
            core = np.reshape(self.core_Q, self.core_shp)
            tmp = np.rollaxis(core, mode, 0)
            for m in modes[::-1]:
                tmp = tl.dot(tmp, self.mtx_MK[m, :self.core_dims_M[m]])
            zeta_K = tmp
        else:
            zeta_K = np.array(self.core_Q)
            for m in modes:
                zeta_K *= self.mtx_MK[m]

        D, K = self.data_shp[mode], self.core_dims_M[mode]
        zeta_DK = np.ones((D, K)) * zeta_K

        # If there are missing entries that are not being imputed, calculate corrections
        if (self.n_missing > 0) and (not self.impute_Y_M[mode]):

            all_missing_subs_D = np.where(self.all_missing_MD[mode])[0]
            if all_missing_subs_D.shape[0] > 0:
                zeta_DK[all_missing_subs_D] = 0

            any_not_all_missing_subs_D = np.where(self.any_not_all_missing_MD[mode])[0]
            if any_not_all_missing_subs_D.shape[0] > 0:
                if self.is_tucker:
                    mask = np.take(self.get_dense_mask(), any_not_all_missing_subs_D, axis=mode)
                    mtxs = list(self.get_matrices(transpose=True))
                    mtxs[mode] = mtxs[mode][any_not_all_missing_subs_D]  # remove rows that have no missing entries
                                                                         # this is the main speedup beyond simply 
                                                                         # calling a tensor operation on the inverted mask
                    correction_DK = uttut(tens=mask,
                                          mode=mode, 
                                          mtxs=mtxs, 
                                          core=core, 
                                          transpose=True)
                else:
                    if self.n_missing > 2500000:  # check this heuristic!
                        mask = np.take(self.get_dense_mask(), any_not_all_missing_subs_D, axis=mode)
                        mtxs = list(self.get_matrices(transpose=True))
                        mtxs[mode] = mtxs[mode][any_not_all_missing_subs_D]  # remove rows that have no missing entries
                                                                             # this is the main speedup beyond simply 
                                                                             # calling a tensor operation on the inverted mask
                        correction_DK = uttkrp(tens=mask, 
                                               mode=mode, 
                                               mtxs=mtxs, 
                                               core=np.array(self.core_Q), 
                                               transpose=True)
                    else:
                        vals = np.ones(self.n_missing)
                        subs = tuple([self.missing_subs_PM[:, m] for m in range(self.n_modes)])
                        mtxs = list(self.get_matrices(transpose=True))
                        correction_DK = sp_uttkrp(subs=subs, 
                                                  vals=vals, 
                                                  mode=mode, 
                                                  mtxs=mtxs, 
                                                  core=np.array(self.core_Q), 
                                                  transpose=True)[any_not_all_missing_subs_D]
                zeta_DK[any_not_all_missing_subs_D] = zeta_DK[any_not_all_missing_subs_D] - correction_DK
        return zeta_DK

    @property
    def core_Q_prior(self):
        """
        Returns the prior shape and rate parameter for the core elements.

        Useful for extension classes that impose hyperiors over the core.
        """
        return self.eps, self.eps

    cdef void _update_core_Q(self, int update_mode):
        cdef:
            np.npy_intp q 
            double prior_shp, prior_rte, shp_q, rte_q
            double[::1] zeta_Q
            gsl_rng * rng

        if update_mode == self._INFER_MODE:
            zeta_Q = self._compute_zeta_Q()

        prior_shp, prior_rte = self.core_Q_prior

        for q in prange(self.n_classes, schedule='static', nogil=True):
            rng = self.rngs[self._get_thread()]
            shp_q, rte_q = prior_shp, prior_rte
            if update_mode == self._INFER_MODE:
                shp_q = shp_q + self.Y_Q[q]
                rte_q = rte_q + zeta_Q[q]

            self.core_Q[q] = _sample_gamma(rng, shp_q, 1./rte_q)

    @property
    def zeta_Q(self):
        return self._compute_zeta_Q()
    
    cdef double[::1] _compute_zeta_Q(self):
        cdef: 
            list vects, mtxs
            tuple subs
            np.ndarray mask
            np.ndarray[double, ndim=1] zeta_Q, vals

        if self.is_tucker:
            if self.n_missing == 0 or self.impute_Y_Q:
                # Compute the Kronecker product of summed arrays
                # This is faster than calling tl.tenalg.kronecker
                zeta_Q = np.take(self.mtx_MK[0], self.subs_QM[:, 0])
                for m in range(1, self.n_modes):
                    zeta_Q *= np.take(self.mtx_MK[m], self.subs_QM[:, m])
                return zeta_Q
            else:
                mtxs = list(self.get_matrices())
                mask = self.get_dense_mask(missing_val_is=0)
                return tl.tucker_to_vec(mask, mtxs)
        else:
            if self.n_missing == 0 or self.impute_Y_Q:
                return np.prod(self.mtx_MK, axis=0)

            elif self.n_missing < 15000000:  # check this heuristic!
                zeta_Q = np.prod(self.mtx_MK, axis=0)
                mtxs = list(self.get_matrices())
                vals = np.ones(self.n_missing)
                subs = tuple([self.missing_subs_PM[:, m] for m in range(self.n_modes)])
                zeta_Q -= np.sum(mtxs[0] * sp_uttkrp(subs=subs,
                                                     vals=vals, 
                                                     mode=0, 
                                                     mtxs=mtxs), axis=1)
                return zeta_Q
            else:
                mtxs = list(self.get_matrices())
                mask = self.get_dense_mask(missing_val_is=0)
                return np.sum(mtxs[0] * uttkrp(tens=mask, mode=0, mtxs=mtxs), axis=1)

    cdef void _update_b_M(self, int update_mode):
        cdef:
             np.npy_intp m
        for m in range(self.n_modes):
            if m not in self.mtx_is_dirichlet:
                self._update_b_m(m, update_mode)

    cdef void _update_b_m(self, int mode, int update_mode):
        cdef:
            double shp, rte

        shp = rte = 10.
        if update_mode == self._INFER_MODE:
            shp += self.eps * self.data_shp[mode] * self.core_dims_M[mode]
            rte += self.eps * np.sum(self.mtx_MK[mode])
        self.b_M[mode] = _sample_gamma(self.rng, shp, 1./rte)
