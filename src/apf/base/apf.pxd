# cython: boundscheck = False
# cython: initializedcheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: language_level = 3

from apf.base.mcmc_model_parallel cimport MCMCModel

cdef class APF(MCMCModel):
    cdef:
        tuple data_shp, core_shp
        int n_modes, n_classes, n_nonzero, n_missing, impute_after
        int max_data_dim, max_core_dim, binary, is_tucker, impute_Y_Q
        double eps
        list mtx_is_dirichlet
        double[::1] core_Q, b_M
        double[:,::1] mtx_MK
        double[:,:,::1] mtx_MKD, P_XMQ, shp_MKD
        int[::1] data_dims_M, core_dims_M, impute_Y_M
        int[:,::1] nonzero_subs_PM, missing_subs_PM, subs_QM
        int[:,::1] any_missing_MD, all_missing_MD, any_not_all_missing_MD
        long[::1] Y_Q, nonzero_data_P, missing_data_P
        long[:,::1] Y_XQ
        long[:,:,::1] Y_MKD
        long[:,:,:,::1] Y_XMKD
        unsigned int[:,:,::1] N_XMQ
        object mask, inv_mask

    cdef void _dummy_update(self, int update_mode) nogil
    cdef void _update_Y_PQ(self, int update_mode)
    # cdef void _reduce_sources(self)
    cdef void _update_b_m(self, int mode, int update_mode)
    cdef void _update_b_M(self, int update_mode)
    cdef void _update_mtx_m_KD(self, int mode, int update_mode)
    cdef void _update_mtx_MKD(self, int update_mode)
    cdef double[:,::1] _compute_zeta_m_DK(self, int mode)
    cdef void _update_core_Q(self, int update_mode)
    cdef double[::1] _compute_zeta_Q(self)