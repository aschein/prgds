# cython: boundscheck = False
# cython: initializedcheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: language_level = 3

import sys
import numpy as np
import numpy.random as rn
cimport numpy as np
from libc.math cimport sqrt

from apf.base.sample cimport _sample_categorical


cdef extern from "gsl/gsl_randist.h" nogil:
    void gsl_ran_multinomial(gsl_rng * r,
                             size_t K,
                             unsigned int N,
                             const double p[],
                             unsigned int n[])

cdef class Allocator:
    """
    Wrapper for a gsl_rng object that exposes all sampling methods to Python.

    Useful for testing or writing pure Python programs.
    """
    def __init__(self, object seed=None):

        self.rng = gsl_rng_alloc(gsl_rng_mt19937)

        if seed is None:
            seed = rn.randint(0, sys.maxint) & 0xFFFFFFFF
        gsl_rng_set(self.rng, seed)

    def __dealloc__(self):
        """
        Free GSL random number generator.
        """

        gsl_rng_free(self.rng)

    def comp_allocate(self, y, subs, matrices, core, P_MQ=None, N_MQ=None, Y_Q=None, Y_MKD=None):
        subs_M = np.array(subs).astype(np.int32)
        core_dims = core.shape
        core_dims_M = np.array(core_dims).astype(np.int32)
        
        if P_MQ is None:
            P_MQ = self.comp_compute_prob(subs, matrices, core)
        
        if N_MQ is None:
            M, Q = P_MQ.shape
            N_MQ = np.zeros((M, Q), dtype=np.uint32)

        if Y_Q is None:
            Q = P_MQ.shape[1]
            Y_Q = np.zeros(Q, dtype=np.int64)

        if Y_MKD is None:
            data_dims = tuple([mx.shape[1] for mx in matrices])
            K = int(np.max(core_dims))
            D = int(np.max(data_dims))
            Y_MKD = np.zeros((M, K, D), dtype=np.int64)

        _allocate(y, subs_M, core_dims_M, Y_MKD, Y_Q, P_MQ, N_MQ, self.rng)
        return Y_MKD, Y_Q

    def comp_compute_prob(self, subs, matrices, core):
        M = len(matrices)
        core_dims = core.shape
        data_dims = tuple([mx.shape[1] for mx in matrices])
        K = int(np.max(core_dims))
        D = int(np.max(data_dims))
        
        core_Q = core.ravel()
        Q = core_Q.size

        P_MQ = np.zeros((M, Q))
        mx_MKD = np.zeros((M, K, D))
        for m, (Km, Dm) in enumerate(zip(core_dims, data_dims)):
            mx_MKD[m, :Km, :Dm] = np.copy(matrices[m])

        subs_M = np.array(subs).astype(np.int32)
        core_dims_M = np.array(core_dims).astype(np.int32)
        
        _comp_compute_prob(subs_M, core_dims_M, core_Q, mx_MKD, P_MQ)
        return P_MQ

cdef void _compute_prob(int[::1] subs_M,
                        int[::1] core_dims_M,
                        double [::1] core_Q,
                        double[:,:,::1] mx_MKD,
                        double[:,::1] P_MQ) nogil:

    if core_Q.shape[0] == core_dims_M[0]:
        _cp_compute_prob(subs_M, core_Q, mx_MKD, P_MQ[0])
    else:
        comp_compute_prob(subs_M, core_dims_M, core_Q, mx_MKD, P_MQ)

cdef void _allocate(int y_p, 
                    int[::1] subs_M,
                    int[::1] core_dims_M,
                    long[:,:,::1] Y_MKD,
                    long[::1] Y_Q,
                    double[:,::1] P_MQ, 
                    unsigned int[:,::1] N_MQ,
                    gsl_rng * rng) nogil:

    if Y_Q.shape[0] == core_dims_M[0]:
        _cp_allocate(y_p, subs_M, Y_MKD, Y_Q, P_MQ[0], N_MQ[0], rng)
    else:
        if y_p < sqrt(P_MQ.shape[1]):
            _comp_token_allocate(y_p, subs_M, core_dims_M, Y_MKD, Y_Q, P_MQ, rng)
        else:
            _comp_count_allocate(y_p, subs_M, core_dims_M, Y_MKD, Y_Q, P_MQ, N_MQ, rng)

cpdef void cp_compute_prob(int[::1] subs_M,
                           double [::1] core_K,
                           double[:,:,::1] mx_MKD,
                           double[::1] P_K) nogil:

    """Thin wrapper for _cp_compute_prob"""
    _cp_compute_prob(subs_M, core_K, mx_MKD, P_K)

cdef void _cp_compute_prob(int[::1] subs_M,
                           double [::1] core_K,
                           double[:,:,::1] mx_MKD,
                           double[::1] P_K) nogil:

    cdef:
        np.npy_intp M, K, k, m, d_m

    M = subs_M.shape[0]
    K = P_K.shape[0]
    for k in range(K):
        P_K[k] = core_K[k]
        for m in range(M):
            d_m = subs_M[m]
            P_K[k] *= mx_MKD[m, k, d_m]

cpdef void comp_compute_prob(int[::1] subs_M,
                             int[::1] core_dims_M,
                             double [::1] core_Q,
                             double[:,:,::1] mx_MKD,
                             double[:,::1] P_MQ) nogil:
    """Thin wrapper for _comp_compute_prob"""
    _comp_compute_prob(subs_M, core_dims_M, core_Q, mx_MKD, P_MQ)

cdef void _comp_compute_prob(int[::1] subs_M,
                             int[::1] core_dims_M,
                             double [::1] core_Q,
                             double[:,:,::1] mx_MKD,
                             double[:,::1] P_MQ) nogil:

    cdef:
        np.npy_intp M, m, d_m, Q_m_, q_m_, q_m, K_m, k_m, offset

    P_MQ[:] = 0
    M = P_MQ.shape[0]
    Q_m_ = P_MQ.shape[1]
    for m in range(M-1, -1, -1):                   
        d_m = subs_M[m]                      # index of mth mode in the data tensor
        K_m = core_dims_M[m]                 # cardinality of mth mode of core tensor
        Q_m_ /= K_m                          # number of latent classes for previous modes
        
        for q_m_ in range(Q_m_):
            offset = q_m_ * K_m
            
            for k_m in range(K_m):
                q_m = offset + k_m
                if m == M - 1:
                    P_MQ[m, q_m] = core_Q[q_m]
                P_MQ[m, q_m] *= mx_MKD[m, k_m, d_m]
                if m > 0:
                    P_MQ[m-1, q_m_] += P_MQ[m, q_m]

cpdef void comp_compute_prob_4M(int[::1] subs_M,
                                double[:,:,:,::1] core_ABCD,
                                double[:,:,::1] mx_MKD,
                                double[::1] P_A,
                                double[:,::1] P_AB,
                                double[:,:,::1] P_ABC,
                                double[:,:,:,::1] P_ABCD) nogil:
    cdef: 
        np.npy_intp a, b, c, d

    P_A[:] = 0
    P_AB[:] = 0
    P_ABC[:] = 0
    for a in range(core_ABCD.shape[0]):
        for b in range(core_ABCD.shape[1]):
            for c in range(core_ABCD.shape[2]):
                for d in range(core_ABCD.shape[3]):
                    P_ABCD[a, b, c, d] = core_ABCD[a, b, c, d] * mx_MKD[3, d, subs_M[3]]
                    P_ABC[a, b, c] += P_ABCD[a, b, c, d]
                P_ABC[a, b, c] *= mx_MKD[2, c, subs_M[2]]
                P_AB[a, b] += P_ABC[a, b, c]
            P_AB[a, b] *= mx_MKD[1, b, subs_M[1]]
            P_A[a] += P_AB[a, b]
        P_A[a] *= mx_MKD[0, a, subs_M[0]]

cdef void _cp_allocate(int y_p, 
                       int[::1] subs_M,
                       long[:,:,::1] Y_MKD,
                       long[::1] Y_K,
                       double[::1] P_K,
                       unsigned int[::1] N_K,
                       gsl_rng * rng) nogil:
    cdef:
        np.npy_intp M, K, k, m, d_m
        double[::1] P_Km
        int[::1] N_Km

    M = subs_M.shape[0]
    K = P_K.shape[0]
    gsl_ran_multinomial(rng,
                        K,
                        y_p,
                        &P_K[0],
                        &N_K[0])
    for k in range(K):
        if N_K[k] > 0:
            Y_K[k] += N_K[k]
            for m in range(M):
                d_m = subs_M[m]
                Y_MKD[m, k, d_m] += N_K[k]

cdef void _comp_token_allocate(int y_p, 
                               int[::1] subs_M,
                               int[::1] core_dims_M,
                               long[:,:,::1] Y_MKD,
                               long[::1] Y_Q,
                               double[:,::1] P_MQ,
                               gsl_rng * rng) nogil:
    cdef:
        np.npy_intp M, _, q, d_m, K_m, k_m
        double[::1] P_Km

    M = core_dims_M.shape[0]
    for _ in range(y_p):
        q = 0
        for m in range(M):
            d_m = subs_M[m]
            K_m = core_dims_M[m]
            P_Km = P_MQ[m, q * K_m : (q + 1) * K_m]
            k_m = _sample_categorical(rng, P_Km)
            Y_MKD[m, k_m, d_m] += 1
            q = q * K_m + k_m
        Y_Q[q] += 1

cdef void _comp_count_allocate(int y_p, 
                               int[::1] subs_M,
                               int[::1] core_dims_M,
                               long[:,:,::1] Y_MKD,
                               long[::1] Y_Q,
                               double[:,::1] P_MQ, 
                               unsigned int[:,::1] N_MQ,
                               gsl_rng * rng) nogil:
    cdef:
        np.npy_intp M, m, d_m, K_m, Q_m_, q_m_, k_m, q
        int y_q_m_
        double[::1] P_Km
        unsigned int[::1] N_Km

    M = P_MQ.shape[0]               # number of tensor modes
    Q = P_MQ.shape[1]
    
    Q_m_ = 1             # number of latent classes for previous modes
    y_q_m_ = y_p         # initialize current allocated subcount to the data
    for m in range(M):
        d_m = subs_M[m]                    # index of mth mode in the data tensor
        K_m = core_dims_M[m]               # cardinality of mth mode of core tensor
        for q_m_ in range(Q_m_):
            if m > 0 :
                y_q_m_ = N_MQ[m-1, q_m_]

            if y_q_m_ > 0:

                P_Km = P_MQ[m, q_m_*K_m : (q_m_+1)*K_m]
                N_Km = N_MQ[m, q_m_*K_m : (q_m_+1)*K_m]

                gsl_ran_multinomial(rng,
                                    K_m, 
                                    y_q_m_,
                                    &P_Km[0], 
                                    &N_Km[0])
                
                for k_m in range(K_m):
                    if N_Km[k_m] > 0:
                        Y_MKD[m, k_m, d_m] += N_Km[k_m]

                        if m == M - 1:
                            q = q_m_ * K_m + k_m
                            Y_Q[q] += N_Km[k_m]
        Q_m_ *= K_m   # don't delete! 
