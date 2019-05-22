# cython: boundscheck = False
# cython: initializedcheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: language_level = 3

import sys
import numpy as np
import numpy.random as rn
cimport numpy as np

cdef extern from "gsl/gsl_rng.h" nogil:
    ctypedef struct gsl_rng_type:
        pass
    ctypedef struct gsl_rng:
        pass
    gsl_rng_type *gsl_rng_mt19937
    gsl_rng *gsl_rng_alloc(gsl_rng_type * T)
    void gsl_rng_set(gsl_rng * r, unsigned long int)
    void gsl_rng_free(gsl_rng * r)

cdef void _compute_prob(int[::1] subs_M,
                        int[::1] core_dims_M,
                        double [::1] core_Q,
                        double[:,:,::1] mx_MKD,
                        double[:,::1] P_MQ) nogil

cdef void _allocate(int y_p, 
                    int[::1] subs_M,
                    int[::1] core_dims_M,
                    long[:,:,::1] Y_MKD,
                    long[::1] Y_Q,
                    double[:,::1] P_MQ, 
                    unsigned int[:,::1] N_MQ,
                    gsl_rng * rng) nogil

cdef void _cp_compute_prob(int[::1] subs_M,
                           double [::1] core_K,
                           double[:,:,::1] mx_MKD,
                           double[::1] P_K) nogil

cpdef void cp_compute_prob(int[::1] subs_M,
                           double [::1] core_K,
                           double[:,:,::1] mx_MKD,
                           double[::1] P_K) nogil

cdef void _comp_compute_prob(int[::1] subs_M,
                             int[::1] core_dims_M,
                             double [::1] core_Q,
                             double[:,:,::1] mx_MKD,
                             double[:,::1] P_MQ) nogil

cpdef void comp_compute_prob(int[::1] subs_M,
                             int[::1] core_dims_M,
                             double [::1] core_Q,
                             double[:,:,::1] mx_MKD,
                             double[:,::1] P_MQ) nogil

cdef void _cp_allocate(int y_p, 
                       int[::1] subs_M,
                       long[:,:,::1] Y_MKD,
                       long[::1] Y_K,
                       double[::1] P_K,
                       unsigned int[::1] N_K,
                       gsl_rng * rng) nogil

cdef void _comp_count_allocate(int y_p, 
                               int[::1] subs_M,
                               int[::1] core_dims_M,
                               long[:,:,::1] Y_MKD,
                               long[::1] Y_Q,
                               double[:,::1] P_MQ, 
                               unsigned int[:,::1] N_MQ,
                               gsl_rng * rng) nogil

cdef void _comp_token_allocate(int y_p, 
                               int[::1] subs_M,
                               int[::1] core_dims_M,
                               long[:,:,::1] Y_MKD,
                               long[::1] Y_Q,
                               double[:,::1] P_MQ, 
                               gsl_rng * rng) nogil

cdef class Allocator:
    """
    Wrapper for a gsl_rng object that exposes all sampling methods to Python.

    Useful for testing or writing pure Python programs.
    """
    cdef:
        gsl_rng * rng

