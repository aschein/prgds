# cython: boundscheck = False
# cython: initializedcheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: language_level = 3
import sys
import numpy as np
cimport numpy as np
from libc.math cimport fabs, exp, log, sqrt

cdef extern from "gsl/gsl_errno.h" nogil:
    ctypedef struct gsl_error_handler_t:
        pass
    gsl_error_handler_t * gsl_set_error_handler_off()
gsl_set_error_handler_off()

cpdef double gsl_kv(double v, double a) nogil:
    return gsl_sf_bessel_Knu(v, a)

cpdef double gsl_iv(double v, double a) nogil:
    return gsl_sf_bessel_Inu(v, a)

cpdef double pmf_unnorm(int y, double v, double a) nogil:
    return _pmf_unnorm(y, v, a)

cpdef double logpmf_unnorm(int y, double v, double a) nogil:
    return _logpmf_unnorm(y, v, a)

cpdef double pmf_norm(double v, double a) nogil:
    return _pmf_norm(v, a)

cpdef double logpmf_norm(double v, double a) nogil:
    return _logpmf_norm(v, a)

cpdef void test_logpmf_norm(double v, double a):
    if isnan(_logpmf_norm(v, a)):
        print('NAN')

    if isinf(_logpmf_norm(v, a)):
        print('INF')

cpdef int mode(double v, double a) nogil:
    return _mode(v, a)

cpdef double mean(double v, double a) nogil:
    return _mean(v, a)

cpdef double mean_naive(double v, double a) nogil:
    return _mean_naive(v, a)

cpdef double variance(double v, double a) nogil:
    return _variance(v, a)

cpdef double quotient(double v, double a) nogil:
    return _quotient(v, a)

cpdef void run_sample(double v, double a, int seed, int[::1] out) nogil:
    
    cdef:
        gsl_rng *rng
        np.npy_intp n, N

    rng = gsl_rng_alloc(gsl_rng_mt19937)
    gsl_rng_set(rng, seed)

    N = out.shape[0]

    for n in range(N):
        out[n] = _sample(rng, v, a)

    gsl_rng_free(rng)


DEF CACHE_SIZE = 4

cpdef void run_sample_from_pmf(double v, double a, int seed, int[::1] out) nogil:
    
    cdef:
        gsl_rng *rng
        np.npy_intp n, N
        double[CACHE_SIZE] cache

    cache[0] = NAN

    rng = gsl_rng_alloc(gsl_rng_mt19937)
    gsl_rng_set(rng, seed)

    N = out.shape[0]

    for n in range(N):
        out[n] = _sample_from_pmf(rng, v, a, cache)

    gsl_rng_free(rng)

cpdef unsigned int test_log_pmf_update(double v, double a, double tol) nogil:
    cdef:
        double z, c, f, g
        unsigned int y
        int m
        np.npy_intp i

    z = _logpmf_norm(v, a)
    y = m = _mode(v, a)
    f = g = _logpmf_unnorm(y, v, a)

    c = 2 * log(a) - log(4)

    for i in range(m):
        y = m - i - 1
        g += log(y + 1) + log(y + v + 1) - c
        if fabs(_logpmf_unnorm(y, v, a) - g) > tol:
            return 0

        y = m + i + 1
        f += c - log(y) - log(y + v)
        if fabs(_logpmf_unnorm(y, v, a) - f) > tol:
            return 0

    y = 2 * m + 1
    for i in range(10000):
        f += c - log(y) - log(y + v)
        if fabs(_logpmf_unnorm(y, v, a) - f) > tol:
            return 0
        else:
            y += 1

    return 1 


cpdef double expected_iter_rejection_1(double v, double a) nogil:
    cdef:
        double z, fm, pm
        int m

    z = _logpmf_norm(v, a)
    m = _mode(v, a)
    fm = _logpmf_unnorm(m, v, a)
    pm = exp(fm - z)
    return 4 + pm

cpdef void run_rejection_1(double v, double a, int seed, int[::1] out) nogil:
    
    cdef:
        gsl_rng *rng
        np.npy_intp N, n
        double[CACHE_SIZE] cache

    cache[0] = NAN

    rng = gsl_rng_alloc(gsl_rng_mt19937)
    gsl_rng_set(rng, seed)

    N = out.shape[0]

    for n in range(N):
        out[n] = _rejection_1(rng, v, a, cache)

    gsl_rng_free(rng)


cpdef double expected_iter_rejection_2(double v, double a) nogil:
    cdef:
        double z, fm, pm, s, q
        int m

    m = _mode(v, a)
    z = _logpmf_norm(v, a)
    fm = _logpmf_unnorm(m, v, a)
    pm = exp(fm - z)

    s = sqrt(_variance(v, a))
    q = 1. / (s * sqrt(648))
    if q > (1. / 3):
        q = (1. / 3)

    return pm * (1 + 4. / q)

cpdef void run_rejection_2(double v, double a, int seed, int[::1] out) nogil:
    
    cdef:
        gsl_rng *rng
        np.npy_intp N, n
        double[CACHE_SIZE] cache

    cache[0] = NAN

    rng = gsl_rng_alloc(gsl_rng_mt19937)
    gsl_rng_set(rng, seed)

    N = out.shape[0]

    for n in range(N):
        out[n] = _rejection_2(rng, v, a, cache)

    gsl_rng_free(rng)


cpdef double expected_iter_rejection_3(double v, double a) nogil:
    cdef:
        double z, fm, pm, s, q
        int m

    m = _mode(v, a)
    z = _logpmf_norm(v, a)
    fm = _logpmf_unnorm(m, v, a)
    pm = exp(fm - z)

    s = sqrt(_variance_bound(v, a))
    q = 1. / (s * sqrt(648))
    if q > (1. / 3):
        q = (1. / 3)

    return pm * (1 + 4. / q)

cpdef void run_rejection_3(double v, double a, int seed, int[::1] out) nogil:
    
    cdef:
        gsl_rng *rng
        np.npy_intp N, n
        double[CACHE_SIZE] cache

    cache[0] = NAN

    rng = gsl_rng_alloc(gsl_rng_mt19937)
    gsl_rng_set(rng, seed)

    N = out.shape[0]

    for n in range(N):
        out[n] = _rejection_3(rng, v, a, cache)

    gsl_rng_free(rng)


cpdef double expected_iter_double_poisson(double v, double a) nogil:
    return exp(a - _logpmf_norm(v, a))

cpdef void run_double_poisson(int v, double a, int seed, int[::1] out) nogil:
    
    cdef:
        gsl_rng *rng
        np.npy_intp  N, n

    rng = gsl_rng_alloc(gsl_rng_mt19937)
    gsl_rng_set(rng, seed)

    N = out.shape[0]

    for n in range(N):
        out[n] = _double_poisson(rng, v, a)

    gsl_rng_free(rng)


cpdef void run_normal_approx(double v, double a, int seed, int[::1] out) nogil:
    
    cdef:
        gsl_rng *rng
        np.npy_intp N, n

    rng = gsl_rng_alloc(gsl_rng_mt19937)
    gsl_rng_set(rng, seed)

    N = out.shape[0]

    for n in range(N):
        out[n] = _normal_approx(rng, v, a)

    gsl_rng_free(rng)
