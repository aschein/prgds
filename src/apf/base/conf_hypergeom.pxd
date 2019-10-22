# cython: boundscheck = False
# cython: initializedcheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: language_level = 3

import sys
import numpy as np
cimport numpy as np
from libc.math cimport exp, log, log1p, floor, lround, lgamma, sqrt, sin, M_PI, hypot, isnan, isinf, NAN, abs

from apf.base.cyutils cimport _log1pexp
from apf.base.sbch cimport mode as sbch_mode

cdef extern from "gsl/gsl_rng.h" nogil:
    ctypedef struct gsl_rng_type:
        pass
    ctypedef struct gsl_rng:
        pass

cdef extern from "gsl/gsl_randist.h" nogil:
    unsigned int gsl_ran_poisson(gsl_rng * r, double)
    double gsl_ran_exponential(gsl_rng * r, double)
    double gsl_rng_uniform(gsl_rng * r)

cdef extern from "gsl/gsl_sf_gamma.h" nogil:
    double gsl_sf_lngamma(double x)
    double gsl_sf_lnfact(unsigned int n)
    double gsl_sf_lnpoch(double a, double x)

cdef extern from "gsl/gsl_sf_hyperg.h" nogil:
    double gsl_sf_hyperg_1F1(double a, double b, double x)

# cpdef int mode(int m, double r) nogil

# cdef inline int _mode(int m, double r) nogil:
#     return int(floor(0.5 * (r + 1 + sqrt(r ** 2 + 2 * r * (2 * m - 1) + 1))))

cpdef double mean(int m, double a, double r) nogil

cdef inline double _mean(int m, double a, double r) nogil:
    if m == 0:
        return r
    else:
        return r*(m+a)/a * gsl_sf_hyperg_1F1(m+a+1, a+1, r) / gsl_sf_hyperg_1F1(m+a, a, r)

# cpdef double variance(int m, double r) nogil

# cdef inline double _variance(int m, double r) nogil:
#     cdef:
#         double mu, out, hyp12, hyp23, hyp34

#     hyp12 = gsl_sf_hyperg_1F1_int(m + 1, 2, r)
#     hyp23 = gsl_sf_hyperg_1F1_int(m + 2, 3, r)
#     hyp34 = gsl_sf_hyperg_1F1_int(m + 3, 4, r)
#     out = hyp23 + hyp34 * r * (m + 2) / 6.
#     out *= r * (m + 1) / hyp12
#     mu = _mean(m, r)
#     out += mu - mu ** 2
#     return out

cpdef double logpmf_unnorm(int n, int m, double a, double r) nogil

cdef inline double _logpmf_unnorm(int n, int m, double a, double r) nogil:
    return gsl_sf_lnpoch(n + a, m) - gsl_sf_lnfact(n) + n * log(r) 

cpdef double logpmf_norm(int m, double a, double r) nogil

cdef inline double _logpmf_norm(int m, double a, double r) nogil:
    return gsl_sf_lnpoch(a, m) + log(gsl_sf_hyperg_1F1(m + a, a, r))

cdef inline int _sample(gsl_rng * rng, 
                        int m,
                        double a,
                        double r) nogil:
    cdef:
        int mode_, n, i
        double norm_const, f_n, g_n, sum_n, cutoff

    if m == 0:
        return gsl_ran_poisson(rng, r)

    # the log normalizing constant for hyperparameters m, r
    norm_const = _logpmf_norm(m, a, r)
    if isinf(norm_const) or isnan(norm_const):
        return -1

    # the mode of the distribution
    # mode_ = _mode(m, r)
    # mode_ = lround(_mean(m, a, r))
    mode_ = sbch_mode(m, r)
    f_n = g_n = sum_n = _logpmf_unnorm(mode_, m, a, r)

    cutoff = norm_const + log(gsl_rng_uniform(rng))
    if sum_n >= cutoff:
        return mode_

    for i in range(mode_):
        n = mode_ - i - 1 
        # Instead of calling _logpmf_unnorm all over again we can 
        # re-use the last value and decrement only the difference:
        # 
        # log P(n; m, r) = log P(n-1; m, r) + log (P(n; m, r) / P(n-1; m, r) )
        # 
        f_n = _logpmf_unnorm(n, m, a, r)
        # f_n += log(n * (n + 1) / (r * (m + n)))
        
        sum_n += _log1pexp(f_n - sum_n)
        if sum_n >= cutoff:
            return n

        n = mode_ + i + 1
        g_n = _logpmf_unnorm(n, m, a, r)
        # g_n -= log((n - 1) * n / (r * (m + n - 1)))
        
        sum_n += _log1pexp(g_n - sum_n)
        if sum_n >= cutoff:
            return n

    n = 2 * mode_ + 1
    while 1:
        f_n = _logpmf_unnorm(n, m, a, r)
        # f_n += log(n * (n + 1) / (r * (m + n)))
        
        sum_n += _log1pexp(f_n - sum_n)
        if sum_n >= cutoff:
            return n

        n += 1
