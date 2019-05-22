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

cdef extern from "gsl/gsl_rng.h" nogil:
    ctypedef struct gsl_rng_type:
        pass
    ctypedef struct gsl_rng:
        pass
    gsl_rng_type *gsl_rng_mt19937
    gsl_rng *gsl_rng_alloc(gsl_rng_type * T)
    void gsl_rng_set(gsl_rng * r, unsigned long int)
    void gsl_rng_free(gsl_rng * r)

cdef extern from "gsl/gsl_randist.h" nogil:
    double gsl_ran_gaussian(gsl_rng * r, double)
    unsigned int gsl_ran_poisson(gsl_rng * r, double)
    void gsl_ran_poisson_array(gsl_rng * r, size_t n, unsigned int array[], double mu)
    double gsl_ran_exponential(gsl_rng * r, double)
    unsigned int gsl_ran_bernoulli(gsl_rng * r, double)
    double gsl_rng_uniform(gsl_rng * r)

cdef extern from "gsl/gsl_sf_bessel.h" nogil:
     double gsl_sf_bessel_Inu(double, double)
     double gsl_sf_bessel_Inu_scaled(double, double)
     double gsl_sf_bessel_Knu(double, double)
     double gsl_sf_bessel_Knu_scaled (double, double)
     double gsl_sf_bessel_lnKnu (double, double)

cpdef double gsl_kv(double v, double a) nogil

cpdef double gsl_iv(double v, double a) nogil

cpdef double pmf_unnorm(int y, double v, double a) nogil

cdef inline double _pmf_unnorm(int y, double v, double a) nogil:
    return exp(_logpmf_unnorm(y, v, a))

cpdef double logpmf_unnorm(int y, double v, double a) nogil

cdef inline double _logpmf_unnorm(int y, double v, double a) nogil:
    return (2 * y + v) * log(a / 2) - lgamma(y + 1) - lgamma(y + v + 1)

cpdef double pmf_norm(double v, double a) nogil

cdef inline double _pmf_norm(double v, double a) nogil:
    return exp(_logpmf_norm(v, a))
    
cdef inline double TOPI = 2. / M_PI

cpdef double logpmf_norm(double v, double a) nogil

cpdef void test_logpmf_norm(double v, double a)

cdef inline double _logpmf_norm(double v, double a) nogil:
    if v >= 0:
        return a + log(gsl_sf_bessel_Inu_scaled(v, a))
    else:
        return log(gsl_sf_bessel_Inu(-v, a) + 2/M_PI * sin(-M_PI*v) * gsl_sf_bessel_Knu(-v, a))

cpdef int mode(double v, double a) nogil

cdef inline int _mode(double v, double a) nogil:
    return <int> floor(0.5 * (hypot(v, a) - v))

cpdef double mean(double v, double a) nogil

cdef inline double _mean(double v, double a) nogil:
    return 0.5 * a * _quotient(v, a)

cpdef double mean_naive(double v, double a) nogil

cdef inline double _mean_naive(double v, double a) nogil:
    return 0.5 * a * _quotient_naive(v, a)

cpdef double variance(double v, double a) nogil

cdef inline double _variance(double v, double a) nogil:
    cdef:
        double q1, q2, c, mu

    q1 = _quotient(v, a)
    q2 = _quotient(v + 1, a)
    c = 0.5 * a
    mu = c * q1
    return mu + c * mu * (q2 - q1)

cdef inline double _variance_bound(double v, double a) nogil:
    cdef:
        double A, B, C, D

    A = hypot(a, v)
    B = hypot(a, v + 1)
    C = a ** 2 / (2 * (v + A))
    D = (1 + (a ** 2 * (1 + B - A)) / (2 * (v + A) * (v + 1 + B))) ** 2
    return C + D

cpdef double quotient(double v, double a) nogil

cdef inline double _quotient_naive(double v, double a) nogil:
    return exp(_logpmf_norm(v + 1, a) - _logpmf_norm(v, a))

DEF TABLE_SIZE = 6

cdef inline double _quotient(double v, double a) nogil:
    """Implements the algorithm for computing the Bessel quotient described in:

    Amos, D. E. (1974). Computation of Modified Bessel Functions and 
    Their Ratios. Mathematics of Computation, 28(125).  See page 245 (Figure 1).

    This algorithm does not require evaluation of Bessel or Gamma functions.

    Inaccurate for (v < 0 and a < 1).
    """
    cdef:
        double R
        np.npy_intp k, m
        double[TABLE_SIZE][TABLE_SIZE] TABLE

    if v < 0 and a < 1:
        return _quotient_naive(v, a)

    for k in range(TABLE_SIZE):
        TABLE[0][k] = a / (v + k + 0.5 + hypot(v + k + 1.5, a))

    for m in range(TABLE_SIZE-1):
        for k in range(TABLE_SIZE-1-m):
            R = TABLE[m][k + 1] / TABLE[m][k]
            TABLE[m + 1][k] = a / (v + k + 1 + hypot(v + k + 1, a * sqrt(R)))

    return TABLE[TABLE_SIZE-1][0]


DEF CACHE_SIZE = 4

cpdef void run_sample(double v, double a, int seed, int[::1] out) nogil

cdef inline int _sample(gsl_rng * rng, double v, double a) nogil:
    cdef:
        double z, fm, pm
        int y, m
        double[CACHE_SIZE] cache

    cache[0] = NAN

    if (v == 0 and a <= 10) or (v == 1 and 4.5 <= a <= 9.5):
        return _double_poisson(rng, <int> v, a)

    elif (v < 0) and (a < 700):
        return _rejection_1(rng, v, a, cache)

    else:
        m = _mode(v, a)
        cache[0] = <double> m
        fm = cache[1] = _logpmf_unnorm(m, v, a)
        z = cache[2] = _logpmf_norm(v, a)

        if (isinf(z) or isnan(z)) or (a > 700):
            return _rejection_2(rng, v, a, cache)

        if (a < 40) or (a < v and v <= 70):
            y = _sample_from_pmf(rng, v, a, cache)
            if y >= 0:
                return y
            else:
                return _rejection_1(rng, v, a, cache)

        pm = cache[3] = exp(fm - z)

        if pm >= 0.115:
            y = _sample_from_pmf(rng, v, a, cache)
            if y >= 0:
                return y
            else:
                return _rejection_1(rng, v, a, cache)
        else:
            return _rejection_1(rng, v, a, cache)


DEF TOL = 1e-10

cpdef void run_sample_from_pmf(double v, double a, int seed, int[::1] out) nogil

cdef inline int _sample_from_pmf(gsl_rng * rng,
                                 double v,
                                 double a,
                                 double[CACHE_SIZE] cache) nogil:
    cdef:
        double z, u, r, s, c, f, g, p
        int m, y
        np.npy_intp i

    if isnan(cache[0]):
        z = _logpmf_norm(v, a)
        m = _mode(v, a)
        f = g = s = _logpmf_unnorm(m, v, a)
    else:
        m = <int> cache[0] 
        f = g = s = cache[1]
        z = cache[2] 

    if isinf(z) or isnan(z):
        return -1

    u = log(gsl_rng_uniform(rng))
    r = z + u

    if s >= r:
        return int(m)

    c = 2 * log(a) - log(4)

    for i in range(m):
        y = m - i - 1
        g += log(y + 1) + log(y + v + 1) - c
        p = _log1pexp(g - s)
        s += p
        if s >= r:
            return y
        if p < TOL:
            return -1

        y = m + i + 1
        f += c - log(y) - log(y + v)
        p = _log1pexp(f - s)
        s += p
        if s >= r:
            return y
        if p < TOL:
            return -1

    y = 2 * m + 1
    while 1:
        f += c - log(y) - log(y + v)
        p = _log1pexp(f - s)
        s += p
        if s >= r:
            return y
        if p < TOL:
            return -1
        y += 1

cpdef unsigned int test_log_pmf_update(double v, double a, double tol) nogil


cpdef double expected_iter_rejection_1(double v, double a) nogil

cpdef void run_rejection_1(double v, double a, int seed, int[::1] out) nogil

cdef inline int _rejection_1(gsl_rng * rng,
                             double v,
                             double a,
                             double[CACHE_SIZE] cache) nogil:
    cdef:
        double z, fm, pm, w, u1, u2, u3, y, e, r, foo, fmx
        unsigned int b
        int m, x, out

    if isnan(cache[0]):
        z = _logpmf_norm(v, a)
        m = _mode(v, a)
        fm = _logpmf_unnorm(m, v, a)
        pm = exp(fm - z)
    else:
        m = <int> cache[0]
        fm = cache[1]
        z = cache[2]
        pm = cache[3]

    if isinf(z) or isnan(z):
        return -1

    w = 1 + 0.5 * pm
    while 1:
        u1 = gsl_rng_uniform(rng)
        if u1 <= (w / (1 + w)):
            u2 = gsl_rng_uniform(rng)
            y = u2 * w / pm
        else:
            e = gsl_ran_exponential(rng, 1.)
            y = (w + e) / pm
        x = lround(y)
        b = gsl_ran_bernoulli(rng, 0.5)
        if b == 0:
            x *= -1
        out = m + x
        if out >= 0:
            u3 = gsl_rng_uniform(rng)
            foo = w - pm * y
            if foo < 0:
                foo += log(u3)
            else:
                foo = log(u3)
            fmx = _logpmf_unnorm(out, v, a)
            r = fmx - fm
            if foo <= r:
                return out


cpdef double expected_iter_rejection_2(double v, double a) nogil

cpdef void run_rejection_2(double v, double a, int seed, int[::1] out) nogil

cdef inline int _rejection_2(gsl_rng * rng,
                             double v,
                             double a,
                             double[CACHE_SIZE] cache) nogil:
    cdef:
        double s, q, u1, u2, e, y, tmp
        unsigned int b
        int m, x, out

    if isnan(cache[0]):
        m = _mode(v, a)
        fm = _logpmf_unnorm(m, v, a)
    else:
        m = <int> cache[0]
        fm = cache[1]

    q = sqrt(_variance(v, a)) * sqrt(648)
    if q < 3:
        q = 3

    while 1:
        u1 = gsl_rng_uniform(rng)
        if u1 < ((1. + 2. * q) / (1. + 4. * q)):
            u2 = gsl_rng_uniform(rng)
            y = u2 * (0.5 + q)
        else:
            e = gsl_ran_exponential(rng, 1.)
            y = 0.5 + q * (1 + e)
        x = lround(y)
        b = gsl_ran_bernoulli(rng, 0.5)
        if b == 0:
            x *= -1
        out = m + x
        if out >= 0:
            fmx = _logpmf_unnorm(out, v, a)
            tmp = 1 + (0.5 - y) / q
            if tmp > 0:
                tmp = 0
            u3 = log(gsl_rng_uniform(rng))
            if (u3 + tmp) <= (fmx - fm):
                return out


cpdef double expected_iter_rejection_3(double v, double a) nogil

cpdef void run_rejection_3(double v, double a, int seed, int[::1] out) nogil

cdef inline int _rejection_3(gsl_rng * rng,
                             double v,
                             double a,
                             double[CACHE_SIZE] cache) nogil:
    cdef:
        double s, q, u1, u2, e, y, tmp
        unsigned int b
        int m, x, out

    if isnan(cache[0]):
        m = _mode(v, a)
        fm = _logpmf_unnorm(m, v, a)
    else:
        m = <int> cache[0]
        fm = cache[1]

    q = sqrt(_variance_bound(v, a)) * sqrt(648)
    if q < 3:
        q = 3

    while 1:
        u1 = gsl_rng_uniform(rng)
        if u1 < ((1. + 2. * q) / (1. + 4. * q)):
            u2 = gsl_rng_uniform(rng)
            y = u2 * (0.5 + q)
        else:
            e = gsl_ran_exponential(rng, 1.)
            y = 0.5 + q * (1 + e)
        x = lround(y)
        b = gsl_ran_bernoulli(rng, 0.5)
        if b == 0:
            x *= -1
        out = m + x
        if out >= 0:
            fmx = _logpmf_unnorm(out, v, a)
            tmp = 1 + (0.5 - y) / q
            if tmp > 0:
                tmp = 0
            u3 = log(gsl_rng_uniform(rng))
            if (u3 + tmp) <= (fmx - fm):
                return out


cpdef double expected_iter_double_poisson(double v, double a) nogil

cpdef void run_double_poisson(int v, double a, int seed, int[::1] out) nogil

# cdef inline unsigned int _double_poisson(gsl_rng * rng, int v, double a) nogil:
#     cdef:
#         double lam
#         unsigned int x, y

#     lam = a / 2.
    
#     while 1:
#         x = gsl_ran_poisson(rng, lam)
#         y = gsl_ran_poisson(rng, lam)
#         if v == (x - y):
#             return y
            
cdef inline unsigned int _double_poisson(gsl_rng * rng, int v, double a) nogil:
    cdef:
        double lam
        unsigned int x, y
        int diff

    lam = a / 2.
    
    while 1:
        x = gsl_ran_poisson(rng, lam)
        y = gsl_ran_poisson(rng, lam)
        diff = x - y
        
        if diff == v:
            return y

        if diff == -v:
            return x

cpdef void run_normal_approx(double v, double a, int seed, int[::1] out) nogil
 
cdef inline int _normal_approx(gsl_rng * rng, double v, double a) nogil:
    cdef:
        double mu, sig, x

    mu = _mean(v, a)
    sig = _variance(v, a)
    x = gsl_ran_gaussian(rng, sig)
    if x < 0:
        return lround(mu)
    else:
        return lround(x + mu)
