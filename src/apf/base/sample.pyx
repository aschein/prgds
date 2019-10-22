# cython: boundscheck = False
# cython: initializedcheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: language_level = 3

import sys
from numpy.random import randint

from apf.base.bessel cimport _sample as _sample_bessel
from apf.base.sbch cimport _sample as _sample_sbch
from apf.base.conf_hypergeom cimport _sample as _sample_conf_hypergeom

cdef class Sampler:
    """
    Wrapper for a gsl_rng object that exposes all sampling methods to Python.

    Useful for testing or writing pure Python programs.
    """
    def __init__(self, object seed=None):

        self.rng = gsl_rng_alloc(gsl_rng_mt19937)

        if seed is None:
            seed = randint(0, sys.maxsize) & 0xFFFFFFFF
        gsl_rng_set(self.rng, seed)

    def __dealloc__(self):
        """
        Free GSL random number generator.
        """

        gsl_rng_free(self.rng)

    cpdef double gamma(self, double a, double b):
        return _sample_gamma(self.rng, a, b)

    cpdef double lngamma(self, double a, double b):
        return _sample_lngamma(self.rng, a, b)

    cpdef double beta(self, double a, double b):
        return _sample_beta(self.rng, a, b)

    cpdef double lnbeta(self, double a, double b):
        return _sample_lnbeta(self.rng, a, b)

    cpdef void dirichlet(self, double[::1] alpha, double[::1] out):
        _sample_dirichlet(self.rng, alpha, out)

    cpdef void lndirichlet(self, double[::1] alpha, double[::1] out):
        _sample_lndirichlet(self.rng, alpha, out)

    cpdef int categorical(self, double[::1] dist):
        return _sample_categorical(self.rng, dist)

    cpdef int discrete(self, double[::1] dist):
        return _sample_discrete(self.rng, dist)

    cpdef int searchsorted(self, double val, double[::1] arr):
        return _searchsorted(val, arr)

    cpdef int crt(self, int m, double r):
        return _sample_crt(self.rng, m, r)

    cpdef int crt_lecam(self, int m, double r, double p_min):
        return _sample_crt_lecam(self.rng, m, r, p_min)

    cpdef int sumcrt(self, int[::1] M, double[::1] R):
        return _sample_sumcrt(self.rng, M, R)

    cpdef int sumlog(self, int n, double p):
        return _sample_sumlog(self.rng, n, p)

    cpdef int poisson(self, double mu):
        return _sample_poisson(self.rng, mu)

    cpdef int trunc_poisson(self, double mu):
        return _sample_trunc_poisson(self.rng, mu)

    cpdef void multinomial(self, unsigned int N, double[::1] p, unsigned int[::1] out):
        _sample_multinomial(self.rng, N, p, out)

    cpdef int bessel(self, double v, double a):
        return _sample_bessel(self.rng, v, a)

    cpdef int sbch(self, int m, double r):
        return _sample_sbch(self.rng, m, r)

    cpdef int conf_hypergeom(self, int m, double a, double r):
        return _sample_conf_hypergeom(self.rng, m, a, r)

    cpdef double slice_sample_gamma_shape(self,
                                          double[::1] obs,
                                          double cons_shp=1.,
                                          double cons_rte=1.,
                                          double prior_shp=0.1,
                                          double prior_rte=0.1,
                                          double x_init=1.,
                                          double x_min=1e-300,
                                          double x_max=1e300,
                                          int max_iter=1000):
        
        return _slice_sample_gamma_shape(self.rng,
                                         obs,
                                         cons_shp,
                                         cons_rte,
                                         prior_shp,
                                         prior_rte,
                                         x_init,
                                         x_min,
                                         x_max,
                                         max_iter)
