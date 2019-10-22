# cython: boundscheck = False
# cython: initializedcheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: language_level = 3

cdef extern from "gsl/gsl_errno.h" nogil:
    ctypedef struct gsl_error_handler_t:
        pass
    gsl_error_handler_t * gsl_set_error_handler_off()
gsl_set_error_handler_off()

cpdef double logpmf_norm(int m, double a, double r) nogil:
    return _logpmf_norm(m, a, r)

cpdef double logpmf_unnorm(int n, int m, double a, double r) nogil:
    return _logpmf_unnorm(n, m, a, r)

# cpdef int mode(int m, double r) nogil:
#     return _mode(m, r)

cpdef double mean(int m, double a, double r) nogil:
    return _mean(m, a, r)

# cpdef double variance(int m, double r) nogil:
#     return _variance(m, r)