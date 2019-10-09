# cython: boundscheck = False
# cython: initializedcheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: language_level = 3

cdef extern from "gsl/gsl_rng.h" nogil:
    ctypedef struct gsl_rng_type:
        pass
    ctypedef struct gsl_rng:
        pass
    gsl_rng_type *gsl_rng_mt19937
    void gsl_rng_free(gsl_rng * r)
    gsl_rng *gsl_rng_alloc(gsl_rng_type * T)
    void gsl_rng_set(gsl_rng * r, unsigned long int)
    unsigned long int gsl_rng_get (const gsl_rng * r)


cdef class MCMCModel(object):
    cdef:
        gsl_rng *rng
        gsl_rng **rngs
        int _total_itns, n_threads, _INFER_MODE, _GENERATE_MODE, _INITIALIZE_MODE
        dict param_list
        int[::1] thread_counts

    cdef int _get_thread(self) nogil
    cdef list _get_variables(self)
    cdef void _generate_state(self)
    cdef void _generate_data(self)
    cdef void _update_cache(self)
    cdef void _initialize_state(self, dict state=?)
    cdef void _print_state(self)
    cdef void _update(self, int n_itns, int verbose, dict schedule)
    cpdef void update(self, int n_itns, int verbose, dict schedule=?)
    cdef void _calc_funcs(self, int n, dict var_funcs, dict out)
    cpdef void set_total_itns(self, int total_itns)

    # cdef void _test(self,
    #                 int n_samples,
    #                 str method,
    #                 dict var_funcs,
    #                 dict schedule)
    # cpdef void geweke(self, int n_samples, dict var_funcs=?, dict schedule=?)
    # cpdef void schein(self, int n_samples, dict var_funcs=?, dict schedule=?)
