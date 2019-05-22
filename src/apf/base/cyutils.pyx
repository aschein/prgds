# cython: boundscheck = False
# cython: initializedcheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: language_level = 3

cpdef int prod_int_vec(int[:] arr) nogil:
    return _prod_int_vec(arr)

cpdef int sum_int_vec(int[:] arr) nogil:
    return _sum_int_vec(arr)

cpdef double sum_double_vec(double[:] arr) nogil:
    return _sum_double_vec(arr)

cpdef int sum_int_mat(int[:,:] arr) nogil:
    return _sum_int_mat(arr)

cpdef double sum_double_mat(double[:,:] arr) nogil:
    return _sum_double_mat(arr)

cpdef double dot_vec(double[:] arr1, double[:] arr2) nogil:
    return _dot_vec(arr1, arr2)

cpdef double log1pexp(double x) nogil:
    return _log1pexp(x)
