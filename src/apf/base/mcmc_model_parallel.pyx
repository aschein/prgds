# cython: boundscheck = False
# cython: initializedcheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: language_level = 3

import os
import shutil
import sys
import numpy as np
cimport numpy as np
import numpy.random as rn

from apf.base.pp_plot import pp_plot
from copy import deepcopy
from time import time
from contextlib import contextmanager
from libc.stdlib cimport malloc, free
from openmp cimport omp_get_max_threads, omp_get_thread_num, omp_set_num_threads


@contextmanager
def timeit_context(name):
    startTime = time()
    yield 
    elapsedTime = time() - startTime
    print('%3.4fms: %s' % (elapsedTime * 1000, name))


def exit_if(func_output, func_desc):
    if func_output is not None:
        if not func_output:
            sys.exit('Error in %s. Exiting.' % func_desc)

cdef class MCMCModel(object):
    
    def __init__(self, object seed=None, object n_threads=None):
        self._INFER_MODE = 0
        self._GENERATE_MODE = 1
        self._INITIALIZE_MODE = 2

        if n_threads is None:
            n_threads = omp_get_max_threads()
            print('Using max num of threads: %d' % n_threads)
        self.n_threads = n_threads
        omp_set_num_threads(self.n_threads)

        self.rngs = <gsl_rng**>malloc(self.n_threads * sizeof(gsl_rng))
        for t in range(self.n_threads):
            self.rngs[t] = gsl_rng_alloc(gsl_rng_mt19937)
        self.rng = self.rngs[0]

        if seed is None:
            seed = rn.randint(0, sys.maxsize) & 0xFFFFFFFF
        gsl_rng_set(self.rng, seed)
        np.random.seed(seed)

        for tid in range(1, self.n_threads):
            seed_tid = gsl_rng_get(self.rng)
            gsl_rng_set(self.rngs[tid], seed_tid)
        gsl_rng_set(self.rng, seed)  # reset the initial seed

        self.thread_counts = np.zeros(n_threads, dtype=np.int32)
        
        self.param_list = {'seed': seed}

        self._total_itns = 0

    property total_itns:
        def __get__(self):
            return self._total_itns

    cpdef void set_total_itns(self, int total_itns):
        self._total_itns = total_itns

    cdef int _get_thread(self) nogil:
        cdef:
            int tid

        tid = omp_get_thread_num()
        self.thread_counts[tid] += 1
        return tid

    def __dealloc__(self):
        """
        Free GSL random number generators.
        """
        for t in range(self.n_threads):
            gsl_rng_free(self.rngs[t])

    def get_params(self):
        """
        Get a copy of the initialization params.

        Inheriting objects should add params to the param_list, e.g.:

        cdef class ExampleModel(MCMCModel):
            
            def __init__(self, double alpha=1., object seed=None):
                
                super(ExampleModel, self).__init__(seed)
                
                self.param_list['alpha'] = alpha

                ...
        """
        return deepcopy(self.param_list)

    cdef list _get_variables(self):
        """
        Return variable names, values, and sampling methods for testing.

        Example:

        return [('foo', self.foo, self._sample_foo),
                ('bar', self.bar, self._sample_bar)]
        """
        pass

    def get_default_schedule(self):
        return {}

    def get_state(self):
        """
        Wrapper around _get_variables(...).

        Returns only the names and values of variables (not update funcs).
        """
        for key, val, update_func in self._get_variables():
            if np.isscalar(val):
                yield key, val
            else:
                yield key, np.array(val)

    def set_state(self, state):
        for key, var, _ in self._get_variables():
            if key in state.keys():
                state_var = state[key]
                assert var.shape == state_var.shape
                if np.isscalar(state_var):
                    raise NotImplementedError
                for idx in np.ndindex(var.shape):
                    var[idx] = state_var[idx]
        self._update_cache()

    cdef void _update_cache(self):
        pass

    cdef void _generate_state(self):
        """
        Generate internal state.
        """
        for key, _, update_func in self._get_variables():
            output = update_func(self, update_mode=self._GENERATE_MODE)
            exit_if(output, 'updating %s' % key)

    cdef void _generate_data(self):
        """
        Generate data given internal state.
        """
        raise NotImplementedError

    def initialize_state(self, state={}):
        self._initialize_state(state)

    cdef void _initialize_state(self, dict state={}):
        """
        Initialize internal state.
        """
        for key, val, update_func in self._get_variables():
            if key in state.keys():
                state_val = state[key]
                if np.isscalar(state_val):
                    assert NotImplementedError
                assert val.shape == state_val.shape
                for idx in np.ndindex(val.shape):
                    val[idx] = state_val[idx]
            else:
                output = update_func(self, update_mode=self._INITIALIZE_MODE)
                exit_if(output, 'updating %s' % key)

    cdef void _print_state(self):
        """
        Print internal state.
        """
        print('ITERATION %d\n' % self._total_itns)

    cdef void _update(self, int n_itns, int verbose, dict schedule):
        """
        Perform inference.
        """

        cdef:
            np.npy_intp n

        for key, _, _ in self._get_variables():
            if key not in schedule.keys():
                schedule[key] = lambda x: True

        for n in range(n_itns):
            for k, _, update_func in self._get_variables():
                if schedule[k](n):
                    if (verbose > 0) and ((n + 1) % verbose == 0):
                        with timeit_context('sampling %s' % k):
                            output = update_func(self, update_mode=self._INFER_MODE)
                            exit_if(output, 'updating %s' % k)

                    else:
                        output = update_func(self, update_mode=self._INFER_MODE)
                        exit_if(output, 'updating %s' % k)

            self._total_itns += 1
            if (verbose > 0) and ((n + 1) % verbose == 0):
                self._print_state()

    cpdef void update(self, int n_itns, int verbose, dict schedule={}):
        """
        Thin wrapper around _update(...).
        """
        self._update(n_itns, verbose, schedule)

    cdef void _calc_funcs(self,
                          int n,
                          dict var_funcs,
                          dict out):
        """
        Helper function for _test. Calculates and stores functions of variables.
        """

        for key, val, _ in self._get_variables():
            if key not in var_funcs.keys():
                continue
            if np.isscalar(val):
                out[key][n] = val
            else:
                for f, func in var_funcs[key].iteritems():
                    if f == 'Geom. Mean' and np.any(np.array(val) < 0):
                        raise RuntimeError('Negative value(s) of %s encountered.' % key)
                    out[key][f][n] = func(val)

    def _test(self, n_samples, method='geweke', var_funcs={}, schedule={}):
        cdef:
            np.npy_intp n

        default_funcs = {'Arith. Mean': np.mean,
                         'Geom. Mean': lambda x: np.exp(np.mean(np.log1p(x))),
                         'Var.': np.var,
                         'Max.': np.max}

        fwd, rev = {}, {}
        var_funcs = deepcopy(var_funcs)  # this method changes var_funcs state
        for key, val, _ in self._get_variables():

            if key not in schedule.keys():
                schedule[key] = lambda x: True

            if not any(schedule[key](n) for n in xrange(n_samples)):
                if key in var_funcs.keys():
                    del var_funcs[key]
                continue

            if key not in var_funcs.keys():
                var_funcs[key] = default_funcs
            assert len(var_funcs[key].keys()) <= 4

            if np.isscalar(val):
                fwd[key] = np.empty(n_samples)
                rev[key] = np.empty(n_samples)
            else:
                fwd[key] = {}
                rev[key] = {}
                for f in var_funcs[key]:
                    fwd[key][f] = np.empty(n_samples)
                    rev[key][f] = np.empty(n_samples)

        if method == 'schein':
            for n in range(n_samples):
                # print('generating state')
                self._generate_state()
                # print('generating data')
                self._generate_data()
                # print('calcing funcs')
                self._calc_funcs(n, var_funcs, fwd)

                # print('updating state')
                self._update(10, 0, schedule)
                # print('updating data')
                self._generate_data()
                # print('calcing funcs')
                self._calc_funcs(n, var_funcs, rev)
                if n % 500 == 0:
                    print(n)
        else:
            for n in range(n_samples):
                self._generate_state()
                self._generate_data()
                self._calc_funcs(n, var_funcs, fwd)
                if n % 500 == 0:
                    print(n)

            self._generate_state()
            for n in range(n_samples):
                self._generate_data()
                self._update(10, 0, schedule)
                self._calc_funcs(n, var_funcs, rev)
                if n % 500 == 0:
                    print(n)

        for key, _, _ in self._get_variables():
            if any(schedule[key](n) for n in xrange(n_samples)):
                pp_plot(fwd[key], rev[key], key, show=False)

    def geweke(self, n_samples, var_funcs={}, schedule={}):
        """
        Wrapper around _test(...).
        """
        self._test(n_samples=n_samples,
                   method='geweke',
                   var_funcs=var_funcs,
                   schedule=schedule)

    def schein(self, n_samples, var_funcs={}, schedule={}):
        """
        Wrapper around _test(...).
        """
        self._test(n_samples=n_samples,
                   method='schein',
                   var_funcs=var_funcs,
                   schedule=schedule)
