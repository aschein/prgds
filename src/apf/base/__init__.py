"""
The numpy import below is necessary. 

Without it, I get an error when trying to
import from a module in base:

ImportError: dlopen(/Users/aaronschein/Documents/dummy_apf/src/apf/models/prgds.cpython-37m-darwin.so, 2): Symbol not found: _cblas_caxpy
  Referenced from: /usr/local/opt/gsl/lib/libgsl.23.dylib
  Expected in: flat namespace
 in /usr/local/opt/gsl/lib/libgsl.23.dylib

Clearly some library is not being linked correctly.
Importing numpy fixes that.
"""
import numpy as np

