import os
import sys

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy as np
from path import Path

# if sys.platform == 'darwin':
# 	os.environ['CC'] = 'gcc-8'
# 	os.environ['CXX'] = 'g++-8'

# include_gsl_dir = '/usr/local/include/'
# lib_gsl_dir = '/usr/local/lib/'

# EXT_MODULES = [Extension(name=str(x.relpath()).replace('/', '.').split('.pyx')[0],
# 					     sources=[str(x.relpath())],
# 						 include_dirs=[np.get_include(),
# 							           include_gsl_dir,
# 							           '.'],
# 						 library_dirs=[lib_gsl_dir],
# 						 libraries=['gsl'],
# 						 extra_compile_args=['-fopenmp'],
# 						 extra_link_args=['-fopenmp'])
# 			   for x in Path('.').walkfiles('*.pyx')]

# setup(name='apf',
# 	  version='1.0',
# 	  description='Allocative Poisson Factorization (APF) framework and examples.',
# 	  author='Aaron Joseph Steriade Schein',
# 	  # packages=['base', 'models', 'tests', 'foo'],
#   	  cmdclass={"build_ext": build_ext},
#       ext_modules=cythonize(EXT_MODULES,
#       					    compiler_directives={'language_level':'3'}))  # fails if set to 3

# if sys.platform == 'darwin':
#     os.environ['CC'] = '/anaconda3/bin/gcc'
#     os.environ['CXX'] = '/anaconda3/bin/g++'

include_gsl_dir = '/usr/local/include/'
lib_gsl_dir = '/usr/local/lib/'


def get_pkgs_and_exts(dir='.'):
    """
    src/
        Makefile
        setup.py
        __init__.py
        base.pyx
        first_pkg/
            __init__.py
            core.pyx
            things/
            __init__.py
                foo.pyx
                foo.pxd
        second_pkg/
            __init__.py
            core.pyx

    get_pkgs_and_exts()
    >>> [...], [first_pkg, first_pkg.things, second_pkg]

    Inspired by:
    https://github.com/cython/cython/wiki/PackageHierarchy
    """
    pkgs = set()
    exts = list()
    for ext_path in Path(dir).walkfiles('*.pyx'):
        ext_name = ext_path.namebase

        subdirs = ext_path.splitall()[1:-1]
        if subdirs:
            pkg = '.'.join(subdirs)
            pkgs.add(pkg)
            ext_name = pkg + '.' + ext_name

        exts.append(make_extension(ext_name, ext_path))

    return list(pkgs), exts


def make_extension(ext_name, ext_path=None):
    if ext_path is None:
        ext_path = Path.joinpath(*ext_name.split('.')) + '.pyx'
    assert Path(ext_path).isfile()

    return Extension(name=ext_name,
                     sources=[ext_path],
                     include_dirs=[np.get_include(),
                                   include_gsl_dir],
                     library_dirs=[lib_gsl_dir],
                     libraries=['gsl'],
                     extra_compile_args=['-fopenmp'],
                     extra_link_args=['-fopenmp'])


pkgs, exts = get_pkgs_and_exts()


setup(name='apf',
      version='1.0',
      description='Allocative Poisson Factorization (APF) framework and examples.',
      author='Aaron Joseph Steriade Schein',
      packages=pkgs,
      ext_modules=cythonize(exts))

