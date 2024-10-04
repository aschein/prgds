# prgds
Poisson-randomized gamma dynamical systems

## Installing
An Anaconda environment is included at [environment.yml](environment.yml). Run the following command in the home directory to create an Anaconda environment with all necessary packages and configurations:
```
conda env create -f environment.yml
```
This will create an environment called `prgds`. To activate it, type:
```
conda activate prgds
```
Then you should be able to compile the code by changing directory to the src/ folder and typing
```
cd src/
make
```

## What's included in src:

* [apf.pyx](src/apf/base/apf.pyx): Allocation-based Poisson factorization (APF). This is the base class for Poisson tensor decomposition models with non-negative priors.
* [bessel.pyx](src/apf/base/bessel.pyx): Sampling algorithms for the Bessel distribution.
* [sbch.pyx](src/apf/base/sbch.pyx): Sampling algorithms for the size-biased confluent hypergeometric (SCH) distribution.
* [pgds.pyx](src/apf/models/pgds.pyx): Tensor generalization of Poisson--gamma dynamical systems (PGDS) of Schein et al. (2016).
* [prgds.pyx](src/apf/models/prgds.pyx): Poisson-randomized gamma dynamical systems (PRGDS).
* [bpmf.pyx](src/apf/models/bpmf.pyx): Bayesian Poisson matrix factorization (BPMF).
* [bpmf_example.ipynb](src/notebooks/bpmf_example.ipynb): Example using BPMF for count or matrix factorization.

## Dependencies:
* [cython](https://cython.org/)
* [numpy](https://www.numpy.org/)
* [scipy](https://www.scipy.org/)
* [pandas](https://pandas.pydata.org/)
* [path](https://anaconda.org/anaconda/path.py)
* [scikit-learn](https://scikit-learn.org/stable/)
* [tensorly](http://tensorly.org/stable/index.html)

OSX users will have to install a version of the [GNU Compiler Collection (GCC)](https://gcc.gnu.org/). If using Anaconda,
```
conda install -c conda-forge gcc
```
and then change the relevant line in [setup.py](src/setup.py). OSX users may run into issues  compiling on MacOS 10.14 or greater which can be fixed by following the solutions in [this thread](https://stackoverflow.com/questions/52509602/cant-compile-c-program-on-a-mac-after-upgrade-to-mojave).

OSX users will also have to install the [GNU Scientific library (GSL)](https://www.gnu.org/software/gsl/doc/html/rng.html). If using Anaconda,
```
conda install -c conda-forge gsl
```

OSX users with problems compiling may find answers to [this question](https://stackoverflow.com/questions/54776301/cython-prange-is-repeating-not-parallelizing) useful.
