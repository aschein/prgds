# prgds
Poisson-randomized gamma dynamical systems

## What's included in src:

* [apf.pyx](src/apf/base/apf.pyx): Allocation-based Poisson factorization (APF). This is the base class for Poisson tensor decomposition models with non-negative priors.
* [bessel.pyx](src/apf/base/bessel.pyx): Sampling algorithms for the Bessel distribution.
* [sbch.pyx](src/apf/base/sbch.pyx): Sampling algorithms for the size-biased confluent hypergeometric (SCH) distribution.
* [pgds.pyx](src/apf/models/pgds.pyx): Tensor generalization of Poisson--gamma dynamical systems (PGDS) of Schein et al. (2016).
* [prgds.pyx](src/apf/models/prgds.pyx): Poisson-randomized gamma dynamical systems (PrGDS).

## Dependencies:

* argparse
* numpy
* path
* pickle
* scikit-learn
* scikit-tensor
