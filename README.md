# prgds
Poisson-randomized gamma dynamical systems

## What's included in src:

* [bptf.py](src/apf): The main code file.  Implements batch variational inference for BPTF.
* [utils.py](https://github.com/aschein/bptf/blob/master/code/utils.py): Utility functions.  Includes some important multilinear algebra functions (e.g., PARAFAC, Khatri-Rao product), preprocessing functions, and serialization functions.
* [anomaly_detection.py](https://github.com/aschein/bptf/blob/master/code/anomaly_detection.py): An example application of using BPTF for anomaly detection.

## Dependencies:

* argparse
* numpy
* path
* pickle
* scikit-learn
* scikit-tensor
