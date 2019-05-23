import sys
from path import Path
sys.path.append(Path(__file__).parent.parent)

import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
import seaborn as sns

from apf.base.sample import Sampler
from apf.base.conf_hypergeom import mean, logpmf_unnorm, logpmf_norm
from IPython import embed

s = Sampler()
def sample_conf_hypergeom(m, a, r, size=1):
	if size == 1:
		return s.conf_hypergeom(m, a, r)
	else:
		return np.array([s.conf_hypergeom(m, a, r) for _ in range(size)])

def test_conf_hypergeom(m, a, r, n_samples=10000):
	samples = sample_conf_hypergeom(m, a, r, size=n_samples)
	assert (samples >= 0).all()

	print('\n%f: empirical mean' % samples.mean())
	print('%f: theoretical mean' % mean(m, a, r))

	print('\n%d: empirical mode' % np.bincount(samples).argmax())
	# print('%d: theoretical mode' % mode(m, r))

	print('\n%f: empirical variance' % samples.var())
	# print('%f: theoretical variance\n' % variance(m, r))

test_conf_hypergeom(m=10, a=1, r=1)

embed()
