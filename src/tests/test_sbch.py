import sys
from path import Path
sys.path.append(Path(__file__).parent.parent)

import numpy as np
import numpy.random as rn
import matplotlib.pyplot as plt
import seaborn as sns

from apf.base.sample import Sampler
from apf.base.sbch import mean, mode, variance, logpmf_unnorm, logpmf_norm
from IPython import embed

s = Sampler()
def sample_sbch(m, r, size=1):
	if size == 1:
		return s.sbch(m, r)
	else:
		return np.array([s.sbch(m, r) for _ in range(size)])

def test_sbch(m, r, n_samples=10000):
	samples = sample_sbch(m, r, size=n_samples)
	assert (samples > 0).all()

	print('%d: empirical mode' % np.bincount(samples).argmax())
	print('%d: theoretical mode' % mode(m, r))

	print('\n%f: empirical mean' % samples.mean())
	print('%f: theoretical mean' % mean(m, r))

	print('\n%f: empirical variance' % samples.var())
	print('%f: theoretical variance\n' % variance(m, r))

def alt_geweke_test(theta, c, size=1000, n_itns=5):
	forward = {'geom': [], 'arith': [], 'var': [], 'sparsity': []}
	backward = {'geom': [], 'arith': [], 'var': [], 'sparsity': []}

	# FORWARD SAMPLES
	n = rn.poisson(theta, size=size)
	forward['geom'].append(np.exp(np.log1p(n).mean()))
	forward['arith'].append(n.mean())
	forward['var'].append(n.var())
	forward['sparsity'].append(np.count_nonzero(n) / float(n.size))

	lam = np.zeros(size)
	lam[n > 0] = rn.gamma(n[n > 0], 1. / c)

	y = np.zeros(size, dtype=int)
	y[lam > 0] = rn.poisson(lam[lam > 0])

	p = 1./(1 + c)
	
	for _ in range(n_itns):
		# BACKWARD SAMPLES
		n[y == 0] = rn.poisson((1-p) * theta, size=(y==0).sum())
		for idx in np.where(y > 0)[0]:
			n[idx] = s.sbch(y[idx], (1-p) * theta)

		lam[:] = 0
		lam[n > 0] = rn.gamma(n[n > 0], 1. / c)

		y[:] = 0
		y[lam > 0] = rn.poisson(lam[lam > 0])

	backward['geom'].append(np.exp(np.log1p(n).mean()))
	backward['arith'].append(n.mean())
	backward['var'].append(n.var())
	backward['sparsity'].append(np.count_nonzero(n) / float(n.size))
	return forward, backward

# test_sbch(m=10, r=10)
alt_geweke_test(10, 1.)

embed()
