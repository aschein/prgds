import sys
from path import Path
sys.path.append(Path(__file__).parent.parent)
from base.allocate import Allocator, comp_compute_prob, cp_compute_prob, comp_compute_prob_4M

import numpy as np
from time import time

core_dims = (3, 5, 4)
data_dims = (10, 11, 12)
matrices = [np.ones((K, D)) for K, D in zip(core_dims, data_dims)]
M = len(matrices)

core = np.random.gamma(1, 1,size=core_dims)
core_Q = core.ravel()
Q = core_Q.size

K = int(np.max(core_dims))
D = int(np.max(data_dims))
mx_MKD = np.zeros((M, K, D))
for m, (Km, Dm) in enumerate(zip(core_dims, data_dims)):
    mx_MKD[m, :Km, :Dm] = np.copy(matrices[m])

a = Allocator()
subs = (1,1,0)

subs_M = np.array(subs).astype(np.int32)
core_dims_M = np.array(core_dims).astype(np.int32)
P_MQ = np.zeros((M, Q))
comp_compute_prob(subs_M, core_dims_M, core_Q, mx_MKD, P_MQ)

tmp = np.einsum('z,zyx->zyx', matrices[0][:, subs[0]], core)
tmp = np.einsum('y,zyx->zyx', matrices[1][:, subs[1]], tmp)
tmp = np.einsum('x,zyx->zyx', matrices[2][:, subs[2]], tmp)

assert np.allclose(P_MQ[2], tmp.ravel())

Y_MKD, Y_Q = a.comp_allocate(10000, subs, matrices, core)
assert np.allclose(*Y_MKD.sum(axis=(1,2)))
print Y_Q.sum(), Y_MKD[0].sum()
assert Y_Q.sum() == Y_MKD[0].sum()

# Bigger data test
core_dims = (20, 20, 6, 3)
data_dims = (400, 400, 20, 52)
matrices = [np.ones((K, D)) for K, D in zip(core_dims, data_dims)]
M = len(matrices)

core = np.random.gamma(1, 1,size=core_dims)
core_Q = core.ravel()
Q = core_Q.size

K = int(np.max(core_dims))
D = int(np.max(data_dims))
mx_MKD = np.zeros((M, K, D))
for m, (Km, Dm) in enumerate(zip(core_dims, data_dims)):
    mx_MKD[m, :Km, :Dm] = np.copy(matrices[m])

a = Allocator()
subs = (1,1,0,2,0)

subs_M = np.array(subs).astype(np.int32)
core_dims_M = np.array(core_dims).astype(np.int32)
P_MQ = np.zeros((M, Q))

n_trials = 100
s = time()
for _ in range(n_trials):
	comp_compute_prob(subs_M, core_dims_M, core_Q, mx_MKD, P_MQ)
print '%fs: compute prob' % ((time() - s) / n_trials)

P_A = np.zeros(core_dims[:-3])
P_AB = np.zeros(core_dims[:-2])
P_ABC = np.zeros(core_dims[:-1])
P_ABCD = np.zeros(core_dims)
s = time()
for _ in range(n_trials):
	comp_compute_prob_4M(subs_M, core, mx_MKD, P_A, P_AB, P_ABC, P_ABCD)
print '%fs: 4-mode compute prob' % ((time() - s) / n_trials)

y = int(np.sqrt(np.prod(core_dims)))

s = time()
for _ in range(n_trials):
	Y_MKD, Y_Q = a.comp_allocate(y, subs, matrices, core, P_MQ=P_MQ)
print '%fs: allocating' % ((time() - s) / n_trials)

# s = time()
# for _ in range(n_trials):
# 	Y_MKD, Y_Q = a.comp_allocate_alt(y, subs, matrices, core, P_MQ=P_MQ)
# print '%fs: alt allocating' % ((time() - s) / n_trials)
