import sys
import numpy as np
import tensorly as tl


def uttkrp(tens, mode, mtxs, core=None, transpose=False):
    """
    Alternative implementation of uttkrp in sktensor library.

    The description of that method is modified below:

    Unfolded tensor times Khatri-Rao product:
    :math:`Z = \\unfold{X}{3} (U_1 \kr \cdots \kr U_N)`
    Computes the _matrix_ product of the unfolding
    of a tensor and the Khatri-Rao product of multiple matrices.
    Efficient computations are perfomed by the respective
    tensor implementations.
    Parameters
    ----------
    tens : input tensor
    mtxs : list of array-likes
        Matrices for which the Khatri-Rao product is computed and
        which are multiplied with the tensor in mode `mode`.
    mode : int
        Mode in which the Khatri-Rao product of `mtxs` is multiplied
        with the tensor.
    core: array-like
        Weights for each component (K-length)
    Returns
    -------
    Z : np.ndarray
        Matrix which is the result of the matrix product of the unfolding of
        the tensor and the Khatri-Rao product of `mtxs`.
    See also
    --------
    For efficient computations of unfolded tensor times Khatri-Rao products
    for specialiized tensors see also
    References
    ----------
    [1] B.W. Bader, T.G. Kolda
        Efficient Matlab Computations With Sparse and Factored Tensors
        SIAM J. Sci. Comput, Vol 30, No. 1, pp. 205--231, 2007
    """
    if transpose:
        mtxs = [mtx.T for mtx in mtxs]

    K, D = mtxs[mode].shape
    order = sorted(range(tens.ndim), key=lambda m: mtxs[m].shape[0])
    order.remove(mode)
    Z = tl.transpose(tens, [mode] + order)

    Z = tl.tenalg.mode_dot(Z, mtxs[order[-1]], -1)
    for m in reversed(order[:-1]):
        Z *= mtxs[m].T
        Z = Z.sum(axis=-2)
    
    if core is not None:
        Z *= core[:, np.newaxis] if not transpose else core 

    return Z.T if not transpose else Z


def sp_uttkrp(subs, vals, mode, mtxs, core=None, transpose=False):
    """Alternative implementation of the sparse version of the uttkrp.
    ----------
    subs : n-tuple of array-likes
        Subscripts of the nonzero entries in the tensor.
        Length of tuple n must be equal to dimension of tensor.
    vals : array-like
        Values of the nonzero entries in the tensor.
    mtxs : list of array-likes
        Matrices for which the Khatri-Rao product is computed and
        which are multiplied with the tensor in mode `mode`.
    mode : int
        Mode in which the Khatri-Rao product of `mtxs` is multiplied
        with the tensor.
    core: array-like
        Weights for each component (K-length)
    Returns
    -------
    out : np.ndarray
        Matrix which is the result of the matrix product of the unfolding of
        the tensor and the Khatri-Rao product of `U`.
    """
    if transpose:
        mtxs = [mtx.T for mtx in mtxs]

    K, D = mtxs[mode].shape

    if core is None:
        core = np.ones(K)

    out_KD = np.zeros((K, D))
    for k in range(K):
        tmp = vals * core[k]
        for m, mtx in enumerate(mtxs):
            if m != mode:
                tmp *= mtx[k, subs[m]]
        out_KD[k] = np.bincount(subs[mode],
                                weights=tmp,
                                minlength=D)
    return out_KD if not transpose else out_KD.T


def get_superdiag_array(shp, diag=None):
    n_modes = len(shp)
    dim = shp[0]
    out = np.zeros(shp)

    if diag is None:
        diag = np.ones(dim)

    for d in range(dim):
        out[(d,) * n_modes] = diag[d]

    return out


def uttkrp_diag(tens, mode, mtxs, core=None, transpose=False):
    """Alternate implementation of uttkrp that calls the uttut
    with a superdiagonal core tensor.
    """
    K = mtxs[0].shape[0] if not transpose else mtxs[0].shape[1]
    diag_core = get_superdiag_array((K,) * tens.ndim, diag=core)
    return uttut(tens, mode, mtxs, diag_core, transpose)


def uttut(tens, mode, mtxs, core, transpose=False):
    """
    Unfolded tensor times unfolded Tucker.
    """
    if core.size > tens.size:
        tens, core, transpose = core, tens, not transpose

    tuck_DN = tl.tucker_to_unfolded(tens, 
                                    mtxs, 
                                    mode=mode,
                                    skip_factor=mode, 
                                    transpose_factors=transpose)
    core_KN = tl.unfold(core, mode)
    return np.dot(core_KN, tuck_DN.T) if not transpose else np.dot(tuck_DN, core_KN.T)


def sp_uttut(subs, vals, mode, mtxs, core, transpose=False):
    if transpose:
        mtxs = [mtx.T for mtx in mtxs]

    K, D = mtxs[mode].shape
    out_KD = np.zeros((K, D))

    for q, core_subs in enumerate(np.ndindex(core.shape)):
        tmp = vals * core[core_subs]
        for m, mtx in enumerate(mtxs):
            if m != mode:
                tmp *= mtx[core_subs[m], subs[m]]

        out_KD[core_subs[mode_dot]] += np.bincount(subs[mode],
                                                   weights=tmp,
                                                   minlength=D)
    return out_KD


def sp_tucker_to_tensor(subs, mtxs, core, transpose=False):
    n_modes = len(mtxs)
    assert len(subs) == n_modes
    out_shp = tuple([mtx.shape[1] for mtx in mtxs])
    out = np.zeros(out_shp)

    P = len(subs[0])

    shp = core.shape
    tmp = core.reshape((core.shape[0], -1))
    for m in range(n_modes):
        mtx_DK = mtxs[m] if transpose else mtxs[m].T
        mtx_PK = mtx_DK[subs[m]]
        if m == 0:
            tmp = np.dot(mtx_PK, tmp)
            tmp = tmp.reshape((P,) + shp[1:])
        elif m < n_modes-1:
            tmp = tmp.reshape(mtx_PK.shape + (-1,))
            tmp = np.einsum('pk,pkz->pz', mtx_PK, tmp)
            tmp = tmp.reshape((P,) + shp[2:])
        else:
            tmp = tmp.reshape(mtx_PK.shape + (-1,))
            vals_P = np.einsum('pk,pkz->p', mtx_PK, tmp)
            out[subs] = vals_P
            return out
        shp = tmp.shape


if __name__ == '__main__':
    import numpy.random as rn
    core_shp = (10, 9, 5, 4)
    data_shp = (50, 40, 20, 10)

    mtxs = [rn.gamma(1, 1, size=(K, D)) for K, D in zip(core_shp, data_shp)]
    core = rn.binomial(1, 0.99, size=data_shp)












