
import sys
from path import Path
from argparse import ArgumentParser

import numpy as np
import numpy.random as rn

from metrics import evaluate
from masking import get_smoothing_mask


def get_forec_eval_data(Y, n_forecast_steps=1, ignore_diag=False):
    if n_forecast_steps > 0:
        Y_train = np.copy(Y[:-n_forecast_steps])
        Y_forec = np.copy(Y[-n_forecast_steps:])
    else:
        Y_train = np.copy(Y)
        Y_forec = np.zeros((0,) + Y_train.shape[1:], dtype=Y.dtype)

    forec_mask = np.ones_like(Y_forec).astype(bool)
    if ignore_diag:
        diag_NN = np.eye(Y.shape[1]).astype(bool)
        forec_mask[:, diag_NN] = False

    eval_forec_subs = np.where(forec_mask)
    eval_forec_vals = Y_forec[eval_forec_subs]
    return Y_train, Y_forec, eval_forec_subs, eval_forec_vals


def get_smooth_eval_data(Y_train, mask, ignore_diag=False):
    if ignore_diag:
        diag_NN = np.eye(Y_train.shape[1]).astype(bool)

    train_mask = ~mask
    if ignore_diag:
        train_mask[:, diag_NN] = False
    train_subs = np.where(train_mask)
    train_vals = Y_train[train_subs]

    smooth_mask = np.copy(mask)
    if ignore_diag:
        smooth_mask[:, diag_NN] = False
    eval_smooth_subs = np.where(smooth_mask)
    eval_smooth_vals = Y_train[eval_smooth_subs]

    return train_subs, train_vals, eval_smooth_subs, eval_smooth_vals


def get_train_data(Y, mask, missing_style='masked', thinning_p=0.5, noising_alpha=1.0, ignore_diag=False, seed=None):

    assert missing_style in ['masked', 'zeroed', 'thinned', 'noised']

    if missing_style == 'masked':
        Y_train = np.copy(Y)
        mask = mask

    elif missing_style == 'zeroed':
        Y_train = Y * (1-mask.astype(int))
        mask = None

    elif missing_style == 'thinned':
        rn.seed(seed)
        Y_train = np.copy(Y)
        Y_train[mask] = rn.binomial(Y[mask], thinning_p)
        mask = None

    elif missing_style == 'noised':
        rn.seed(seed)
        Y_train = np.copy(Y)
        Y_train[mask] = rn.poisson(rn.gamma(noising_alpha, (Y_train[mask] + 0.1)/noising_alpha))
        mask = None

    if ignore_diag:
        if mask is None:
            mask = np.zeros_like(Y_train, dtype=bool)
        assert mask.shape[1] == mask.shape[2]
        diag_NN = np.eye(mask.shape[1]).astype(bool)
        
        mask[:, diag_NN] = True
        Y_train[:, diag_NN] = 0

    return np.ma.array(Y_train, mask=mask)


def load_experiment(experiment_path):
    experiment_dict = np.load(experiment_path)
    
    eval_forec_vals = experiment_dict['eval_forec_vals']
    eval_forec_subs = tuple(experiment_dict['eval_forec_subs'])

    eval_smooth_vals = experiment_dict['eval_smooth_vals']
    eval_smooth_subs = tuple(experiment_dict['eval_smooth_subs'])

    train_data = np.ma.array(experiment_dict['train_data'], mask=experiment_dict['train_mask'])
    return train_data, eval_smooth_vals, eval_smooth_subs, eval_forec_vals, eval_forec_subs


def reconstruct_data(experiment):
    train_data = experiment[0]
    eval_smooth_vals = experiment[1]
    eval_smooth_subs = experiment[2]
    eval_forec_vals = experiment[3]
    eval_forec_subs = experiment[4]
    n_forecast_steps = len(np.unique(eval_forec_subs[0]))
    
    n_timesteps = train_data.shape[0] + n_forecast_steps
    Y = np.zeros((n_timesteps,) + train_data.shape[1:], dtype=int)
    Y[:-n_forecast_steps] = train_data.data
    Y[eval_smooth_subs] = eval_smooth_vals
    Y[-n_forecast_steps:][eval_forec_subs] = eval_forec_vals
    return Y


def get_evaluation_funcs(experiment):
    Y = reconstruct_data(experiment)
    train_data = experiment[0]
    eval_smooth_vals = experiment[1]
    eval_smooth_subs = experiment[2]
    eval_forec_vals = experiment[3]
    eval_forec_subs = experiment[4]
    n_forecast_steps = len(np.unique(eval_forec_subs[0]))

    lag_mae = np.abs(Y[1:] - Y[:-1]).mean(axis=0)
    scale_smooth_vals = lag_mae[eval_smooth_subs[1:]]
    scale_forec_vals = lag_mae[eval_forec_subs[1:]]

    def evaluate_smoothing(pred_smooth_vals):
        eval_dict = evaluate(eval_smooth_vals, pred_smooth_vals)

        abs_scaled_err = np.abs(pred_smooth_vals - eval_smooth_vals)
        abs_scaled_err[scale_smooth_vals > 0] /= scale_smooth_vals[scale_smooth_vals > 0]
        abs_scaled_err[scale_smooth_vals == 0] = np.nan
        eval_dict['mase'] = np.nanmean(abs_scaled_err)
        eval_dict['mase-z'] = np.nanmean(abs_scaled_err[eval_smooth_vals == 0])
        eval_dict['mase-nz'] = np.nanmean(abs_scaled_err[eval_smooth_vals > 0])

        return dict([('s-%s' % k, v) for k, v in eval_dict.items()])

    def evaluate_forecasting(pred_forec_vals):
        if n_forecast_steps == 0:
            return {}

        eval_dict = evaluate(eval_forec_vals, pred_forec_vals)
        abs_scaled_err = np.abs(pred_forec_vals - eval_forec_vals)
        abs_scaled_err[scale_forec_vals > 0] /= scale_forec_vals[scale_forec_vals > 0]
        abs_scaled_err[scale_forec_vals == 0] = np.nan
        eval_dict['mase'] = np.nanmean(abs_scaled_err)
        eval_dict['mase-z'] = np.nanmean(abs_scaled_err[eval_forec_vals == 0])
        eval_dict['mase-nz'] = np.nanmean(abs_scaled_err[eval_forec_vals > 0])
        
        return dict([('f-%s' % k, v) for k, v in eval_dict.items()])

    return evaluate_smoothing, evaluate_forecasting


def main(cmd=None):
    p = ArgumentParser()
    p.add_argument('-d', '--data', type=Path, required=True)
    p.add_argument('-o', '--out', type=Path, default=Path('.'))

    p.add_argument('--time_mode', type=int, default=0,
        help='Which mode of the data tensor corresponds to time.')
    
    p.add_argument('--ignore_diag', action='store_true', default=False,
        help='Whether to ignore the diagonal of actor-actor data.')

    # when missing
    p.add_argument('--n_periods', type=int, default=2)
    p.add_argument('--period_length', type=int, default=1)
    p.add_argument('--n_forecast_steps', type=int, default=1)

    p.add_argument('--masking_style', type=str, default='all', 
        choices=['all', 'random', 'structured'],
        help='Style in which missing/corrupted entries are chosen.')
    
    p.add_argument('--masking_p', type=float, default=0.5, 
        help='The proportion of missing/corrupted entries when held out at random.')

    p.add_argument('--mask_SHP_style', type=str, default=':10',
        help='Style of missing mask when missing/corrupted entries are structured.\
              Currently the only supported format is i:j that gives the range of actors\
              to hold out as a block.')

    p.add_argument('--mask_SHP', type=Path, 
        help='Path to a .npz file that contains the structured missing mask.')

    p.add_argument('--missing_style', type=str, default='masked',
        choices=['masked', 'zeroed', 'thinned', 'noised'],
        help='Style in which entries are made missing or corrupted.')

    p.add_argument('--noising_alpha', type=float, default=0.5,
        help='Value of alpha when noising corrupted entries with gamma noise.')

    p.add_argument('--thinning_p', type=float, default=0.5,
        help='Thinning probability parameter (lower means fewer tokens observed).')

    p.add_argument('--seed', type=int, default=rn.choice(2**32-1))
    args = p.parse_args() if cmd is None else p.parse_args(cmd)

    Y = np.load(args.data)['Y']
    if args.time_mode != 0:
        Y = np.rollaxis(Y, args.time_mode, 0)

    n_timesteps = Y.shape[0]
    assert args.n_forecast_steps < n_timesteps-1

    # where missing
    mask_SHP = None
    if args.masking_style == 'structured':
        if args.mask_SHP is not None:
            mask_SHP = np.load(args.mask_SHP)['mask_SHP']
        else:
            mask_SHP = np.zeros(Y.shape[1:]).astype(bool)
            n_actors = mask_SHP.shape[0]
            assert n_actors == mask_SHP.shape[1]
            i, j = args.mask_SHP_style.strip().split(':')
            i = 0 if len(i) == 0 else int(i)
            j = n_actors-1 if len(j) == 0 else int(j)
            assert (i < n_actors) and (j < n_actors)
            mask_SHP[i:j, i:j] = True
        assert mask_SHP.shape == Y.shape[1:]
        assert mask_SHP.dtype == bool

    assert 0 < args.masking_p and args.masking_p < 1.
    assert 0 < args.thinning_p and args.thinning_p < 1.

    print('Splitting training and forecast data...')

    Y_train, Y_forec, eval_forec_subs, eval_forec_vals = get_forec_eval_data(Y=Y,
                                                                             n_forecast_steps=args.n_forecast_steps,
                                                                             ignore_diag=args.ignore_diag)
    print('Generating smoothing mask...')

    mask = get_smoothing_mask(shp=Y_train.shape,
                              masking_style=args.masking_style, 
                              masking_p=args.masking_p, 
                              mask_SHP=mask_SHP,
                              n_periods=args.n_periods, 
                              period_length=args.period_length,
                              seed=args.seed)

    print('Getting smoothing evaluation data...')

    train_subs, train_vals, eval_smooth_subs, eval_smooth_vals = get_smooth_eval_data(Y_train=Y_train, 
                                                                            mask=mask,
                                                                            ignore_diag=args.ignore_diag)

    print('Masking/corrupting training data...')

    train_data = get_train_data(Y_train, 
                                mask=mask,
                                missing_style=args.missing_style,
                                thinning_p=args.thinning_p,
                                noising_alpha=args.noising_alpha,
                                ignore_diag=args.ignore_diag,
                                seed=args.seed)

    assert np.abs(train_data.data[train_subs] - train_vals).mean() == 0
    if args.missing_style in ['thinning', 'noising']:
        assert np.abs(train_data.data[eval_smooth_subs] - eval_smooth_vals).mean() != 0

    curr_nums = [int(x.namebase.split('_experiment')[0]) for x in args.out.files('*_experiment.npz')]
    experiment_num = np.max(curr_nums) + 1 if curr_nums else 1
    experiment_path = args.out.joinpath('%d_experiment.npz' % experiment_num)

    print('Serializing...')

    np.savez_compressed(experiment_path,
                        train_data=train_data.data,
                        train_mask=train_data.mask,
                        eval_smooth_subs=eval_smooth_subs,
                        eval_smooth_vals=eval_smooth_vals,
                        eval_forec_subs=eval_forec_subs,
                        eval_forec_vals=eval_forec_vals,
                        **vars(args))

    print(experiment_path)
    return experiment_path

def test():
    data_path = Path('/Users/aaronschein/Documents/research/mlds/data/icews/tensors/1995-2000-M.npz')
    Y = np.rollaxis(np.load(data_path)['Y'], 3, 0)

    cmd=['-d=%s' % data_path,
         '--ignore_diag',
         '--time_mode=3',
         '--n_forecast_steps=2',
         '--missing_style=noised',
         '--masking_style=structured',
         '--mask_SHP_style=6:12']
    experiment_path = main(cmd)

    assert np.allclose(Y, reconstruct_data(load_experiment(experiment_path)))

    data, eval_smooth_vals, eval_smooth_subs, eval_forec_vals, eval_forec_subs = load_experiment(experiment_path)
    missing_T = np.unique(eval_smooth_subs[0])
    n_forecast_steps = len  (np.unique(eval_forec_subs[0]))

    Y_train = Y[:-n_forecast_steps]
    Y_forec = Y[-n_forecast_steps:]
    assert Y_train.shape == data.shape

    assert np.allclose(Y_train[:, 12:, 12:], data.data[:, 12:, 12:])
    assert np.allclose(Y_train[:, :6, :6], data.data[:, :6, :6])

    missing_T = np.unique(eval_smooth_subs[0])
    assert len(missing_T) == 2
    diag_NN = np.eye(Y_forec.shape[1]).astype(bool)

    assert not np.allclose(eval_smooth_vals, data.data[eval_smooth_subs])
    assert not np.allclose(eval_smooth_vals, data.data[missing_T, 6:12, 6:12][:, ~diag_NN[6:12, 6:12]].ravel())
    
    assert np.allclose(eval_smooth_vals, Y_train[missing_T, 6:12, 6:12][:, ~diag_NN[6:12, 6:12]].ravel())
    assert np.allclose(eval_smooth_vals, Y_train[eval_smooth_subs])

    assert np.allclose(Y_forec[eval_forec_subs], eval_forec_vals)

    test_replication(experiment_path)

def test_replication(experiment_path):
    experiment_dict = np.load(experiment_path)
    
    non_arg_keys = ['train_data',
                    'train_mask', 
                    'eval_smooth_subs', 
                    'eval_smooth_vals', 
                    'eval_forec_subs', 
                    'eval_forec_vals']
    cmd = []
    for k in experiment_dict.files:
        if k not in non_arg_keys:
            v = str(experiment_dict[k])
            if k == 'ignore_diag' and v == 'True':
                cmd.append('--ignore_diag')
            elif v != 'None':
                cmd.append('--%s=' % k + v)
    print(' '.join(cmd))
    replicated_path = main(cmd)

    experiment = load_experiment(experiment_path)
    replicated = load_experiment(replicated_path)

    for x1, x2 in zip(experiment, replicated):
        if type(x1) is tuple:
            x1 = np.array(x1)
            x2 = np.array(x2)
        assert np.allclose(x1, x2)

    Path.remove(replicated_path)

if __name__ == '__main__':
    # test()
    main()
