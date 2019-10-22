"""Script for running a model on an experiment file.

This does NOT perform any house-keeping on the output directory.
It takes the results_dir as an argument and plops results into it.
If there are already results in it, this script may overwrite them.
"""
import sys
import pickle

import numpy as np
import numpy.random as rn
import scipy.stats as st
import scipy.special as sp
import pandas as pd
from tqdm import tqdm

from path import Path
from argparse import ArgumentParser

from create_experiment import load_experiment, get_evaluation_funcs
sys.path.append(Path(__file__).parent.parent)
from apf.models.gpdptf import GPDPTF
from apf.models.prgds import PRGDS
from apf.models.pgds import PGDS
from apf.base.apf import APF


def run_initialized_model(model,
                          experiment,
                          schedule={},
                          results_dir=Path('tmp'), 
                          n_itns=50,
                          n_epochs=100,
                          avg_after_itn=1000,
                          get_model_epoch_args=lambda x: {'itn': x.total_itns},
                          restart=False):
    
    results_dir.makedirs_p()

    train_data = experiment[0]
    eval_smooth_vals = experiment[1]
    eval_smooth_subs = experiment[2]
    eval_forec_vals = experiment[3]
    eval_forec_subs = experiment[4]
    n_forecast_steps = len(np.unique(eval_forec_subs[0]))

    evaluate_smoothing, evaluate_forecasting = get_evaluation_funcs(experiment)

    pred_smooth_vals_N_ = []
    prob_smooth_vals_N_ = []
    pred_forec_vals_N_ = []
    prob_forec_vals_N_ = []

    results_df = pd.DataFrame()

    if restart:
        print('Restarting chain...')
        samples_dir = results_dir.joinpath('samples')
        for sample_file in tqdm(samples_dir.files('state_*.npz')):
            itn = int(sample_file.namebase.split('_')[1])

            if itn >= avg_after_itn:
                model.set_state(np.load(sample_file))
                pred_smooth_vals = model.reconstruct(subs=eval_smooth_subs)
                pred_forec_vals = model.forecast(n_timesteps=n_forecast_steps, subs=eval_forec_subs)
                pred_smooth_vals_N_.append(pred_smooth_vals)
                prob_smooth_vals_N_.append(st.poisson.logpmf(eval_smooth_vals, pred_smooth_vals))
                pred_forec_vals_N_.append(pred_forec_vals)
                prob_forec_vals_N_.append(st.poisson.logpmf(eval_forec_vals, pred_forec_vals))
    
        results_df = pd.read_pickle(results_dir.joinpath('results_df.pkl'))

        last_itn = max([int(x.namebase.split('_')[1]) for x in samples_dir.files('state_*.npz')])
        sample_file = samples_dir.joinpath('state_%d.npz' % last_itn)
        model.set_state(np.load(sample_file))
        model.set_total_itns(last_itn)

    if not restart:
        n_epochs += 1

    for epoch in range(n_epochs):
        n_gibbs_itns = n_itns
        if epoch == 0 and model.total_itns == 0:
            n_gibbs_itns = 0

        model.fit(train_data,
                  initialize=False,
                  verbose=n_itns,
                  n_itns=n_gibbs_itns,
                  schedule=schedule)

        pred_smooth_vals = model.reconstruct(subs=eval_smooth_subs)
        pred_forec_vals = model.forecast(n_timesteps=n_forecast_steps, subs=eval_forec_subs)

        if model.total_itns >= avg_after_itn:
            pred_smooth_vals_N_.append(pred_smooth_vals)
            prob_smooth_vals_N_.append(st.poisson.logpmf(eval_smooth_vals, pred_smooth_vals))
            pred_forec_vals_N_.append(pred_forec_vals)
            prob_forec_vals_N_.append(st.poisson.logpmf(eval_forec_vals, pred_forec_vals))

        df_row = {**get_model_epoch_args(model),
                  **evaluate_smoothing(pred_smooth_vals),
                  **evaluate_forecasting(pred_forec_vals)}

        results_df = results_df.append(df_row, ignore_index=True)
        results_df.to_pickle(results_dir.joinpath('results_df.pkl'))

        state_dir = results_dir.joinpath('samples')
        state_dir.makedirs_p()
        np.savez_compressed(state_dir.joinpath('state_%d.npz' % model.total_itns), **dict(model.get_state()))

    pred_smooth_vals_N_ = np.asarray(pred_smooth_vals_N_)
    prob_smooth_vals_N_ = np.asarray(prob_smooth_vals_N_)
    pred_forec_vals_N_ = np.asarray(pred_forec_vals_N_)
    prob_forec_vals_N_ = np.asarray(prob_forec_vals_N_)
    n_avg_itns = pred_smooth_vals_N_.shape[0]

    avg_pred_smooth_vals = pred_smooth_vals_N_.mean(axis=0)
    avg_prob_smooth_vals = sp.logsumexp(prob_smooth_vals_N_, axis=0)
    avg_pred_forec_vals = pred_forec_vals_N_.mean(axis=0)
    avg_prob_forec_vals = sp.logsumexp(prob_forec_vals_N_, axis=0)

    perp_s = np.exp(-avg_prob_smooth_vals.mean())
    perp_s_z = np.exp(-avg_prob_smooth_vals[eval_smooth_vals == 0].mean())
    perp_s_nz = np.exp(-avg_prob_smooth_vals[eval_smooth_vals > 0].mean())
    perp_f = np.exp(-avg_prob_forec_vals.mean())
    perp_f_z = np.exp(-avg_prob_forec_vals[eval_forec_vals == 0].mean())
    perp_f_nz = np.exp(-avg_prob_forec_vals[eval_forec_vals > 0].mean())

    np.savez_compressed(results_dir.joinpath('averages.npz'),
                        avg_pred_smooth_vals=avg_pred_smooth_vals,
                        avg_prob_smooth_vals=avg_prob_smooth_vals,
                        avg_pred_forec_vals=avg_pred_forec_vals,
                        avg_prob_forec_vals=avg_prob_forec_vals,
                        n_avg_itns=n_avg_itns,
                        perp_s=perp_s,
                        perp_s_z=perp_s_z,
                        perp_s_nz=perp_s_nz,
                        perp_f=perp_f,
                        perp_f_z=perp_f_z,
                        perp_f_nz=perp_f_nz,
                        **evaluate_smoothing(avg_pred_smooth_vals),
                        **evaluate_forecasting(avg_pred_forec_vals))

def main(cmd=None):
    p = ArgumentParser()
    
    # Experiment params
    p.add_argument('-e', '--experiment', type=Path, required=True)
    p.add_argument('-r', '--results_dir', type=Path, default=Path('.'))
    p.add_argument('--n_itns', type=int, default=50)
    p.add_argument('--n_epochs', type=int, default=80)
    p.add_argument('--avg_after_itn', type=int, default=1000)

    # All APF model params
    p.add_argument('--model_type', type=str, default='prgds',
                   choices=['apf', 'gpdptf', 'pgds', 'prgds'])
    p.add_argument('--core_shp', type=int, nargs='*', default=[100])
    p.add_argument('--mtx_is_dirichlet', type=int, nargs='*', default=[-1])
    p.add_argument('--binary', type=bool, default=False)
    p.add_argument('--n_threads', type=int, default=1)
    p.add_argument('--seed', type=int, default=rn.choice(2**32-1))
    p.add_argument('-i', '--init_state', type=Path)
    p.add_argument('--restart', action="store_true", default=False)
    
    # GPDPTF/PGDS/PrGDS params
    p.add_argument('--gam', type=float, default=25.)
    p.add_argument('--eps', type=float, default=0.1)
    p.add_argument('--tau', type=float, default=1.0)
    p.add_argument('--stationary', type=bool, default=True)
    
    # PrGDS params
    p.add_argument('--diagonal', type=bool, default=False)
    p.add_argument('--nu_eps', type=float, default=1.0)
    p.add_argument('--theta_eps', type=float, default=0.0)
    p.add_argument('--pi_is_dirichlet', type=bool, default=True)
    args = p.parse_args(cmd) if cmd is not None else p.parse_args()

    experiment = load_experiment(args.experiment)
    train_data = experiment[0]

    data_shp = train_data.shape
    core_shp = tuple(args.core_shp)
    assert len(core_shp) in [1, len(data_shp)]

    mtx_is_dirichlet = args.mtx_is_dirichlet
    if mtx_is_dirichlet == [-1]:
        mtx_is_dirichlet = list(range(1, len(data_shp)))

    if args.model_type == 'pgds':
        model = PGDS(data_shp=data_shp,
                     core_shp=core_shp,
                     eps=args.eps,
                     binary=args.binary,
                     seed=args.seed,
                     n_threads=args.n_threads,
                     time_mode=0,
                     gam=args.gam,
                     tau=args.tau,
                     stationary=args.stationary)

    elif args.model_type == 'prgds':
        model = PRGDS(data_shp=data_shp,
                      core_shp=core_shp,
                      eps=args.eps,
                      binary=args.binary,
                      mtx_is_dirichlet=mtx_is_dirichlet,
                      pi_is_dirichlet=args.pi_is_dirichlet,
                      block_sample_nu_and_g=args.nu_eps == 0,
                      block_sample_Theta_and_H=args.theta_eps == 0,
                      seed=args.seed,
                      n_threads=args.n_threads,
                      time_mode=0,
                      gam=args.gam,
                      theta_eps=args.theta_eps,
                      stationary=args.stationary,
                      diagonal=args.diagonal)

    elif args.model_type == 'gpdptf':
        raise NotImplementedError

    elif args.model_type == 'apf':
        raise NotImplementedError

    schedule = model.get_default_schedule()

    init_state = {}
    if args.init_state is not None:
        init_state = np.load(args.init_state)
    model.initialize_state(dict(init_state))

    args.results_dir.makedirs_p()
    with open(args.results_dir.joinpath('experiment_params.pkl'), 'wb') as f:
        pickle.dump(vars(args), f)

    with open(args.results_dir.joinpath('model_params.pkl'), 'wb') as f:
        pickle.dump(model.get_params(), f)

    run_initialized_model(model,
                          experiment,
                          schedule=schedule,
                          results_dir=args.results_dir, 
                          n_itns=args.n_itns,
                          n_epochs=args.n_epochs,
                          avg_after_itn=args.avg_after_itn,
                          get_model_epoch_args=lambda x: {'itn': x.total_itns},
                          restart=args.restart)


if __name__ == '__main__':
    main()
