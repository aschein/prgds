
import sys
from path import Path
from subprocess import Popen, PIPE
import shutil
import itertools as it

PYTHON_INSTALLATION = '~/anaconda3/bin/python3.7'
CODE_DIR = Path('~/research/apf/src/experiments')
DATA_DIR = Path('/mnt/nfs/work1/wallach/aschein/data')
RESULTS_DIR = Path('/mnt/nfs/work1/wallach/aschein/results/NEURIPS19/camera_ready/tensors')
EXPERIMENTS_DIR = Path('/mnt/nfs/work1/wallach/aschein/results/thesis')


def qsub(cmd, job_name=None, stdout=None, stderr=None, depend=None, n_cores=None):
    print(cmd)
    if type(depend) is str:
        depend = [depend]
    args = ['qsub']
    if n_cores:
        args.extend(['-pe', 'generic', '%d' % n_cores])
    if job_name:
        args.extend(['-N', job_name])
    if stderr:
        args.extend(['-e', stderr])
    if stdout:
        args.extend(['-o', stdout])
    if depend:
        args.extend(['-hold_jid', ','.join(depend)])
    out = Popen(args, stdin=PIPE, stdout=PIPE, encoding='utf8').communicate('%s\n' % cmd)[0]
    print(out.rstrip())
    job_id = out.split()[2]
    return job_id


def main():
    n_jobs = 0
    n_threads = 12
    core_shp = 100

    mtx_is_gamma = True

    for exp_num in [1, 2]:
        for experiment_file in EXPERIMENTS_DIR.walkfiles('%d_experiment.npz' % exp_num):
            if 'periods6' not in str(experiment_file):
                continue
            if 'masked' not in str(experiment_file):
                continue
            if 'structured' in str(experiment_file):
                continue

            subdirs = Path(experiment_file.parent.split(EXPERIMENTS_DIR)[1]).splitall()[1:]
            exp_dir = RESULTS_DIR.joinpath(*subdirs)
            exp_dir.makedirs_p()

            shutil.copyfile(experiment_file, exp_dir.joinpath(experiment_file.namebase + '.npz'))
            exp_dir = exp_dir.joinpath('%s_results' % experiment_file.namebase)
            exp_dir.makedirs_p()

    #         for model_type in ['prgds-v2', 'pgds']:

    #             for i, (mtx_is_gamma, theta_eps, nu_eps) in enumerate(it.product([False, True], [0, 1], [0, 1])):
    #                 if (model_type == 'pgds') and i > 0:
    #                     continue

    #                 for seed in [617, 781]:
    #                     out_dir = exp_dir.joinpath(model_type)
    #                     if model_type == 'prgds-v2':
    #                         out_dir = out_dir.joinpath('theta_eps_%d' % theta_eps)
    #                         out_dir = out_dir.joinpath('nu_eps_%d' % nu_eps)
    #                         if mtx_is_gamma:
    #                             out_dir = out_dir.joinpath('mtx_is_gamma')
    #                         else:
    #                             out_dir = out_dir.joinpath('mtx_is_dirichlet')
    #                     out_dir = out_dir.joinpath('core_shp_%d' % core_shp, 'seed_%d' % seed)
    #                     out_dir.makedirs_p()

    #                     cmd = '%s %s ' % (PYTHON_INSTALLATION, CODE_DIR.joinpath('run_experiment.py'))
    #                     cmd += '--experiment=%s ' % experiment_file
    #                     cmd += '--results_dir=%s ' % out_dir
    #                     cmd += '--model_type=%s ' % model_type
    #                     cmd += '--theta_eps=%d ' % theta_eps
    #                     cmd += '--nu_eps=%d ' % nu_eps
    #                     cmd += '--core_shp %d ' % core_shp
    #                     cmd += '--n_threads=%d ' % n_threads
    #                     cmd += '--seed=%d ' % seed
    #                     if mtx_is_gamma:
    #                         cmd += '--mtx_is_dirichlet '

    #                     # cmd += '--n_epochs=%d ' % 5
    #                     # cmd += '--n_itns %d ' % 1
    #                     # cmd += '--avg_after=%d ' % 1

    #                     job_name = model_type
    #                     stdout = out_dir.joinpath('output-train.out')
    #                     stderr = out_dir.joinpath('errors-train.out')
    #                     jid = qsub(cmd, job_name=job_name, stdout=stdout, stderr=stderr, n_cores=n_threads)

    # print('%d jobs submitted.' % n_jobs)

if __name__ == '__main__':
    main()
