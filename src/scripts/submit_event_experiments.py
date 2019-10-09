
import sys
from path import Path
from subprocess import Popen, PIPE
import shutil
import itertools as it

PYTHON_INSTALLATION = '~/anaconda3/bin/python3.7'
CODE_DIR = Path('~/research/prgds/src/scripts')
DATA_DIR = Path('/mnt/nfs/work1/wallach/aschein/data')
EXPERIMENTS_DIR = Path('/mnt/nfs/work1/wallach/aschein/results/NEURIPS19/camera_ready/tensors')


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

    nu_eps = 1
    mtx_is_gamma = True

    n_itns = 20
    n_total_itns = 5000
    n_epochs = n_total_itns / n_itns
    n_avg_after_itn = 1000

    for experiment_file in EXPERIMENTS_DIR.walkfiles('%d_experiment.npz' % exp_num):
        exp_dir = experiment_file.parent
        for (model_type, theta_eps) in [('prgds', 1), ('prgds', 0), ('pgds', None)]:
            for seed in [617, 781]:
                out_dir = exp_dir.joinpath('%s_results' % experiment_file.namebase)
                out_dir = out_dir.joinpath(model_type)
                if model_type == 'prgds':
                    out_dir = out_dir.joinpath('nu_eps_%d' % nu_eps)
                    if mtx_is_gamma:
                        out_dir = out_dir.joinpath('mtx_is_gamma')
                    else:
                        out_dir = out_dir.joinpath('mtx_is_dirichlet')
                out_dir = out_dir.joinpath('core_shp_%d' % core_shp, 'seed_%d' % seed)
                out_dir.makedirs_p()

                cmd = '%s %s ' % (PYTHON_INSTALLATION, CODE_DIR.joinpath('run_experiment.py'))
                cmd += '--experiment=%s ' % experiment_file
                cmd += '--results_dir=%s ' % out_dir
                cmd += '--model_type=%s ' % model_type
                cmd += '--theta_eps=%d ' % theta_eps
                cmd += '--nu_eps=%d ' % nu_eps
                cmd += '--core_shp %d ' % core_shp
                cmd += '--n_threads=%d ' % n_threads
                cmd += '--seed=%d ' % seed
                if mtx_is_gamma:
                    cmd += '--mtx_is_dirichlet '

                # cmd += '--n_epochs=%d ' % 5
                # cmd += '--n_itns %d ' % 1
                # cmd += '--avg_after=%d ' % 1

                job_name = model_type
                stdout = out_dir.joinpath('output-train.out')
                stderr = out_dir.joinpath('errors-train.out')
                jid = qsub(cmd, job_name=job_name, stdout=stdout, stderr=stderr, n_cores=n_threads)
                n_jobs += 1

print('%d jobs submitted.' % n_jobs)

if __name__ == '__main__':
    main()
