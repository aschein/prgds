import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


gray = (102/255.0, 102/255.0, 102/255.0, 1.0)
light_gray = (238/255.0, 238/255.0, 238/255.0, 1.0)

# sns.set_style({'font.family': 'Abel'})
sns.set_style({'axes.facecolor': light_gray})
sns.set_style({'xtick.color': gray})
sns.set_style({'text.color': gray})
sns.set_style({'ytick.color': gray})
sns.set_style({'axes.grid': False})


def _cdf(data):
    """
    Returns the empirical CDF (a function) for the specified data.

    Arguments:

    data -- data from which to compute the CDF
    """

    tmp = np.empty_like(data)
    tmp[:] = data
    tmp.sort()

    def f(x):
        return np.searchsorted(tmp, x, 'right') / float(len(tmp))

    return f


def pp_plot(a, b, t=None, plot_dir=None, show=True):
    """
    Generates a P-P plot.
    """
    if not show:
        if t is None:  # plot name is required for out path
            raise NotImplementedError
        if plot_dir is None:
            plot_dir = 'pp_plots'
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        plot_path = os.path.join(plot_dir, '%s.png' % t)

    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys()
        for n, (k, v) in enumerate(a.items()):
            plt.subplot(221 + n)
            x = np.sort(np.asarray(v))
            if len(x) > 10000:
                step = int(len(x) / 5000)
                x = x[::step]
            plt.plot(_cdf(v)(x), _cdf(b[k])(x), lw=3, alpha=0.7)
            plt.plot([0, 1], [0, 1], ':', c='k', lw=4, alpha=0.7)
            if t is not None:
                plt.title(t + ' (' + k + ')')
            plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(plot_path)
        plt.clf()
    else:
        x = np.sort(np.asarray(a))
        if len(x) > 10000:
            step = int(len(x) / 5000)
            x = x[::step]
        plt.plot(_cdf(a)(x), _cdf(b)(x), lw=3, alpha=0.7)
        plt.plot([0, 1], [0, 1], ':', c='k', lw=4, alpha=0.7)
        if t is not None:
            plt.title(t)
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(plot_path)
        plt.clf()


def test(num_samples=100000):
    """
    Test code.
    """

    a = np.random.normal(20.0, 5.0, num_samples)
    b = np.random.normal(20.0, 5.0, num_samples)
    pp_plot(a, b)


if __name__ == '__main__':
    test()
