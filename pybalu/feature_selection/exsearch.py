__all__ = ['exsearch']

import numpy as np
from itertools import combinations
from pybalu.feature_analysis.score import score
import warnings

import tqdm


def choose(n, k):
    return int(np.math.factorial(n) / (np.math.factorial(n - k) * np.math.factorial(k)))


def exsearch(features, classes, n_features, *, method='fisher', options=None, show=False):
    if options is None:
        options = dict()

    tot_feats = features.shape[1]
    N = choose(tot_feats, n_features)

    if N > 10000:
        warnings.warn(
            f'Doing more than 10.000 iterations ({N}). This may take a while...')

    def _calc_score(ii):
        feats = features[:, ii]
        return score(feats, classes, method=method, **options)

    _combinations = combinations(range(tot_feats), n_features)

    if show:
        _combinations = zip(tqdm.trange(N,
                                        desc='Combinations checked',
                                        unit_scale=True,
                                        unit=' combinations'),
                            _combinations)

        _combinations = (ii for _, ii in _combinations)

    chosen_feats = max(_combinations, key=_calc_score)

    return np.array(chosen_feats)
