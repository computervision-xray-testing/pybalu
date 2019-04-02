__all__ = ['sfs', 'SFSWarning']

import numpy as np
import warnings
import tqdm
from pybalu.feature_analysis.score import score


class SFSWarning(Warning):
    pass


def sfs(features, classes, n_features, *, force=False, method='fisher', options=None, show=False):

    N, M = features.shape
    remaining_feats = set(np.arange(M))
    selected = list()
    curr_feats = np.zeros((N, 0))
    if options is None:
        options = dict()

    def _calc_score(i):
        feats = np.hstack([curr_feats, features[:, i].reshape(-1, 1)])
        return score(feats, classes, method=method, **options)

    if show:
        _range = tqdm.trange(
            n_features, desc='Selecting Features', unit_scale=True, unit=' features')
    else:
        _range = range(n_features)

    for _ in _range:
        new_selected = max(remaining_feats, key=_calc_score)
        selected.append(new_selected)
        remaining_feats.remove(new_selected)
        curr_feats = np.hstack(
            [curr_feats, features[:, new_selected].reshape(-1, 1)])

    return np.array(selected)
