import numpy as np
from pybalu.classification import structure
from pybalu.performance_eval import performance
from .sp100 import sp100
from .jfisher import jfisher

__all__ = ['score']


def score(features, classification, *, method='fisher', param=None):
    if param is None:
        dn = classification.max() - classification.min() + 1  # number of classes
        p = np.ones((dn, 1)) / dn
    else:
        p = param

    if method == 'mi':  # mutual information
        raise NotImplementedError()

    # maximal relevance
    elif method == 'mr':
        raise NotImplementedError()

    # minimal redundancy and maximal relevance
    elif method == 'mrmr':
        raise NotImplementedError()

    # fisher
    elif method == 'fisher':
        return jfisher(features, classification, p)

    elif method == 'sp100':
        return sp100(features, classification)

    else:
        return 0
