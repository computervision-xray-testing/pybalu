import numpy as np
from pybalu.classification import structure
from pybalu.performance_eval import performance
from ._mutual_info import mutual_info
from ._relevance import relevance
from ._mRMR import mRMR
from ._sp100 import sp100
from ._jfisher import jfisher

__all__ = ['score']


def score(features, classification, *, method='fisher', param=None):
    if param is None:
        dn = classification.max() - classification.min() + 1 # number of classes
        p = np.ones((dn, 1)) / dn
    else:
        p = param

    if method == 'mi': # mutual information
        return mutual_info(features, classification, p)
    
    # maximal relevance
    elif method == 'mr': 
        return relevance(features, classification, p)

    # minimal redundancy and maximal relevance
    elif method == 'mrmr': 
        return mRMR(features, classification, p)

    # fisher
    elif method == 'fisher':
        return jfisher(features, classification, p)

    elif method == 'sp100':
        return sp100(features, classification)

    else:
        return 0