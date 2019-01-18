__all__ = ['clean']

import numpy as np
import warnings

def clean(features, show=False):
    n_features = features.shape[1]
    ip = np.ones(n_features, dtype=int)

    # cleaning correlated features
    warnings.filterwarnings('ignore')
    C = np.abs(np.corrcoef(features, rowvar=False))
    idxs = np.vstack(np.where(C > .99))
    
    # remove pairs of same feature ( feature i will have a correlation of 1 whit itself )
    idxs = idxs[:, idxs[0,:] != idxs[1,:]]
    
    # remove correlated features
    if idxs.size > 0:
        ip[np.max(idxs, 0)] = 0
    
    # remove constant features
    s = features.std(axis=0, ddof=1)
    ip[s < 1e-8] = 0
    p = np.where(ip.astype(bool))[0]

    if show:
        print(f'Clean: number of features reduced from {n_features} to {p.size}.')

    return p