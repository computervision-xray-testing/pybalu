__all__ = ['stratify']

import numpy as np

def stratify(labels, portion):

    if portion < 0 or portion > 1:
        raise ValueError(f'portion must be a value between 0 and 1, not {portion}')
    
    labels = labels.squeeze()
    np.unique(labels, return_counts=True)

    idxs_chosen = []
    idxs_not_chosen = []
    for cls, count in zip(*np.unique(labels, return_counts=True)):
        useful_idxs = np.where(labels == cls)[0]
        np.random.shuffle(useful_idxs)
        cls_portion = int(np.round(count * portion))
        idxs_chosen.append(useful_idxs[:cls_portion])
        idxs_not_chosen.append(useful_idxs[cls_portion:])


    return np.hstack(idxs_chosen), np.hstack(idxs_not_chosen)

