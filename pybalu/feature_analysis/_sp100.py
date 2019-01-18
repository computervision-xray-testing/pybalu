__all__ = ['sp100']


def sp100(features, classification):
    norm = (classification.flatten() -  classification.min()).astype(bool)
    max_class = norm.max()

    if max_class > 1:
        raise Exception('sp100 works only for two classes')

    c1_features = features[norm == 1,:]

    min_feat = c1_features.min(0)
    max_feat = c1_features.max(0)

    z1 = (features >= min_feat) & (features <= max_feat)
    dr = z1.all(1)
    
    TP = (dr * norm).sum()
    FP = dr.sum() - TP

    TN = (~dr * ~norm).sum()
    return TN / (FP+TN)