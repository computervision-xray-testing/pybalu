__all__ = ['jfisher']

import numpy as np

def jfisher(features, classification, p=None): 
    m = features.shape[1]
    
    norm = classification.ravel() - classification.min()
    max_class = norm.max() + 1
    
    if p is None:
        p = np.ones(shape=(max_class, 1)) / max_class
        
    # Centroid of all samples
    features_mean = features.mean(0)

    # covariance within class 
    cov_w = np.zeros(shape=(m, m))
    
    # covariance between classes
    cov_b = np.zeros(shape=(m, m))

    for k in range(max_class):
        ii = (norm == k)                                   # indices from class k
        class_features = features[ii,:]                    # samples of class k
        class_mean = class_features.mean(0)                # centroid of class k 
        class_cov = np.cov(class_features, rowvar=False)   # covariance of class k
        
        cov_w += p[k] * class_cov                          # within-class covariance
        
        dif = (class_mean - features_mean).reshape((m, 1))
        cov_b += p[k] * dif @ dif.T                        # between-class covariance
    try:
        return np.trace(np.linalg.inv(cov_w) @ cov_b)
    except np.linalg.LinAlgError:
        return - np.inf


