__all__ = ['crossval']
import numpy as np
from pybalu.classification import structure
from .performance import performance
from scipy.stats import t, norm


def crossval(features, classification, classifier, n_folds, *, confidence=.95, classifier_opts=None, strat=False):
    '''\
    crossval(features, classification, classifier, n_folds, *, confidence=.95, options=None, strat=False)

    n-fold Cross Validation in n groups of given samples and classification 
    according to the given classifier.

    Parameters 
    ----------
    features: numerical 2 dimensional ndarray
        Corresponds to the sample features over which to perform cross-validation.
    classification: integer ndarray
        Corresponds to the classification of the given samples.
    classifier: Class that defines train and predict methods
        The classifier to train or predict with. It can be a reference to a class that defines two methods:
            - The `fit` method should accept a subset of `features` and `classification` as input and 
            should train the classifier.
            - The `predict` method should accept a subset of `features` as input and return a numerical array-like.
        The classifier will be instanced with `classifier_opts` as initializer values.
    n_folds: integer
        Number of groups (folders) of the cross-validation
    condifence: float in range [0, 1]
        The probability of the confidence interval.
    classifier_opts: dictionary, optional
        Represents the keyword arguments used to instantiate the classifier.
    strat: bool, optional
        If set to True, the classes are stratified within the folds. Default value is False

    Returns
    -------
    performance: float in range [0, 1]
        The mean performance of classifier.
    confidence_interval: (float, float)
        Confidence intervals for `confidence` probability.

    Examples
    --------
    ( TODO )
    '''
    N = features.shape[0]
    if N != classification.size:
        raise ValueError(
            f'Dim 0 of `features` must be equal to `classification` size. ({N} != {classification.size})')

    # shuffle features and then sort features by classification
    perm = np.random.permutation(np.arange(N))
    features = features[perm]
    classification = classification[perm].squeeze()
    idx_sort = np.argsort(classification)
    features = features[idx_sort]
    classification = classification[idx_sort]

    classes, counts = np.unique(classification, return_counts=True)
    n_cls = len(classes)

    if strat and any(counts < n_folds):
        raise ValueError(
            'Some classes have less occurences than the number of folds!')

    if n_folds == 1:
        f_train = features
        f_test = features
        d_train = classification
        d_test = classification

        d_fitted = structure(classifier,
                             train_data=f_train,
                             test_data=f_test,
                             train_classes=d_train,
                             classifier_opts=classifier_opts)

        return performance(d_test, d_fitted), (0., 0.)
    if strat:
        # generate blocks
        cls_size = counts // n_folds
        starts = np.tril(np.ones((n_cls, n_cls), dtype=int), -1) @ counts
        ends = np.hstack([starts[1:], N])
        idx_vec = np.hstack([np.arange(s, s+v)
                             for s, v in zip(starts, cls_size)])
        diff = np.repeat(cls_size, cls_size)
        starts += cls_size * (n_folds - 1)
        first_blocks = [idx_vec + diff*v for v in range(n_folds-1)]
        last_block = [np.hstack([np.arange(s, e)
                                 for s, e in zip(starts, ends)])]
        blocks = first_blocks + last_block
    else:
        block_size = N // n_folds
        first_blocks = [np.arange(block_size*i, block_size*(i+1))
                        for i in range(n_folds-1)]
        last_block = [np.arange(block_size*(n_folds-1), N)]
        blocks = first_blocks + last_block

    pp = np.zeros(n_folds)
    for n, block in enumerate(blocks):
        other = np.delete(np.arange(N), block)
        f_train = features[block]
        f_test = features[other]
        d_train = classification[block]
        d_test = classification[other]
        d_fitted, _ = structure(classifier,
                                train_data=f_train,
                                test_data=f_test,
                                train_classes=d_train,
                                classifier_opts=classifier_opts)

        pp[n] = performance(d_test, d_fitted)

    pm = pp.mean()
    std = np.sqrt((pm * (1 - pm) / N))
    if n_folds > 20:
        return pm, norm.interval(confidence, loc=pm, scale=std)

    return pm, t.interval(confidence, n_folds - 1, loc=pm, scale=std)
