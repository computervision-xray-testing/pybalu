from sklearn.base import BaseEstimator, TransformerMixin
import tqdm
from sklearn.pipeline import make_pipeline

__all__ = ['FeatureExtractor', 'ImageProcessor']


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Base class for all feature extractors in PyBalu


    Notes (from sklearn.base.BaseEstimator)
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    def __init__(self):
        self.show = False

    def _get_iterator(self, X, *, desc=None):
        if self.show:
            if desc is not None:
                desc = f'Extracting Features({desc})'
            else:
                desc = 'Extracting Features'
            return tqdm.tqdm(X, desc=desc, total=len(X), unit_scale=True)
        return iter(X)

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged
        This method is just there to implement the usual API and hence
        work in pipelines.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Training data.
        """
        return self


class ImageProcessor(BaseEstimator, TransformerMixin):
    """Base class for all image transformers in PyBalu


    Notes (from sklearn.base.BaseEstimator)
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    def __init__(self):
        self.show = False

    def _get_iterator(self, X, *, desc=None):
        if self.show:
            if desc is not None:
                desc = f'Processing Images({desc})'
            else:
                desc = 'Processing Images'
            return tqdm.tqdm(X, desc=desc, total=len(X), unit_scale=True)
        return iter(X)

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged
        This method is just there to implement the usual API and hence
        work in pipelines.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Training data.
        """
        return self
