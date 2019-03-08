import numpy as np
from pybalu.base import FeatureExtractor
__all__ = ['all', 'AllExtractor']


def all(image):
    return image.flatten()


class AllExtractor(FeatureExtractor):
    def transform(self, X):
        return np.array([all(x) for x in self._get_iterator(X)])
