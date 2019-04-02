__all__ = ['gupta_features', 'GuptaExtractor']

import numpy as np

from pybalu.base import FeatureExtractor
# pylint: disable=no-name-in-module
from .geometric_utils import bw_perim
# pylint: enable=no-name-in-module

_gupta_labels = ['Gupta-moment 1',
                 'Gupta-moment 2',
                 'Gupta-moment 3']


def gupta_features(image, *, show=False, labels=False):
    '''\
gupta_features(image, *, show=False, labels=False)

Return an array of with the 3 Gupta moments of a binary image.

Parameters
----------
image : a numpy 2 dimensional int array
    It represents a binary image. Non binary arrays will return nonsensical results.
show : bool, optional
    Wether to print or not messages during execution
labels : bool, optional
    Wether to return a second array that contains the label of each value. 

Returns
-------
labels: ndarray, optional
    A one dimensional string ndarray that contains the labels to each of the features.
    This array is only returned if `labels` is set to True.
features: ndarray
    A float ndarray that contains the 3 Gupta moments extracted from `image`.

See Also
--------
basic_geo : Extract the standard geometric features from a binary image.
fit_ellipse : <Not Implemented Yet>
flusser : Extract the four Flusser moments from binary image.
hugeo : Extract the seven Hu moments from a binary image.

Examples
--------
Load an image and get its binary representation, then proceed to get its features:

>>> from pybalu.feature_extraction import gupta_features
>>> from pybalu.img_processing import segbalu
>>> from pybalu.io import imread
>>> img = imread('testimg.png')
>>> binary_img, _, _ = segbalu(img)
>>> features = gupta_features(binary_img)

Print a binary image features:

>>> from pybalu.io import print_features
>>> labels, features = gupta_features(binary_img, labels=True)
>>> print_features(labels, features)
Gupta-moment 1:  0.41745
Gupta-moment 2: -0.13898
Gupta-moment 3:  2.25704
'''
    if show:
        print('--- extracting Gupta moments...')

    i_perim, j_perim = np.where(bw_perim(image, 4).astype(bool))
    im_perim = i_perim + j_perim * 1j
    ix = i_perim.mean()
    jx = j_perim.mean()
    centre = ix + jx * 1j
    z = np.abs(im_perim - centre)
    m1 = z.mean()

    mur1 = z - m1
    mur2 = mur1 * mur1
    mur3 = mur1 * mur2
    mur4 = mur2 * mur2

    mu2 = mur2.mean()
    mu3 = mur3.mean()
    mu4 = mur4.mean()

    F1 = (mu2 ** .5) / m1
    F2 = mu3 / (mu2 * (mu2 ** .5))
    F3 = mu4 / mu2 ** 2

    gupta_moments = np.array([F1, F2, F3])

    if labels:
        return np.array(_gupta_labels), gupta_moments
    return gupta_moments


class GuptaExtractor(FeatureExtractor):

    def transform(self, X):
        return np.array([gupta_features(x) for x in self._get_iterator(X)])

    def get_labels(self):
        return np.array(_gupta_labels)
