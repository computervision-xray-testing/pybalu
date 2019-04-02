__all__ = ['fourier_des_features', 'FourierDesExtractor']

import numpy as np
from pybalu.base import FeatureExtractor
# pylint: disable=no-name-in-module
from .geometric_utils import bw_boundaries
# pylint: enable=no-name-in-module


def fourier_des_features(image, *, n_des=16, show=False, labels=False):
    '''\
fourier_des_features(image, *, n_des=16, show=False, labels=False)

Return an array of with the Fourier descriptors of a binary image.

Parameters
----------
image : a numpy 2 dimensional int array
    It represents a binary image. Non binary arrays will return nonsensical results.
n_des : int, optional
    The number of descriptors to extract from the image. Default is 16
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
    A float ndarray that contains `n_des` Fourier descriptors extracted from `image`.

See Also
--------
fit_ellipse : <Not Implemented Yet>
flusser : Extract the four Flusser moments from binary image.
gupta : Extract the three Gupta moments from binary image.
hugeo : Extract the seven Hu moments from a binary image.

Examples
--------
Load an image and get its binary representation, then proceed to get its features:

>>> from pybalu.feature_extraction import fourier_des_features
>>> from pybalu.img_processing import segbalu
>>> from pybalu.io import imread
>>> img = imread('testimg.png')
>>> binary_img, _, _ = segbalu(img)
>>> features = fourier_des_features(binary_img)

Print a binary image features:

>>> from pybalu.io import print_features
>>> labels, features = fourier_des_features(binary_img, labels=True)
>>> print_features(labels, features)
Fourier-des  1:  0.66428
Fourier-des  2:  0.92453
Fourier-des  3:  1.00341
Fourier-des  4:  0.49639
Fourier-des  5:  1.05206
Fourier-des  6:  0.34365
Fourier-des  7:  0.06147
Fourier-des  8:  0.12616
Fourier-des  9:  0.01778
Fourier-des 10:  0.10414
Fourier-des 11:  0.04698
Fourier-des 12:  0.13555
Fourier-des 13:  0.42163
Fourier-des 14:  0.06778
Fourier-des 15:  0.01811
Fourier-des 16:  0.05512
'''
    if show:
        print('--- extracting Fourier descriptors...')

    B = bw_boundaries(image)
    V = B[:, 1] + 1j * B[:, 0]
    m = B.shape[0]

    r = np.zeros(m, dtype=complex)
    phi = np.zeros(m)
    dphi = np.zeros(m)
    l = np.zeros(m)
    dl = np.zeros(m)

    r[0] = V[0] - V[m-1]
    r[1:] = V[1:] - V[:m-1]

    dl = np.abs(r)
    phi = np.angle(r)

    dphi[:m-1] = np.mod(phi[1:] - phi[:m-1] + np.pi, 2 * np.pi) - np.pi
    dphi[m-1] = np.mod(phi[0] - phi[m-1] + np.pi, 2 * np.pi) - np.pi

    l[0] = dl[0]
    for k in range(1, m):
        l[k] = l[k-1] + dl[k]

    l = l * (2 * np.pi / l[m-1])
    descriptors = np.zeros(n_des)

    for n in range(1, n_des + 1):
        an = (dphi * np.sin(l * n)).sum()
        bn = (dphi * np.cos(l * n)).sum()
        an = -an / n / np.pi
        bn = bn / n / np.pi
        imagi = an + 1j * bn
        descriptors[n-1] = np.abs(imagi)

    if labels:
        return np.array([f'Fourier-des {n+1:>2d}' for n in range(n_des)]), descriptors
    return descriptors


class FourierDesExtractor(FeatureExtractor):
    def __init__(self, *, n_des=16):
        self.n_des = n_des

    def transform(self, X):
        params = self.get_params()
        return np.array([fourier_des_features(x, **params) for x in self._get_iterator(X)])

    def get_labels(self):
        return np.array([f'Fourier-des {n+1:>2d}' for n in range(self.n_des)])
