__all__ = ['huint_features', 'HuIntExtractor']

import numpy as np

from pybalu.base import FeatureExtractor
# pylint: disable=no-name-in-module
from .geometric_utils import moments
# pylint: enable=no-name-in-module


hu_labels = ['Hu-moment-int 1',
             'Hu-moment-int 2',
             'Hu-moment-int 3',
             'Hu-moment-int 4',
             'Hu-moment-int 5',
             'Hu-moment-int 6',
             'Hu-moment-int 7']


def huint_features(image, region=None, *, show=False, labels=False):
    '''\
huint_features(image, region=None, *, show=False, labels=False)

Return an array of with the 7 Hu moments of a grayscale image.

Parameters
----------
image : a numpy 2 dimensional int array
    It represents a grayscale image (or only one color channel).
region: a numpy 2 dimensional binary or boolean array, optional
    Represents the region over which to extract the features. If none is
    given, the whole input image will be analyzed
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
    A float ndarray that contains the 7 Hu moments with intensity extracted from `image`.

See Also
--------
haralick: 
clp:
fourier:
dct: 
lbp:

Examples
--------
Load an image and get its binary representation, then proceed to get its features:

>>> from pybalu.feature_extraction import huint_features
>>> from pybalu.img_processing import segbalu
>>> from pybalu.io import imread
>>> img = imread('testimg.png')
>>> region, _, _ = segbalu(img)
>>> features = huint_features(img, region)

Print a grayscale image features:

>>> from pybalu.io import print_features
>>> labels, features = huint_features(img, labels=True)
>>> print_features(labels, features)
Hu-moment-int 1:  0.38161
Hu-moment-int 2:  0.00845
Hu-moment-int 3:  0.00560
Hu-moment-int 4:  0.00057
Hu-moment-int 5: -0.00000
Hu-moment-int 6: -0.00005
Hu-moment-int 7: -0.00000
'''
    if show:
        print('--- extracting Hu moments with intensity...')

    if region is None:
        region = np.ones_like(image).astype(int)

    useful_img = image[np.where(region)]
    m = moments(region, centered=True)

    if useful_img.any():
        area = useful_img.sum()
    else:
        area = m[0, 0]

    area_sq = area ** 2
    area_25 = area ** 2.5

    n02 = m[0, 2] / area_sq
    n20 = m[2, 0] / area_sq
    n11 = m[1, 1] / area_sq
    n12 = m[1, 2] / area_25
    n21 = m[2, 1] / area_25
    n03 = m[0, 3] / area_25
    n30 = m[3, 0] / area_25

    f1 = n20 + n02
    f2 = (n20-n02)**2 + 4*n11**2
    f3 = (n30-3*n12)**2+(3*n21-n03)**2
    f4 = (n30+n12)**2+(n21+n03)**2
    f5 = (n30-3*n12)*(n30+n12)*((n30+n12)**2 - 3*(n21+n03)**2) + \
        (3*n21-n03)*(n21+n03)*(3*(n30+n12)**2 - (n21+n03)**2)
    f6 = (n20-n02)*((n30+n12)**2 - (n21+n03)**2) + 4*n11*(n30+n12)*(n21+n03)
    f7 = (3*n21-n03)*(n30+n12)*((n30+n12)**2 - 3*(n21+n03)**2) - \
        (n30-3*n12)*(n21+n03)*(3*(n30+n12)**2 - (n21+n03)**2)

    hu_moments = np.array([f1, f2, f3, f4, f5, f6, f7])

    if labels:
        return np.array(hu_labels), hu_moments
    return hu_moments


class HuIntExtractor(FeatureExtractor):

    def transform(self, X):
        return np.array([huint_features(x) for x in self._get_iterator(X)])

    def get_labels(self):
        return np.array(hu_labels)
