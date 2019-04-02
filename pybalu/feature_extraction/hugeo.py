__all__ = ['hugeo_features', 'HuGeoExtractor']

import numpy as np

from pybalu.base import FeatureExtractor
# pylint: disable=no-name-in-module
from .geometric_utils import moments
# pylint: enable=no-name-in-module


hu_labels = ['Hu-moment 1',
             'Hu-moment 2',
             'Hu-moment 3',
             'Hu-moment 4',
             'Hu-moment 5',
             'Hu-moment 6',
             'Hu-moment 7']


def hugeo_features(image, *, show=False, labels=False):
    '''\
hugeo_features(image, *, show=False, labels=False)

Return an array of with the 7 Hu moments of a binary image.

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
    A float ndarray that contains the 7 Hu moments extracted from `image`.

See Also
--------
basic_geo : Extract the standard geometric features from a binary image.
fit_ellipse : <Not Implemented Yet>
flusser : Extract the four Flusser moments from binary image.
gupta : Extract the three Gupta moments from a binary image.

Examples
--------
Load an image and get its binary representation, then proceed to get its features:

>>> from pybalu.feature_extraction import hugeo_features
>>> from pybalu.img_processing import segbalu
>>> from pybalu.io import imread
>>> img = imread('testimg.png')
>>> binary_img, _, _ = segbalu(img)
>>> features = hugeo_features(binary_img)

Print a binary image features:

>>> from pybalu.io import print_features
>>> labels, features = hugeo_features(binary_img, labels=True)
>>> print_features(labels, features)
Hu-moment 1:  0.38161
Hu-moment 2:  0.00845
Hu-moment 3:  0.00560
Hu-moment 4:  0.00057
Hu-moment 5: -0.00000
Hu-moment 6: -0.00005
Hu-moment 7: -0.00000
'''
    if show:
        print('--- extracting Hu moments...')

    m = moments(image, centered=True)

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


class HuGeoExtractor(FeatureExtractor):

    def transform(self, X):
        return np.array([hugeo_features(x) for x in self._get_iterator(X)])

    def get_labels(self):
        return np.array(hu_labels)
