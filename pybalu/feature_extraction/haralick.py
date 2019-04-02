__all__ = ['haralick_features', 'HaralickExtractor']

import numpy as np
from collections import Sequence

from pybalu.base import FeatureExtractor
# pylint: disable=no-name-in-module
from .haralick_utils import norm_cooc_mtrx, cooc_features
# pylint: enable=no-name-in-module


def haralick_features(image, region=None, *, distance=3, show=False, labels=False):
    '''\
haralick_features(image, region=None, *, distance=3, show=False, labels=False)

Return an array of with the mean and range of 14 different measures over the
coocurrence matrix of a grayscale image.

Parameters
----------
image : a numpy 2 dimensional float array
    It represents a grayscale image or just one dimension of color (eg: green channel).
    Its values should range from 0 to 255.
region : a numpy 2 dimensional array, optional
    A ndarray of the same dimensions as `image` of boolean type. Only the pixels that are
    True in this array will be analyzed. If it is not set, as a default the whole image
    will be used.
distance : (int, array_like), optional
    If `distance` is an int, it defines the distance for which to analyze pixel coocurrence.
    If `distance` is an array_like, it is expected to be a sequence of ints, all of which
    define the distance for which to analyze pixel coocurrence. All features are then extracted
    for each of these distances.
    Default value is 3
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
    A float ndarray that contains the 28 * len(`distance`) values of various measures
    extracted from the coocurrence matrix of `image`.

See Also
--------
clp : <Not Implemented Yet>
dct : <Not Implemented Yet>
fourier : Extracts the fourier features of an image.
gabor : <Not Implemented Yet> 
lbp : <Not Implemented Yet>
Examples
--------
Load an image and get its grayscale representation, then proceed to get its features
for 3 distances:

>>> from pybalu.feature_extraction import haralick_features
>>> from pybalu.io import imread
>>> img = imread('testimg.png')
>>> features = haralick_features(img, distance=[3, 4, 5])

Print the haralick_features features for distance 12:

>>> from pybalu.io import print_features
>>> labels, features = haralick_features(img, distance=12, labels=True)
>>> print_features(labels, features)
Tx 1 , d 12 (mean) :  0.43780
Tx 2 , d 12 (mean) :  13.47477
Tx 3 , d 12 (mean) :  72.48584
Tx 4 , d 12 (mean) :  13.47477
Tx 5 , d 12 (mean) :  0.69266
Tx 6 , d 12 (mean) :  4.77283
Tx 7 , d 12 (mean) :  3.66382
Tx 8 , d 12 (mean) :  1.10901
Tx 9 , d 12 (mean) :  1.33820
Tx 10, d 12 (mean) :  0.05783
Tx 11, d 12 (mean) :  0.91347
Tx 12, d 12 (mean) : -0.03575
Tx 13, d 12 (mean) :  0.04514
Tx 14, d 12 (mean) :  0.05285
Tx 1 , d 12 (range):  0.49309
Tx 2 , d 12 (range):  15.46938
Tx 3 , d 12 (range):  107.59946
Tx 4 , d 12 (range):  15.46938
Tx 5 , d 12 (range):  0.79220
Tx 6 , d 12 (range):  4.79386
Tx 7 , d 12 (range):  3.68360
Tx 8 , d 12 (range):  1.12357
Tx 9 , d 12 (range):  1.36838
Tx 10, d 12 (range):  0.07199
Tx 11, d 12 (range):  0.96111
Tx 12, d 12 (range):  0.13119
Tx 13, d 12 (range):  0.16454
Tx 14, d 12 (range):  0.20168
'''
    image = image.astype('double')

    if region is None:
        region = np.ones_like(image).astype(int)

    if show:
        print('--- extracting Haralick texture features...')

    if not isinstance(distance, Sequence):
        distance = [distance]
    dseq = np.array(distance, dtype=int).ravel()

    features = []
    label_list = []
    for d in dseq:
        cooc = norm_cooc_mtrx(image, region, d)
        feats = np.vstack([cooc_features(P) for P in cooc])
        features.append(np.hstack([feats.mean(0), np.abs(feats).max(0)]))
        if labels:
            label_list.extend(
                [f"Tx {i:<2d}, d {d:<2d} (mean)" for i in range(1, 15)])
            label_list.extend(
                [f"Tx {i:<2d}, d {d:<2d} (range)" for i in range(1, 15)])

    haralick_features = np.hstack(features)

    if labels:
        return np.array(label_list), haralick_features
    return haralick_features


class HaralickExtractor(FeatureExtractor):
    def __init__(self, *, distance=3, show=False):
        self.show = show
        self.distance = distance

    def transform(self, X):
        params = self.get_params()
        params.update({'show': False})
        return np.array([haralick_features(x, **params) for x in self._get_iterator(X, desc='haralick')])

    def get_labels(self):
        if isinstance(self.distance, Sequence):
            distance = self.distance
        else:
            distance = [self.distance]
        labels = []
        for d in distance:
            labels.extend(
                [f"Tx {i:<2d}, d {d:<2d} (mean)" for i in range(1, 15)])
            labels.extend(
                [f"Tx {i:<2d}, d {d:<2d} (range)" for i in range(1, 15)])
        return np.array(labels)
