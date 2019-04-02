__all__ = ['basic_int_features', 'BasicIntExtractor']

import numpy as np
import scipy.stats
# pylint: disable=no-name-in-module
from pybalu.img_processing import fst_deriv, snd_deriv
from .geometric_utils import bw_perim
# pylint: enable=no-name-in-module
from pybalu.base import FeatureExtractor

int_labels = ['Intensity Mean',
              'Intensity StdDev',
              'Intensity Kurtosis',
              'Intensity Skewness',
              'Mean Laplacian',
              'Mean Boundary Gradient']


def basic_int_features(image, region=None, *, mask=15, show=False, labels=False):
    '''\
basic_int_features(image, region=None, *, mask=15, show=False, labels=False)

Return an array of with the basic intensity features of a grayscale image.

Parameters
----------
image : a numpy 2 dimensional float array
    It represents a grayscale image or just one dimension of color (eg: green channel)
region : a numpy 2 dimensional array, optional
    A ndarray of the same dimensions as `image` of boolean type. Only the pixels that are
    True in this array will be analyzed. If it is not set, as a default the whole image
    will be used.
mask: int, optional
    Gauss mask value for gradient computation. Default is 15
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
    A float ndarray that contains all 6 intensity features extracted from pixels in 
    `image` contained in the `region` given.

See Also
--------
clp : <Not Implemented Yet>
dct : <Not Implemented Yet>
fourier : Extracts the fourier features of an image.
gabor : <Not Implemented Yet> 
haralick : Extracts 28 haralick texture features from an image.
lbp : <Not Implemented Yet>

Examples
--------
Load an image on its grayscale representation, then proceed to get its features:

>>> from pybalu.feature_extraction import basic_int_features
>>> from pybalu.img_processing import segbalu
>>> from pybalu.io import imread
>>> img = imread('testimg.png', flatten=True) # to grayscale
>>> region, _, _ = segbalu(img)
>>> features = basic_int_features(img, region)

Print image features:

>>> from pybalu.io import print_features
>>> labels, features = basic_int_features(img, region, labels=True)
>>> print_features(labels, features)
Intensity Mean        :  224.86193
Intensity StdDev      :  66.68996
Intensity Kurtosis    :  6.26327
Intensity Skewness    : -2.15027
Mean Laplacian        : -23.24741
Mean Boundary Gradient:  48.57239
'''
    if region is None:
        region = np.ones(shape=image.shape, dtype=int)
    if show:
        print('--- extracting basic intensity features...')

    r_perim = bw_perim(region, 4).astype(bool)
    region = region.astype(bool)

    image = image.astype(float)

    im1, _, _ = fst_deriv(image, mask=mask)
    im2 = snd_deriv(image)

    if not region.all():
        boundary_gradient = np.abs(im1[r_perim]).mean()
    else:
        boundary_gradient = -1

    useful_img = image[region]

    intensity_mean = useful_img.mean()
    intensity_std = useful_img.std(ddof=1)
    intensity_kurtosis = scipy.stats.kurtosis(useful_img, fisher=False)
    intensity_skewness = scipy.stats.skew(useful_img)
    mean_laplacian = im2[region].mean()

    int_features = np.array([intensity_mean,
                             intensity_std,
                             intensity_kurtosis,
                             intensity_skewness,
                             mean_laplacian,
                             boundary_gradient])

    if labels:
        return np.array(int_labels), int_features
    return int_features


class BasicIntExtractor(FeatureExtractor):
    def __init__(self, *, mask=15):
        self.mask = mask

    def transform(self, X):
        params = self.get_params()
        return np.array([basic_int_features(x, **params) for x in self._get_iterator(X)])

    def get_labels(self):
        return np.array(int_labels)
