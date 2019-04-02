import numpy as np
import scipy.ndimage as img
# pylint: disable=no-name-in-module
from .geometric_utils import bbox, convex_area, moments, perimeter
# pylint: enable=no-name-in-module
from pybalu.base import FeatureExtractor

__all__ = ['basic_geo_features', 'BasicGeoExtractor']

geo_labels = ['center of grav i [px]',
              'center of grav j [px]',
              'Height [px]',
              'Width [px]',
              'Area [px]',
              'Perimeter [px]',
              'Roundness',
              'Danielsson factor',
              'Euler Number',
              'Equivalent Diameter [px]',
              'MajorAxisLength [px]',
              'MinorAxisLength [px]',
              'Orientation [deg]',
              'Solidity',
              'Extent',
              'Eccentricity',
              'Convex Area [px]',
              'Filled Area [px]']


def basic_geo_features(image, *, show=False, labels=False):
    '''\
basic_geo_features(image, *, show=False, labels=False)

Return an array of with the standard geometric features of a binary image.

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
    A float ndarray that contains all 18 geometric features extracted from `image`.

See Also
--------
fit_ellipse : <Not Implemented Yet>
flusser : Extract the four Flusser moments from binary image.
gupta : Extract the three Gupta moments from binary image.
hugeo : Extract the seven Hu moments from a binary image.

Examples
--------
Load an image and get its binary representation, then proceed to get its features:

>>> from pybalu.feature_extraction import basic_geo_features
>>> from pybalu.img_processing import segbalu
>>> from pybalu.io import imread
>>> img = imread('testimg.png')
>>> binary_img, _, _ = segbalu(img)
>>> features = basic_geo_features(binary_img)

Print a binary image features:

>>> from pybalu.io import print_features
>>> labels, features = basic_geo_features(binary_img, labels=True)
>>> print_features(labels, features)
center of grav i [px]   :  51.49202
center of grav j [px]   :  74.95331
Height [px]             :  72.00000
Width [px]              :  90.00000
Area [px]               :  2506.00000
Perimeter [px]          :  534.00000
Roundness               :  0.11044
Danielsson factor       :  13.58911
Euler Number            :  1.00000
Equivalent Diameter [px]:  56.48662
MajorAxisLength [px]    :  97.43620
MinorAxisLength [px]    :  76.20511
Orientation [grad]      :  2.27367
Solidity                :  0.49767
Extent                  :  0.38673
Eccentricity            :  0.62315
Convex Area [px]        :  5035.50000
Filled Area [px]        :  2506.00000
'''

    if show:
        print('--- extracting standard geometric features...')

    # Center of gravity
    I, J = np.where(image.astype(bool))
    i_m = I.mean()
    j_m = J.mean()

    # Area
    area = I.size

    # Perimeter
    L = perimeter(image)

    # Roundness
    roundness = 4 * area * np.math.pi / L ** 2 if L > 0 else np.nan

    # 'EulerNumber'
    filled_region = img.binary_fill_holes(image)
    _, n_objects = img.label(image)
    _, n_holes = img.label(filled_region - image)
    euler = n_objects - n_holes

    # 'FilledArea'
    filled_area = filled_region.sum()

    # 'ConvexArea'
    c_area = convex_area(image)

    # 'EquivDiameter'
    equiv_diameter = np.math.sqrt(4 * area / np.math.pi)

    # 'Solidity'
    solidity = area / c_area if c_area > 0 else np.nan

    # Bounding box
    row_slice, col_slice = bbox(image)
    height = row_slice.stop - row_slice.start
    width = col_slice.stop - col_slice.start
    bbox_area = height * width

    # 'Extent'
    extent = area / bbox_area if bbox_area > 0 else np.nan

    # MajorAxisLength, MinorAxisLength, Orientation and Eccentricity
    if area > 0:
        mu = moments(image, centered=True, order=2)

        a = mu[2, 0] / mu[0, 0]
        b = mu[1, 1] / mu[0, 0]
        c = mu[0, 2] / mu[0, 0]

        # eigen values of inertia tensor
        l1 = (a + c) / 2 + np.sqrt(4 * b ** 2 + (a - c) ** 2) / 2
        l2 = (a + c) / 2 - np.sqrt(4 * b ** 2 + (a - c) ** 2) / 2

        major_axis_len = 4 * np.sqrt(l1)
        minor_axis_len = 4 * np.sqrt(l2)
        eccentricity = 0 if l1 == 0 else np.sqrt(1 - l2 / l1)
        orientation = .5 if a - \
            c == 0 else np.math.atan2(2 * b, (a - c)) / np.pi
        if orientation < 0:
            orientation = (orientation + 1) * 90
        else:
            orientation = (orientation - 1) * 90

        # Danielsson shape factor (see Danielsson, 1977)
        td = img.morphology.distance_transform_cdt(image)
        dm = td[image.astype(bool)].mean()
        danielsson = area / 9 / np.pi / dm**2
    else:
        major_axis_len = np.nan
        minor_axis_len = np.nan
        orientation = np.nan
        eccentricity = np.nan
        danielsson = np.nan

    geo_features = np.array([i_m,
                             j_m,
                             height,
                             width,
                             area,
                             L,
                             roundness,
                             danielsson,
                             euler,
                             equiv_diameter,
                             major_axis_len,
                             minor_axis_len,
                             orientation,
                             solidity,
                             extent,
                             eccentricity,
                             c_area,
                             filled_area])

    if labels:
        return np.array(geo_labels), geo_features
    return geo_features


class BasicGeoExtractor(FeatureExtractor):

    def transform(self, X):
        return np.array([basic_geo_features(x) for x in self._get_iterator(X)])

    def get_labels(self):
        return np.array(geo_labels)
