__all__ = ['lbp_features', 'LBPExtractor']

import numpy as np
from pybalu.misc import im2col
from skimage.feature import local_binary_pattern as _lbp

from pybalu.base import FeatureExtractor


def lbp_features(image, region=None, *, show=False, labels=False, **kwargs):
    '''\
    lbp_features(image, region=None, *, show=False, labels=False, **kwargs)

    Calculates the Local Binary Patterns over a regular grid of patches. It returns an array
    of uniform lbp82 descriptors for `image`, made by the concatenating histograms of each grid 
    cell in the image. Grid size is `hdiv` * `vdiv`

    Parameters 
    ----------
    image: 2 dimensional ndarray
        It represents a grayscale image or just one dimension of color (eg: green channel)
    region: 2 dimensional ndarray, optional
        If not None, must be of same dimensions an image and of a bool-like type. All the pixels
        not set in this region will be set to 0 on `image` before performing LBP calculations.
    hdiv: positive integer
        Number of horizontal divisions to perform on image.
    vdiv: positive integer
        Number of vertical divisions to perform on image.
    samples: positive integer, optional
        Number of circularly symmetric neighbour set points (quantization of the angular space).
        default value is 8 (all neighbours in 2d)
    norm: bool, optional
        If set to True, the output array is normalized so that the sum of all its features 
        equals 1. Default value is False.
    mapping: string, optional
        Reprsents the kind of LBP performed over each block. Options are:
            - 'default': original local binary pattern which is gray scale but not
               rotation invariant.
            - 'ror': extension of default implementation which is gray scale and
               rotation invariant.
            - 'uniform': improved rotation invariance with uniform patterns and
               finer quantization of the angular space which is gray scale and
               rotation invariant.
            - 'nri_uniform': non rotation-invariant uniform patterns variant
               which is only gray scale invariant.
        default value is 'default'.
    radius: integer, optional
        Radius of circle (spatial resolution of the operator). Default is set depending on number 
        of samples
    ret_centers: bool, optional
        If set to True, an array with the centers of each block over which LBP was performed is 
        returned. Default value is False.
    show: bool, optional
        Wether to print or not messages during execution. Default is False.
    labels: bool, optional
        Wether to return a second array that contains the label of each value. 

    Returns
    -------
    labels: ndarray, optional
        A one dimensional string ndarray that contains the labels to each of the features.
        This array is only returned if `labels` is set to True.
    features: ndarray
        A numeric ndarray that contains (`hdiv`*`vdiv`) * `num_patterns` features extracted from 
        `image`. `num_patterns` depends on `mapping` and `samples` and is usually 10, 59 or 256.
    x_centers: integer ndarray, optional
        A one dimensional array of size `hdiv`*`vdiv` that contains the x center 
        coordinate of the blocks generated on image division. only returned if `ret_centers` is 
        set to True.
    y_centers: integer ndarray, optional
        A one dimensional array of size `hdiv`*`vdiv` that contains the y center 
        coordinate of the blocks generated on image division. only returned if `ret_centers` is 
        set to True.

    Examples
    --------
    ( TODO )
    '''
    vdiv = kwargs.pop('vdiv', None)
    hdiv = kwargs.pop('hdiv', None)
    if vdiv is None or hdiv is None:
        raise ValueError('`vdiv` and `hdiv` must be given to lbp.')

    if region is None:
        region = np.ones_like(image)

    samples = kwargs.pop('samples', 8)
    normalize = kwargs.pop('norm', False)
    integral = kwargs.pop('integral', False)
    max_d = kwargs.pop('max_d', None)
    if integral and max_d is None:
        raise ValueError('`max_d` must be set if `integral` is set to True.')

    weight = kwargs.pop('weight', 0)
    mapping = kwargs.pop('mapping', 'default')

    if mapping == 'ror' or mapping == 'default':
        num_patterns = 2 ** samples
    elif mapping == 'uniform':
        num_patterns = samples + 2
    elif mapping == 'nri_uniform':
        num_patterns = 59
    else:
        raise ValueError(f"Unknown mapping: '{mapping}'")

    radius = kwargs.pop('radius', None)
    if radius is None:
        radius = np.log(samples) / np.log(2) - 1

    ret_centers = kwargs.pop('ret_centers', False)

    if len(kwargs) > 0:
        unknowns = "'" + "', '".join(kwargs.keys()) + "'"
        raise ValueError(f"Unknown options given to lbp: {unknowns}")

    if show:
        print('--- extracting local binary patterns features...')
    label = 'LBP'

    # set pixels not within region to 0
    image = image.copy()
    image[~region.astype(bool)] = 0

    code_img = _lbp(image, P=samples, R=radius, method=mapping)
    n, m = code_img.shape
    N, M = image.shape
    Ilbp = np.zeros_like(image)
    i1 = round((N - n)/2)
    j1 = round((M - m)/2)
    Ilbp[i1:i1+n, j1:j1+m] = code_img

# TODO:
#     if integral:
#         hx = inthist(Ilbp+1, max_d)

    ylen = int(np.ceil(n / vdiv))
    xlen = int(np.ceil(m / hdiv))
    grid_img = im2col(code_img, ylen, xlen) + 1

    if weight > 0:
        label = 'W-' + label
        pass
    else:
        desc = np.vstack([np.histogram(grid_img[:, i], num_patterns)[0]
                          for i in range(grid_img.shape[1])])

    lbp_feats = desc.ravel()
    N, M = desc.shape

    if normalize:
        lbp_feats = lbp_feats / lbp_feats.sum()

    if not labels and not ret_centers:
        return lbp_feats

    if labels:
        lbp_labels = []
        for i in range(N):
            for j in range(M):
                lbp_labels.append(
                    f"{label}({i+1},{j+1:>2d}) [{samples},'{mapping}']")
        ret = np.array(lbp_labels), lbp_feats
    else:
        ret = (lbp_feats,)

    if ret_centers:
        dx = 1 / hdiv
        dy = 1 / vdiv
        x = np.linspace(dx / 2, 1 - dx / 2, hdiv)
        y = np.linspace(dy / 2, 1 - dy / 2, vdiv)
        ret = (ret,) + (x, y)

    return ret


class LBPExtractor(FeatureExtractor):
    def __init__(self, *, hdiv=None, vdiv=None, samples=8, norm=False, mapping="default", radius=None):
        self.hdiv = hdiv
        self.vdiv = vdiv
        self.samples = samples
        self.norm = norm
        self.mapping = mapping
        self.radius = radius

    def transform(self, X):
        params = self.get_params()
        return np.array([lbp_features(x, **params) for x in self._get_iterator(X)])

    def get_labels(self):
        return "LBP"
