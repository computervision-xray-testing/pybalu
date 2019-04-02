__all__ = ['segbalu', 'SegBaluSegmentator']

import numpy as np
from pybalu.base import ImageProcessor
from skimage.filters import threshold_otsu
from .rgb2hcm import rgb2hcm
from .morphoreg import morphoreg


def segbalu(image, p=-.05):
    '''\
    segbalu(image, p=-.05)

    Segmentation of an object with homogeneous background.

    Parameters
    ----------
    image: 3 dimensional ndarray
        The RGB input image.
    p: float between -1 and 1, optional
        A positive value is used to dialte the segmentation. A negative erodes.
        default value is -0.05

    Returns
    -------
    region: 2 dimensional ndarray
        Binary image of the object
    edges: 2 dimensional ndarray
        Binary image of the edges of the object
    hcm: 2 dimensional ndarray
        high contrast grayscale representation of input image

    See Also
    --------
    (TODO)

    Examples
    --------
    (TODO)
    '''
    hcm = rgb2hcm(image.astype('double'))
    threshold = threshold_otsu(hcm)
    region, edge = morphoreg(hcm, threshold + p)
    return region, edge, hcm


class SegBaluSegmentator(ImageProcessor):
    def __init__(self, *, p=-.05, returns='all', show=False):
        self.show = show

        self.p = p
        self.returns = returns
        if returns == 'all':
            self.idx = slice(0, 3)
        elif returns in ['region', 'edges', 'hcm']:
            self.idx = ['region', 'edges', 'hcm'].index(returns)
        else:
            raise ValueError(
                f"Invalid value for <returns>: '{returns}'"
            )

    def transform(self, X):
        return np.array([segbalu(x, self.p)[self.idx] for x in self._get_iterator(X, desc=f'segbalu_{self.returns}')])
