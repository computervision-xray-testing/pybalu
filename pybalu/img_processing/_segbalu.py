__all__ = ['segbalu', 'SegBaluSegmentator']

from ._image_processor import ImageProcessorBase
from ._rgb2hcm import rgb2hcm
from ._morphoreg import morphoreg
from skimage.filters import threshold_otsu

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

class SegBaluSegmentator(ImageProcessorBase):
    processing_func = segbalu