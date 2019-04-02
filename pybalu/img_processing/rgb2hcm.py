__all__ = ['rgb2hcm']

from skimage.transform import resize
from scipy.optimize import minimize
import numpy as np

_k = np.ones(3)


def rgb2hcm(image):
    '''\
    rgb2hcm(image)

    Segmentation of an object with homogeneous background.

    Parameters
    ----------
    image: 3 dimensional ndarray
        The RGB input image.

    Returns
    -------
    hcm: 2 dimensional ndarray
        high contrast grayscale representation of input image


    Examples
    --------
    (TODO)
    '''
    if image.ndim < 3:
        I = image
    else:
        img_resize = resize(image, (64, 64), order=3,
                            mode='reflect', anti_aliasing=False)
        k = minimize(monochrome_std, [1., 1.], args=(img_resize,))['x']
        _k[:2] = k
        I = image @ _k
    J = I - I.min()
    J = J / J.max()
    n = J.shape[0] // 4
    m = J.shape[1] // 4

    if (J[:n, :m].mean() > .4):
        J = 1 - J
    return J


def monochrome_std(k, image):
    _k[:2] = k
    I = image @ _k
    return - I.std(ddof=1) / (I.max() - I.min())
