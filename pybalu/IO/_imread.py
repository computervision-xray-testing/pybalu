__all__ = ['imread']


import numpy as _np
from imageio import imread as _imread


def imread(filename, *, normalize=False, flatten=False):
    '''\
    imread(filename, flatten=False)

    Loads an image as a numerical matrix of either 2 or 3 dimensions.

    Parameters
    ----------
    filename: string
        The path of the image to load
    normalize: boolean, optional
        If set to true, the return value will be an array with float values between 0 and 1.
        If set to false, the reurn value will be an array with uint8 values between 0 and 255.
        default value is True.
    flatten: boolean, optional
        If set to true, the return value is a 2 dimensional ndarray with the grayscale
        representation of the loaded image.

    Returns
    -------
    image: ndarray
        matrix representation of the loaded image, either 2 or 3 dimensional depending on the
        file and the `flatten` parameter.

    Examples
    --------
    (TODO)
    '''
    img = _imread(filename)
    if flatten:
        img = img @ [0.299, 0.587, 0.114]
    if normalize:
        return (img / 255)
    return img.astype(_np.uint8)
