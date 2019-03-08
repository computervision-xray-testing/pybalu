__all__ = ["im2col", "im2row"]

import numpy as _np


def im2col(image, n, m):
    '''\
    im2col(image, n, m)

    Rearranges discrete image blocks of size m-by-n into columns, and returns the concatenated columns. 
    im2col pads `image` with zeros, if necessary.

    Parameters 
    ----------
    image: a numpy 2 dimensional array
        Represents the input image of dimensions N-by-M
    n: non-zero positive integer value
        The size of the blocks on the first dimension.
    m: non-zero positive integer value
        The size of the blocks on the second dimension.

    Returns
    -------
    blocks: 2 dimensional ndarray of size (n * m)-by-(ceil(N / n) * ceil(M / m)) of the same type as `image`.
        Each column represents a diferent block of (n * m) values on the original image.
        The blocks are iterated from (0, 0) increasing initially in the first dimension.
        ie: (0, 0), (n, 0), (2*n, 0), ... , (x*n, 0), (0, m), (n, m), ... ,(x*n, m), ...

    Examples
    --------
    Generate a 4-by-5 matrix and transform it to columns in 2-by-3 blocks

    >>> import numpy as np
    >>> from pybalu.misc import im2col
    >>> img = np.arange(20).reshape(4, 5)
    >>> print(img)
    [[ 0  1  2  3  4]
     [ 5  6  7  8  9]
     [10 11 12 13 14]
     [15 16 17 18 19]]
    >>> print(im2col(img, 2, 3))
    [[ 0 10  3 13]
     [ 5 15  8 18]
     [ 1 11  4 14]
     [ 6 16  9 19]
     [ 2 12  0  0]
     [ 7 17  0  0]]
    '''
    image = image.T
    n, m = m, n
    N, M = image.shape
    dn, rn = divmod(N, n)
    dm, rm = divmod(M, m)
    if rn or rm:
        if rn:
            _N = (dn+1)*n
        else:
            _N = N
        if rm:
            _M = (dm+1)*m
        else:
            _M = M
        _img = _np.zeros((_N, _M), dtype=image.dtype)
        _img[:N, :M] = image
    else:
        _img = _np.ascontiguousarray(image)

    arr_shape = _np.array(_img.shape)
    block_shape = _np.array([n, m])
    new_shape = tuple(arr_shape // block_shape) + tuple(block_shape)
    new_strides = tuple(_img.strides * block_shape) + _img.strides
    blocks = _np.lib.stride_tricks.as_strided(
        _img, shape=new_shape, strides=new_strides)
    _n, _m, *_ = new_shape
    return blocks.reshape(_n*_m, -1).T


def im2row(image, n, m):
    '''\
    im2row(image, n, m)

    Divides the given image into overlaping blocks that slide accross the image and returns the 
    concatenated rows. The block size is such that it is the biggest possible sliding window that
    divides the image into n-by-m distinct blocks. Depending on dimensions, it may or may not
    cover the whole image.

    Parameters 
    ----------
    image: a numpy 2 dimensional array
        Represents the input image of dimensions N-by-M
    n: non-zero positive integer value
        The number of sliding blocks in the first dimension.
    m: non-zero positive integer value
        The size of sliding blocks in the second dimension

    Returns
    -------
    windows: 2 dimensional ndarray of the same type as `image`.
        of size (n * m)-by-(N // (n + 1) * M // (m + 1) * 4). Each row represents a 
        diferent view of the sliding window over the original image. The window is  
        moved from (0, 0) increasing initially in the first dimension.
        ie: (0, 0), (N // (n + 1), 0), ... , (n * N // (n + 1), 0), 
        (0, M // (m + 1)), (, ... ,((n * N // (n + 1),  M // (m + 1)), ...

    Examples
    --------
    Generate a 4-by-4 matrix and transform it to rows in a 3-by-3 window division

    >>> import numpy as np
    >>> from pybalu.misc import im2row
    >>> img = np.arange(16).reshape(4, 4)
    >>> print(img)
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [12 13 14 15]]
    >>> print(im2row(img, 3, 3))
    [[ 0  1  4  5]
     [ 1  2  5  6]
     [ 2  3  6  7]
     [ 4  5  8  9]
     [ 5  6  9 10]
     [ 6  7 10 11]
     [ 8  9 12 13]
     [ 9 10 13 14]
     [10 11 14 15]]
    '''
    _img = _np.ascontiguousarray(image)
    N, M = _img.shape
    hh = N // (n + 1)
    hw = M // (m + 1)
    vs, hs = _img.strides

    new_shape = (n, m, hh*2, hw*2)
    new_strides = (vs * hh, hs * hw, vs, hs)
    windows = _np.lib.stride_tricks.as_strided(
        _img, shape=new_shape, strides=new_strides)
    return windows.reshape(n * m, -1)
