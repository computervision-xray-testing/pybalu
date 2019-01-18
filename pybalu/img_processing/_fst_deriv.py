__all__ = ['fst_deriv']

import numpy as np 
from scipy import signal

def fst_deriv(image, mask):
    if mask % 2 == 0:
        raise Exception('Mask must be of odd size')
    sigma = mask / 8.5
    c = (mask - 1) / 2
    x = np.tile(np.arange(mask) + 1 - c, (mask, 1))
    y = x.T
    exp = np.exp(-(x**2 + y**2) / 2 / sigma ** 2)
    gauss_x = x * exp
    gauss_y = y * exp
    mgx = np.abs(gauss_x).sum() / 2 * (0.3192 * mask - .3543)
    gauss_x /= mgx
    gauss_y /= mgx
    dx = signal.convolve2d(image, gauss_x, 'same')
    dy = signal.convolve2d(image, gauss_y, 'same')
    return np.sqrt(dx**2 + dy**2), dx, dy