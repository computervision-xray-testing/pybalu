__all__ = ['snd_deriv']

import numpy as np

def snd_deriv(image):
    m = image.shape[0]
    n = image.shape[1]
    
    out = np.zeros((m,n))
    
    out = -4 * image
    out[1:, :] += image[:m-1, :]
    out[:m-1, :] += image[1:, :]
    out[:, 1:] += image[:, :n-1]
    out[:, :n-1] += image[:, 1:]
    return out