cimport cython
cimport numpy as np
import numpy as np
from .geometric_utils import *

f_labels = ['Flusser-moment 1',
            'Flusser-moment 2',
            'Flusser-moment 3',
            'Flusser-moment 4']

@cython.wraparound(False)
@cython.boundscheck(False)
def flusser(np.ndarray[np.int_t, ndim=2] image, *, show=False, labels=False):
    cdef np.ndarray[np.double_t, ndim=2] m
    cdef np.double_t f_moment1, f_moment2, f_moment3, f_moment4

    m = moments(image, centered=True, order=3)

    f_moment1 = (m[2, 0] * m[0, 2] - m[1, 1] ** 2) / m[0, 0] ** 4
    f_moment2 = (m[3, 0] ** 2 * m[0, 3] ** 2 - 6 * m[3,0] * m[2, 1] * m[1, 2] * m[0, 3] + 4 * m[3, 0] * m[1, 2] ** 3 + 4 * m[2, 1] ** 3 * m[0, 3] - 3 * m[2, 1] ** 2 * m[1, 2] ** 2)/ m[0, 0] ** 10
    f_moment3 = (m[2, 0] * (m[2, 1] * m[0, 3]-m[1, 2] ** 2)-m[1, 1] * (m[3, 0] * m[0, 3] - m[2, 1] * m[1, 2]) + m[0, 2] * (m[3, 0] * m[1, 2]-m[2, 1] ** 2)) / m[0, 0] ** 7
    f_moment4 = (m[2, 0] ** 3 * m[0, 3] ** 2 - 6 * m[2, 0] ** 2 * m[1, 1] * m[1, 2] * m[0, 3] - 6 * m[2, 0] ** 2 * m[0, 2] * m[2, 1] * m[0, 3] + 9 * m[2, 0] ** 2 * m[0, 2] * m[1, 2] ** 2 + 12 * m[2, 0] * m[1, 1] ** 2 * m[2, 1] * m[0, 3] + 6 * m[2, 0]*m[1, 1]*m[0, 2] * m[3, 0] * m[0, 3]-18 * m[2, 0]*m[1, 1] * m[0, 2] * m[2, 1] *m[1, 2]-8 * m[1, 1] ** 3 * m[3, 0]*m[0, 3]- 6 * m[2, 0]*m[0, 2] ** 2 * m[3, 0]*m[1, 2]+9 * m[2, 0]*m[0, 2] ** 2 * m[2, 1]+12 * m[1, 1] ** 2 * m[0, 2]*m[3, 0] * m[1, 2]-6 * m[1, 1] * m[0, 2] ** 2 * m[3, 0]*m[2, 1]+m[0, 2] ** 3 * m[3, 0] ** 2)/m[0, 0] ** 11
 
    f_moments = np.array([f_moment1, f_moment2, f_moment3, f_moment4])
    if labels:
        return np.array(f_labels), f_moments
    return f_moments
