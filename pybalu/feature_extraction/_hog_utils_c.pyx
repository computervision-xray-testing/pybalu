__all__ = ['sep_into_bins']

cimport cython
cimport numpy as np
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def sep_into_bins(double[:] vals, long[:] bins, int n_bins):
    cdef double[:] out
    cdef int i, b, N
    cdef float val
    
    N = bins.size
    out = np.zeros(n_bins)
    for i in range(N):
        out[bins[i]] += vals[i]
        
    return np.asarray(out)