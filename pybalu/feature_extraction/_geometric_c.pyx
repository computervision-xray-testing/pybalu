__all__ = ['perimeter', 'moments', 'convex_area', 'bbox', 'bw_perim']

cimport cython
cimport numpy as np
import numpy as np
import scipy.spatial

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef float perimeter(np.ndarray[np.int_t, ndim=2] region):

    cdef int h = region.shape[0]
    cdef int w = region.shape[1]
    # first account for border pixels
    cdef float p = region[0].sum() + region[h-1].sum() + region[1:h-1,0].sum() + region[1:h-1,w-1].sum()
    
    cdef int i, j
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            if region[i,j] == 0: continue
            #connectivity 4
            if region[i-1, j] == 0 or region[i+1, j] == 0:
                p += 1
                continue
            if region[i, j-1] == 0 or region[i, j+1] == 0:
                p += 1
                continue

            #connectivity 8
            if region[i-1, j-1] == 0 or region[i-1, j+1] == 0:
                p += .25
                continue
            if region[i+1, j-1] == 0 or region[i+1, j+1] == 0:
                p += .25
                continue
    return p

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef np.ndarray[np.int_t, ndim=2] bw_perim(np.ndarray[np.int_t, ndim=2] region, int conn=4):
    if not (conn == 4 or conn == 8):
        raise Exception('Neighbourhood connectivity must be either 4 or 8, not ' + str(conn))
    
    cdef int h = region.shape[0]
    cdef int w = region.shape[1]

    cdef np.ndarray[np.int_t, ndim=2] out = np.zeros(shape=(h, w), dtype=int)
    
    cdef int i, j
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            if not region[i,j]: continue
            #connectivity 4
            if not region[i-1, j] or not region[i+1, j]:
                out[i, j] = 1
                continue
            if not region[i, j-1] or not region[i, j+1]:
                out[i, j] = 1
                continue
            if conn == 4: continue
            #connectivity 8
            if not region[i-1, j-1] or not region[i-1, j+1]:
                out[i, j] = 1
                continue
            if not region[i+1, j-1] or not region[i+1, j+1]:
                out[i, j] = 1
                continue
    return out


@cython.wraparound(False)
@cython.boundscheck(False)
def moments(np.ndarray[np.int_t, ndim=2] region, centered=False, order=3):
    cdef int i, j
    cdef np.ndarray[np.double_t, ndim=2] m, I_, J_
    
    I, J = np.where(region == 1)
    I_ = np.ones((order+1, I.size))
    J_ = np.ones((order+1, J.size))
    I_[1,:] = I
    J_[1,:] = J

    if centered:
        I_[1,:] -= I.mean()
        J_[1,:] -= J.mean()
        
    for i in range(2, order + 1):
        I_[i,:] = I_[i-1,:] * I_[1,:]
        J_[i,:] = J_[i-1,:] * J_[1,:]

    m = np.zeros(shape=(order+1,order+1), dtype=float)
    for i in range(order + 1):    
        for j in range(order + 1):
            m[i,j] = I_[i,:] @ J_[j,:]
    return m


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef bbox(np.ndarray[np.int_t, ndim=2] region):
    
    cdef int n, m, i, j, min_i, max_i, min_j, max_j
    
    max_i = max_j = 0
    n = min_i = region.shape[0]
    m = min_j = region.shape[1]
    
    for i in range(n):
        for j in range(m):
            if region[i, j] == 1:
                if i < min_i: min_i = i
                if i > max_i: max_i = i
                if j < min_j: min_j = j
                if j > max_j: max_j = j
    
    return slice(min_i, max_i + 1), slice(min_j, max_j + 1)


@cython.wraparound(False)
@cython.boundscheck(False)
def convex_area(np.ndarray[np.int_t, ndim=2] region):
    cdef np.ndarray[np.int_t, ndim=1] ii, jj
    cdef np.ndarray[np.double_t, ndim=1] X, Y
    cdef np.ndarray[np.double_t, ndim=2] dx, dy
    cdef list hull
    cdef int i, p, q, N
    cdef float val
    
    
    if region.size < 1: return 0

    ii, jj = np.where(bw_perim(region).astype(bool))
    dx = np.array([-0.5, -0.5, 0.5, 0.5]).reshape(4, 1)
    dy = np.array([-0.5, 0.5, -0.5, 0.5]).reshape(4, 1)
    X = (ii + dx).flatten()
    Y = (jj + dy).flatten()
    
    if X.size == 0:
        return 0

    hull = []
    
    area = 0
    i = 0
    p = 0
    q = 0
    
    while True:
        hull.append(p)
        q = (p + 1) % X.size;
        for i in range(X.size):
            val = (Y[i] - Y[p]) * (X[q] - X[i]) - (X[i] - X[p]) * (Y[q] - Y[i])
            if val < 0:
                q = i
        p = q
        if p == 0: 
            break
    hull.append(0)
    X, Y = X[hull], Y[hull]
    N = X.size
    return (X[:N-1] @ Y[1:] - X[1:] @ Y[:N-1]) / 2

@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline tuple nnext_cw(tuple curr, tuple _from):
    cdef int i, j, ci, cj
    i, j = curr
    ci, cj = _from
    
    if i == ci:
        if j < cj:
            return i-1, j
        return i+1, j
    if j == cj:
        if i < ci:
            return i, j+1
        return i, j-1
    if i < ci:
        if j < cj:
            return i, cj
        return ci, j
    if j < cj:
        return ci, j
    return i, cj

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef bw_boundaries(np.ndarray[np.int_t, ndim=2] R):
    cdef tuple s, b, p, c
    cdef Py_ssize_t i, j, N, M
    cdef list B
    cdef np.ndarray[np.int_t, ndim=2] _R

    N = R.shape[0]
    M = R.shape[1]
    _R = np.zeros((N+2, M+2), dtype=int)
    _R[1:N+1,1:M+1] = R
    
    s = None
    i = N
    j = 1
    
    while i > 0 and s is None:
        j = 1
        while j < M:
            if _R[i,j] == 1:
                s = (i,j)
                break
            j += 1
        i -= 1
    if j == 1:
        b = i + 2, j
    else:
        b = i + 1, j - 1
    B = [s]
    p = s
    c = nnext_cw(b, p)
    while c != s:
        if _R[c]:
            B.append(c)
            b = p
            p = c
        else:
            b = c
        c = nnext_cw(b, p)
    return np.array(B) - 1