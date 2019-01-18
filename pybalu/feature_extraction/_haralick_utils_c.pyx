__all__ = ['noorm_cooc_mtrx', 'cooc_features']

cimport numpy as np
cimport cython
import numpy as np

@cython.wraparound(False)
@cython.boundscheck(False)
def norm_cooc_mtrx(np.ndarray I, np.ndarray[np.int_t, ndim=2] R, int d):
    
    cdef int i, j, v1, v2, h_tot, v_tot, d1_tot, d2_tot
    cdef np.ndarray[np.double_t, ndim=3] co_occ
    cdef np.ndarray[np.int_t, ndim=2] V
    cdef np.ndarray[np.int_t, ndim=1] ii, jj, tot
    
    # normalize for only 8 pixel values and get rid of values not in R
    V = np.zeros(shape=(I.shape[0] + 2*d, I.shape[1] + 2*d), dtype=int)
    V[d:I.shape[0]+d, d:I.shape[1]+d] = np.floor(I / 32).astype(int)
    ii, jj = np.where(R != 1)
    V[ii+d, jj+d] = -1

    co_occ = np.zeros(shape=(4, 8, 8), dtype=float)
    tot = np.zeros(4, dtype=int)
    
    for i in range(d, I.shape[0]):
        for j in range(d, I.shape[1]):
            v1 = V[i, j] 
            if v1 < 0: continue
            # vertical
            v2 = V[i+d, j]
            if v1 >= 0 and v2 >= 0:
                co_occ[0, v1, v2] += 1
                co_occ[0, v2, v1] += 1
                tot[0] += 2
            # diagonal 1
            v2 = V[i+d, j-d]
            if v1 >= 0 and v2 >= 0:
                co_occ[1, v1, v2] += 1
                co_occ[1, v2, v1] += 1
                tot[1] += 2
            # horizontal
            v2 = V[i, j+d]
            if v1 >= 0 and v2 >= 0:
                co_occ[2, v1, v2] += 1
                co_occ[2, v2, v1] += 1
                tot[2] += 2
            # diagonal 2
            v2 = V[i+d, j+d]
            if v1 >= 0 and v2 >= 0:
                co_occ[3, v1, v2] += 1
                co_occ[3, v2, v1] += 1
                tot[3] += 2
    return co_occ / tot.reshape(4, 1, 1)
    

@cython.wraparound(False)
@cython.boundscheck(False)
def cooc_features(np.ndarray[np.double_t, ndim=2] P):
    
    cdef N, i, j

    N = 8
    
    Pij = P.ravel()
    pxi = P.sum(1)
    pyj = P.sum(0)
    ux = pxi.mean()
    uy = pyj.mean()
    sx = pxi.std(ddof=1)
    sy = pyj.std(ddof=1)
    
    ii, jj = np.where(P >= 0)
    dif_ij = ii - jj
    dif_ij_sq = dif_ij * dif_ij
    
    pxy1 = np.zeros(N * 2 - 1)
    pxy2 = np.zeros(N)
     
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            pxy1[i+j] += P[i, j]
            pxy2[abs(i - j)] += P[i, j]
    
    # angular second moment (??)
    energy = Pij @ Pij
    
    # simplified method of calculating -> same as the paper
    contrast = dif_ij_sq @ Pij
    
    # holy f*ck what is this??
    correlation = (((ii+1) * (jj+1)) @ Pij - ux * uy * N ** 2) / sx / sy
    
    # this is the same as contrast (??)
    sum_of_sq = contrast

    # same as paper!
    inverse_dif_moment =  Pij @ (1 / (dif_ij_sq + 1))
    
    sum_avg = (ii+jj+2) @ Pij
    
    sum_entropy = - pxy1 @ np.log(pxy1, where=pxy1>0)
    
    sum_var = (np.arange(2, N*2+1) - sum_entropy) @ pxy1
    
    # same as paper!
    entropy = - Pij @ np.log(Pij, where=(Pij>0))
    
    # Difference Variance
    difference_var = pxy2.var(ddof=1)
    
    # Difference Entropy
    difference_entropy = -pxy2 @ np.log(pxy2, where=pxy2>0)
    
    pxipyj = pxi[ii] * pyj[jj]
    pxipyj_log = np.log(pxipyj, where=pxipyj>0)

    HXY = entropy
    HXY1 = -Pij @ pxipyj_log
    HXY2 = -pxipyj @ pxipyj_log
    HX = -pxi @ np.log(pxi, where=pxi>0)
    HY = -pyj @ np.log(pyj, where=pyj>0)
    
    # Measures of correlation
    corr_1 =  (HXY - HXY1) / max(HX, HY) if HX > 0 or HY > 0 else 0
    corr_2 = 1 - np.exp(-2*(HXY2 - HXY))
    
    # Maximal Correlation Coefficient
    Q = np.divide(P, pyj, where=pyj>0) @ np.divide(P, pyj, where=pxi>0)
    eigQ = np.linalg.eigvalsh(Q)
    max_corr_coef = eigQ[7]
    
    return np.array([
        energy,
        contrast,
        correlation,
        sum_of_sq,
        inverse_dif_moment,
        sum_avg,
        sum_var,
        sum_entropy,
        entropy,
        difference_var,
        difference_entropy,
        corr_1,
        corr_2,
        max_corr_coef
    ])