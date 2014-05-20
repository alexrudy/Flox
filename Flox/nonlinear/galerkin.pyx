# -*- coding: utf-8 -*-
# 
#  galerkin.pyx
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-18.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

#cython: overflowcheck=False
#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True
#cython: profile=False

from __future__ import division

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel cimport prange

from Flox._flox cimport DTYPE_t

cdef DTYPE_t pi = np.pi

cpdef int galerkin_sin(int J, int K, DTYPE_t[:,:] G_curr, DTYPE_t[:,:] V_curr, DTYPE_t[:,:] dVdz, DTYPE_t[:,:] O_curr, DTYPE_t[:,:] dOdz, DTYPE_t a, DTYPE_t[:] npa, DTYPE_t factor) nogil:
    
    cdef int j, k, kp, kpp
    cdef DTYPE_t p2a = pi / (2.0 * a)
    
    for j in prange(J, nogil=True):
    
        for k in range(K):
            # n=0 special case.
            G_curr[j,0] += -factor * p2a * k * (V_curr[j, k] * dOdz[j, k] + O_curr[j, k] * dVdz[j, k])
        
            # Terms applied everywhere.
            G_curr[j,k] += -factor * npa[k] * O_curr[j, k] * dVdz[j, 0]
        
            for kp in range(1, K):
                # 1st term, 1st Delta
                kpp = k - kp
                if 0 < kpp < K:
                    G_curr[j, k] += -factor * p2a * (-kp * dOdz[j, kpp] * V_curr[j, kp] + kpp * O_curr[j, kpp] * dVdz[j, kp])
            
                # 2nd term, 1st Delta
                kpp = kp + k
                if 0 < kpp < K:
                    G_curr[j, k] += -factor * p2a * (kp * dOdz[j, kpp] * V_curr[j, kp] + kpp * O_curr[j, kpp] * dVdz[j, kp])
                
                # 2nd term, 2nd Delta
                kpp = kp - k
                if 0 < kpp < K:
                    G_curr[j, k] += -factor * p2a * (kp * dOdz[j, kpp] * V_curr[j, kp] + kpp * O_curr[j, kpp] * dVdz[j, kp])

    return 0
    
cpdef int galerkin_cos(int J, int K, DTYPE_t[:,:] G_curr, DTYPE_t[:,:] V_curr, DTYPE_t[:,:] dVdz, DTYPE_t[:,:] O_curr, DTYPE_t[:,:] dOdz, DTYPE_t a, DTYPE_t factor) nogil:
    
    cdef int j, k, kp, kpp
    cdef DTYPE_t p2a = pi / (2.0 * a)
    
    for j in prange(J, nogil=True):
        for k in range(K):
            for kp in range(1, K):
                # 1st term, 1st delta
                kpp = k - kp
                if 0 < kpp < K:
                    G_curr[j, k] += -factor * p2a * (-kp * dOdz[j, kpp] * V_curr[j, kp] + kpp * O_curr[j, kpp] * dVdz[j, kp])
                # 2nd term, 1st delta
                kpp = kp + k
                if 0 < kpp < K:
                    G_curr[j, k] += -factor * p2a * ( -1.0 * (kp * dOdz[j, kpp] * V_curr[j, kp] + kpp * O_curr[j, kpp] * dVdz[j, kp]))
                # 2nd term, 2nd delta
                kpp = kp - k
                if 0 < kpp < K:
                    G_curr[j, k] += -factor * p2a * (kp * dOdz[j, kpp] * V_curr[j, kp] + kpp * O_curr[j, kpp] * dVdz[j, kp])
                    
    return 0
