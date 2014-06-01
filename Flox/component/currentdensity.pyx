# -*- coding: utf-8 -*-
# 
#  currentdensity.pyx
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-19.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

#cython: overflowcheck=False
#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True
from __future__ import division

import numpy as np
cimport numpy as np
cimport cython

from cython.parallel cimport prange
from Flox._flox cimport DTYPE_t
from Flox.component._solve cimport Solver

cpdef int currentdensity(int J, int K, DTYPE_t[:,:] J_curr, DTYPE_t[:,:] A_curr, DTYPE_t[:,:] dAdzz, DTYPE_t[:] npa) nogil:
    
    cdef int j, k
    cdef DTYPE_t npa_s
    
    for k in prange(K):
        npa_s = npa[k] * npa[k]
        for j in range(J):
            J_curr[j, k] = -1.0 * ( dAdzz[j, k] - npa_s * A_curr[j, k])
    
    return 0

cdef class CurrentDensitySolver(Solver):
    
    cpdef int compute_base(self, DTYPE_t[:,:] A_curr, DTYPE_t[:,:] dAdzz, DTYPE_t[:] npa):
        
        return currentdensity(self.nz, self.nx, self.V_curr, A_curr, dAdzz, npa)