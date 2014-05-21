# -*- coding: utf-8 -*-
# 
#  _transform.pyx
#  flox
#  
#  Created by Alexander Rudy on 2014-05-20.
#  Copyright 2014 University of California. All rights reserved.
# 

#cython: overflowcheck=False
#cython: wraparound=False
#cython: boundscheck=True
#cython: cdivision=True
#cython: profile=False

from __future__ import division

import numpy as np
cimport numpy as np
cimport cython
from cpython.array cimport array, clone
from cython.parallel cimport prange

from Flox._flox cimport DTYPE_t

cpdef int transform(int J, int K, int Kx, DTYPE_t[:,:] V_trans, DTYPE_t[:,:] V_curr, DTYPE_t[:,:] _transform) nogil:
    
    cdef int j, k, x
    for j in range(J):
        for k in range(K):
            for x in range(Kx):
                V_trans[j,x] += _transform[k,x] * V_curr[j,k]
    return 0