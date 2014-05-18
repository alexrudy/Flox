# -*- coding: utf-8 -*-
# 
#  finitedifference.pyx
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-18.
#  Copyright 2014 University of California. All rights reserved.
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
from cpython.array cimport array, clone
from cython.parallel cimport prange

from Flox._flox cimport DTYPE_t

# Also known as 'clear_walrus'
cpdef int clear_values(int J, DTYPE_t[:] val) nogil:
    
    cdef int j
    
    for j in range(0, J-1):
        val[j] = 0.0
    
    return 0


cpdef int second_derivative2D(int J, int K, DTYPE_t[:,:] ddf, DTYPE_t[:,:] f, DTYPE_t dz, DTYPE_t[:] f_p, DTYPE_t[:] f_m, DTYPE_t factor) nogil:
    
    cdef DTYPE_t dzs = dz * dz
    cdef int k, j, r = 0
    
    for k in prange(K):
        j = 0
        ddf[j,k] += factor * (f[j+1,k] - 2.0 * f[j,k] + f_m[k])/(dzs)
        for j in range(1, J-1):
            ddf[j,k] += factor * (f[j+1,k] - 2.0 * f[j,k] + f[j-1,k])/(dzs)
        
        j = J-1
        ddf[j,k] += factor * (f_p[k] - 2.0 * f[j,k] + f[j-1,k])/(dzs)
    return 0

cpdef int first_derivative2D(int J, int K, DTYPE_t[:,:] df, DTYPE_t[:,:] f, DTYPE_t dz, DTYPE_t[:] f_p, DTYPE_t[:] f_m, DTYPE_t factor) nogil:

    cdef DTYPE_t dzs = 2.0 * dz
    cdef int k, j, r = 0
    
    for k in prange(K):
        j = 0
        df[j,k] += factor * (f[j+1,k] - f_m[k])/(dzs)
        for j in range(1, J-1):
            df[j,k] += factor * (f[j+1,k] - f[j-1,k])/(dzs)
        
        j = J-1
        df[j,k] += factor * (f_p[k] - f[j-1,k])/(dzs)
    
    return 0


cpdef int second_derivative(int J, DTYPE_t[:] ddf, DTYPE_t[:] f, DTYPE_t dz, DTYPE_t f_p, DTYPE_t f_m, DTYPE_t factor) nogil:
    
    cdef int j
    cdef DTYPE_t dzs = dz * dz
    
    j = 0
    ddf[j] += factor * (f[j+1] - 2.0 * f[j] + f_m)/(dzs)
    
    for j in range(1, J-1):
        ddf[j] += factor * (f[j+1] - 2.0 * f[j] + f[j-1])/(dzs)
        
    j = J-1
    ddf[j] += factor * (f_p - 2.0 * f[j] + f[j-1])/(dzs)
    
    return 0
    
cpdef int first_derivative(int J, DTYPE_t[:] df, DTYPE_t[:] f, DTYPE_t dz, DTYPE_t f_p, DTYPE_t f_m, DTYPE_t factor) nogil:
    
    cdef int j
    cdef DTYPE_t dzs = 2.0 * dz
    
    j = 0
    df[j] += factor * (f[j+1] - f_m)/(dzs)
    
    for j in range(1, J-1):
        df[j] += factor * (f[j+1] - f[j-1])/(dzs)
    
    j = J-1
    df[j] += factor * (f_p - f[j-1])/(dzs)
    
    return 0
    