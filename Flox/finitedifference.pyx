# -*- coding: utf-8 -*-
# 
#  finitedifference.pyx
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-18.
#  Copyright 2014 University of California. All rights reserved.
# 

from __future__ import division

import numpy as np
cimport numpy as np
cimport cython
from cpython.array cimport array, clone

from Flox._flox cimport DTYPE_t

cpdef int second_derivative(int J, DTYPE_t[:] ddf, DTYPE_t[:] f, DTYPE_t dz, DTYPE_t f_p, DTYPE_t f_m):
    
    cdef int j
    cdef DTYPE_t dzs
    
    dzs = dz * dz
    
    j = 0
    ddf[j] = (f[j+1] - 2.0 * f[j] + f_m)/(dzs)
    
    for j in range(1, J-2):
        ddf[j] = (f[j+1] - 2.0 * f[j] + f[j-1])/(dzs)
        
    j = J-1
    ddf[j] = (f_p - 2.0 * f[j] + f[j-1])/(dzs)
    
    return 0
    

    