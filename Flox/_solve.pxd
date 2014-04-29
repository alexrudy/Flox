# -*- coding: utf-8 -*-
# 
#  _solve.pyx
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-22.
#  Copyright 2014 University of California. All rights reserved.
# 

from __future__ import division

import numpy as np
cimport numpy as np
cimport cython
from cpython.array cimport array, clone

from Flox._flox cimport DTYPE_t

cdef class Solver:
    
    cdef int nz, nx
    cdef DTYPE_t[:,:] G_curr
    cdef DTYPE_t[:,:] G_prev
    cdef DTYPE_t[:,:] V_curr
    cdef DTYPE_t[:] V_p
    cdef DTYPE_t[:] V_m
    
    cpdef int advance(self, DTYPE_t deltaT)
    cpdef int get_state(self, DTYPE_t[:,:] V_curr, DTYPE_t[:,:] G_curr, DTYPE_t[:,:] G_prev)
    
cdef class Evolver:
    cdef readonly DTYPE_t time
    cpdef DTYPE_t delta_time(self)
    cpdef int step(self, DTYPE_t delta_time)
    cpdef int evolve(self, DTYPE_t time, int max_iterations)
