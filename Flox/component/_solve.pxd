# -*- coding: utf-8 -*-
# 
#  _solve.pyx
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-22.
#  Copyright 2014 University of California. All rights reserved.
# 

from __future__ import division

cimport cython

from Flox._flox cimport DTYPE_t

cdef class Solver:
    
    cdef readonly int nz, nx
    cdef DTYPE_t[:,:] V_curr
    cdef DTYPE_t[:,:] dVdz
    cdef DTYPE_t[:] V_p
    cdef DTYPE_t[:] V_m
    cdef DTYPE_t[:,:] _transform
    cdef bint transform_ready
    
    cpdef int prepare(self, DTYPE_t dz)
    
    cpdef int transform(self, int Kx, DTYPE_t[:,:] V_trans)
    
    
cdef class TimeSolver(Solver):

    cdef DTYPE_t[:,:] G_curr
    cdef DTYPE_t[:,:] G_prev
    cdef DTYPE_t timestep
    cdef bint ready

    cpdef int advance(self, DTYPE_t deltaT)