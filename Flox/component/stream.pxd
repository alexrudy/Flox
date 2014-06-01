# -*- coding: utf-8 -*-
# 
#  stream.pxd
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-18.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from Flox._flox cimport DTYPE_t
from Flox.tridiagonal._tridiagonal cimport TridiagonalSolver
    
cdef class StreamSolver(TridiagonalSolver):
    cdef DTYPE_t[:,:] Vx
    cdef DTYPE_t[:,:] Vz
    cdef DTYPE_t[:,:] Vx_transform
    cdef DTYPE_t[:,:] Vz_transform
    cdef DTYPE_t[:,:] Velocity
    cdef DTYPE_t maxV
    cpdef int setup(self, DTYPE_t dz, DTYPE_t[:] npa)
    cpdef int setup_transform(self, DTYPE_t[:] npa)
    cpdef int compute_velocity(self)