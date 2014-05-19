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
    cpdef int setup(self, DTYPE_t dz, DTYPE_t[:] npa)
    