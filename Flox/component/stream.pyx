# -*- coding: utf-8 -*-
# 
#  stream.pyx
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-18.
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
from Flox.tridiagonal._tridiagonal cimport TridiagonalSolver

cdef class StreamSolver(TridiagonalSolver):
    
    cpdef int setup(self, DTYPE_t dz, DTYPE_t[:] npa):
        
        cdef int j, k
        cdef DTYPE_t dzs = dz * dz
        cdef DTYPE_t dzI = -1.0 / dzs
        cdef DTYPE_t npa_s
        
        for k in range(self.K):
            npa_s = (npa[k] * npa[k])
            self.sub[0,k] = 0.0
            self.sup[0,k] = 0.0
            self.dia[0,k] = 1.0
            for j in range(1, self.J-1):
                self.sub[j,k] = dzI
                self.sup[j,k] = dzI
                self.dia[j,k] = 2.0/dzs + npa_s
            self.sub[self.J-1,k] = 0.0
            self.sup[self.J-1,k] = 0.0
            self.dia[self.J-1,k] = 1.0
        
        return self._warm_work()
        
    cpdef int solve(self, DTYPE_t[:,:] rhs, DTYPE_t[:,:] sol):
        
        return TridiagonalSolver.solve(self, rhs, sol)