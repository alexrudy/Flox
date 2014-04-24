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
    
    def __cinit__(self, int nz, int nx, DTYPE_t[:,:] curr):
        
        cdef int j
        self.V_curr = curr
        self.nx = nx
        self.nz = nz
        self.V_curr = curr
        self.V_next = np.empty((nz, nx), dtype=np.float)
        self.G_curr = np.empty((nz, nx), dtype=np.float)
        self.G_prev = np.empty((nz, nx), dtype=np.float)
        
        for j in range(self.size):
            self.G_prev[j] = 0.0
        
        
    cpdef int advance(self, DTYPE_t deltaT):
        
        cdef int j, k
        
        for k in range(self.nx):
            for j in range(self.nz):
                self.V_next[j,k] = self.V_curr[j,k] + deltaT / 2.0 * (3.0 * self.G_curr[j,k] - self.G_prev[j,k])
                self.G_prev[j,k] = self.G_curr[j,k]
        
        return 0
        
    # User should implement some function which can compute G_curr at each timestep!
    # We don't implement a method here, because its signature will vary greatly!
    
cdef class Evolver:
    
    cdef DTYPE_t delta_time(self):
        
        return 0.0
    
    cpdef int step(self, DTYPE_t delta_time):
        
        return 0
        
    cpdef int evolve(self, DTYPE_t time, int max_iterations):
        
        cdef DTYPE_t delta_time, ctime = self.time
        cdef int j, r = 0
        
        for j in range(max_iterations):
            if self.time > time:
                break
            
            delta_time = self.delta_time()
            r += self.step(delta_time)
            if r == 0:
                self.time += delta_time
            else:
                break
        
        self.time = ctime
        return r
    
