# -*- coding: utf-8 -*-
# 
#  _solve.pyx
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-22.
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

from Flox._flox cimport DTYPE_t
from Flox.finitedifference cimport first_derivative2D

cdef class Solver:
    
    def __cinit__(self, int nz, int nx):
        
        self.nx = nx
        self.nz = nz
        self.V_curr = np.zeros((nz, nx), dtype=np.float)
        self.dVdz = np.zeros((nz, nx), dtype=np.float)
        self.V_p = np.zeros((nx,), dtype=np.float)
        self.V_m = np.zeros((nx,), dtype=np.float)
        
    cpdef int prepare(self, DTYPE_t dz) except -1:
        self.dVdz[...] = 0.0    
        return first_derivative2D(self.nz, self.nx, self.dVdz, self.V_curr, dz, self.V_p, self.V_m, 1.0)
        
    property Value:
        
        def __get__(self):
            return np.asanyarray(self.V_curr)
            
        def __set__(self, value):
            self.V_curr = np.asanyarray(value)
    
    # User should implement some function which can compute G_curr at each timestep!
    # We don't implement a method here, because its signature will vary greatly!
    
cdef class TimeSolver(Solver):
    
    def __cinit__(self, int nz, int nx):
        
        self.nx = nx
        self.nz = nz
        self.ready = False
        self.G_curr = np.zeros((nz, nx), dtype=np.float)
        self.G_prev = np.zeros((nz, nx), dtype=np.float)

    cpdef int prepare(self, DTYPE_t dz) except -1:
        
        self.G_curr[...] = 0.0
        self.ready = True
        return Solver.prepare(self, dz)
        
        
    cpdef int advance(self, DTYPE_t deltaT) except -1:
    
        cdef int j, k
    
        for k in range(self.nx):
            for j in range(self.nz):
                self.V_curr[j,k] = self.V_curr[j,k] + deltaT / 2.0 * (3.0 * self.G_curr[j,k] - self.G_prev[j,k])
                self.G_prev[j,k] = self.G_curr[j,k]
        
        self.ready = False
        return 0
        
    property dValuedt:

        def __get__(self):
            return np.asanyarray(self.G_prev)

        def __set__(self, value):
            self.G_prev = np.asanyarray(value)
        
    property Value_p:
    
        def __get__(self):
            return np.asanyarray(self.V_p)
        
        def __set__(self, value):
            self.V_p = np.asanyarray(value)
    
    property Value_m:
    
        def __get__(self):
            return np.asanyarray(self.V_m)
    
        def __set__(self, value):
            self.V_m = np.asanyarray(value)
