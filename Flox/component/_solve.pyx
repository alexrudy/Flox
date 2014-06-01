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

from __future__ import division

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel cimport prange
from cpython.array cimport array, clone

from Flox._flox cimport DTYPE_t
from Flox.finitedifference cimport first_derivative2D
from Flox.component._transform import transform

cdef class Solver:
    
    def __cinit__(self, int nz, int nx):
        
        self.nx = nx
        self.nz = nz
        self.V_curr = np.zeros((nz, nx), dtype=np.float)
        self.dVdz = np.zeros((nz, nx), dtype=np.float)
        self.V_p = np.zeros((nx,), dtype=np.float)
        self.V_m = np.zeros((nx,), dtype=np.float)
        self.transform_ready = False
        
    cpdef int prepare(self, DTYPE_t dz):
        self.dVdz[...] = 0.0    
        return first_derivative2D(self.nz, self.nx, self.dVdz, self.V_curr, dz, self.V_p, self.V_m, 1.0)
    
    cpdef int transform(self, int Kx, DTYPE_t[:,:] V_trans):
        
        return transform(self.nz, self.nx, Kx, V_trans, self.V_curr, self._transform)
    
    property Value:
        
        def __get__(self):
            return np.asanyarray(self.V_curr)
            
        def __set__(self, value):
            self.V_curr = np.asanyarray(value).copy()
    
    property Transform:
        
        def __get__(self):
            return np.asanyarrray(self._transform)
        
        def __set__(self, value):
            self._transform = np.asanyarray(value).copy()
    
    # User should implement some function which can compute G_curr at each timestep!
    # We don't implement a method here, because its signature will vary greatly!
    
cpdef int advance_cdt(int J, int K, DTYPE_t[:,:] V_curr, DTYPE_t[:,:] G_prev, DTYPE_t[:,:] G_curr, DTYPE_t deltaT) nogil:
    
    cdef int j, k
    
    for j in prange(J):
        for k in range(K):
            V_curr[j,k] = V_curr[j,k] + deltaT / 2.0 * (3.0 * G_curr[j,k] - G_prev[j,k])
            G_prev[j,k] = G_curr[j,k]
    
    return 0
    
cpdef int advance_vdt(int J, int K, DTYPE_t[:,:] V_curr, DTYPE_t[:,:] G_prev, DTYPE_t[:,:] G_curr, DTYPE_t deltaT, DTYPE_t deltaTp) nogil:
    
    cdef int j, k
    cdef DTYPE_t c1, c2
    
    c1 = (1.0 + deltaT / (2.0 * deltaTp))
    c2 = (deltaT / (2.0 * deltaTp))
    
    for j in prange(J):
        for k in range(K):
            V_curr[j,k] = V_curr[j,k] + deltaT * (c1 * G_curr[j, k] - c2 * G_prev[j, k])
            G_prev[j,k] = G_curr[j,k]
    
    return 0
    
cdef class TimeSolver(Solver):
    
    def __cinit__(self, int nz, int nx):
        
        self.nx = nx
        self.nz = nz
        self.ready = False
        self.timestep = 0.0
        self.G_curr = np.zeros((nz, nx), dtype=np.float)
        self.G_prev = np.zeros((nz, nx), dtype=np.float)

    cpdef int prepare(self, DTYPE_t dz):
        
        self.G_curr[...] = 0.0
        self.ready = True
        return Solver.prepare(self, dz)
        
    cpdef int advance(self, DTYPE_t deltaT):
        
        cdef int r
        if self.timestep == 0.0:
            r = advance_cdt(self.nz, self.nx, self.V_curr, self.G_prev, self.G_curr, deltaT)
        elif deltaT == self.timestep:
            r = advance_cdt(self.nz, self.nx, self.V_curr, self.G_prev, self.G_curr, deltaT)
        else:
            r = advance_vdt(self.nz, self.nx, self.V_curr, self.G_prev, self.G_curr, deltaT, self.timestep)
            self.timestep = deltaT
        
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
