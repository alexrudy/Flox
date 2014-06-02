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
from Flox.component._transform cimport transform
from Flox.transform import setup_transform

cdef class StreamSolver(TridiagonalSolver):
    
    def __cinit__(self, int nz, int nx):
        self.Velocity = np.zeros((nz, nx), dtype=np.float)
        self.Vx = np.zeros((nz, nx), dtype=np.float)
        self.Vz = np.zeros((nz, nx), dtype=np.float)
    
    cpdef int setup_transform(self, DTYPE_t[:] npa):
        
        self.Vx_transform = setup_transform(np.sin, self.nx, self.nx)
        self.Vz_transform = setup_transform(np.cos, self.nx, self.nx)
        for k in range(self.nx):
            for kp in range(self.nx):
                self.Vx_transform[k,kp] *= -1.0
                self.Vz_transform[k,kp] *= npa[k]
        self.transform_ready = True
    
    cpdef int compute_velocity(self):
        # Compute and update the internal variable handling the maximum fluid velocity.
        cdef int r, j, k
        cdef DTYPE_t Vz, Vx
        self.Vx[...] = 0.0
        self.Vz[...] = 0.0
        self.maxV = 0.0
        r = transform(self.nz, self.nx, self.nx, self.Vx, self.dVdz, self.Vx_transform)
        r += transform(self.nz, self.nx, self.nx, self.Vz, self.V_curr, self.Vz_transform)
        for j in range(self.nz):
            for k in range(self.nx):
                Vz = (self.Vz[j,k])*(self.Vz[j,k])
                Vx = (self.Vx[j,k])*(self.Vx[j,k])
                self.Velocity[j,k] = Vx + Vz
                if Vz > self.maxV:
                    self.maxV = Vz
        return r
    
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