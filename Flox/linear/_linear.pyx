# -*- coding: utf-8 -*-
# 
#  _linear.pyx
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-24.
#  Copyright 2014 University of California. All rights reserved.
# 

#cython: overflowcheck=True
#cython: wraparound=False
#cython: boundscheck=True
#cython: cdivision=True

from __future__ import division

import numpy as np
cimport numpy as np
cimport cython
from cpython.array cimport array, clone

from Flox._flox cimport DTYPE_t
from Flox.finitedifference cimport second_derivative2D
from Flox._solve cimport Solver, Evolver
from Flox.tridiagonal._tridiagonal cimport TridiagonalSolver

cpdef int temperature(int J, int K, DTYPE_t[:,:] d_T, DTYPE_t[:,:] T_curr, DTYPE_t[:,:] P_curr, DTYPE_t dz, DTYPE_t[:] npa, DTYPE_t[:] f_p, DTYPE_t[:] f_m):
    
    cdef int j, k
    # The last term in equation (2.10)
    # This resets the values in T_next
    for k in range(K):
        for j in range(J):
            d_T[j,k] =  -T_curr[j,k] * npa[k] * npa[k]
    
    # The second last term in equation (2.10)
    r1 = second_derivative2D(J, K, d_T, T_curr, dz, f_p, f_m, 1.0)
    
    return r1

cdef class TemperatureSolver(Solver):
    
    def __cinit__(self, int nz, int nx):
        # Boundary Conditions:
        # T(n=0,z=0) = 1.0
        self.V_m[0] = 1.0
    
    cpdef int compute(self, DTYPE_t[:,:] P_curr, DTYPE_t dz, DTYPE_t[:] npa):
        
        cdef int r
        cdef DTYPE_t npa_i
        r = temperature(self.nz, self.nx, self.G_curr, self.V_curr, P_curr, dz, npa, self.V_p, self.V_m)
        
        for k in range(self.nx):
            npa_i = npa[k]
            for j in range(self.nz):
                self.G_curr[j, k] += npa_i * P_curr[j,k]
        
        return r

cpdef int vorticity(int J, int K, DTYPE_t[:,:] d_V, DTYPE_t[:,:] V_curr, DTYPE_t[:,:] T_curr, DTYPE_t dz, DTYPE_t[:] npa, DTYPE_t Pr, DTYPE_t Ra, DTYPE_t[:] f_p, DTYPE_t[:] f_m):
    
    cdef int j, k
    
    # The second term and fourth in equation (2.11)
    for k in range(K):
        for j in range(J):
            d_V[j,k] = (Ra * Pr * npa[k] * T_curr[j,k]) - (Pr * npa[k] * npa[k] * V_curr[j,k])
        
    # The second last term in equation (2.11)
    # Boundary Conditions:
    # w(z=0) = 0.0
    # w(z=1) = 0.0
    r1 = second_derivative2D(J, K, d_V, V_curr, dz, f_p, f_m, Pr)
    
    return r1
    

cdef class VorticitySolver(Solver):
    
    cpdef int compute(self, DTYPE_t[:,:] T_curr, DTYPE_t dz, DTYPE_t[:] npa, DTYPE_t Pr, DTYPE_t Ra):
        
        return vorticity(self.nz, self.nx, self.G_curr, self.V_curr, T_curr, dz, npa, Pr, Ra, self.V_p, self.V_m)
    

cdef class StreamSolver(TridiagonalSolver):
    
    cpdef int setup(self, DTYPE_t dz, DTYPE_t[:] npa):
        
        cdef int j, k
        cdef DTYPE_t dzs = dz * dz
        cdef DTYPE_t dzI = -1.0 / dzs
        
        for j in range(self.J):
            self.sub[j,0] = dzI
            self.sup[j,0] = 0.0
            self.dia[j,0] = 1.0
            for k in range(1, self.K-1):
                self.sub[j,k] = dzI
                self.sup[j,k] = dzI
                self.dia[j,k] = npa[k] * npa[k] + 2.0/dzs
            self.sub[j,self.K-1] = 0.0
            self.sup[j,self.K-1] = dzI
            self.dia[j,self.K-1] = 1.0
        
        return self._warm_work()
        
    
    
cdef class LinearEvolver(Evolver):
    
    def __cinit__(self, int nz, int nx, DTYPE_t[:] npa, DTYPE_t Pr, DTYPE_t Ra, DTYPE_t dz, DTYPE_t time, DTYPE_t safety):
        
        self.time = time
        self.npa = npa
        self.Pr = Pr
        self.Ra = Ra
        self.dz = dz
        self.Temperature = TemperatureSolver(nz, nx)
        self.Vorticity = VorticitySolver(nz, nx)
        self.Stream = StreamSolver(nz, nx)
        self.Stream.setup(self.dz, self.npa)
        self.safety = safety
        
    
    cpdef int get_state(self, DTYPE_t[:,:] Temperature, DTYPE_t[:,:] Vorticity, DTYPE_t[:,:] Stream):
        
        Temperature[...] = self.Temperature.V_curr
        Vorticity[...] = self.Vorticity.V_curr
        Stream[...] =  self.Stream.V_curr
        return 0
    
    cpdef int set_state(self, DTYPE_t[:,:] Temperature, DTYPE_t[:,:] Vorticity, DTYPE_t[:,:] Stream, DTYPE_t time):
        
        self.Temperature.V_curr = Temperature
        self.Vorticity.V_curr = Vorticity
        self.Stream.V_curr = Stream
        self.time = time
        return 0
        
    
    cpdef DTYPE_t delta_time(self):
        
        return (self.dz * self.dz) / 4.0 * self.safety
        
    cpdef int step(self, DTYPE_t delta_time):
        
        cdef DTYPE_t time = self.time
        # Compute the derivatives
        self.Temperature.compute(self.Stream.V_curr, self.dz, self.npa)
        self.Vorticity.compute(self.Temperature.V_curr, self.dz, self.npa, self.Pr, self.Ra)
        self.Stream.solve(self.Vorticity.V_curr, self.Stream.V_curr)
        
        # Advance the derivatives
        self.Temperature.advance(delta_time)
        self.Vorticity.advance(delta_time)
        
        self.time = time + delta_time
        
        return 0
    
    
    