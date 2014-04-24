# -*- coding: utf-8 -*-
# 
#  _tridiagonal.pyx
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-17.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import division

import numpy as np
cimport numpy as np
cimport cython
from cpython.array cimport array, clone

from Flox._flox cimport DTYPE_t
from Flox.finitedifference cimport second_derivative2D
from Flox._solve cimport Solver, Evolver

cpdef int temperature(int J, int K, DTYPE_t[:,:] d_T, DTYPE_t[:,:] T_curr, DTYPE_t dz, DTYPE_t[:] npa):
    
    cdef int j, k
    
    # The last term in equation (2.10)
    # This resets the values in T_next
    for k in range(0, K):
        for j in range(0, J):
            d_T[j,k] = T_curr[j,k] * npa[k] * npa[k]
    
    # The second last term in equation (2.10)
    # Boundary Conditions:
    # T(z=0) = 1.0
    # T(z=1) = 0.0
    r1 = second_derivative2D(J, K, d_T, T_curr, dz, 0.0, 1.0, 1.0)
    
    return r1

cdef class TemperatureSolver(Solver):
    
    cpdef int compute(self, DTYPE_t dz, DTYPE_t[:] npa):
        
        return temperature(self.nz, self.nx, self.G_curr, self.V_curr, dz, npa)

cpdef int vorticity(int J, int K, DTYPE_t[:,:] d_V, DTYPE_t[:,:] V_curr, DTYPE_t[:,:] T_curr, DTYPE_t dz, DTYPE_t[:] npa, DTYPE_t Pr, DTYPE_t Ra):
    
    cdef int j, k
    
    # The second term and fourth in equation (2.11)
    for k in range(0, K):
        for j in range(0, J):
            d_V[j,k] = (Ra * Pr * npa[k] * T_curr[j,k]) - (Pr * npa[k] * npa[k] * V_curr[j,k])
        
    # The second last term in equation (2.11)
    # Boundary Conditions:
    # w(z=0) = 0.0
    # w(z=1) = 0.0
    r1 = second_derivative2D(J, K, d_V, V_curr, dz, 0.0, 0.0, Pr)
    
    return r1
    

cdef class VorticitySolver(Solver):
    
    cpdef int compute(self, DTYPE_t[:,:] T_curr, DTYPE_t dz, DTYPE_t[:] npa, DTYPE_t Pr, DTYPE_t Ra):
        
        return vorticity(self.nz, self.nx, self.G_curr, self.V_curr, T_curr, dz, npa, Pr, Ra)
    
    
cdef class LinearEvolver(Evolver):
    
    def __cinit__(self, DTYPE_t[:,:] Temperature, DTYPE_t[:,:] Vorticity, DTYPE_t[:] npa, DTYPE_t Pr, DTYPE_t Ra, DTYPE_t dz, DTYPE_t time):
        
        nz = Temperature.shape[0]
        nx = Temperature.shape[1]
        self.time = time
        self.Temperature = TemperatureSolver(nz, nx, Temperature)
        self.Vorticity = VorticitySolver(nz, nx, Vorticity)
        self.npa = npa
        self.Pr = Pr
        self.Ra = Ra
        self.dz = dz
        
    
    cpdef int get_state(self, DTYPE_t[:,:] Temperature, DTYPE_t[:,:] Vorticity):
        
        Temperature = self.Temperature.V_curr
        Vorticity = self.Vorticity.V_curr
        return 0
    
    cpdef int set_state(self, DTYPE_t[:,:] Temperature, DTYPE_t[:,:] Vorticity, DTYPE_t time):
        
        self.Temperature.V_curr = Temperature
        self.Vorticity.V_curr = Vorticity
        self.time = time
        return 0
        
    
    cpdef DTYPE_t delta_time(self):
        
        return (self.dz * self.dz) / 4.0
        
    cpdef int step(self, DTYPE_t delta_time):
        
        # Compute the derivatives
        self.Temperature.compute(self.dz, self.npa)
        self.Vorticity.compute(self.Temperature.V_curr, self.dz, self.npa, self.Pr, self.Ra)
        
        # Advance the derivatives
        self.Temperature.advance(delta_time)
        self.Vorticity.advance(delta_time)
        
        self.time += delta_time
    
    
    