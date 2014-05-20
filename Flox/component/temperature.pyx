# -*- coding: utf-8 -*-
# 
#  temperature.pyx
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-18.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

#cython: overflowcheck=True
#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True

from __future__ import division

import numpy as np
cimport numpy as np
cimport cython
from cython.parallel cimport prange

from Flox._flox cimport DTYPE_t
from Flox.finitedifference cimport second_derivative2D
from Flox._solve cimport TimeSolver
from Flox.nonlinear.galerkin cimport galerkin_sin

cpdef int temperature(int J, int K, DTYPE_t[:,:] d_T, DTYPE_t[:,:] T_curr, DTYPE_t dz, DTYPE_t[:] npa, DTYPE_t[:] f_p, DTYPE_t[:] f_m) nogil:
    
    cdef int j, k
    cdef DTYPE_t npa_s
    # The last term in equation (2.10)
    # This resets the values in T_next
    for k in prange(K):
        npa_s = npa[k] * npa[k]
        for j in range(J):
            d_T[j,k] =  -T_curr[j,k] * npa_s
    
    # The second last term in equation (2.10)
    r1 = second_derivative2D(J, K, d_T, T_curr, dz, f_p, f_m, 1.0)
    
    return r1

cpdef int temperature_linear(int J, int K, DTYPE_t[:,:] d_T, DTYPE_t[:,:] P_curr, DTYPE_t[:] npa) nogil:
    
    cdef int j, k
    cdef DTYPE_t npa_i
    
    for k in prange(K):
        npa_i = npa[k]
        for j in range(J):
            d_T[j, k] += npa_i * P_curr[j,k]
    
    return 0

cdef class TemperatureSolver(TimeSolver):
    
    def __cinit__(self, int nz, int nx):
        # Boundary Conditions:
        # T(n=0,z=0) = 1.0
        self.V_m[0] = 1.0
    
    cpdef int compute_base(self, DTYPE_t dz, DTYPE_t[:] npa):
        
        return temperature(self.nz, self.nx, self.G_curr, self.V_curr, dz, npa, self.V_p, self.V_m)
        
    cpdef int compute_linear(self, DTYPE_t dz, DTYPE_t[:] npa, DTYPE_t[:,:] P_curr):
        
        return temperature_linear(self.nz, self.nx, self.G_curr, P_curr, npa)
        
    cpdef int compute_nonlinear(self, DTYPE_t[:,:] P_curr, DTYPE_t[:,:] dPdz, DTYPE_t a, DTYPE_t[:] npa):
    
        # Now we do the non-linear terms from equation 4.6
        return galerkin_sin(self.nz, self.nx, self.G_curr, self.V_curr, self.dVdz, P_curr, dPdz, a, npa, 1.0)
    

