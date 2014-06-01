# -*- coding: utf-8 -*-
# 
#  vorticity.pyx
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
from Flox.finitedifference cimport second_derivative2D
from Flox.component._solve cimport TimeSolver
from Flox.nonlinear.galerkin cimport galerkin_cos_grad_cos


cpdef int vorticity(int J, int K, DTYPE_t[:,:] d_V, DTYPE_t[:,:] V_curr, DTYPE_t[:,:] T_curr, DTYPE_t dz, DTYPE_t[:] npa, DTYPE_t Pr, DTYPE_t Ra, DTYPE_t[:] f_p, DTYPE_t[:] f_m) nogil:
    
    cdef int j, k
    
    # The second term and fourth in equation (2.11)
    for k in range(K):
        for j in range(J):
            d_V[j,k] += (Ra * Pr * npa[k] * T_curr[j,k]) - (Pr * npa[k] * npa[k] * V_curr[j,k])
        
    # The second last term in equation (2.11)
    # Boundary Conditions:
    # w(z=0) = 0.0
    # w(z=1) = 0.0
    r1 = second_derivative2D(J, K, d_V, V_curr, dz, f_p, f_m, Pr)
    
    return r1
    
cpdef int linear_lorentz(int J, int K, DTYPE_t[:,:] d_V, DTYPE_t[:,:] dJdz, DTYPE_t factor) nogil:
    cdef int j, k
    for k in prange(K):
        for j in range(j):
            d_V[j, k] += factor * dJdz[j , k]
    return 0

cdef class VorticitySolver(TimeSolver):
    
    cpdef int compute_base(self, DTYPE_t[:,:] T_curr, DTYPE_t dz, DTYPE_t[:] npa, DTYPE_t Pr, DTYPE_t Ra):
        
        return vorticity(self.nz, self.nx, self.G_curr, self.V_curr, T_curr, dz, npa, Pr, Ra, self.V_p, self.V_m)
        
    cpdef int compute_nonlinear(self, DTYPE_t[:,:] P_curr, DTYPE_t[:,:] dPdz, DTYPE_t a):
    
        # Now we do the non-linear terms from equation 4.6
        return galerkin_cos_grad_cos(self.nz, self.nx, self.G_curr, self.V_curr, self.dVdz, P_curr, dPdz, a, 1.0)
    
    cpdef int compute_lorentz(self, DTYPE_t[:,:] A_curr, DTYPE_t[:,:] dAdz, DTYPE_t[:,:] J_curr, DTYPE_t[:,:] dJdz, DTYPE_t a, DTYPE_t Q, DTYPE_t Pr, DTYPE_t q):
        cdef int r
        # Compute the linear magnetic lorentz force due to the background field.
        r = linear_lorentz(self.nz, self.nx, self.G_curr, dJdz, -1.0 * (Q * Pr)/q)
        # Compute the magnetic lorentz force from equation 
        r += galerkin_cos_grad_cos(self.nz, self.nx, self.G_curr, J_curr, dJdz, A_curr, dAdz, a, -1.0 * (Q * Pr)/q)
        return r
