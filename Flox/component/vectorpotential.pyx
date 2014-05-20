# -*- coding: utf-8 -*-
# 
#  vectorpotential.pyx
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
from Flox.finitedifference cimport second_derivative2D_nb
from Flox._solve cimport TimeSolver
from Flox.nonlinear.galerkin cimport galerkin_sin

cpdef int vectorpotential(int J, int K, DTYPE_t[:,:] d_A, DTYPE_t[:,:] A_curr, DTYPE_t[:,:] dAdzz, DTYPE_t dz, DTYPE_t[:] npa, DTYPE_t q) nogil:
    
    cdef int j, k, r
    cdef DTYPE_t npa_s
    # The last term in equation (11.25)
    # This resets the values in d_A
    for k in prange(K):
        npa_s = npa[k] * npa[k]
        for j in range(J):
            d_A[j,k] = (dAdzz[j, k] - A_curr[j,k] * npa_s) / q
        
    return 0

cpdef int vectorpotential_linear(int J, int K, DTYPE_t[:,:] d_A, DTYPE_t[:,:] dPdz) nogil:
    
    cdef int j, k
    
    for k in prange(K):
        for j in range(J):
            d_A[j, k] += dPdz[j, k]
    
    return 0
    
cpdef int vectorpotential_dzz(int J, int K, DTYPE_t[:,:] dAdzz, DTYPE_t[:,:] A_curr, DTYPE_t dz, DTYPE_t factor) nogil:
    
    cdef int j, k, r1
    for k in range(K):
        # This takes care of the top and bottom boundaries in (11.25)
        # Equation (11.37): Bottom boundary, first derivative vanishes.
        j = 0
        dAdzz[j,k] += factor * (2.0 * (A_curr[j+1,k] - A_curr[j,k])) / (dz * dz)
        # Equation (11.38): Top boundary, first derivative vanishes.
        j = J - 1
        dAdzz[j,k] += factor * (2.0 * (A_curr[j-1,k] - A_curr[j,k])) / (dz * dz)
    
    # The second last term in equation (11.25)
    r1 = second_derivative2D_nb(J, K, dAdzz, A_curr, dz, factor)
    # We handle the boundary conditions separately, above
    return r1
    
cdef class VectorPotentialSolver(TimeSolver):
    
    def __cinit__(self, int nz, int nx):
        # Initialize the second derivative array here.
        self.dVdzz = np.zeros((nz, nx), dtype=np.float)
    
    cpdef int prepare(self, DTYPE_t dz):
        # Compute the first and second z derivatives of the vector potential here.
        cdef int r
        r = TimeSolver.prepare(self, dz)
        r += vectorpotential_dzz(self.nz, self.nx, self.dVdzz, self.V_curr, dz, 1.0)
        return r
        
    
    cpdef int compute_base(self, DTYPE_t dz, DTYPE_t[:] npa, DTYPE_t q):
    
        return vectorpotential(self.nz, self.nx, self.G_curr, self.V_curr, self.dVdzz, dz, npa, q)
    
    cpdef int compute_linear(self, DTYPE_t[:,:] dPdz):
    
        return vectorpotential_linear(self.nz, self.nx, self.G_curr, dPdz)
    
    cpdef int compute_nonlinear(self, DTYPE_t[:,:] P_curr, DTYPE_t[:,:] dPdz, DTYPE_t a, DTYPE_t[:] npa, DTYPE_t dz):
        
        cdef int k,j
        # First we tweak the advection on the boundaries.
        for k in range(self.nx):
            j = 0
            dPdz[ j, k ] = P_curr[j + 1, k] / dz
            j = self.nz - 1
            dPdz[ j, k ] = - P_curr[j - 1, k] / dz
        
        # Now we do the non-linear terms from equation 11.25
        return galerkin_sin(self.nz, self.nx, self.G_curr, self.V_curr, self.dVdz, P_curr, dPdz, a, npa, 1.0)

