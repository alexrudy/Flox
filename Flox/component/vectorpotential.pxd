# -*- coding: utf-8 -*-
# 
#  vectorpotential.pxd
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-19.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from Flox._flox cimport DTYPE_t
from Flox._solve cimport Solver

cpdef int vectorpotential(int J, int K, DTYPE_t[:,:] d_A, DTYPE_t[:,:] A_curr, DTYPE_t[:,:] dAdzz, DTYPE_t dz, DTYPE_t[:] npa, DTYPE_t q) nogil

cpdef int vectorpotential_linear(int J, int K, DTYPE_t[:,:] d_T, DTYPE_t[:,:] P_curr) nogil

cpdef int vectorpotential_dzz(int J, int K, DTYPE_t[:,:] dAdzz, DTYPE_t[:,:] A_curr, DTYPE_t dz, DTYPE_t factor) nogil

cdef class VectorPotentialSolver(Solver):
    cdef DTYPE_t[:,:] dVdzz
    cpdef int compute_base(self, DTYPE_t dz, DTYPE_t[:] npa, DTYPE_t q)
    cpdef int compute_linear(self, DTYPE_t[:,:] dPdz)
    cpdef int compute_nonlinear(self, DTYPE_t[:,:] P_curr, DTYPE_t[:,:] dPdz, DTYPE_t a, DTYPE_t[:] npa)