# -*- coding: utf-8 -*-
# 
#  vorticity.pxd
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-18.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from Flox._flox cimport DTYPE_t
from Flox._solve cimport TimeSolver

cpdef int vorticity(int J, int K, DTYPE_t[:,:] d_V, DTYPE_t[:,:] V_curr, DTYPE_t[:,:] T_curr, DTYPE_t dz, DTYPE_t[:] npa, DTYPE_t Pr, DTYPE_t Ra, DTYPE_t[:] f_p, DTYPE_t[:] f_m) nogil

cpdef int linear_lorentz(int J, int K, DTYPE_t[:,:] d_V, DTYPE_t[:,:] dJdz, DTYPE_t factor) nogil

cdef class VorticitySolver(TimeSolver):
    cpdef int compute_base(self, DTYPE_t[:,:] T_curr, DTYPE_t dz, DTYPE_t[:] npa, DTYPE_t Pr, DTYPE_t Ra)
    cpdef int compute_nonlinear(self, DTYPE_t[:,:] P_curr, DTYPE_t[:,:] dPdz, DTYPE_t a)
    cpdef int compute_lorentz(self, DTYPE_t[:,:] A_curr, DTYPE_t[:,:] dAdz, DTYPE_t[:,:] J_curr, DTYPE_t[:,:] dJdz, DTYPE_t a, DTYPE_t Q, DTYPE_t Pr, DTYPE_t q)