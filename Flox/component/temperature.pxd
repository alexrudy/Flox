# -*- coding: utf-8 -*-
# 
#  temperature.pxd
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-18.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from Flox._flox cimport DTYPE_t
from Flox._solve cimport TimeSolver

cpdef int temperature(int J, int K, DTYPE_t[:,:] T_next, DTYPE_t[:,:] T_curr, DTYPE_t dz, DTYPE_t[:] npa, DTYPE_t[:] f_p, DTYPE_t[:] f_m) nogil

cpdef int temperature_linear(int J, int K, DTYPE_t[:,:] d_T, DTYPE_t[:,:] P_curr, DTYPE_t[:] npa) nogil

cpdef int temperature_forcing(int J, int K, DTYPE_t[:,:] d_T, DTYPE_t[:,:] T_curr, DTYPE_t[:] T_s, DTYPE_t tau, int js) nogil

cdef class TemperatureSolver(TimeSolver):
    cdef DTYPE_t[:] T_s
    cdef int Z_interface
    
    cpdef int compute_base(self, DTYPE_t dz, DTYPE_t[:] npa)
    cpdef int compute_linear(self, DTYPE_t dz, DTYPE_t[:] npa, DTYPE_t[:,:] P_curr)
    cpdef int compute_nonlinear(self, DTYPE_t[:,:] P_curr, DTYPE_t[:,:] dPdz, DTYPE_t a, DTYPE_t[:] npa)
    cpdef int compute_forcing(self, DTYPE_t tau)