# -*- coding: utf-8 -*-
# 
#  _linear.pxd
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-18.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from Flox._flox cimport DTYPE_t
from Flox._solve cimport Solver, Evolver
from Flox.linear._linear cimport StreamSolver

cpdef int galerkin_sin(int J, int K, DTYPE_t[:,:] G_curr, DTYPE_t[:,:] V_curr, DTYPE_t[:,:] dVdz, DTYPE_t[:,:] O_curr, DTYPE_t[:,:] dOdz, DTYPE_t a, DTYPE_t[:] npa) nogil

cpdef int galerkin_cos(int J, int K, DTYPE_t[:,:] G_curr, DTYPE_t[:,:] V_curr, DTYPE_t[:,:] dVdz, DTYPE_t[:,:] O_curr, DTYPE_t[:,:] dOdz, DTYPE_t a) nogil

cdef class VorticitySolver(Solver):
    cpdef int compute(self, DTYPE_t[:,:] T_curr, DTYPE_t[:,:] P_curr, DTYPE_t[:,:] dPdz, DTYPE_t dz, DTYPE_t a, DTYPE_t[:] npa, DTYPE_t Pr, DTYPE_t Ra)
    
cdef class TemperatureSolver(Solver):
    cpdef int compute(self, DTYPE_t[:,:] P_curr, DTYPE_t[:,:] dPdz, DTYPE_t dz, DTYPE_t a, DTYPE_t[:] npa)
    
cdef class NonlinearEvolver(Evolver):
    cdef readonly DTYPE_t Pr
    cdef readonly DTYPE_t Ra
    cdef readonly DTYPE_t dz
    cdef readonly DTYPE_t safety
    cdef DTYPE_t[:] npa
    cdef VorticitySolver _Vorticity
    cdef TemperatureSolver _Temperature
    cdef StreamSolver _Stream
    