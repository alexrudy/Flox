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
from Flox.tridiagonal._tridiagonal cimport TridiagonalSolver


cpdef int temperature(int J, int K, DTYPE_t[:,:] T_next, DTYPE_t[:,:] T_curr, DTYPE_t[:,:] P_curr, DTYPE_t dz, DTYPE_t[:] npa, DTYPE_t[:] f_p, DTYPE_t[:] f_m)

cpdef int vorticity(int J, int K, DTYPE_t[:,:] d_V, DTYPE_t[:,:] V_curr, DTYPE_t[:,:] T_curr, DTYPE_t dz, DTYPE_t[:] npa, DTYPE_t Pr, DTYPE_t Ra, DTYPE_t[:] f_p, DTYPE_t[:] f_m)

cdef class VorticitySolver(Solver):
    cpdef int compute(self, DTYPE_t[:,:] T_curr, DTYPE_t dz, DTYPE_t[:] npa, DTYPE_t Pr, DTYPE_t Ra)
    
cdef class TemperatureSolver(Solver):
    cpdef int compute(self, DTYPE_t[:,:] P_curr, DTYPE_t dz, DTYPE_t[:] npa)
    
cdef class StreamSolver(TridiagonalSolver):
    cdef DTYPE_t[:,:] V_curr
    cpdef int setup(self, DTYPE_t dz, DTYPE_t[:] npa)
    
cdef class NonlinearEvolver(Evolver):
    cdef readonly DTYPE_t Pr
    cdef readonly DTYPE_t Ra
    cdef readonly DTYPE_t dz
    cdef readonly DTYPE_t safety
    cdef DTYPE_t[:] npa
    cdef VorticitySolver Vorticity
    cdef TemperatureSolver Temperature
    cdef StreamSolver Stream
    
    cpdef int get_state(self, DTYPE_t[:,:] Temperature, DTYPE_t[:,:] Vorticity, DTYPE_t[:,:] Stream)
    cpdef int set_state(self, DTYPE_t[:,:] Temperature, DTYPE_t[:,:] Vorticity, DTYPE_t[:,:] Stream, DTYPE_t time)