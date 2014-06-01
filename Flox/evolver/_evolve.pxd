# -*- coding: utf-8 -*-
# 
#  _evolve.pxd
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-18.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import division

cimport cython

from Flox._flox cimport DTYPE_t
from Flox.component.temperature cimport TemperatureSolver
from Flox.component.vorticity cimport VorticitySolver
from Flox.component.stream cimport StreamSolver
from Flox.component.vectorpotential cimport VectorPotentialSolver
from Flox.component.currentdensity cimport CurrentDensitySolver

cdef class Evolver:
    cdef public DTYPE_t Time
    cdef public DTYPE_t dz
    cdef public DTYPE_t a
    cdef public DTYPE_t safety
    cdef bint timestep_ready
    cdef public int checkCFL
    cdef DTYPE_t timestep
    cdef DTYPE_t[:] npa
    
    cdef VorticitySolver _Vorticity
    cdef TemperatureSolver _Temperature
    cdef StreamSolver _Stream
    cdef VectorPotentialSolver _VectorPotential
    cdef CurrentDensitySolver _CurrentDensity
    
    cpdef DTYPE_t delta_time(self)
    cpdef int step(self, DTYPE_t delta_time)
    cpdef int evolve(self, DTYPE_t time, int max_iterations)
    cpdef int prepare(self)
    cpdef int compute(self)
    cpdef int advance(self, DTYPE_t delta_time)
    cpdef int solve(self)
    
