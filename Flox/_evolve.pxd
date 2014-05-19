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

cdef class Evolver:
    cdef public DTYPE_t Time
    cdef VorticitySolver _Vorticity
    cdef TemperatureSolver _Temperature
    cdef StreamSolver _Stream
    
    cpdef DTYPE_t delta_time(self)
    cpdef int step(self, DTYPE_t delta_time)
    cpdef int evolve(self, DTYPE_t time, int max_iterations)