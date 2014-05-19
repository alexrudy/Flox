# -*- coding: utf-8 -*-
# 
#  _linear.pyx
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-24.
#  Copyright 2014 University of California. All rights reserved.
# 

#cython: overflowcheck=True
#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True

from __future__ import division

import numpy as np
cimport numpy as np
cimport cython

from Flox._flox cimport DTYPE_t
from Flox._evolve cimport Evolver
from Flox.component.temperature cimport TemperatureSolver
from Flox.component.vorticity cimport VorticitySolver
from Flox.component.stream cimport StreamSolver

cdef class LinearEvolver(Evolver):
    
    def __cinit__(self, int nz, int nx, DTYPE_t[:] npa, DTYPE_t Pr, DTYPE_t Ra, DTYPE_t dz, DTYPE_t safety):
        
        self.npa = npa
        self.Pr = Pr
        self.Ra = Ra
        self.dz = dz
        self._Temperature = TemperatureSolver(nz, nx)
        self._Vorticity = VorticitySolver(nz, nx)
        self._Stream = StreamSolver(nz, nx)
        self._Stream.setup(self.dz, self.npa)
        self.safety = safety
        
    
    cpdef DTYPE_t delta_time(self):
        
        return (self.dz * self.dz) / 4.0 * self.safety
        
    cpdef int step(self, DTYPE_t delta_time):
        
        cdef DTYPE_t time = self.Time
        # Compute the derivatives
        self._Temperature.compute_base(self.dz, self.npa)
        self._Temperature.compute_linear(self.dz, self.npa, self._Stream.V_curr)
        self._Vorticity.compute_base(self._Temperature.V_curr, self.dz, self.npa, self.Pr, self.Ra)
        
        # Advance the derivatives
        self._Temperature.advance(delta_time)
        self._Vorticity.advance(delta_time)
        self._Stream.solve(self._Vorticity.V_curr, self._Stream.V_curr)
        
        self.Time = time + delta_time
        
        return 0
    
