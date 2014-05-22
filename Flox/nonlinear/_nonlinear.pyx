# -*- coding: utf-8 -*-
# 
#  _linear.pyx
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-24.
#  Copyright 2014 University of California. All rights reserved.
# 

#cython: overflowcheck=False
#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True
#cython: profile=False

from __future__ import division

import numpy as np
cimport numpy as np
cimport cython
from cpython.array cimport array, clone
from cython.parallel cimport prange

from Flox._flox cimport DTYPE_t
from Flox.evolver._evolve cimport Evolver
from Flox.component.temperature cimport TemperatureSolver
from Flox.component.vorticity cimport VorticitySolver
from Flox.component.stream cimport StreamSolver


cdef class NonlinearEvolver(Evolver):
    
    def __cinit__(self, int nz, int nx, DTYPE_t[:] npa, DTYPE_t dz, DTYPE_t a, DTYPE_t safety):
        
        self.dz = dz
        self.npa = npa
        self.a = a
        self._Temperature = TemperatureSolver(nz, nx)
        self._Vorticity = VorticitySolver(nz, nx)
        self._Stream = StreamSolver(nz, nx)
        self._Stream.setup(self.dz, self.npa)
        self.safety = safety
        
    
    cpdef DTYPE_t delta_time(self):
        
        cdef DTYPE_t dt_a, dt_b
        if self.timestep_ready:
            return Evolver.delta_time(self)
        dt_a = (self.dz * self.dz) / 4.0
        dt_b = (2.0 * np.pi) / (50.0 * np.sqrt(self.Ra * self.Pr))
        dt = np.min([dt_a, dt_b])
        self.timestep = dt
        self.timestep_ready = True
        return Evolver.delta_time(self)
        
    cpdef int step(self, DTYPE_t delta_time):
        
        cdef DTYPE_t time = self.Time
        cdef int j, k
        
        # Prepare the computation, resetting arrays and computing first spatial derivatives.
        self._Temperature.prepare(self.dz)
        self._Vorticity.prepare(self.dz)
        self._Stream.prepare(self.dz)
        
        # Compute the time derivatives
        
        # First the regular linear terms.
        self._Temperature.compute_base(self.dz, self.npa)
        # Then the nonlinear galerkin terms.
        self._Temperature.compute_nonlinear(self._Stream.V_curr, self._Stream.dVdz, self.a, self.npa)
        
        # First the regular linear terms.
        self._Vorticity.compute_base(self._Temperature.V_curr, self.dz, self.npa, self.Pr, self.Ra)
        # Then the nonlinear galerkin terms.
        self._Vorticity.compute_nonlinear(self._Stream.V_curr, self._Stream.dVdz, self.a)
        
        # Advance the variables.
        self._Temperature.advance(delta_time)
        self._Vorticity.advance(delta_time)
        
        # Advance the stream function.
        self._Stream.solve(self._Vorticity.V_curr, self._Stream.V_curr)
        
        
        self.Time = time + delta_time
        
        return 0
    
    