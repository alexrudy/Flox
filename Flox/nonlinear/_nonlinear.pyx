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
from Flox._evolve cimport Evolver
from Flox.component.temperature cimport TemperatureSolver
from Flox.component.vorticity cimport VorticitySolver
from Flox.component.stream cimport StreamSolver


cdef class NonlinearEvolver(Evolver):
    
    def __cinit__(self, int nz, int nx, DTYPE_t[:] npa, DTYPE_t Pr, DTYPE_t Ra, DTYPE_t dz, DTYPE_t a, DTYPE_t safety):
        
        self.npa = npa
        self.Pr = Pr
        self.Ra = Ra
        self.dz = dz
        self.a = a
        self._Temperature = TemperatureSolver(nz, nx)
        self._Vorticity = VorticitySolver(nz, nx)
        self._Stream = StreamSolver(nz, nx)
        self._Stream.setup(self.dz, self.npa)
        self.safety = safety
        
    
    cpdef DTYPE_t delta_time(self):
        
        cdef DTYPE_t dt_a, dt_b
        # return 3.6e-6 * self.safety
        dt_a =  (self.dz * self.dz) / 4.0
        dt_b = (2.0 * np.pi) / (50.0 * np.sqrt(self.Ra * self.Pr)) 
        if dt_a > dt_b:
            return dt_b * self.safety
        else:
            return dt_a * self.safety
        
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
    
    property Temperature:
    
        """Temperature"""
    
        def __get__(self):
            return np.asanyarray(self._Temperature.V_curr)
        
        def __set__(self, value):
            self._Temperature.V_curr = np.asanyarray(value).copy()
        
    property dTemperature:
    
        """Derivative of Temperature with Time"""
    
        def __get__(self):
            return np.asanyarray(self._Temperature.G_prev)

        def __set__(self, value):
            self._Temperature.G_prev = np.asanyarray(value).copy()

    property Vorticity:

        """Vorticity"""

        def __get__(self):
            return np.asanyarray(self._Vorticity.V_curr)

        def __set__(self, value):
            self._Vorticity.V_curr = np.asanyarray(value).copy()

    property dVorticity:

        """Derivative of Vorticity with Time"""

        def __get__(self):
            return np.asanyarray(self._Vorticity.G_prev)

        def __set__(self, value):
            self._Vorticity.G_prev = np.asanyarray(value).copy()
        
    property Stream:
    
        """Stream function"""
    
        def __get__(self):
            return np.asanyarray(self._Stream.V_curr)

        def __set__(self, value):
            self._Stream.V_curr = np.asanyarray(value).copy()
    