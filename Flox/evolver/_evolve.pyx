# -*- coding: utf-8 -*-
# 
#  _evolve.pyx
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-18.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

#cython: overflowcheck=False
#cython: wraparound=False
#cython: boundscheck=False
#cython: cdivision=True

from __future__ import division

import numpy as np
cimport numpy as np
cimport cython
from cpython.array cimport array, clone

from Flox._flox cimport DTYPE_t

cdef class Evolver:
    
    def __cinit__(self, *args, **kwargs):
        self.timestep_ready = False
    
    cpdef DTYPE_t delta_time(self):
        
        return self.timestep * self.safety
    
    cpdef int prepare(self):
        
        return 0
        
    cpdef int compute(self):
        
        return 0
    
    cpdef int advance(self, DTYPE_t delta_time):
        
        self.Time += delta_time
        
        return 0
    
    cpdef int solve(self):
        
        return 0
    
    
    cpdef int step(self, DTYPE_t delta_time):
        
        cdef int r
        
        r = self.prepare()
        r += self.compute()
        r += self.advance(delta_time)
        r += self.solve()
        
        
        return r
        
    cpdef int evolve(self, DTYPE_t time, int max_iterations):
        
        cdef int j, r = 0, cfl = 0
        cdef DTYPE_t timestep
        self.timestep_ready = False
        
        for j in range(max_iterations):
            if self.Time > time:
                break
            
            timestep = self.delta_time()
            
            cfl += 1
            # We check the CFL every N iterations and at the first iteration.
            if cfl >= self.checkCFL or j==0:
                cfl = 0
                self.timestep_ready = False
            
            r += self.step(timestep)
            
            if r == 0:
                pass
            else:
                break
        
        return r
    
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
 
    property VectorPotential:
    
        """Magnetic Vector Potential"""
        
        def __get__(self):
            return np.asanyarray(self._VectorPotential.V_curr)

        def __set__(self, value):
            self._VectorPotential.V_curr = np.asanyarray(value).copy()
            
    property dVectorPotential:

        """Derivative of Magnetic Vector Potential with Time"""

        def __get__(self):
            return np.asanyarray(self._VectorPotential.G_prev)

        def __set__(self, value):
            self._VectorPotential.G_prev = np.asanyarray(value).copy()
            
    property CurrentDensity:

        """Magnetic Current Density"""

        def __get__(self):
            return np.asanyarray(self._CurrentDensity.V_curr)

        def __set__(self, value):
            self._CurrentDensity.V_curr = np.asanyarray(value).copy()