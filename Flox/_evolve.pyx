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
#cython: profile=False

from __future__ import division

import numpy as np
cimport numpy as np
cimport cython
from cpython.array cimport array, clone

from Flox._flox cimport DTYPE_t

cdef class Evolver:
    
    cpdef DTYPE_t delta_time(self):
        
        return 0.0
    
    cpdef int step(self, DTYPE_t delta_time):
        
        return 0
        
    cpdef int evolve(self, DTYPE_t time, int max_iterations):
        
        cdef int j, r = 0
        
        for j in range(max_iterations):
            if self.Time > time:
                break
            
            r += self.step(self.delta_time())
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
            self._Temperature.V_curr = np.asanyarray(value)
        
    property dTemperature:
    
        """Derivative of Temperature with Time"""
    
        def __get__(self):
            return np.asanyarray(self._Temperature.G_prev)

        def __set__(self, value):
            self._Temperature.G_prev = np.asanyarray(value)

    property Vorticity:

        """Vorticity"""

        def __get__(self):
            return np.asanyarray(self._Vorticity.V_curr)

        def __set__(self, value):
            self._Vorticity.V_curr = np.asanyarray(value)

    property dVorticity:

        """Derivative of Vorticity with Time"""

        def __get__(self):
            return np.asanyarray(self._Vorticity.G_prev)

        def __set__(self, value):
            self._Vorticity.G_prev = np.asanyarray(value)
        
    property Stream:
    
        """Stream function"""
    
        def __get__(self):
            return np.asanyarray(self._Stream.V_curr)

        def __set__(self, value):
            self._Stream.V_curr = np.asanyarray(value)
 
    property VectorPotential:
    
        """Magnetic Vector Potential"""
        
        def __get__(self):
            return np.asanyarray(self._VectorPotential.V_curr)

        def __set__(self, value):
            self._VectorPotential.V_curr = np.asanyarray(value)