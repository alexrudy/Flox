# -*- coding: utf-8 -*-
# 
#  _magneto.pyx
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-19.
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
from cython.parallel cimport prange

from Flox._flox cimport DTYPE_t
from Flox._evolve cimport Evolver
from Flox.component.temperature cimport TemperatureSolver
from Flox.component.vorticity cimport VorticitySolver
from Flox.component.stream cimport StreamSolver
from Flox.component.vectorpotential cimport VectorPotentialSolver
from Flox.component.currentdensity cimport CurrentDensitySolver

cdef class MagnetoEvolver(Evolver):
    
    def __cinit__(self, int nz, int nx, DTYPE_t[:] npa, DTYPE_t dz, DTYPE_t a, DTYPE_t safety):
        
        self.dz = dz
        self.npa = npa
        self._Temperature = TemperatureSolver(nz, nx)
        self._Vorticity = VorticitySolver(nz, nx)
        self._Stream = StreamSolver(nz, nx)
        self._Stream.setup(dz, npa)
        self._VectorPotential = VectorPotentialSolver(nz, nx)
        self._VectorPotential.prepare(dz) # Called becasue this is needed before the first iteration to compute J.
        self._CurrentDensity = CurrentDensitySolver(nz, nx)
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
        if not self._VectorPotential.ready:
            self._VectorPotential.prepare(self.dz)
        # Compute the time derivatives
        
        # Vector Potential
        self._VectorPotential.compute_base(self.dz, self.npa, self.q)
        self._VectorPotential.compute_nonlinear(self._Stream.V_curr, self._Stream.dVdz, self.a, self.npa, self.dz)
        self._VectorPotential.compute_linear(self._Stream.dVdz)
        
        # Tempearture
        self._Temperature.compute_base(self.dz, self.npa)
        self._Temperature.compute_nonlinear(self._Stream.V_curr, self._Stream.dVdz, self.a, self.npa)
        
        # Vorticity
        self._Vorticity.compute_base(self._Temperature.V_curr, self.dz, self.npa, self.Pr, self.Ra)
        self._Vorticity.compute_nonlinear(self._Stream.V_curr, self._Stream.dVdz, self.a)
        self._Vorticity.compute_lorentz(self._VectorPotential.V_curr, self._VectorPotential.dVdz, self._CurrentDensity.V_curr, self._CurrentDensity.dVdz, self.a, self.Q, self.Pr, self.q)
        
        # Advance the variables.
        self._Temperature.advance(delta_time)
        self._Vorticity.advance(delta_time)
        self._VectorPotential.advance(delta_time)
        
        # Advance the stream function.
        self._Stream.solve(self._Vorticity.V_curr, self._Stream.V_curr)
        # Advance the current denstiy function.
        self._VectorPotential.prepare(self.dz) # This is called to set the second-derivatives correctly.
        self._CurrentDensity.compute_base(self._VectorPotential.V_curr, self._VectorPotential.dVdzz, self.npa)
        
        
        self.Time = time + delta_time
        
        return 0
    
    