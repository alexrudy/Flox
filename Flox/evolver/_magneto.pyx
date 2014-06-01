# -*- coding: utf-8 -*-
# 
#  _magneto.pyx
#  Flox
#  
#  Created by Alexander Rudy on 2014-06-01.
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
from cython.parallel cimport prange

from Flox._flox cimport DTYPE_t
from Flox.evolver._hydro cimport HydroEvolver
from Flox.component.vectorpotential cimport VectorPotentialSolver
from Flox.component.currentdensity cimport CurrentDensitySolver

cdef class MagnetoEvolver(HydroEvolver):
    
    def __cinit__(self, int nz, int nx, DTYPE_t[:] npa, DTYPE_t dz, DTYPE_t a, DTYPE_t safety, int checkCFL):
        
        self._VectorPotential = VectorPotentialSolver(nz, nx)
        self._CurrentDensity = CurrentDensitySolver(nz, nx)
        self.maxAlfven = 0.0
    
    cpdef DTYPE_t delta_time(self):
        
        cdef DTYPE_t dt_a, dt_b, dt_c, dt_d, dt
        if self.timestep_ready:
            return HydroEvolver.delta_time(self)
        if self.maxAlfven == 0.0:
            self.maxAlfven = self.Q * self.Pr / self.q
        dt_a = (self.dz * self.dz) / 4.0
        dt_b = (2.0 * np.pi) / (50.0 * np.sqrt(self.Ra * self.Pr))
        dt_c = (self.dz * self.dz) * self.q / 4.0
        dt_d = self.dz / np.sqrt(self.maxAlfven)
        dt = np.min([dt_a, dt_b, dt_c, dt_d])
        self.timestep = dt
        self.timestep_ready = True
        return HydroEvolver.delta_time(self)
        
    cpdef int prepare(self):
        cdef int r = 0
        
        r += HydroEvolver.prepare(self)
        if not self._VectorPotential.ready:
            r += self._VectorPotential.prepare(self.dz)
        r += self._CurrentDensity.prepare(self.dz)
        return r
        
    cpdef int compute(self):
        cdef int r = 0
        
        # Vector Potential
        r += self._VectorPotential.compute_base(self.dz, self.npa, self.q)
        if not self._linear:
            r += self._VectorPotential.compute_nonlinear(self._Stream.V_curr, self._Stream.dVdz, self.a, self.dz)
        r += self._VectorPotential.compute_linear(self._Stream.dVdz)
        
        # Hydro Computations
        r += HydroEvolver.compute(self)
        
        # Lorentz Force
        r += self._Vorticity.compute_lorentz(self._VectorPotential.V_curr, self._VectorPotential.dVdz, self._CurrentDensity.V_curr, self._CurrentDensity.dVdz, self.a, self.Q, self.Pr, self.q)
        
        return r
        
    cpdef int advance(self, DTYPE_t delta_time):
        cdef int r = 0
        # Advance the Hydro Variables
        r += HydroEvolver.advance(self, delta_time)
        # Advance the Vector potential
        r += self._VectorPotential.advance(delta_time)
        return r
        
    cpdef int solve(self):
        cdef int r = 0
        # Solve the Stream function.
        r += HydroEvolver.solve(self)
        
        # Solve the current denstiy function.
        r += self._VectorPotential.prepare(self.dz) # This is called to set the second-derivatives correctly.
        r += self._CurrentDensity.compute_base(self._VectorPotential.V_curr, self._VectorPotential.dVdzz, self.npa)
        
        # If requried, do things to compute the alfven velocity.
        if not self._VectorPotential.transform_ready:
            r += self._VectorPotential.setup_transform(self.npa)
        if not self.timestep_ready:
            r += self._VectorPotential.compute_alfven(self.Q, self.q, self.Pr)
            self.maxAlfven = self._VectorPotential.maxAlfven
        
        return r
        
