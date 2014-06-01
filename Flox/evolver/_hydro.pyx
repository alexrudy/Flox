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


cdef class HydroEvolver(Evolver):
    
    def __cinit__(self, int nz, int nx, DTYPE_t[:] npa, DTYPE_t dz, DTYPE_t a, DTYPE_t safety, int checkCFL):
        
        self.dz = dz
        self.npa = npa
        self.a = a
        self._Temperature = TemperatureSolver(nz, nx)
        self._Vorticity = VorticitySolver(nz, nx)
        self._Stream = StreamSolver(nz, nx)
        self._Stream.setup(self.dz, self.npa)
        self.safety = safety
        self.checkCFL = checkCFL
        self.maxV = 0.0
        
    
    cpdef DTYPE_t delta_time(self):
        
        cdef DTYPE_t dt_diffusion, dt_gmode, dt_velocity, dt
        if self.timestep_ready:
            return Evolver.delta_time(self)
        dt_diffusion = (self.dz * self.dz) / 4.0
        dt_gmode = (2.0 * np.pi) / (50.0 * np.sqrt(self.Ra * self.Pr))
        if self.maxV == 0:
            dt_velocity = dt_diffusion
        else:
            dt_velocity = (self.dz) / self.maxV
        
        # Check the timestep.
        dt = dt_diffusion
        if dt > dt_gmode:
            dt = dt_gmode
        if dt > dt_velocity:
            dt = dt_velocity
        
        self.timestep = dt
        self.timestep_ready = True
        return Evolver.delta_time(self)
        
    cpdef int prepare(self):
        # Prepare the computation, resetting arrays and computing first spatial derivatives.
        cdef int r = 0
        r += Evolver.prepare(self)
        r += self._Temperature.prepare(self.dz)
        r += self._Vorticity.prepare(self.dz)
        r += self._Stream.prepare(self.dz)
        return r
        
    cpdef int compute(self):
        # First the regular linear terms.
        cdef int r = 0
        r += Evolver.compute(self)
        r += self._Temperature.compute_base(self.dz, self.npa)
        if self._linear:
            # Then the linear only terms.
            r += self._Temperature.compute_linear(self.dz, self.npa, self._Stream.V_curr)
        else:
            # Then the nonlinear galerkin terms.
            r += self._Temperature.compute_nonlinear(self._Stream.V_curr, self._Stream.dVdz, self.a, self.npa)
        
        # First the regular linear terms.
        r += self._Vorticity.compute_base(self._Temperature.V_curr, self.dz, self.npa, self.Pr, self.Ra)
        # Then the nonlinear galerkin terms.
        if not self._linear:
            r += self._Vorticity.compute_nonlinear(self._Stream.V_curr, self._Stream.dVdz, self.a)
        
        return r
    
    
    cpdef int advance(self, DTYPE_t delta_time):
        cdef int r = 0
        r += Evolver.advance(self, delta_time)
        # Advance the variables.
        r += self._Temperature.advance(delta_time)
        r += self._Vorticity.advance(delta_time)
        
        return r
    
    cpdef int solve(self):
        cdef int r = 0
        r += Evolver.solve(self)
        # Advance the stream function.
        r += self._Stream.solve(self._Vorticity.V_curr, self._Stream.V_curr)
        
        # If requried, do things to compute the fluid velocity.
        if not self._Stream.transform_ready:
            r += self._Stream.setup_transform(self.npa)
        if not self.timestep_ready:
            r += self._Stream.compute_velocity()
            self.maxV = self._Stream.maxV
        
        return r
    
    property linear:
        
        "Set the evolver into linear-only mode."
        
        def __set__(self, value):
            self._linear = value
            
        def __get__(self):
            if self._linear:
                return True
            else:
                return False
    
    def set_T_bounds(self, T_p, T_m):
        self._Temperature.Value_p = T_p
        self._Temperature.Value_m = T_m
    
    
    