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
#cython: boundscheck=True
#cython: cdivision=True
#cython: profile=True

from __future__ import division

import numpy as np
cimport numpy as np
cimport cython
from cpython.array cimport array, clone

from Flox._flox cimport DTYPE_t
from Flox.finitedifference cimport second_derivative2D
from Flox._solve cimport Solver, Evolver
from Flox.linear._linear cimport vorticity, temperature, StreamSolver

cdef DTYPE_t pi = np.pi

cdef class TemperatureSolver(Solver):
    
    def __cinit__(self, int nz, int nx):
        # Boundary Conditions:
        # T(n=0,z=0) = 1.0
        self.V_m[0] = 1.0
        self.V_p[0] = 0.0
    
    cpdef int compute(self, DTYPE_t[:,:] P_curr, DTYPE_t[:,:] dPdz, DTYPE_t dz, DTYPE_t a, DTYPE_t[:] npa):
        
        cdef int r, j, k, kp, kpp, kt = 2, jt = 33
        cdef DTYPE_t p2a = pi / (2.0 * a)
        # This equation handles the linear terms. It is only slightly modified from the linear version.
        r = temperature(self.nz, self.nx, self.G_curr, self.V_curr, P_curr, dz, npa, self.V_p, self.V_m)
        # Now we do the non-linear terms from equation 4.6
        
        for j in range(self.nz):
            
            for k in range(self.nx):
                # n=0 special case.
                self.G_curr[j,0] += -p2a * k * (self.V_curr[j, k] * dPdz[j, k] + P_curr[j, k] * self.dVdz[j, k])
                
                # Terms applied everywhere.
                self.G_curr[j,k] += -npa[k] * P_curr[j, k] * self.dVdz[j, 0]
                
                for kp in range(1, self.nx):
                    # 1st term, 1st Delta
                    kpp = k - kp
                    if 0 < kpp < self.nx:
                        self.G_curr[j, k] += -p2a * (-kp * dPdz[j, kpp] * self.V_curr[j, kp] + kpp * P_curr[j, kpp] * self.dVdz[j, kp])
                    
                    # 2nd term, 1st Delta
                    kpp = kp + k
                    if 0 < kpp < self.nx:
                        self.G_curr[j, k] += -p2a * (kp * dPdz[j, kpp] * self.V_curr[j, kp] + kpp * P_curr[j, kpp] * self.dVdz[j, kp])
                        
                    # 2nd term, 2nd Delta
                    kpp = kp - k
                    if 0 < kpp < self.nx:
                        self.G_curr[j, k] += -p2a * (kp * dPdz[j, kpp] * self.V_curr[j, kp] + kpp * P_curr[j, kpp] * self.dVdz[j, kp])

        return r


cdef class VorticitySolver(Solver):
    
    cpdef int compute(self, DTYPE_t[:,:] T_curr, DTYPE_t[:,:] P_curr, DTYPE_t[:,:] dPdz, DTYPE_t dz, DTYPE_t a, DTYPE_t[:] npa, DTYPE_t Pr, DTYPE_t Ra):
        
        cdef int r, j, k, kp, kpp
        cdef DTYPE_t p2a = pi / (2.0 * a), term
        
        r = vorticity(self.nz, self.nx, self.G_curr, self.V_curr, T_curr, dz, npa, Pr, Ra, self.V_p, self.V_m)
        
        for j in range(self.nz):
            for k in range(self.nx):
                for kp in range(1, self.nx):
                    # 1st term, 1st delta
                    kpp = k - kp
                    if 0 < kpp < self.nx:
                        self.G_curr[j, k] += -p2a * (-kp * dPdz[j, kpp] * self.V_curr[j, kp] + kpp * P_curr[j, kpp] * self.dVdz[j, kp])
                    # 2nd term, 1st delta
                    kpp = kp + k
                    if 0 < kpp < self.nx:
                        self.G_curr[j, k] += p2a * (kp * dPdz[j, kpp] * self.V_curr[j, kp] + kpp * P_curr[j, kpp] * self.dVdz[j, kp])
                    # 2nd term, 2nd delta
                    kpp = kp - k
                    if 0 < kpp < self.nx:
                        self.G_curr[j, k] += -p2a * (kp * dPdz[j, kpp] * self.V_curr[j, kp] + kpp * P_curr[j, kpp] * self.dVdz[j, kp])
        return r
    
    
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
        
        # return 3.6e-6 * self.safety
        return (self.dz * self.dz) / 4.0 * self.safety
        
    cpdef int step(self, DTYPE_t delta_time):
        
        cdef DTYPE_t time = self.Time
        cdef int j, k
        
        # Prepare the computation
        self._Temperature.prepare(self.dz)
        self._Vorticity.prepare(self.dz)
        self._Stream.prepare(self.dz)
        
        # Compute the derivatives
        self._Temperature.compute(self._Stream.V_curr, self._Stream.dVdz, self.dz, self.a, self.npa)
        self._Vorticity.compute(self._Temperature.V_curr, self._Stream.V_curr, self._Stream.dVdz, self.dz, self.a, self.npa, self.Pr, self.Ra)
        
        # Advance the derivatives
        self._Temperature.advance(delta_time)
        self._Vorticity.advance(delta_time)
        
        # Advance the stream function
        self._Stream.solve(self._Vorticity.V_curr, self._Stream.V_curr)
        
        
        self.Time = time + delta_time
        
        return 0
    
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
    