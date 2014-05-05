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
    
    cpdef int compute(self, DTYPE_t[:,:] P_curr, DTYPE_t[:,:] dPdz, DTYPE_t dz, DTYPE_t a, DTYPE_t[:] npa):
        
        cdef int r, j, k, kp, kpp
        cdef DTYPE_t p2a = pi / (2.0 * a)
        # This equation handles the linear terms. It is only slightly modified from the linear version.
        r = temperature(self.nz, self.nx, self.G_curr, self.V_curr, P_curr, dz, npa, self.V_p, self.V_m)
        # Now we do the non-linear terms from equation 4.6
        for j in range(self.nz):
            for k in range(self.nx):
                # n=0 special case.
                self.G_curr[j,0] += -p2a * k * self.V_curr[j, k] * dPdz[j, k] + P_curr[j, k] * self.dVdz[j, k]
                
                # Terms applied everywhere.
                self.G_curr[j,k] += -npa[k] * P_curr[j, k] * self.dVdz[j, 0]
                for kp in range(1, self.nx):
                    # 2nd term, 2nd Delta
                    kpp = kp - k
                    if 0 < kpp < self.nx:
                        self.G_curr[j, k] += -p2a * (kp * dPdz[j,kpp] * self.V_curr[j, kp] + kpp * P_curr[j, kpp] * self.dVdz[j, kp])
                        
                    # 2nd term, 1st Delta
                    kpp = kp + k
                    if 0 < kpp < self.nx:
                        self.G_curr[j, k] += -p2a * (kp * dPdz[j,kpp] * self.V_curr[j, kp] + kpp * P_curr[j, kpp] * self.dVdz[j, kp])
                    
                    # 1st term, 1st Delta
                    kpp = k - kp
                    if 0 < kpp < self.nx:
                        self.G_curr[j, k] += p2a * (kp * dPdz[j, kpp] * self.V_curr[j, kp] + kpp * P_curr[j, kpp] * self.dVdz[j, kp])
        
        return r


cdef class VorticitySolver(Solver):
    
    cpdef int compute(self, DTYPE_t[:,:] T_curr, DTYPE_t[:,:] P_curr, DTYPE_t[:,:] dPdz, DTYPE_t dz, DTYPE_t a, DTYPE_t[:] npa, DTYPE_t Pr, DTYPE_t Ra):
        
        cdef int r, j, k, kp, kpp
        cdef DTYPE_t p2a = pi / (2.0 * a)
        
        r = vorticity(self.nz, self.nx, self.G_curr, self.V_curr, T_curr, dz, npa, Pr, Ra, self.V_p, self.V_m)
        
        for j in range(self.nz):
            for k in range(self.nx):
                for kp in range(1, self.nx):
                    # 2nd term, 2nd delta
                    kpp = kp - k
                    if 0 < kpp < self.nx:
                        self.G_curr[j, k] += -p2a * (kp * dPdz[j, kpp] * self.V_curr[j, kp] + kpp * P_curr[j, kpp] * self.dVdz[j, kp])
                    # 2nd term, 1st delta
                    kpp = kp + k
                    if 0 < kpp < self.nx:
                        self.G_curr[j, k] += p2a * (kp * dPdz[j, kpp] * self.V_curr[j, kp] + kpp * P_curr[j, kpp] * self.dVdz[j, kp])
                    
                    # 1st term, 1st delta
                    kpp = k - kp
                    if 0 < kpp < self.nx:
                        self.G_curr[j, k] += p2a * (-kp * dPdz[j, kpp] * self.V_curr[j, kp] + kpp * P_curr[j, kpp] * self.dVdz[j, kp])
        
        return r
    
    
cdef class NonlinearEvolver(Evolver):
    
    def __cinit__(self, int nz, int nx, DTYPE_t[:] npa, DTYPE_t Pr, DTYPE_t Ra, DTYPE_t dz, DTYPE_t a, DTYPE_t time, DTYPE_t safety):
        
        self.time = time
        self.npa = npa
        self.Pr = Pr
        self.Ra = Ra
        self.dz = dz
        self.a = a
        self.Temperature = TemperatureSolver(nz, nx)
        self.Vorticity = VorticitySolver(nz, nx)
        self.Stream = StreamSolver(nz, nx)
        self.Stream.setup(self.dz, self.npa)
        self.safety = safety
        
    
    cpdef int get_state(self, DTYPE_t[:,:] Temperature, DTYPE_t[:,:] Vorticity, DTYPE_t[:,:] Stream):
        
        Temperature[...] = self.Temperature.V_curr
        Vorticity[...] = self.Vorticity.V_curr
        Stream[...] =  self.Stream.V_curr
        return 0
    
    cpdef int set_state(self, DTYPE_t[:,:] Temperature, DTYPE_t[:,:] Vorticity, DTYPE_t[:,:] Stream, DTYPE_t time):
        
        self.Temperature.V_curr = Temperature
        self.Vorticity.V_curr = Vorticity
        self.Stream.V_curr = Stream
        self.time = time
        return 0
        
    
    cpdef DTYPE_t delta_time(self):
        
        return 3.6e-6
        # return (self.dz * self.dz) / 4.0 * self.safety
        
    cpdef int step(self, DTYPE_t delta_time):
        
        cdef DTYPE_t time = self.time
        # Prepare the computation
        self.Temperature.prepare(self.dz)
        self.Vorticity.prepare(self.dz)
        self.Stream.prepare(self.dz)
        
        # Compute the derivatives
        self.Temperature.compute(self.Stream.V_curr, self.Stream.dVdz, self.dz, self.a, self.npa)
        self.Vorticity.compute(self.Temperature.V_curr, self.Stream.V_curr, self.Stream.dVdz, self.dz, self.a, self.npa, self.Pr, self.Ra)
        
        # Advance the derivatives
        self.Temperature.advance(delta_time)
        self.Vorticity.advance(delta_time)
        
        # Advance the stream function
        self.Stream.solve(self.Vorticity.V_curr, self.Stream.V_curr)
        
        self.time = time + delta_time
        
        return 0
    
    
    