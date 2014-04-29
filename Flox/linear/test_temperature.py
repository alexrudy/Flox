# -*- coding: utf-8 -*-
# 
#  test_linear.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-24.
#  Copyright 2014 University of California. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import numpy as np
import nose.tools as nt

def w_dtemperature(T, dz, npa):
    """Derivative of temperature solver."""
    from ._linear import temperature
    nz = T.shape[0]
    nx = T.shape[1]
    dT = np.zeros_like(T)
    fp = np.zeros(nx, np.float)
    fm = np.zeros(nx, np.float)
    rv = temperature(nz, nx, dT, T, dz, npa, fp, fm)
    return dT

def w_temperature(T, dz, dt, npa):
    """TempeartureSolver Wrapper."""
    from ._linear import TemperatureSolver
    nz = T.shape[0]
    nx = T.shape[1]
    Tn = np.zeros_like(T)
    dTn = np.zeros_like(T)
    dTo = np.zeros_like(T)
    TS = TemperatureSolver(nz, nx, T)
    TS.compute(dz, npa)
    TS.advance(dt)
    TS.get_state(Tn, dTn, dTo)
    return Tn, dTn, dTo

def s_temperature(nx, nz, dz, dt, a=1.0):
    """Setup for the temperature equations."""
    z = np.tile(np.arange(-nz/2 * dz, nz/2 * dz, dz), (nx, 1)).T
    npa = np.tile(np.arange(1,nx+1).T, (nz, 1)).astype(np.float)
    T = z**3 + 2 * z**2
    ddT = 6 * z + 4
    dT = -T * npa * npa + ddT
    Tn = dt/2.0 * 3.0 * dT + T
    return T, dT, Tn, npa[0,:]
    
def test_temperature_derivative():
    """Derivative of temperature."""
    dz = 0.1
    dt = 1.0
    nx, nz = 5, 10
    T, dT, Tn, npa = s_temperature(nx, nz, dz, dt)
    dTc = w_dtemperature(T, dz, npa)
    # This test cuts out the boundary points, and doesn't care about them.
    assert np.allclose(dT[1:-1,:], dTc[1:-1,:])

def test_temperature_solver():
    """Solver for temperature."""
    dt = 1.0
    dz = 0.1
    nx, nz = 5, 10
    T, dT, Tn, npa = s_temperature(nx, nz, dz, dt)
    Tnc, dTnc, dToc = w_temperature(T, dz, dt, npa)
    assert np.isfinite(Tn).all()
    assert np.isfinite(dTnc).all()
    assert np.allclose(Tn[1:-1,:], Tnc[1:-1,:])
    assert np.allclose(dT[1:-1,:], dTnc[1:-1,:])


    