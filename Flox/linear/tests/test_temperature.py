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
from .system import PolynomialSystem, FourierSystem

def get_system():
    """Get the appropriate system for use with this test."""
    return FourierSystem(
        dz = 0.5,
        dt = 1.0,
        a = 1.0,
        nx = 2,
        nz = 6,
        Ra = 10.0,
        Pr = 5.0,
    )

def w_dtemperature(System):
    """Derivative of temperature solver."""
    from .._linear import temperature
    dT = np.zeros_like(System.Temperature)
    fp = np.zeros(System.nx, np.float)
    fm = np.zeros(System.nx, np.float)
    rv = temperature(System.nz, System.nx, dT, System.Temperature, System.Stream, System.dz, System.npa[0,:], fp, fm)
    return dT

def w_temperature(System):
    """TemperatureSolver Wrapper."""
    from .._linear import TemperatureSolver
    Tn = np.zeros_like(System.Temperature)
    dTn = np.zeros_like(System.Temperature)
    dTo = np.zeros_like(System.Temperature)
    TS = TemperatureSolver(System.nz, System.nx)
    TS.Value = System.Temperature
    TS.compute(System.Stream, System.dz, System.npa[0,:])
    TS.advance(System.dt)
    return TS.Value, TS.dValuedt
    
def test_temperature_derivative():
    """Derivative of temperature."""
    System = get_system()
    dTc = w_dtemperature(System)
    dTa = System.d_Temperature_simple
    
    dTr = dTa / dTc
    # This test cuts out the boundary points, and doesn't care about them.
    assert np.allclose(dTa, dTc)

def test_temperature_solver():
    """Solver for temperature."""
    System = get_system()
    Tnc, dTnc = w_temperature(System)
    assert np.isfinite(Tnc).all()
    assert np.isfinite(dTnc).all()
    assert np.allclose(System.evolved("Temperature")[1:-1,:], Tnc[1:-1,:])
    assert np.allclose(System.d_Temperature[1:-1,:], dTnc[1:-1,:])


    