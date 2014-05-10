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
from .system import PolynomialSystem

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
    System = PolynomialSystem(
        dz = 0.8,
        dt = 0.4,
        a = 0.25,
        nx = 2,
        nz = 6,
        Ra = 10.0,
        Pr = 5.0,
    )
    dTc = w_dtemperature(System)
    # This test cuts out the boundary points, and doesn't care about them.
    assert np.allclose(System.d_Temperature_simple[1:-1,:], dTc[1:-1,:])

def test_temperature_solver():
    """Solver for temperature."""
    System = PolynomialSystem(
        dz = 0.8,
        dt = 0.4,
        a = 0.25,
        nx = 2,
        nz = 6,
        Ra = 10.0,
        Pr = 5.0,
    )
    Tnc, dTnc = w_temperature(System)
    assert np.isfinite(Tnc).all()
    assert np.isfinite(dTnc).all()
    assert np.allclose(System.evolved("Temperature")[1:-1,:], Tnc[1:-1,:])
    assert np.allclose(System.d_Temperature[1:-1,:], dTnc[1:-1,:])


    