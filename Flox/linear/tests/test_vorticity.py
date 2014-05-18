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
from .system import PolynomialSystem, FourierSystem

def get_system():
    """Get the required system for this test."""
    return FourierSystem(
        dz = 0.8,
        dt = 0.4,
        a = 0.25,
        nx = 2,
        nz = 6,
        Ra = 10.0,
        Pr = 5.0,
    )

def w_dvorticity(System):
    """Derivative of vorticity solver."""
    from .._linear import vorticity
    dV = np.zeros_like(System.Vorticity)
    fp = np.zeros(System.nx, np.float)
    fm = np.zeros(System.nx, np.float)
    rv = vorticity(System.nz, System.nx, dV, System.Vorticity, System.Temperature, System.dz, System.npa[0,:], System.Pr, System.Ra, fp, fm)
    return dV

def w_vorticity(System):
    """TempeartureSolver Wrapper."""
    from .._linear import VorticitySolver
    Vn = np.zeros_like(System.Vorticity)
    dVn = np.zeros_like(System.Vorticity)
    dVo = np.zeros_like(System.Vorticity)
    VS = VorticitySolver(System.nz, System.nx)
    VS.Value = System.Vorticity
    VS.compute(System.Temperature, System.dz, System.npa[0,:], System.Pr, System.Ra)
    VS.advance(System.dt)
    return VS.Value, VS.dValuedt
    
def test_vorticity_derivative():
    """Derivative of vorticity."""
    System = get_system()
    dVc = w_dvorticity(System)
    # This test cuts out the boundary points, and doesn't care about them.
    assert np.allclose(System.d_Vorticity[1:-1,:], dVc[1:-1,:])
    
def test_vorticity_solver():
    """Solver for vorticity."""
    System = get_system()
    Vnc, dVc = w_vorticity(System)
    assert np.isfinite(Vnc).all()
    assert np.isfinite(dVc).all()
    assert np.allclose(Vnc[1:-1,:], System.evolved("Vorticity")[1:-1,:])
    assert np.allclose(dVc[1:-1,:], System.d_Vorticity[1:-1,:])


    