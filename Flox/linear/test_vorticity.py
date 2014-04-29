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

def w_dvorticity(V, T, dz, npa, Ra, Pr):
    """Derivative of temperature solver."""
    from ._linear import vorticity
    nz = V.shape[0]
    nx = V.shape[1]
    dV = np.zeros_like(V)
    rv = vorticity(nz, nx, dV, V, T, dz, npa, Pr, Ra)
    return dV

def w_vorticity(V, T, dz, dt, npa, Ra, Pr):
    """TempeartureSolver Wrapper."""
    from ._linear import VorticitySolver
    nz = V.shape[0]
    nx = V.shape[1]
    Vn = np.zeros_like(V)
    dVn = np.zeros_like(V)
    dVo = np.zeros_like(V)
    VS = VorticitySolver(nz, nx, V)
    VS.compute(T, dz, npa, Pr, Ra)
    VS.advance(dt)
    VS.get_state(Vn, dVn, dVo)
    return Vn, dVn, dVo

def s_vorticity(nx, nz, dz, dt, a=1.0, Ra=1, Pr=1):
    """Setup for the vorticity equations."""
    z = np.tile(np.arange(-nz/2 * dz, nz/2 * dz, dz), (nx, 1)).T
    npa = np.tile(np.arange(1,nx+1).T, (nz, 1)).astype(np.float)
    T = z**3 + 2 * z**2
    V = z**3 - (2 * z**2)
    ddV = 6 * z - 4
    dV = (Ra * Pr * T * npa) - (Pr * npa * npa * V) + Pr * ddV
    Vn = dt/2.0 * 3.0 * dV + V
    return T, V, dV, Vn, npa[0,:]
    
def test_vorticity_derivative():
    """Derivative of vorticity."""
    dz = 0.1
    dt = 2.0
    a = 0.5
    nx, nz = 10, 10
    Ra, Pr = 5.0, 10.0
    T, V, dV, Vn, npa = s_vorticity(nx, nz, dz, dt, a, Ra, Pr)
    dVc = w_dvorticity(V, T, dz, npa, Ra, Pr)
    # This test cuts out the boundary points, and doesn't care about them.
    assert np.allclose(dV[1:-1,:], dVc[1:-1,:])

def test_vorticity_solver():
    """Solver for temperature."""
    dz = 0.1
    dt = 1.0
    a = 0.5
    nx, nz = 10, 10
    Ra, Pr = 2.0, 5.0
    T, V, dV, Vn, npa = s_vorticity(nx, nz, dz, dt, a, Ra, Pr)
    Vnc, dVc, dVo = w_vorticity(V, T, dz, dt, npa, Ra, Pr)
    assert np.isfinite(Vnc).all()
    assert np.isfinite(dVc).all()
    assert np.allclose(Vnc[1:-1,:], Vn[1:-1,:])
    assert np.allclose(dV[1:-1,:], dVc[1:-1,:])


    