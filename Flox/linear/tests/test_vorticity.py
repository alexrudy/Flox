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

def vorticity_linearterms(system):
    """Derivative of vorticity solver."""
    from ...component.vorticity import vorticity
    dV = np.zeros_like(system.Vorticity)
    f_m, f_p = system.b_Vorticity
    rv = vorticity(system.nz, system.nn, dV, system.Vorticity, system.Temperature, system.dz, system.npa[0,:], system.Pr, system.Ra, f_p, f_m)
    return dV

def w_vorticity(system):
    """TempeartureSolver Wrapper."""
    from ...component.vorticity import VorticitySolver
    VS = VorticitySolver(system.nz, system.nn)
    VS.Value = system.Vorticity
    VS.Value_m, VS.Value_p = system.b_Vorticity
    VS.compute_base(system.Temperature, system.dz, system.npa[0,:], system.Pr, system.Ra)
    VS.advance(system.dt)
    return VS.Value, VS.dValuedt
    
def test_vorticity_derivative(system):
    """Derivative of vorticity."""
    dVc = vorticity_linearterms(system)
    assert np.allclose(system.d_Vorticity, dVc)
    
def test_vorticity_solver(system):
    """Solver for vorticity."""
    Vnc, dVc = w_vorticity(system)
    assert np.isfinite(Vnc).all()
    assert np.isfinite(dVc).all()
    assert np.allclose(Vnc, system.evolved("Vorticity"))
    assert np.allclose(dVc, system.d_Vorticity)


    