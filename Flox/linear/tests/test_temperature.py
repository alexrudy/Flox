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

def temperature_linearterms(system):
    """Operate the derivative of temperature linear term computer."""
    from ...component.temperature import temperature
    dT = np.zeros_like(system.Temperature)
    f_m, f_p = system.b_Temperature
    assert not temperature(system.nz, system.nx, dT, system.Temperature, system.dz, system.npa[0,:], f_p, f_m)
    return dT

def temperature_linearsolver(system):
    """Solve the linear temperature terms, and return the current value and the time derivative."""
    from ...component.temperature import TemperatureSolver
    Tn = np.zeros_like(system.Temperature)
    dTn = np.zeros_like(system.Temperature)
    dTo = np.zeros_like(system.Temperature)
    TS = TemperatureSolver(system.nz, system.nx)
    TS.Value = system.Temperature
    TS.Value_m, TS.Value_p = system.b_Temperature
    TS.compute_base(system.dz, system.npa[0,:])
    TS.compute_linear(system.dz, system.npa[0,:], system.Stream)
    TS.advance(system.dt)
    return TS.Value, TS.dValuedt
    
def test_temperature_linearterms(system):
    """Derivative of temperature."""
    dTc = temperature_linearterms(system)
    dTa = system.d_Temperature_simple
    assert np.allclose(dTa, dTc)

def test_temperature_linearsolver(system):
    """Solver for temperature."""
    Tnc, dTnc = temperature_linearsolver(system)
    assert np.isfinite(Tnc).all()
    assert np.isfinite(dTnc).all()
    assert np.allclose(system.evolved("Temperature"), Tnc)
    assert np.allclose(system.d_Temperature, dTnc)


    