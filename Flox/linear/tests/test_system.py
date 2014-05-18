# -*- coding: utf-8 -*-
# 
#  test_system.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-09.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import numpy as np

from .system import PolynomialSystem, FourierSystem
import pytest



@pytest.fixture(params = [
        PolynomialSystem(
                dz = 1.0,
                dt = 1.0,
                a = 1.0,
                nx = 3,
                nz = 4,
                Ra = 1.0,
                Pr = 1.0,
            ),
        FourierSystem(
                dz = 1.0,
                dt = 1.0,
                a = 1.0,
                nx = 3,
                nz = 4,
                Ra = 1.0,
                Pr = 1.0,
            ),
        PolynomialSystem(
                dz = 0.1,
                dt = 2.0,
                a = 0.5,
                nx = 10,
                nz = 8,
                Ra = 5.0,
                Pr = 10.0,
            ),
        FourierSystem(
                dz = 0.1,
                dt = 2.0,
                a = 0.5,
                nx = 10,
                nz = 8,
                Ra = 5.0,
                Pr = 10.0,
            )
    ])
def system(request):
    """Generate individual systems."""
    return request.param
    
@pytest.fixture(params=["Temperature", "Vorticity", "Stream"])
def fluid_componet(request):
    """Parameterized across the fluid components."""
    return request.param

def test_system_shapes(system):
    """System array shapes"""
    for prop in ["z", "Temperature", "Vorticity", "Stream"]:
        assert getattr(system, prop).shape == (system.nz, system.nx)

def test_system_z(system):
    """System z array steps"""
    assert np.allclose(np.diff(system.z, axis=0), system.dz)
    
def test_system_derivative(system, fluid_componet):
    """System derivative using gradients."""
    component = getattr(system, fluid_componet)
    dd_componet = getattr(system, "dd_"+fluid_componet)
    dCdz = np.gradient(component)[0] # Take only the z-component
    dCdzz = np.gradient(dCdz)[0] # Take only the z-component again!
    
    assert np.allclose(dCdzz, dd_componet)
    