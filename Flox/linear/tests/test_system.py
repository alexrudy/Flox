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

from .system import second_derivative

def test_system_shapes(system):
    """System array shapes"""
    for prop in ["z", "Temperature", "Vorticity", "Stream"]:
        assert getattr(system, prop).shape == (system.nz, system.nn)

def test_system_z(system):
    """System z array steps"""
    assert np.allclose(np.diff(system.z, axis=0), system.dz)
    
def test_system_derivative(system, fluid_componet):
    """System derivative using gradients."""
    dd_componet = getattr(system, "dd_"+fluid_componet)
    dd_finite = second_derivative(system, fluid_componet)
    
    
    assert np.allclose(dd_finite, dd_componet)
    