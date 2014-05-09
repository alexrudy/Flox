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

from .system import PolynomialSystem

def test_polynomial_system_shapes():
    """System array shapes"""
    System = PolynomialSystem(
        dz = 0.1,
        dt = 2.0,
        a = 0.5,
        nx = 10,
        nz = 8,
        Ra = 5.0,
        Pr = 10.0,
    )
    
    for prop in ["z", "Temperature", "Vorticity", "Stream"]:
        assert getattr(System, prop).shape == (System.nz, System.nx)


def test_system_z():
    """System z array steps"""
    System = PolynomialSystem(
        dz = 0.1,
        dt = 2.0,
        a = 0.5,
        nx = 10,
        nz = 8,
        Ra = 5.0,
        Pr = 10.0,
    )
    
    assert np.allclose(np.diff(System.z, axis=0), System.dz) 