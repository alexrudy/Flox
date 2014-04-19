# -*- coding: utf-8 -*-
# 
#  test_finitedifference.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-18.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import numpy as np
import nose.tools as nt

def w_second_derivative(f, dz, f_p, f_m):
    """Second derivative warpper."""
    from ..finitedifference import second_derivative
    ddf = np.zeros_like(f)
    J = f.shape[0]
    rval = second_derivative(J, ddf, f, dz, f_p, f_m)
    return ddf

def test_simple_secondderivative():
    """Simple second derivative"""
    n = 1000
    dz = 0.1
    z = np.arange(-n/2 * dz, n/2 * dz, dz)
    f = z**3 + 2 * z**2
    z_m = np.min(z) - dz
    z_p = np.max(z) + dz
    f_m = z_m**3 + 2 * z_m**2
    f_p = z_p**3 + 2 * z_p**2
    ddf_sol = 6 * z + 4
    ddf = w_second_derivative(f, dz, f_p, f_m)
    assert np.allclose(ddf_sol, ddf)
    
    
    