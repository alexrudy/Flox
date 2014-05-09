# -*- coding: utf-8 -*-
# 
#  test_stream.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-05.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 


from __future__ import (absolute_import, unicode_literals, division, print_function)

import numpy as np
import nose.tools as nt
from .system import PolynomialSystem, ConstantSystem
from ...tests.test_finitedifference import w_2d_second_derivative

def test_stream_solver():
    """Solver for the stream function."""
    System = ConstantSystem(
        dz = 0.8,
        dt = 0.4,
        a = 0.25,
        nx = 3,
        nz = 6,
        Ra = 10.0,
        Pr = 5.0,
    )
    stream = w_stream(System)
    vorticity = w_vorticity(System, stream)
    s_vorticity = System.Vorticity
    assert np.allclose(vorticity, s_vorticity)
    
def dd_stream(System, stream):
    """Compute the second derivative of the stream function."""
    return w_2d_second_derivative(stream, System.dz, 0.0, 0.0)
    
def w_stream(System):
    """Compute the stream function from the vorticity."""
    from .._linear import StreamSolver
    
    stream = np.zeros_like(System.Vorticity)
    
    print(System.npa[0,:])
    SS = StreamSolver(System.nz, System.nx)
    SS.setup(System.dz, System.npa[0,:])
    SS.solve(System.Vorticity, stream)
    
    return stream
    
def w_vorticity(System, stream):
    """Compute the vorticity from a stream function."""
    _dd_stream = dd_stream(System, stream)
    return - _dd_stream + System.npa**2 * stream
    