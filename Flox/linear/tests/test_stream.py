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
from .system import PolynomialSystem, ConstantSystem, FourierSystem
from ...tests.test_finitedifference import w_2d_second_derivative

def get_system():
    """Get the currently in use system, with appropriate parameters."""
    return FourierSystem(
        dz = 0.8,
        dt = 0.4,
        a = 0.25,
        nx = 3,
        nz = 6,
        Ra = 10.0,
        Pr = 5.0,
    )

def test_stream_solver():
    """Solver for the stream function."""
    System = get_system()
    stream = w_stream(System)
    vorticity = w_vorticity(System, stream)
    s_vorticity = System.Vorticity
    assert np.allclose(vorticity[1:-1], s_vorticity[1:-1])
    
def test_stream_matrix():
    """Matrix used for the stream solver function."""
    from .._linear import StreamSolver
    System = get_system()
    SS = StreamSolver(System.nz, System.nx)
    SS.setup(System.dz, System.npa[0,:])
    
    sup, dia, sub = m_stream(System)
    
    assert sup.shape == SS.sup.shape
    
    assert np.allclose(sup, SS.sup)
    
    assert np.allclose(sub, SS.sub)
    
    assert np.allclose(dia, SS.dia)
    
    
def m_stream(System):
    """Extract the matrix components for Stream."""
    sup = np.empty((System.nz + 2, System.nx))
    sup[...] = -1.0 / (System.dz)**2.0
    dia = np.empty((System.nz + 2, System.nx))
    dia[1:-1] = System.npa ** 2.0 + 2.0 / (System.dz) ** 2.0
    sub = np.empty((System.nz + 2, System.nx))
    sub[...] = -1.0 / (System.dz)**2.0
    
    sub[0] = 0.0
    sup[0] = 0.0
    dia[0] = 1.0
    dia[-1] = 1.0
    sub[-1] = 0.0
    sup[-1] = 0.0
    
    return sup, dia, sub
    
    
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
    