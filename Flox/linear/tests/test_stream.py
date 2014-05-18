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
from .system import _second_derivative

def test_stream_solver(system):
    """Solver for the stream function."""
    stream = stream_solver(system)
    vorticity = vorticity_from_stream(system, stream)
    assert np.allclose(vorticity, system.Vorticity)
    
def test_stream_matrix(system):
    """Matrix used for the stream solver function."""
    from .._linear import StreamSolver
    SS = StreamSolver(system.nz, system.nx)
    SS.setup(system.dz, system.npa[0,:])
    
    sup, dia, sub = stream_matrix(system)
    
    assert sup.shape == SS.sup.shape
    
    assert np.allclose(sup, SS.sup)
    
    assert np.allclose(sub, SS.sub)
    
    assert np.allclose(dia, SS.dia)
    
    
def stream_matrix(system):
    """Extract the matrix components for Stream."""
    # Matrix contents
    sup = np.empty((system.nz + 2, system.nx))
    sup[...] = -1.0 / (system.dz)**2.0
    
    dia = np.empty((system.nz + 2, system.nx))
    dia[1:-1] = system.npa ** 2.0 + 2.0 / (system.dz) ** 2.0
    
    sub = np.empty((system.nz + 2, system.nx))
    sub[...] = -1.0 / (system.dz)**2.0
    
    # Boundary Conditions
    sub[0] = 0.0
    sup[0] = 0.0
    dia[0] = 1.0
    dia[-1] = 1.0
    sub[-1] = 0.0
    sup[-1] = 0.0
    
    return sup, dia, sub
    
def stream_solver(system):
    """Compute the stream function from the vorticity."""
    from .._linear import StreamSolver
    
    stream = np.zeros_like(system.Vorticity)
    
    SS = StreamSolver(system.nz, system.nx)
    SS.setup(system.dz, system.npa[0,:])
    SS.solve(system.Vorticity, stream)
    
    return stream
    
def stream_second_derivative(stream, dz):
    """Compute the second derivative of a stream function given our boundary conditions."""
    f_p = np.zeros(stream.shape[1])
    f_m = np.zeros(stream.shape[1])
    return _second_derivative(stream, f_p, f_m, dz)
    
def vorticity_from_stream(system, stream):
    """Compute the vorticity from a stream function."""
    stream_ddz = stream_second_derivative(stream, system.dz)
    return -stream_ddz + system.npa**2 * stream
    