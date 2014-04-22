# -*- coding: utf-8 -*-
# 
#  test_tridiagonal.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-17.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 


from __future__ import (absolute_import, unicode_literals, division, print_function)

import nose.tools as nt
from nose.plugins.attrib import attr
import numpy as np
from astropy.utils.misc import NumpyRNGContext

@nt.nottest
def assemble_tridiagonal_matrix(n, eps=1e-2, seed=5):
    """Assemble a tridiagonal matrix."""
    with NumpyRNGContext(seed):
        from numpy import random
        
        dia = np.diag(random.randn(n))
        sup = eps * np.diag(random.randn(n-1),-1)
        sub = eps * np.diag(random.randn(n-1),1)
    return np.matrix(dia + sup + sub)
    
def assemble_solution_matrix(n, seed=5):
    """Assemble a solution matrix"""
    
    with NumpyRNGContext(seed):
        from numpy import random
        
        sol = random.randn(n)
        
    return np.matrix(sol).T
    
def test_tridiagonal_solve():
    """Tridiagonal solver basic"""
    from . import tridiagonal_from_matrix
    seed = 5
    n = int(1e2)
    eps = 1e-2
    
    mat = assemble_tridiagonal_matrix(n, eps, seed=seed)
    sol = assemble_solution_matrix(n, seed=seed)
    
    rhs = np.array(mat * sol)[:,0]
    solar = np.array(sol)[:,0]
    res = np.array(np.zeros_like(sol))[:,0]
    
    status = tridiagonal_from_matrix(rhs, res, mat)
    assert np.allclose(res, solar)
    

def test_tridiagonal_solve_linear():
    """Tridiagonal solver linear"""
    from . import tridiagonal_from_matrix
    seed = 5
    n = int(1e2)
    eps = 1e-2
    with NumpyRNGContext(seed):
        from numpy import random
        factor = random.randn()
    

    mat = assemble_tridiagonal_matrix(n, eps, seed=seed)
    sol = assemble_solution_matrix(n, seed=seed)
    
    rhs = np.array(mat * sol)[:,0]
    solar = np.array(sol)[:,0]
    res = np.array(np.zeros_like(sol))[:,0]

    status = tridiagonal_from_matrix(factor*rhs, res, factor*mat)
    assert np.allclose(res, solar)
    
def test_tridiagonal_object():
    """TridiagonalSolver"""
    from . import TridiagonalSolver
    seed = 5
    n = int(1e2)
    eps = 1e-2
    
    mat = assemble_tridiagonal_matrix(n, eps, seed=seed)
    sol = assemble_solution_matrix(n, seed=seed)
    
    rhs = np.array(mat * sol)[:,0]
    solar = np.array(sol)[:,0]
    res = np.array(np.zeros_like(sol))[:,0]
    
    TS = TridiagonalSolver(n)
    TS.matrix(mat)
    TS.solve(rhs, res)    
    assert np.allclose(res, solar)
    TS.solve(5 * rhs, res)
    assert np.allclose(res, 5 * solar)