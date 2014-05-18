# -*- coding: utf-8 -*-
# 
#  test_tridiagonal.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-17.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 


from __future__ import (absolute_import, unicode_literals, division, print_function)

import numpy as np
import numpy.random
import pytest

def assemble_tridiagonal_matrix(n, eps=1e-2, mag=1, func=np.ones):
    """Assemble a tridiagonal matrix."""
    # from numpy import random
    # dia = mag * random.randn(n)
    # _sup = eps * mag * random.randn(n-1)
    # _sub = eps * mag * random.randn(n-1)
    dia = mag * func(n)
    _sup = eps * mag * func(n-1)
    _sub = eps * mag * func(n-1)
    sup = np.zeros_like(dia)
    sup[:-1] = _sup
    sub = np.zeros_like(dia)
    sub[1:] = _sub
    dia += sup + sub
    mat = np.matrix(np.diag(dia) + np.diag(_sup,-1) + np.diag(_sub,1))
    return mat, sub, dia, sup
    
def assemble_solution_matrix(n, mag=1, func=np.ones):
    """Assemble a solution matrix"""
    sol = mag * func(n)
    return np.matrix(sol).T
    

@pytest.fixture(params=[
    (int(5), 1e-1, 1.0, lambda n : np.ones(n)),
    (int(5), 1e-1, 1.0, lambda n : np.arange(n) + 1),
    (int(5), 1e-1, 1.0, lambda n : np.arange(n)[::-1] + 1),
    (int(5), 1e-1, 1.0, lambda n : np.linspace(1,2,n)[::-1]),
    (int(5), 1e-1, 1.0, lambda n : np.hstack((np.linspace(1,3,n//2)[::-1],np.linspace(-1,2,n-n//2)))),
    (int(5), 0.0, 1.0, lambda n : np.random.randn(n) + 10),
    pytest.mark.xfail((int(3), 0.1, 1.0, lambda n : np.random.rand(n) + 10)),
    ])
def tridiagonal_group(request):
    """Create a tridiagonal group."""
    n, eps, mag, func = request.param
    mat, sub, dia, sup = assemble_tridiagonal_matrix(n, eps, mag, func)
    sol = assemble_solution_matrix(n, mag, func)
    rhs = np.array(mat * sol)[:,0]
    sol = np.array(sol)[:,0]
    return (mat, sub, dia, sup, sol, rhs)
    
@pytest.fixture
def boundary_tridiagonal_group(tridiagonal_group):
    """Set up a tridiagonal group with boundaries."""
    (_mat, sub, dia, sup, _sol, _rhs) = tridiagonal_group
    n = dia.shape[0]
    
    mat = np.matrix(np.zeros((n+2, n+2)))
    mat[1:-1,1:-1] = _mat
    mat[0,0] = 1.0
    mat[-1,-1] = 1.0
    sol = np.matrix(np.zeros((n+2))).T
    sol[1:-1] = _sol[:,np.newaxis]
    rhs = np.array(mat * sol)[:,0]
    sol = np.array(sol)
    
    return (mat, sub, dia, sup, sol, rhs)
    
def test_tridiagonal_solve(tridiagonal_group):
    """Tridiagonal solver basic"""
    from ._tridiagonal import tridiagonal_do_work, tridiagonal_from_work
    mat, sub, dia, sup, sol, rhs = tridiagonal_group
    wk1 = np.zeros_like(sol)
    wk2 = np.zeros_like(sol)
    res = np.zeros_like(sol)
    assert not tridiagonal_do_work(rhs.shape[0], sub, dia, sup, wk1, wk2)
    assert not tridiagonal_from_work(rhs.shape[0], rhs, res, wk1, wk2, sub)
    assert np.allclose(res, sol)
    
@pytest.mark.xfail
def test_tridiagonal_solve_linear(boundary_tridiagonal_group):
    """Tridiagonal solver linear"""
    from . import tridiagonal_from_matrix
    (mat, sub, dia, sup, sol, rhs) = boundary_tridiagonal_group
    factor = random.randn()
    
    assert not tridiagonal_from_matrix(n+2, factor*rhs, res, factor*mat)
    assert np.allclose(res, solar)
    
@pytest.mark.xfail
def test_tridiagonalsolver(boundary_tridiagonal_group):
    """TridiagonalSolver"""
    from . import TridiagonalSolver, tridiagonal_split_matrix
    (mat, sub, dia, sup, sol, rhs) = boundary_tridiagonal_group
    
    nz = dia.shape[0]
    nx = 10
    rhsar = np.zeros((nz, nx))
    matar = np.zeros((nz, nz, nx))
    solar = np.zeros((nz, nx))
    resar = np.zeros((nz, nx))
    
    assert mat.shape == (nz+2, nz+2)
    assert sol.shape == (nz+2, 1)
    assert np.array(mat * sol).shape == (nz+2, 1)
    
    rhsar[...] = np.array(mat * sol)[1:-1,0,np.newaxis] * (np.arange(nx)[np.newaxis,:] + 1.0)
    solar[...] = np.array(sol)[1:-1,0,np.newaxis] * (np.arange(nx)[np.newaxis,:] + 1.0)
    matar[...] = (np.arange(nx)[np.newaxis,:] + 1.0) * np.array(mat)[1:-1,1:-1,np.newaxis]
    
    # Test some basics before we start.
    assert matar.shape == (nz, nz, nx)
    assert rhsar.shape == (nz, nx)
    assert solar.shape == (nz, nx)
    
    TS = TridiagonalSolver(nz, nx)
     
    assert not TS.matrix(matar)
    assert not TS.solve(rhsar, resar)
    assert np.isfinite(rhsar).all()
    assert np.isfinite(resar).all()
    
    assert np.allclose(resar, solar)
    
    
    TS.solve(5 * rhsar, resar)
    assert np.allclose(resar, 5 * solar)