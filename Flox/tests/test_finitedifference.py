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
import numpy.random

def _check_2D_bounds(K, *args):
    """Check 2-dimesnional bounds."""
    r = []
    for f in args:
        if np.asarray(f).ndim < 1:
            r.append(np.array([f]*K))
        else:
            r.append(f)
    return tuple(r)

def second_derivative(f, dz, f_p, f_m, factor=1.0):
    """Second derivative warpper."""
    from ..finitedifference import second_derivative as _second_derivative
    ddf = np.zeros_like(f)
    J = f.shape[0]
    assert not _second_derivative(J, ddf, f, dz, f_p, f_m, factor)
    return ddf
    
def second_derivative_2D(f, dz, f_p, f_m, factor=1.0):
    """Second derivative warpper."""
    from ..finitedifference import second_derivative2D as _second_derivative2D
    ddf = np.zeros_like(f)
    J = f.shape[0]
    K = f.shape[1]
    
    f_p, f_m = _check_2D_bounds(K, f_p, f_m)
    
    assert not _second_derivative2D(J, K, ddf, f, dz, f_p, f_m, factor)
    return ddf
    
def second_derivative_2D_nb(f, dz, factor=1.0):
    """Second derivative warpper."""
    from ..finitedifference import second_derivative2D_nb as _second_derivative2D_nb
    ddf = np.zeros_like(f)
    J = f.shape[0]
    K = f.shape[1]
    
    assert not _second_derivative2D_nb(J, K, ddf, f, dz, factor)
    return ddf
    
def first_derivative_2D(f, dz, f_p, f_m, factor=1.0):
    """Second derivative warpper."""
    from ..finitedifference import first_derivative2D as _first_derivative2D
    df = np.zeros_like(f)
    J = f.shape[0]
    K = f.shape[1]
    
    f_p, f_m = _check_2D_bounds(K, f_p, f_m)
    
    assert not _first_derivative2D(J, K, df, f, dz, f_p, f_m, factor)
    return df
    
def first_derivative(f, dz, f_p, f_m, factor=1.0):
    """Second derivative warpper."""
    from ..finitedifference import first_derivative as _first_derivative
    ddf = np.zeros_like(f)
    J = f.shape[0]
    assert not _first_derivative(J, ddf, f, dz, f_p, f_m, factor)
    return ddf

def test_secondderivative(functional_form):
    """Second derivative"""
    functional_form.ndim = 1
    factor = np.random.randn(1)
    ddfx = second_derivative(functional_form.fx, functional_form.dx, functional_form.f_p, functional_form.f_m, factor) / factor
    assert np.allclose(functional_form.ddfx, ddfx)
    
def test_secondderivative_2D(functional_form):
    """2D second derivative"""
    functional_form.ndim = 2
    factor = np.random.randn(1)
    ddfx = second_derivative_2D(functional_form.fx, functional_form.dx, functional_form.f_p, functional_form.f_m, factor) / factor
    assert np.allclose(functional_form.ddfx, ddfx)
    
def test_secondderivative_2D_nb(functional_form):
    """2D second derivative"""
    functional_form.ndim = 2
    factor = np.random.randn(1)
    ddfx = second_derivative_2D_nb(functional_form.fx, functional_form.dx, factor) / factor
    assert np.allclose(functional_form.ddfx[1:-1,:], ddfx[1:-1,:])
    assert np.allclose(0.0, ddfx[[0,-1],:])
    
def test_firstderivative(functional_form):
    """Second derivative"""
    functional_form.ndim = 1
    factor = np.random.randn(1)
    dfx = first_derivative(functional_form.fx, functional_form.dx, functional_form.f_p, functional_form.f_m, factor) / factor
    assert np.allclose(functional_form.dfx, dfx)
    
def test_firstderivative_2D(functional_form):
    """2D second derivative"""
    functional_form.ndim = 2
    factor = np.random.randn(1)
    dfx = first_derivative_2D(functional_form.fx, functional_form.dx, functional_form.f_p, functional_form.f_m, factor) / factor
    assert np.allclose(functional_form.dfx, dfx)