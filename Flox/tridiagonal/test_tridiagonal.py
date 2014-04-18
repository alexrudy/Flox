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
        
        matrix = np.diag(random.randn(n))
        
        # Handle the off-diagonal terms.
        matrix[1,0] = eps * random.randn(1)
        for i in range(1,n-1):
            matrix[i-1,i] = eps * random.randn(1)
            matrix[i+1,i] = eps * random.randn(1)
        matrix[n-2,n-1] = eps * random.randn(1)
        
    return np.matrix(matrix)
    
def assemble_solution_matrix(n, seed=5):
    """Assemble a solution matrix"""
    
    with NumpyRNGContext(seed):
        from numpy import random
        
        sol = random.randn(n)
        
    return np.matrix(sol).T
    
def test_tridiagonal_solve():
    """Test a tridiagonal solver"""
    from . import tridiagonal_from_matrix
    seed = 5
    n = int(1e4)
    eps = 1e-7
    
    mat = assemble_tridiagonal_matrix(n, seed=seed)
    sol = assemble_solution_matrix(n, seed=seed)
    
    rhs = np.array(mat * sol)[:,0]
    solar = np.array(sol)[:,0]
    res = np.array(np.zeros_like(sol))[:,0]
    
    status = tridiagonal_from_matrix(rhs, res, mat)
    assert np.allclose(res, solar)
    

    
    