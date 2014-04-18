# -*- coding: utf-8 -*-
# 
#  __init__.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-17.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

from ._tridiagonal import tridiagonal_solver, tridiagonal_from_matrix, tridiagonal_split_matrix

def tridiagonal_combine_matrix(sub, dia, sup):
    """Combine a tridiagonal matrix."""
    return np.matrix(np.diag(sub, -1) + np.diag(dia, 0) + np.diag(sup, 1))

__all__ = ['tridiagonal_solver', 'tridiagonal_from_matrix', 'tridiagonal_split_matrix', 'tridiagonal_combine_matrix']