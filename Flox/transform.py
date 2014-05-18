# -*- coding: utf-8 -*-
# 
#  transform.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-01.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import numpy as np

def spectral_transform(func, data, nx=1.0, a=1.0, perturbed=False):
    """Do a function transform along the 1st index."""
    data = np.asanyarray(data)
    
    nz = data.shape[0]
    nm = data.shape[1]
    
    x = np.linspace(0, 1, nx+2)[1:-1] * a
    n = np.arange(nm)
    
    # This array is n * x large.
    # It will convert from modes to x-space.
    cs = func(n[:,np.newaxis] * np.pi / a * x[np.newaxis,:])
    
    result = np.zeros((nz, nx), dtype=data.dtype)
    for i_n in range(int(perturbed), data.shape[1]):
        result[:,:,...] += cs[np.newaxis,i_n,:] * data[:,i_n,np.newaxis,...]
    return result
    