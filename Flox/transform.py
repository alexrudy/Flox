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

def spectral_transform(func, data, x_width=1.0, a=1.0, perturbed=False):
    """Do a function transform along the 1st index."""
    data = np.asanyarray(data)
    
    x = np.arange(data.shape[1])/data.shape[1] * x_width
    n = np.arange(data.shape[1])
    
    # This array is n * x large.
    # It will convert from modes to x-space.
    cs = func(n[:,np.newaxis] * np.pi / a * x[np.newaxis,:])
    
    result = np.zeros_like(data)
    for i_n in range(int(perturbed), data.shape[1]):
        result[:,:,...] += cs[np.newaxis,i_n,:] * data[:,i_n,np.newaxis,...]
    return result
    