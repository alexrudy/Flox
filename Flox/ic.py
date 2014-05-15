# -*- coding: utf-8 -*-
# 
#  ic.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-02.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

"Handle initial conditions."

from __future__ import (absolute_import, unicode_literals, division, print_function)

import numpy as np

def z_array(System):
    """Z position array, appropriate for initialzing."""
    return (np.arange(System.nz + 2) / (System.nz + 1))[1:-1]

def stable_temperature_gradient(System):
    """Apply the stable temperature gradient to the system."""
    if System.deltaT > 0.0:
        step = -1
    else:
        step = 1
    System.Temperature[:,0] = z_array(System)[::-1]
    

def single_mode_linear_perturbation(System, mode=1, eps=1):
    """Make a standard mode for this perturbation."""
    System.Temperature[:,mode] = eps * np.sin(np.pi * z_array(System))

def standard_linear_perturbation(System):
    """Apply the standard linear perturbation from Ch. 3"""
    System.Temperature[:,1:] = np.sin(np.pi * z_array(System))[:,np.newaxis]
    

def standard_nonlinear_perturbation(System, eps=1e-2):
    """Apply the standard nonlinear perturbation from Ch. 4"""
    import numpy.random
    System.Temperature[:,1:-1] = eps * 2 * (np.random.rand(System.nx - 2)[np.newaxis,:] - 0.5) * np.sin(np.pi * z_array(System))[:,np.newaxis]