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

def stable_temperature_gradient(System):
    """Apply the stable temperature gradient to the system."""
    if System.deltaT > 0.0:
        step = -1
    else:
        step = 1
    System.Temperature[:,0,System.it] = np.arange(System.nz)[::step] / System.nz
    

def single_mode_linear_perturbation(System, mode=1):
    """Make a standard mode for this perturbation."""
    System.Temperature[:,mode,System.it] = np.sin(np.pi * np.arange(System.nz)/System.nz)

def standard_linear_perturbation(System):
    """Apply the standard linear perturbation from Ch. 3"""
    System.Temperature[:,1:,System.it] = np.sin(np.pi * np.arange(System.nz)/System.nz)[:,np.newaxis]
    

