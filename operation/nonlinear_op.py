#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
#  linear_op.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-22.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import os, os.path
import time

import numpy as np
from Flox.system import NDSystem2D
from Flox.input import FloxConfiguration
from Flox.nonlinear import NonlinearEvolver
from Flox.io import HDF5Writer
from Flox.ic import stable_temperature_gradient, standard_nonlinear_perturbation, single_mode_linear_perturbation
from pyshell.util import ipydb, askip

def filename(extension=".yml"):
    """docstring for filename"""
    base = os.path.splitext(__file__)[0]
    return base + extension


if __name__ == '__main__':
    
    Config = FloxConfiguration.fromfile(filename(".yml"))
    System = NDSystem2D.from_params(Config["system"])
    chunks = System.nt - System.it - 1
    stable_temperature_gradient(System)
    single_mode_linear_perturbation(System, 1)
    print(System)
    print(System.diagnostic_string())
    NLE = NonlinearEvolver.from_system(System)
    NLE.evolve_system(System, Config['time'], int(Config["iterations"]), chunks)
    print("")
    print(System)
    print(System.diagnostic_string())
    Writer = HDF5Writer(filename(".hdf5"))
    Writer.write(System,'main')
    
    