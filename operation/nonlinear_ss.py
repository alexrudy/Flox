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
from Flox.linear import LinearEvolver
from Flox.io import HDF5Writer
from Flox.ic import stable_temperature_gradient, standard_nonlinear_perturbation, single_mode_linear_perturbation
from pyshell.util import ipydb, askip

def filename(extension=".yml", base=None):
    """Filenames related to this file!"""
    directory = os.path.dirname(__file__)
    base = os.path.splitext(os.path.basename(__file__))[0] if base is None else base
    return os.path.join(directory, base + extension)
    


if __name__ == '__main__':
    ipydb()
    
    Config = FloxConfiguration.fromfile(filename(".yml", "nonlinear_op"))
    System = NDSystem2D.from_params(Config["system"])
    iterations = int(Config["iterations"])
    chunks = System.nt - System.it - 1
    Writer = HDF5Writer(filename(".hdf5"))
    # System.Rayleigh = critical_raleigh_number(System, 1) * 0.8
    stable_temperature_gradient(System)
    # single_mode_linear_perturbation(System, 49)
    # single_mode_linear_perturbation(System, 48)
    single_mode_linear_perturbation(System, 1)
    # single_mode_linear_perturbation(System, 2)
    
    print("INITIAL")
    print(System)
    print(System.diagnostic_string())
    NLE = NonlinearEvolver.from_system(System)
    LE = LinearEvolver.from_system(System)
    # LE.update_from_system(System)
    NLE.update_from_system(System)
    NLE.evolve_system(System, Config['time'], iterations, chunks)
    # print("NONLINEAR")
    # print(System)
    # print(System.diagnostic_string())
    # LE.evolve_system(System, Config['time'], iterations, chunks)
    print("LINEAR")
    print(System)
    for it in range(System.nit+1):
        System.it = it
        print(System.diagnostic_string())
    # Writer.write(System,'main')