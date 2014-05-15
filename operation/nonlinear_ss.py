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
from Flox.nonlinear.plot import setup_plots, setup_plots_watch

import matplotlib.pyplot as plt
import matplotlib

def filename(extension=".yml", base=None):
    """Filenames related to this file!"""
    directory = os.path.dirname(__file__)
    base = os.path.splitext(os.path.basename(__file__))[0] if base is None else base
    return os.path.join(directory, base + extension)
    


if __name__ == '__main__':
    ipydb()
    plt.ion()
    
    Config = FloxConfiguration.fromfile(filename(".yml", "nonlinear_op"))
    System = NDSystem2D.from_params(Config["system"])
    LSystem = NDSystem2D.from_params(Config["system"])
    iterations = int(Config["iterations"])
    chunks = System.nt - System.it - 1
    stable_temperature_gradient(System)
    LSystem.read_packet(System.create_packet())
    
    
    print("INITIAL")
    print(System)
    print(System.diagnostic_string())
    LE = LinearEvolver.from_system(LSystem)
    LE.read_packet(LSystem.create_packet())
    NLE = NonlinearEvolver.from_system(System)
    NLE.read_packet(System.create_packet())
    matplotlib.rcParams['text.usetex'] = False
    MVC = setup_plots_watch(plt.figure(figsize=(10, 10)), stability=1, zmode=System.nz-1)
    MVC.update(System)
    print(NLE.delta_time())
    # NLE.evolve_system(System, Config["time"], chunks=80, chunksize=1)
    for i in range(200):
        NLE.step(NLE.delta_time())
        LE.step(NLE.delta_time())
        LSystem.read_packet(LE.create_packet())
        System.read_packet(NLE.create_packet())
        MVC.update(System)
        print("Step {:d}".format(i))
        print(LSystem)
        print(LSystem.diagnostic_string())
        print(System)
        print(System.diagnostic_string())
        askip()()
    