#!/usr/bin/env /opt/local/Library/Frameworks/Python.framework/Versions/3.3/bin/python3.3
# -*- coding: utf-8 -*-
# 
#  manage_evolve.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-10.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import os.path, os
if "VIRTUAL_ENV" in os.environ:
    activate_this = os.path.join(os.environ["VIRTUAL_ENV"],'bin/activate_this.py')
    exec(open(activate_this).read(), dict(__file__=activate_this))

from Flox.system import NDSystem2D
from Flox.input import FloxConfiguration
from Flox.nonlinear import NonlinearEvolver
from Flox.io import HDF5Writer
from Flox.ic import stable_temperature_gradient, standard_linear_perturbation, single_mode_linear_perturbation

from Flox.nonlinear.plot import setup_plots, setup_plots_watch, setup_movie

import os, os.path
import queue
from astropy.utils.console import ProgressBar
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
from Flox.process.evolver import EvolverProcessing

from pyshell.util import ipydb, askip



if __name__ == '__main__':
    
    mode = 1
    
    Config = FloxConfiguration.fromfile(os.path.join(os.path.dirname(__file__),"linear_op.yml"))
    System = NDSystem2D.from_params(Config["system"])
    System.Rayleigh = 1e6
    System.Prandtl = 0.5
    System.nz = 100
    system.nn = 50
    System.aspect = 3
    System.nt = 500
    System.initialize_arrays()
    stable_temperature_gradient(System)
    for m in range(1, system.nn//10):
        single_mode_linear_perturbation(System, m, eps=5e-3)
    # single_mode_linear_perturbation(System, 1, eps=5e-1)
    matplotlib.rcParams['text.usetex'] = False
    MVC = setup_movie(plt.figure(figsize=(10, 10)))
    MVC.update(System)
    
    EM = EvolverProcessing(buffer_length=0, timeout=300)
    EM.register_evolver(NonlinearEvolver)
    with EM:
        # EM.evolve(NonlinearEvolver, System, Config['time'], chunks=System.nt - 1, chunksize=100)
        EM.animate_evolve(NonlinearEvolver, System, MVC, Config['time'], chunks=System.nt - 1, chunksize=50)
        print(System)
        print(System.diagnostic_string())
    Writer = HDF5Writer(os.path.expanduser(os.path.join("~/Documents/","nonlinear.hdf5")))
    Writer.write(System, 'nonlinear')
    print("Done!")
        