#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
#  manage_evolve.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-10.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

from Flox.system import NDSystem2D
from Flox.input import FloxConfiguration
from Flox.linear import LinearEvolver
from Flox.io import HDF5Writer
from Flox.ic import stable_temperature_gradient, standard_linear_perturbation, single_mode_linear_perturbation

from Flox.linear.plot import setup_plots
from Flox.nonlinear.plot import setup_plots_watch

import os, os.path
import queue
from astropy.utils.console import ProgressBar
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
from Flox.process.evolver import EvolverProcessing


if __name__ == '__main__':
    
    mode = 1
    
    Config = FloxConfiguration.fromfile(os.path.join(os.path.dirname(__file__),"linear_op.yml"))
    System = NDSystem2D.from_params(Config["system"])
    System.Rayleigh = 100
    # stable_temperature_gradient(System)
    # single_mode_linear_perturbation(System, mode)
    
    matplotlib.rcParams['text.usetex'] = False
    MVC = setup_plots_watch(plt.figure(figsize=(10, 10)), stability=mode)
    MVC.update(System)
    
    EM = EvolverProcessing(timeout=1, buffer_length=0)
    EM.register_evolver(LinearEvolver)
    with EM:
        EM.animate_evolve(LinearEvolver, System, MVC, Config['time'], chunks=System.nt - 1, chunksize=100)
        print(System)
        print(System.diagnostic_string())
    print("Done!")
        