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

import os, os.path
import queue
from astropy.utils.console import ProgressBar
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
from Flox.process.evolver import EvolverManager


def recieve(i, System, MVC, Q, PBar=None):
    """Recieve data"""
    if PBar is not None:
        PBar.update(i)
    packet = Q.get(timeout=1)
    System.read_packet(packet)
    if System.it > 2:
        MVC.update(System)

if __name__ == '__main__':
    
    mode = 1
    
    Config = FloxConfiguration.fromfile(os.path.join(os.path.dirname(__file__),"linear_op.yml"))
    System = NDSystem2D.from_params(Config["system"])
    stable_temperature_gradient(System)
    single_mode_linear_perturbation(System, mode)
    
    EM = EvolverManager()
    with mp.Manager() as QM:
        Q = QM.Queue()
        EM.register_evolver(LinearEvolver)
        matplotlib.rcParams['text.usetex'] = False
        MVC = setup_plots(plt.figure(figsize=(10, 10)), stability=mode)
    
        EM.start()
        LE = EM.LinearEvolver(System)
        packet = System.create_packet()
        LE.read_packet(packet)
        time = System.nondimensionalize(Config["time"]).value
        LE.evolve_async(time, queue=Q)
        with ProgressBar(System.nt-1) as PBar:
            anim = animation.FuncAnimation(MVC.figure, recieve, frames=System.nt-2, interval=1, repeat=False, fargs=(System, MVC, Q, PBar))
            plt.show()
        print(System)
        print(System.diagnostic_string())
        EM.stop()
        EM.join()
    print("Done!")
        