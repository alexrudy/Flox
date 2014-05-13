#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
#  plot_op.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-24.
#  Copyright 2014 University of California. All rights reserved.
# 


from __future__ import (absolute_import, unicode_literals, division, print_function)

import os, os.path
import time
import matplotlib.pyplot as plt

import argparse

from Flox.system import NDSystem2D
from Flox.input import FloxConfiguration
from Flox.linear import LinearEvolver
from Flox.io import HDF5Writer
from Flox.plot import GridView, MultiViewController, EvolutionViewStabilityTest
from pyshell.util import ipydb
from matplotlib import animation
from matplotlib import rcParams
from matplotlib.colors import SymLogNorm

def filename(extension=".yml", base=None):
    """Filenames related to this file!"""
    directory = os.path.dirname(__file__)
    base = os.path.splitext(os.path.basename(__file__))[0] if base is None else base
    return os.path.join(directory, base + extension)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base', nargs="?", default='linear_op')
    parser.add_argument('-s','--stability', type=int, help='Show stability plots')
    opt = parser.parse_args()
    rcParams['text.usetex'] = False
    ipydb()
    Config = FloxConfiguration.fromfile(filename(".yml", base=opt.base))
    System = NDSystem2D.from_params(Config["system"])
    Writer = HDF5Writer(filename(".hdf5", base=opt.base))
    Writer.read(System, 'main')
    print(System)
    fig = plt.figure(figsize=(10, 10))
    rows = 2 if opt.stability else 1
    MVC = MultiViewController(fig, rows, 3)
    MVC[0,0] = GridView("Temperature", perturbed=False)
    MVC[0,1] = GridView("Vorticity", cmap='Blues')
    MVC[0,2] = GridView("Stream", cmap='Greens')
    if opt.stability:
        MVC[1,0] = EvolutionViewStabilityTest("Temperature", opt.stability, 33)
        MVC[1,1] = EvolutionViewStabilityTest("Vorticity", opt.stability, 33)
        MVC[1,2] = EvolutionViewStabilityTest("Stream", opt.stability, 33)
    System.it = 2
    MVC.update(System)
    System.infer_iteration()
    def update(i):
        System.it = i
        MVC.update(System)
    anim = animation.FuncAnimation(fig, update, frames=System.nit, interval=1)
    plt.show()
    
