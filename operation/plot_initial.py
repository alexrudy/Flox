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
    opt = parser.parse_args()
    rcParams['text.usetex'] = False
    ipydb()
    Config = FloxConfiguration.fromfile(filename(".yml", base=opt.base))
    System = NDSystem2D.from_params(Config["system"])
    Writer = HDF5Writer(filename(".hdf5", base=opt.base))
    Writer.read(System, 'main')
    System.it = 1
    print(System)
    print(System.diagnostic_string())
    fig = plt.figure(figsize=(10, 10))
    MVC = MultiViewController(fig, 2, 2)
    MVC[0,0] = GridView("Temperature")
    MVC[1,0] = EvolutionViewStabilityTest("Temperature", 1, 33)
    MVC[0,1] = GridView("Vorticity", cmap='Blues', vmin=-1e-7, vmax=1e-7, norm=SymLogNorm(1e-9), perturbed=True)
    MVC[1,1] = EvolutionViewStabilityTest("Vorticity", 1, 33)
    MVC.update(System)
    plt.show()
    
