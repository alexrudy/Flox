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
from Flox.io import HDF5Writer
from Flox.nonlinear.plot import setup_movie
from pyshell.util import ipydb
from matplotlib import animation
from matplotlib import rcParams
from matplotlib.colors import SymLogNorm
from astropy.utils.console import ProgressBar

def filename(extension=".yml", base=None):
    """Filenames related to this file!"""
    directory = os.path.dirname(__file__)
    base = os.path.splitext(os.path.basename(__file__))[0] if base is None else base
    return os.path.join(directory, base + extension)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base', nargs="?", default='nonlinear')
    parser.add_argument('-s','--stability', type=int, help='Show stability plots')
    opt = parser.parse_args()
    rcParams['text.usetex'] = False
    ipydb()
    Config = FloxConfiguration.fromfile(filename(".yml", base="linear_op"))
    System = NDSystem2D.from_params(Config["system"])
    Writer = HDF5Writer(filename(".hdf5", base=opt.base))
    Writer.read(System, 'nonlinear')
    print(System)
    fig = plt.figure(figsize=(6,3.5))
    MVC = setup_movie(fig, kwargs=[dict(cmap="hot", interpolation='bilinear')])
    System.it = 2
    MVC.update(System)
    System.infer_iteration()
    with ProgressBar(System.it) as PBar:
        def update(i):
            System.it = i
            MVC.update(System)
            PBar.update(i)
        
        anim = animation.FuncAnimation(fig, update, frames=int(System.nit), interval=1)
        anim.save("convection.mp4", writer='ffmpeg', dpi=300)
    
