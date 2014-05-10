#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
#  linear_stability.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-02.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 
"""
This script runs the linear stability tests for the fluid code.

The results are saved in HDF5 for analysis, but are also analyzed in this code.

"""

from __future__ import (absolute_import, unicode_literals, division, print_function)

import numpy as np
import argparse
import os, os.path

from matplotlib import animation

from Flox.system import NDSystem2D
from Flox.input import FloxConfiguration
from Flox.linear import LinearEvolver
from Flox.io import HDF5Writer
from Flox.ic import stable_temperature_gradient, single_mode_linear_perturbation
from Flox.plot import GridView, MultiViewController, EvolutionViewStabilityTest
from pyshell.util import ipydb

def parse_options():
    """Handle argument parsing."""
    parser = argparse.ArgumentParser()
    actions = {"run","plot","animate", "analyze"}
    parser.add_argument('action', choices=actions, nargs="+", help='One of {!r}'.format(actions))
    default_filename = os.path.splitext(__file__)[0]
    
    parser.add_argument('-f', dest='filename', default=default_filename+".hdf5", help="Data file.")
    parser.add_argument('-c', dest='configuration', default=default_filename+".yml", help="Configuration file.")
    parser.add_argument('-m', dest='movie', default=default_filename+".mp4", help="Movie file.")
    parser.add_argument('-s', dest='plot', default=default_filename+".png", help="Plot file.")
    parser.add_argument('--mode', type=int, default=1, help='Mode to test.')
    return parser.parse_args()
    
def initialize(Config, opt):
    """Initialize the linear problem with the correct setup."""
    System = NDSystem2D.from_params(Config["system"])
    stable_temperature_gradient(System)
    single_mode_linear_perturbation(System, mode=opt.mode)
    return System
    
def critical_raleigh_number(System, mode):
    """Return the critical raleigh number for a given mode."""
    return (np.pi/System.aspect)**4 * (mode**2 + System.aspect**2)**3 / mode**2
    
def run(opt):
    """Run the systems."""
    
    Config = FloxConfiguration.fromfile(opt.configuration)
    Writer = HDF5Writer(opt.filename)
    
    iterations = int(Config["iterations"])
    chunks = Config["system.nt"] - 1
    
    Stable = initialize(Config, opt)
    Stable.Rayleigh = critical_raleigh_number(Stable, opt.mode) * 0.8
    
    Critical = initialize(Config, opt)
    Critical.Rayleigh = critical_raleigh_number(Critical, opt.mode)
    
    Unstable = initialize(Config, opt)
    Unstable.Rayleigh = critical_raleigh_number(Unstable, opt.mode) * 1.2
    
    print(Stable)
    LE = LinearEvolver.from_grids(Stable)
    LE.evolve_many(Stable, Config['time'], iterations, chunks)
    Writer.write(Stable, 'stable')
    
    print(Critical)
    LE = LinearEvolver.from_grids(Critical)
    LE.evolve_many(Critical, Config['time'], iterations, chunks)
    Writer.write(Critical, 'critical')
    
    print(Unstable)
    LE = LinearEvolver.from_grids(Unstable)
    LE.evolve_many(Unstable, Config['time'], iterations, chunks)
    Writer.write(Unstable, 'unstable')
    
def plot(opt):
    """Plot just the stability criteria."""
    import matplotlib.pyplot as plt
    Config = FloxConfiguration.fromfile(opt.configuration)
    Writer = HDF5Writer(opt.filename)
    
    fig = plt.figure(figsize=(11, 8.5))
    
    for i,system in enumerate("stable critical unstable".split()):
        MVC = MultiViewController(fig, 3, 3, wspace=0.6, hspace=0.4)
        System = NDSystem2D.from_params(Config["system"])
        Writer.read(System, system)
        print(System)
        MVC[0,i] = EvolutionViewStabilityTest("Temperature", opt.mode, System.nz//3)
        MVC[1,i] = EvolutionViewStabilityTest("Vorticity", opt.mode, System.nz//3)
        MVC[2,i] = EvolutionViewStabilityTest("StreamFunction", opt.mode, System.nz//3)
        MVC.views[0].ax.text(0.5, 1.25, system.capitalize(), transform=MVC.views[0].ax.transAxes, ha='center')
        MVC.update(System)
    
    fig.savefig(opt.plot, dpi=300)
    
def animate(opt):
    """Animate the data sets."""
    from astropy.utils.console import ProgressBar
    import matplotlib.pyplot as plt
    Config = FloxConfiguration.fromfile(opt.configuration)
    Writer = HDF5Writer(opt.filename)
    
    fig = plt.figure(figsize=(11, 8.5))
    
    Plots = []
    
    for i,system in enumerate("stable critical unstable".split()):
        MVC = MultiViewController(fig, 3, 3, wspace=0.6, hspace=0.4)
        System = NDSystem2D.from_params(Config["system"])
        Writer.read(System, system)
        print(System)
        MVC[0,i] = EvolutionViewStabilityTest("Temperature", opt.mode, System.nz//3)
        MVC[1,i] = EvolutionViewStabilityTest("Vorticity", opt.mode, System.nz//3)
        MVC[2,i] = EvolutionViewStabilityTest("StreamFunction", opt.mode, System.nz//3)
        System.it = 0
        MVC.update(System)
        Plots.append((MVC, System))
        
    with ProgressBar(System.nit) as pbar:
        def update(i):
            """Animation"""
            for MVC, System in Plots:
                System.it = i
                MVC.update(System)
            pbar.update(i)
        
        anim = animation.FuncAnimation(fig, update, frames=int(System.nit), interval=0.1)
        anim.save(opt.movie, writer='ffmpeg')
        # plt.show()
    
def analyze(opt):
    """Analyze the data, showing the late time values of the stability criterion."""
    import matplotlib.pyplot as plt
    Config = FloxConfiguration.fromfile(opt.configuration)
    Writer = HDF5Writer(opt.filename)
    
    for i,system in enumerate("stable critical unstable".split()):
        System = NDSystem2D.from_params(Config["system"])
        Writer.read(System, system)
        print(System)
        System.it = 0
        for array in System.list_arrays():
            if array == "Time":
                continue
            data = getattr(System, array)[System.nz//3, 1]
            ln_data = np.log(np.abs(data))
            print("{:15.15s}: {}".format(array, np.diff(ln_data)[-4:]))



if __name__ == '__main__':
    opt = parse_options()
    ipydb()
    if "run" in opt.action:
        run(opt)
    if "plot" in opt.action:
        plot(opt)
    if "animate" in opt.action:
        animate(opt)
    if "analyze" in opt.action:
        analyze(opt)