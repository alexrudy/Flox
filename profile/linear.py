#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 
#  linear.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-24.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import cProfile
import pstats

import os, os.path
import time

import astropy.units as u

from Flox.system import NDSystem2D
from Flox.input import FloxConfiguration
from Flox.linear import LinearEvolver
# from pyshell.util import ipydb

def filename(extension=".yml"):
    """Make a filename"""
    base = os.path.splitext(__file__)[0]
    return base + extension

if __name__ == '__main__':
    
    
    Config = FloxConfiguration.fromfile(filename(".yml"))
    System = NDSystem2D.from_params(Config["system"])
    print(System.npa.shape)
    iterations = int(Config["iterations"])
    time = Config["time"].to(u.s).value / 100
    LE = LinearEvolver.from_grids(System)
    LE.to_grids(System, 1)
    cProfile.runctx("LE.evolve(time, iterations)", globals(), locals(), filename(".prof"))
    s = pstats.Stats(filename(".prof"))
    s.strip_dirs().sort_stats("time").print_stats()