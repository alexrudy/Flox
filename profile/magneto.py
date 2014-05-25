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

from Flox.magneto.system import MagnetoSystem
from Flox.input import FloxConfiguration
from Flox.magneto import MagnetoEvolver
from Flox._threads import omp_set_num_threads
# from pyshell.util import ipydb

def filename(extension=".yml"):
    """Make a filename"""
    base = os.path.splitext(__file__)[0]
    return base + extension

if __name__ == '__main__':
    
    
    Config = FloxConfiguration.fromfile(filename(".yml"))
    System = MagnetoSystem.from_params(Config["system"])
    iterations = int(Config["iterations"])
    time = Config["time"].to(u.s).value / 100
    ME = MagnetoEvolver.from_system(System)
    print("Profiling {} for {:d} iterations".format(ME, iterations))
    cProfile.runctx("omp_set_num_threads(int(Config.get('threads',1)))\nME.evolve(time, iterations)", globals(), locals(), filename(".prof"))
    s = pstats.Stats(filename(".prof"))
    s.strip_dirs().sort_stats("time").print_stats()
    if os.path.exists(filename("-old.prof")):
        s = pstats.Stats(filename("-old.prof"))
        s.strip_dirs().sort_stats("time").print_stats()
    
    