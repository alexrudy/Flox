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

from Flox.system import NDSystem2D
from Flox.input import FloxConfiguration
from Flox.linear import LinearEvolver
from Flox.io import HDF5Writer
from pyshell.util import ipydb

def filename(extension=".yml"):
    """docstring for filename"""
    base = os.path.splitext(__file__)[0]
    return base + extension


if __name__ == '__main__':
    ipydb()
    
    Config = FloxConfiguration.fromfile(filename(".yml"))
    System = NDSystem2D.from_params(Config["system"])
    iterations = int(Config["iterations"])
    chunks = System.nt - System.it - 1
    Writer = HDF5Writer(filename(".hdf5"))
    System.Temperature[:,:,System.it] = 0.1
    System.Temperature[:,0,System.it] = 0.5
    print(System)
    LE = LinearEvolver.from_grids(System)
    LE.evolve_many(System, Config['time'], iterations, chunks)
    print(System.Temperature[...,System.it])
    print(System)
    Writer.write(System,'main')