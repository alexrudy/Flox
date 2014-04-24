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

from Flox.system import NDSystem2D
from Flox.input import FloxConfiguration
from Flox.grids import Grids2D
from Flox.linear import LinearEvolver

from pyshell.util import ipydb

def filename(extension=".yml"):
    """docstring for filename"""
    base = os.path.splitext(__file__)[0]
    return base + extension


if __name__ == '__main__':
    ipydb()
    
    Config = FloxConfiguration.fromfile(filename(".yml"))
    System = NDSystem2D.from_params(Config["system"])
    LE = LinearEvolver.from_grids(System)
    LE.step(LE.delta_time())
    print(LE)
    LE.evolve(20*LE.delta_time(), 15)
    print(LE)