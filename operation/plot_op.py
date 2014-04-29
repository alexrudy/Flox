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


from Flox.system import NDSystem2D
from Flox.input import FloxConfiguration
from Flox.linear import LinearEvolver
from Flox.io import HDF5Writer
from Flox.plot import GridView, MultiViewController
from pyshell.util import ipydb

def filename(extension=".yml", base=None):
    """Filenames related to this file!"""
    directory = os.path.dirname(__file__)
    base = os.path.splitext(os.path.basename(__file__))[0] if base is None else base
    return os.path.join(directory, base + extension)


if __name__ == '__main__':
    ipydb()
    
    Config = FloxConfiguration.fromfile(filename(".yml", base="linear_op"))
    System = NDSystem2D.from_params(Config["system"])
    Writer = HDF5Writer(filename(".hdf5", base="linear_op"))
    Writer.read(System, 'main')
    print(System)
    plt.ion()
    fig = plt.figure()
    MVC = MultiViewController(fig, 1, 1)
    MVC[0,0] = GridView("Temperature")
    print(System.it)
    for i in range(System.it):
        System.it = i
        MVC.update(System)
        # print(i)
        # raw_input("...")
    
