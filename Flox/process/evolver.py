# -*- coding: utf-8 -*-
# 
#  evolver.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-10.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

"""
Evolver managed by a process.
"""

from __future__ import (absolute_import, unicode_literals, division, print_function)

import multiprocessing.managers as mm
import multiprocessing as mp

from .manager import AsynchronousManager, AsynchronousProxy

class EvolverManager(AsynchronousManager):
    """Manage Evolver objects"""
    
    def register_evolver(self, evolver):
        """Register an evolver with a generic proxy."""
        self.register(evolver.__name__, evolver.from_system, AsynchronousProxy)