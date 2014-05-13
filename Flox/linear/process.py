# -*- coding: utf-8 -*-
# 
#  process.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-12.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

from . import LinearEvolver

import multiprocessing as mp

class LinearEvolverProcess(mp.Process):
    """A process for handling the linear evolver."""
    def __init__(self, System, Writer, write_args, evolve_args, *args, **kwargs):
        super(LinearEvolverProcess, self).__init__()
        self.System = System
        self.Writer = Writer
        self.write_args = write_args
        self.evolve_args = evolve_args

    def run(self):
        """Run this process."""
        LE = LinearEvolver.from_system(self.System)
        LE.evolve_system(self.System, *self.evolve_args, quiet=True)
        self.Writer.write(self.System, *self.write_args)
        print(self.System)
        print(self.System.diagnostic_string())