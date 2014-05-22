# -*- coding: utf-8 -*-
# 
#  __init__.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-22.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

from ._nonlinear import NonlinearEvolver as _NonlinearEvolver
from ..evolver.base import Evolver

class NonlinearEvolver(_NonlinearEvolver, Evolver):
    """Nonlinear evolver"""
    def __init__(self, *args, **kwargs):
        super(NonlinearEvolver, self).__init__()
    
    def get_packet_list(self):
        """Variables"""
        return [ "Temperature", "dTemperature", "Vorticity", "dVorticity", "Stream", "Time"]
    
    @classmethod
    def from_system(cls, system, saftey=0.5):
        """Load the grid parameters into the LE"""
        ev = cls(
            system.nz, system.nx,
            system.nondimensionalize(system.npa).value,
            system.nondimensionalize(system.dz).value,
            system.nondimensionalize(system.aspect).value,
            saftey
            )
        ev.Pr = system.nondimensionalize(system.Prandtl).value
        ev.Ra = system.nondimensionalize(system.Rayleigh).value
        return ev
            