# -*- coding: utf-8 -*-
# 
#  __init__.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-19.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

from ._magneto import MagnetoEvolver as _MagnetoEvolver
from ..evolver import Evolver

class MagnetoEvolver(_MagnetoEvolver, Evolver):
    """Nonlinear evolver"""
    def __init__(self, *args, **kwargs):
        super(MagnetoEvolver, self).__init__()
    
    def get_packet_list(self):
        """Variables"""
        return [ "Temperature", "dTemperature", "Vorticity", "dVorticity", "Stream", "VectorPotential", "dVectorPotential", "CurrentDensity", "Time"]
    
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
        ev.Prandtl = system.nondimensionalize(system.Prandtl).value
        ev.Rayleigh = system.nondimensionalize(system.Rayleigh).value
        return ev