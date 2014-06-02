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
from ..evolver.base import Evolver

class MagnetoEvolver(_MagnetoEvolver, Evolver):
    """Nonlinear evolver"""
    def __init__(self, *args, **kwargs):
        super(MagnetoEvolver, self).__init__()
    
    @classmethod
    def get_parameter_list(cls):
        """Retrieve the relevant parameters."""
        return []
    
    def get_data_list(self):
        """Variables"""
        return [ "Temperature", "dTemperature", "Vorticity", "dVorticity", "Stream", "VectorPotential", "dVectorPotential", "CurrentDensity", "Time"]
    
    @classmethod
    def from_system(cls, system, safety=0.5, checkCFL=10, LinearOnly=False):
        """Load the grid parameters into the LE"""
        ev = cls(
            system.nz, system.nn,
            system.nondimensionalize(system.npa).value,
            system.nondimensionalize(system.dz).value,
            system.nondimensionalize(system.aspect).value,
            safety
            )
        ev.Pr = system.nondimensionalize(system.Prandtl).value
        ev.Ra = system.nondimensionalize(system.Rayleigh).value
        ev.Q = system.nondimensionalize(system.Chandrasekhar).value
        ev.q = system.nondimensionalize(system.Roberts).value
        ev.checkCFL = checkCFL
        ev.LinearOnly = LinearOnly
        return ev