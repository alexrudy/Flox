# -*- coding: utf-8 -*-
# 
#  evolver.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-31.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

from ..evolver._hydro import HydroEvolver as _HydroEvolver
from ..evolver.base import Evolver

class HydroEvolver(_HydroEvolver, Evolver):
    """Nonlinear evolver"""
    def __init__(self, *args, **kwargs):
        super(HydroEvolver, self).__init__()
    
    @classmethod
    def get_parameter_list(cls):
        """Retrieve the relevant parameters."""
        return []
    
    def get_data_list(self):
        """Variables"""
        return super(HydroEvolver, self).get_data_list() + [ "Temperature", "dTemperature", "Vorticity", "dVorticity", "Stream", "Time"]
    
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
        ev.linear = system.linear
        ev.set_T_bounds(*system._T_Bounds())
        ev.Pr = system.nondimensionalize(system.Prandtl).value
        ev.Ra = system.nondimensionalize(system.Rayleigh).value
        return ev