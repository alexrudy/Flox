# -*- coding: utf-8 -*-
# 
#  __init__.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-22.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import astropy.units as u
from astropy.utils.console import ProgressBar
from ._linear import LinearEvolver as _LinearEvolver
from ..evolver import Evolver
from ..packet import Packet2D

class LinearEvolver(Evolver, _LinearEvolver):
    """Linear evolver"""
    def __init__(self, *args, **kwargs):
        super(LinearEvolver, self).__init__()
        
    @classmethod
    def from_system(cls, system, saftey):
        """Load the grid parameters into the LE"""
        return cls(
            system.nz, system.nx,
            system.nondimensionalize(grids.npa).value,
            system.nondimensionalize(grids.Prandtl).value,
            system.nondimensionalize(grids.Rayleigh).value,
            system.nondimensionalize(grids.dz).value, 
            system.nondimensionalize(grids.time).value,
            0.5
            )
        
    def create_packet(self):
        """Create a packet from the LinearEvolver state."""
        pass
        
    def update_from_grids(self, grids):
        """Update the state from a set of grids."""
        self.set_state(
            grids.Temperature[...,grids.it].copy(),
            grids.Vorticity[...,grids.it].copy(),
            grids.StreamFunction[...,grids.it].copy(),
            grids.nondimensionalize(grids.time).value
            )
        
    def to_grids(self, grids, iteration):
        """Load the LE data back into a grid set."""
        grids.it = iteration
        self.get_state(grids.Temperature[...,grids.it], grids.Vorticity[...,grids.it], grids.StreamFunction[...,grids.it])
        grids.Time[grids.it] = self.time
    