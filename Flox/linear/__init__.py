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

class LinearEvolver(_LinearEvolver):
    """Linear evolver"""
    def __init__(self, *args, **kwargs):
        super(LinearEvolver, self).__init__()
        
    def __repr__(self):
        """Represent this Linear Evolver."""
        try:
            return "<{} at Time={}>".format(self.__class__.__name__, self.time)
        except:
            return super(LinearEvolver, self).__repr__()
        
    def evolve_many(self, grids, total_time, chunksize=int(1e3), chunks=1000):
        """Evolve over many iterations with a given total time."""
        self.update_from_grids(grids)
        start_time = grids.dimensionalize(self.time * grids.nondimensional_unit(total_time.unit))
        end_time = self.time + grids.nondimensionalize(total_time)
        for i in ProgressBar(chunks):
            self.evolve(end_time, chunksize)
            self.to_grids(grids, grids.it+1)
            if (grids.time - start_time) > total_time:
                break
        
    
    @classmethod
    def from_grids(cls, grids):
        """Load the grid parameters into the LE"""
        return cls(
            grids.Temperature[...,grids.it].copy(), 
            grids.Vorticity[...,grids.it].copy(), 
            grids.nondimensionalize(grids.npa).value,
            grids.nondimensionalize(grids.Prandtl).value,
            grids.nondimensionalize(grids.Reynolds).value,
            grids.nondimensionalize(grids.dz).value, 
            grids.nondimensionalize(grids.time).value,
            )
        
    def update_from_grids(self, grids):
        """Update the state from a set of grids."""
        self.set_state(
            grids.Temperature[...,grids.it].copy(),
            grids.Vorticity[...,grids.it].copy(),
            grids.nondimensionalize(grids.time).value
            )
        
    def to_grids(self, grids, iteration):
        """Load the LE data back into a grid set."""
        grids.it = iteration
        self.get_state(grids.Temperature[...,grids.it], grids.Vorticity[...,grids.it])
        grids.Time[grids.it] = self.time
        
            