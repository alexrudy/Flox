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
from ._nonlinear import NonlinearEvolver as _NonlinearEvolver

class NonlinearEvolver(_NonlinearEvolver):
    """Nonlinear evolver"""
    def __init__(self, *args, **kwargs):
        super(NonlinearEvolver, self).__init__()
        
    def __repr__(self):
        """Represent this Linear Evolver."""
        try:
            return "<{} at Time={}>".format(self.__class__.__name__, self.time)
        except:
            return super(NonlinearEvolver, self).__repr__()
        
    def evolve_many(self, grids, total_time, chunksize=int(1e3), chunks=1000):
        """Evolve over many iterations with a given total time."""
        self.update_from_grids(grids)
        start_time = grids.dimensionalize(self.time * grids.nondimensional_unit(total_time.unit))
        end_time = self.time + grids.nondimensionalize(total_time).value
        with ProgressBar(chunks) as pbar:
            for i in range(chunks):
                if grids.time >= total_time:
                    break
                else:
                    self.evolve(end_time, chunksize)
                    self.to_grids(grids, grids.it+1)
                    pbar.update(i)
    
    @classmethod
    def from_grids(cls, grids):
        """Load the grid parameters into the LE"""
        return cls(
            grids.nz, grids.nx,
            grids.nondimensionalize(grids.npa).value,
            grids.nondimensionalize(grids.Prandtl).value,
            grids.nondimensionalize(grids.Rayleigh).value,
            grids.nondimensionalize(grids.dz).value, 
            grids.nondimensionalize(grids.time).value,
            0.5
            )
        
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
        
            