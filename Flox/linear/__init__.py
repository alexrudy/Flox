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
        
    def evolve_many(self, grids, total_time, chunksize=int(1e3)):
        """Evolve over many iterations with a given total time."""
        self.update_from_grids(grids)
        tstart = grids.Time[grids.it]
        with grids.bases("nondimensional"):
            nd_total_time = self.time + total_time.to(grids.Time.unit).value
        itotal = 0
        with ProgressBar(grids.nt) as pb:
            while (grids.Time[grids.it] - tstart) < total_time:
                self.evolve(nd_total_time, chunksize)
                self.to_grids(grids, grids.it+1)
                itotal += 1
                pb.update(itotal)
        
    
    @classmethod
    def from_grids(cls, grids):
        """Load the grid parameters into the LE"""
        return cls(grids.Temperature[...,grids.it].copy(), grids.Vorticity[...,grids.it].copy(), 
            grids.npa.copy(), grids.Prandtl, grids.Reynolds, grids.dz, 
            grids.Time[grids.it])
        
    def update_from_grids(self, grids):
        """Update the state from a set of grids."""
        self.set_state(grids.Temperature[...,grids.it].copy(), grids.Vorticity[...,grids.it].copy(), grids.Time[grids.it])
        
    def to_grids(self, grids, iteration):
        """Load the LE data back into a grid set."""
        grids.it = iteration
        self.get_state(grids.Temperature[...,grids.it], grids.Vorticity[...,grids.it])
        grids.Time[grids.it] = self.time
        
            