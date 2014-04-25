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
        with grids.in_nondimensional():
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
        with grids.in_nondimensional():
            return cls(grids.Temperature[...,grids.it].value.copy(), grids.Vorticity[...,grids.it].value.copy(), 
                grids.npa.value.copy(), grids.Prandtl.value, grids.Reynolds.value, grids.dz.value, 
                grids.Time[grids.it].value)
        
    def update_from_grids(self, grids):
        """Update the state from a set of grids."""
        with grids.in_nondimensional():
            self.set_state(grids.Temperature[...,grids.it].value.copy(), grids.Vorticity[...,grids.it].value.copy(), grids.Time[grids.it].value)
        
    def to_grids(self, grids, iteration):
        """Load the LE data back into a grid set."""
        grids.it = iteration
        with grids.in_nondimensional():
            time_unit = grids.Time.unit
            times = grids.Time
            self.get_state(grids.Temperature[...,grids.it], grids.Vorticity[...,grids.it])
            times[grids.it] = self.time * time_unit
            grids.Time = times
        
            