# -*- coding: utf-8 -*-
# 
#  __init__.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-22.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

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
        
    @classmethod
    def from_grids(cls, grids):
        """Load the grid parameters into the LE"""
        with grids.in_nondimensional():
            return cls(grids.Temperature[...,-1].value, grids.Vorticity[...,-1].value, 
                grids.npa.value, grids.Prandtl.value, grids.Reynolds.value, grids.dz.value, 
                grids.Time[...,-1].value)
        
    def to_grids(self, grids):
        """Load the LE data back into a grid set."""
        raise NotImplementedError()