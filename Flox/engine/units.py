# -*- coding: utf-8 -*-
# 
#  units.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-25.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import collections

from pyshell.astron.units import recompose
import astropy.units as u

from ..process.packet import PickleInterface

class BasesView(object):
    """docstring for BasesView"""
    def __init__(self, parent, bases):
        super(BasesView, self).__init__()
        self.parent = parent.__class__.__name__ # Used for __repr__
        self.bases = bases
        
    def __repr__(self):
        return "<{} of {}.bases>".format(self.__class__.__name__, self.parent)
        
    def __getitem__(self, key):
        """Return the base set for this key."""
        return set(self.bases[key].values())

class WithUnitBases(PickleInterface):
    """An object with unit bases"""
    
    def __init__(self, **kwargs):
        super(WithUnitBases, self).__init__(**kwargs)
        self.setup_bases()
    
    _bases = None
    
    def setup_bases(self):
        """This method is called to setup the unit bases systems."""
        pass
    
    def add_bases(self, name, bases):
        """Add a bases state."""
        if self._bases is None:
            self._bases = {}
        if isinstance(bases, collections.Set):
            self._bases[name] = { unit.physical_type:unit for unit in bases }
        elif isinstance(bases, collections.Mapping):
            self._bases[name] = dict(**bases)
        
    
    def __setstate__(self, parameters):
        """Set the state of this system from pickling."""
        super(WithUnitBases, self).__setstate__(parameters)
        self.setup_bases()
    
    @property
    def bases(self):
        """Return the bases view object."""
        if self._bases is None:
            self._bases = {}
        return BasesView(self, self._bases)
    
    def nondimensionalize(self, quantity):
        """Nondimensionalize a given quantity for use somewhere."""
        return recompose(quantity, self.bases['nondimensional'])
        
    def dimensionalize(self, quantity):
        """Dimensionalize a given quantity or value."""
        return recompose(quantity, self.bases['standard'])
        
    