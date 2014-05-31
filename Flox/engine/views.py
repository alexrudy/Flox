# -*- coding: utf-8 -*-
# 
#  views.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-25.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import abc
import six
import copy
import numpy as np
import astropy.units as u
from pyshell.astron.units import recompose, recompose_unit

class BaseView(object):
    """The view interface"""
    def __init__(self, source):
        super(BaseView, self).__init__()
        self.source = source
        
    def __getdata__(self, system, key):
        """Get data belonging to a system with a key."""
        return self.source.__getdata__(system, key)
        
    def __setdata__(self, system, key, value):
        """Set data belonging to a system with a key"""
        self.source.__setdata__(system, key, value)

class UnitView(BaseView):
    """A view for retrieving units, etc."""
    
    def __unit__(self, system, key):
        """Get the unit for a key."""
        return getattr(self.source.type, key).unit
        
    def __ndunit__(self, system, key):
        """Non-dimensional unit for a key."""
        unit = getattr(self.source.type, key).unit
        return recompose_unit(unit, system.bases['nondimensional'])

class DimensionalView(UnitView):
    """Return a dimensionalizer for the array engine."""
        
    def __getdata__(self, system, key):
        """Get item."""
        value = super(DimensionalView, self).__getdata__(system, key)
        return (value * self.__ndunit__(system, key)).to(self.__unit__(system, key))
        
    def __setdata__(self, system, key, value):
        """Set an item."""
        value = u.Quantity(value, unit=self.__unit__(system, key)).to(self.__ndunit__(system, key)).value
        return super(DimensionalView, self).__setdata__(system, key, value)
        
class NonDimensionalView(UnitView):
    """Return a dimensionalizer for the array engine."""
    def __init__(self, source):
        super(NonDimensionalView, self).__init__(source)
        
    def __getdata__(self, system, key):
        """Get item."""
        value = super(NonDimensionalView, self).__getdata__(system, key)
        return (value * self.__ndunit__(system, key))
        
    def __setdata__(self, system, key, value):
        """Set an item."""
        value = u.Quantity(value, unit=self.__ndunit__(system, key)).value
        return super(NonDimensionalView, self).__setitem__(system, key, value)
        
class TransformedView(DimensionalView):
    """docstring for TransformedView"""
    
    def __getdata__(self, system, key):
        """Get item."""
        descriptor = getattr(self.source.type, key)
        return (descriptor.itransform(system) * self.__ndunit__(system, key)).to(self.__unit__(system, key))
        
    def __setdata__(self, system, key, value):
        """Set item."""
        raise AttributeError("{}: Can't set a transformed array! Key={}".format(system, key))
        
class PerturbedTransformedView(TransformedView):
    """docstring for PerturbedTransformedView"""
    
    def __getdata__(self, system, key):
        """Get item."""
        descriptor = getattr(self.source.type, key)
        return (descriptor.itransform(system, perturbed=True) * self.__ndunit__(system, key)).to(self.__unit__(system, key))
    