# -*- coding: utf-8 -*-
# 
#  array.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-28.
#  Copyright 2014 University of California. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import abc
import collections
import six
import numpy as np
import astropy.units as u
from pyshell.astron.units import UnitsProperty, HasUnitsProperties, recompose
from pyshell.util import setup_kwargs

@six.add_metaclass(abc.ABCMeta)
class ArrayEngine(collections.MutableMapping):
    """An engine for handling array properties behind the scenes.
    
    Must implement the mapping interface!
    
    Should refuse to implicitly allocate arrays, and all operations should happen in-place.
    """
    
    @classmethod
    def get_parameter_list(cls):
        """Get the parameter list for YAML."""
        return []
    
    @abc.abstractmethod
    def allocate(self, name, shape):
        """Allocate a given array."""
        raise NotImplementedError()
    
    @classmethod
    def from_engine(cls, engine):
        """Convert from one engine to another."""
        new_engine = cls()
        for key in engine.keys():
            old_array = engine[key]
            new_engine.allocate(key, old_array.shape)
            new_engine[key] = old_array
        return new_engine

class ArrayYAMLSupport(object):
    """An empty class for Array YAML support with built-in types."""
    pass


class ArrayProperty(UnitsProperty):
    """Custom subclass used to assist with Array allocation."""
    def __init__(self, name, unit, shape=tuple(), engine='engine', **kwargs):
        super(ArrayProperty, self).__init__(name, unit, **kwargs)
        self._shape = shape
        self._engine = engine
    
    
    def shape(self, obj):
        """Retrieve inexplicit shapes"""
        shape = tuple([self._get_shape_part(obj, part) for part in self._shape])
        if len(shape) < 1:
            raise ValueError("Got 0-dimensional array: {!r}".format(shape))
        if shape == tuple((0,)):
            raise ValueError("Got unindexable array: {!r}".format(shape))
        return shape
        
    def _get_shape_part(self, obj, part):
        """Get a part of the shape tuple"""
        if isinstance(part, int):
            return part
        elif isinstance(part, six.string_types):
            return getattr(obj, part)
        else:
            raise ValueError("Implicit shape '{}' is not valid.".format(part))
        
    def allocate(self, obj):
        """Allocate the array in the proper area, with the proper shape."""
        getattr(obj, self._engine).allocate(self._attr, self.shape(obj))
    
    def set(self, obj, value):
        """Short out the setter so that it doesn't use units, but uses the allocated array space."""
        getattr(obj, self._engine)[self._attr] = value
        
    def get(self, obj):
        """Get this object."""
        return getattr(obj, self._engine)[self._attr]
        
class NumpyArrayEngine(ArrayYAMLSupport, dict, ArrayEngine):
    """A numpy-based array engine"""
    
    def __init__(self, dtype=np.float):
        """Initialize this object."""
        super(NumpyArrayEngine, self).__init__()
        self._dtype = dtype
        
    @property
    def dtype(self):
        """dtype"""
        return np.dtype(self._dtype).str
        
    @classmethod
    def get_parameter_list(cls):
        """Get the parameter list pairs."""
        return ['dtype']
        
    def allocate(self, name, shape):
        """Allocate arrays with empty numpy arrays"""
        self[name] = np.empty(shape, dtype=self._dtype)
        