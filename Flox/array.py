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

from .transform import spectral_transform

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
        
    def shape(self, obj, shape):
        """Hook into modification of the array shape."""
        return shape
    
    def _get(self, obj, name):
        """Engine caller to the underlying get method."""
        return self[name]
    
    def _set(self, obj, name, value):
        """Engine caller to the underlying set method."""
        self[name] = value
    
    @classmethod
    def from_engine(cls, engine):
        """Convert from one engine to another."""
        new_engine = cls()
        for key in engine.keys():
            old_array = engine[key]
            new_engine.allocate(key, old_array.shape)
            new_engine[key] = old_array
        return new_engine

class TimekeepingEngine(ArrayEngine):
    """An engine which handles timekeeping using the system iteration interface."""
    
    def shape(self, obj, shape):
        """Hook into modification of the array shape."""
        return shape + tuple([obj.nt])
    
    def _get(self, obj, name):
        """Engine caller to the underlying get method."""
        return self[name][...,obj.it]
        
    def _set(self, obj, name, value):
        """Engine caller to the underlying set method."""
        self[name][...,obj.it] = value


class ArrayYAMLSupport(object):
    """An empty class for Array YAML support with built-in types."""
    pass


class ArrayProperty(UnitsProperty):
    """Custom subclass used to assist with Array allocation."""
    def __init__(self, name, unit, shape=tuple(), engine='engine', **kwargs):
        super(ArrayProperty, self).__init__(name, unit, **kwargs)
        self._shape = shape
        self._engine = engine
        self._attr = name
    
    
    def shape(self, obj):
        """Retrieve inexplicit shapes"""
        shape = tuple([self._get_shape_part(obj, part) for part in self._shape])
        if shape == tuple((0,)):
            raise ValueError("Got unindexable array: {!r}".format(shape))
        return getattr(obj, self._engine).shape(obj, shape)
        
    def _get_shape_part(self, obj, part):
        """Get a part of the shape tuple"""
        if isinstance(part, int):
            return part
        elif isinstance(part, six.string_types):
            return getattr(obj, part)
        else:
            raise ValueError("Implicit shape '{!r}' is not a valid type: {!r}".format(part, [int, six.string_types]))
        
    def allocate(self, obj):
        """Allocate the array in the proper area, with the proper shape."""
        getattr(obj, self._engine).allocate(self._attr, self.shape(obj))
    
    def set(self, obj, value):
        """Short out the setter so that it doesn't use units, but uses the allocated array space."""
        getattr(obj, self._engine)._set(obj, self._attr, value)
        
    def get(self, obj):
        """Get this object."""
        return getattr(obj, self._engine)._get(obj, self._attr)
        
class SpectralArrayProperty(ArrayProperty):
    """An array with spectral property support"""
    def __init__(self, name, unit, func, **kwargs):
        super(SpectralArrayProperty, self).__init__(name, unit, **kwargs)
        self._func = func
        
    def itransform(self, obj):
        """Perform the inverse transform."""
        # The width here is 1.0, because this function takes ND variables.
        return spectral_transform(self._func, self.get(obj), obj.nx, obj.aspect.value)
        
    def p_itransform(self, obj):
        """Perturbed inverse transform."""
        return spectral_transform(self._func, self.get(obj), obj.nx, obj.aspect.value, perturbed=True)

class NumpyArrayEngine(ArrayYAMLSupport, dict, TimekeepingEngine):
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
        self[name] = np.zeros(shape, dtype=self._dtype)
        
class NumpyFrameEngine(ArrayYAMLSupport, dict, ArrayEngine):
    """A numpy-based array engine which only holds a single frame."""
    
    def __init__(self, dtype=np.float):
        """Initialize this object."""
        super(NumpyFrameEngine, self).__init__()
        self._dtype = dtype
        
    def allocate(self, name, shape):
        """Allocate arrays with empty numpy arrays"""
        self[name] = np.zeros(shape, dtype=self._dtype)
        
    @classmethod
    def get_parameter_list(cls):
        """Get the parameter list pairs."""
        return ['dtype']