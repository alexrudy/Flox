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
from pyshell.astron.units import UnitsProperty, HasUnitsProperties, recompose, recompose_unit
from pyshell.util import setup_kwargs

from .process.packet import PacketInterface
from .transform import spectral_transform
from .view import DictionaryView

class EngineStateError(Exception):
    """An error for when an engine is in the wrong state."""
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
        return spectral_transform(self._func, self.get(obj), obj.nx, obj.aspect.value)
        
    def p_itransform(self, obj):
        """Perturbed inverse transform."""
        return spectral_transform(self._func, self.get(obj), obj.nx, obj.aspect.value, perturbed=True)

class ExistingView(DictionaryView):
    """A view which only allows actions on existing elements."""
    
    def __setitem__(self, key, value):
        """Set item."""
        if key not in self.source:
            raise KeyError("Can't set not-present key {}".format(key))
            
    def __delitem__(self, key):
        """Delete item."""
        raise KeyError("Keys can't be deleted from this view.")

class UnitView(ExistingView):
    """A view for retrieving units, etc."""
    
    def __init__(self, source):
        super(UnitView, self).__init__(source)
        self._units = {}
    
    def unit(self, key):
        """Get the unit for a key."""
        if key not in self._units:
            self._unit(key)
        return self._units[key][0]
        
    def _unit(self, key):
        """Cache a unit."""
        unit = getattr(self.source._system_type, key).unit(self.source._system)
        self._units[key] = (unit, recompose_unit(unit, set(self.source._system._bases['nondimensional'].values())))
        
    def nd_unit(self, key):
        """Non-dimensional unit for a key."""
        if key not in self._units:
            self._unit(key)
        return self._units[key][1]

class DimensionalView(UnitView):
    """Return a dimensionalizer for the array engine."""
        
    def __getitem__(self, key):
        """Get item."""
        value = super(DimensionalView, self).__getitem__(key)
        return (value * self.nd_unit(key)).to(self.unit(key))
        
    def __setitem__(self, key, value):
        """Set an item."""
        value = u.Quantity(value, unit=self.unit(key)).to(self.nd_unit(key)).value
        return super(DimensionalView, self).__setitem__(key, value)
        
class NonDimensionalView(DictionaryView):
    """Return a dimensionalizer for the array engine."""
    def __init__(self, source):
        super(NonDimensionalView, self).__init__(source)
        
    def __getitem__(self, key):
        """Get item."""
        value = super(NonDimensionalView, self).__getitem__(key)
        return (value * self.nd_unit(key))
        
    def __setitem__(self, key, value):
        """Set an item."""
        value = u.Quantity(value, unit=self.nd_unit(key)).to(self.unit(key)).value
        return super(DimensionalView, self).__setitem__(key, value)
        
class TransformedView(DimensionalView):
    """docstring for TransformedView"""
    
    def __getitem__(self, key):
        """Get item."""
        descriptor = getattr(self.source._system_type, key)
        return (descriptor.itransform(self.source._system) * self.nd_unit(key)).to(self.unit(key))
        
    def __setitem__(self, key, value):
        """Set item."""
        raise ValueError("Can't set a transformed array!")

class EngineIterator(collections.Iterator):
    """An iterator wrapper for the engine class."""
    def __init__(self, engine):
        super(EngineIterator, self).__init__()
        self.engine = engine
        self.engine.iteration = 0
        self.expired = False
        
    def __iter__(self):
        """Return self as the iterator"""
        return self
        
    def __next__(self):
        """Advance this iterator."""
        if self.expired:
            raise StopIteration
        try:
            self.engine.iteration += 1
        except IndexError as e:
            self.expired = True
            raise StopIteration
        else:
            return self.engine._system
    
    def __len__(self):
        """A best-estimate length of this iterator. May be unreliable"""
        return self.engine.iterations
        

@six.add_metaclass(abc.ABCMeta)
class ArrayEngine(collections.Mapping, PacketInterface):
    """An engine for handling array properties behind the scenes.
    
    Must implement the mapping interface!
    
    Should refuse to implicitly allocate arrays, and all operations should happen in-place.
    """
    
    def __init__(self, system):
        super(ArrayEngine, self).__init__()
        self._system = system
        self._system_type = type(system)
        self._iteration = 0
        self._setup_views()
    
    def _setup_views(self):
        """Set up the views."""
        self.dimensional = DimensionalView(self)
        self.nondimensional = NonDimensionalView(self)
        self.transformed = TransformedView(self)
        self.raw = ExistingView(self)
    
    @property
    def iteration(self):
        """The current iteration index."""
        return self._iteration
        
    @iteration.setter
    def iteration(self, value):
        """Set the current iteration index."""
        if value >= self.iterations:
            raise IndexError("Exceeded maximum iterations.")
    
    @abc.abstractproperty
    def iterations(self):
        """Number of available iterations."""
        pass
        
    @abc.abstractproperty
    def length(self):
        """Total potential length of this object."""
        pass
    
    def iterator(self):
        """Return an iterator."""
        return EngineIterator(self)
    
    def __getstate__(self):
        """Return the pickling state for this object. 
        Should return an empty dictionary, as we don't want to actually pickle the contents of this array engine."""
        return { name:getattr(self, name) for name in self.get_parameter_list() }
    
    def __setstate__(self, state):
        """Return the state, resetting the caching views."""
        [self.__setattr__(name, value) for name, value in state.items()]
        self._system_type = type(self._system)
        self._setup_views()
    
    @classmethod
    def get_parameter_list(cls):
        """Get the parameter list for YAML."""
        return [ '_system', '_iteration' ]
    
    def get_data_list(self):
        """Return the packet list."""
        return self.keys()
    
    def initialize_arrays(self):
        """Initialize data arrays for this engine."""
        for attr_name in self._system.list_arrays():
            getattr(self._system_type, attr_name).allocate(self._system)
    
    @abc.abstractmethod
    def allocate(self, name, shape):
        """Allocate a given array."""
        raise NotImplementedError()
        
    def shape(self, obj, shape):
        """Hook into modification of the array shape."""
        return shape
    
    def _get(self, obj, name):
        """Engine caller to the underlying get method."""
        return self.raw[name]
    
    def _set(self, obj, name, value):
        """Engine caller to the underlying set method."""
        self.raw[name] = value
        
    def create_packet(self):
        """Create the packet."""
        packet = dict()
        for key in self.get_data_list():
            packet[key] = self._get(self._system, key)
        return packet

class TimekeepingEngine(ArrayEngine):
    """An engine which handles timekeeping using the system iteration interface."""
    
    def _get(self, obj, name):
        """Engine caller to the underlying get method."""
        return self.raw[name][...,self.iteration]
        
    def _set(self, obj, name, value):
        """Engine caller to the underlying set method."""
        self.raw[name][...,self.iteration] = value

class NumpyArrayEngine(dict, TimekeepingEngine):
    """A numpy-based array engine"""
    
    def __init__(self, system, length=None, dtype=np.float):
        """Initialize this object."""
        TimekeepingEngine.__init__(self, system)
        super(NumpyArrayEngine, self).__init__()
        self._dtype = dtype
        self._length = length
        self._iterations = 0
        
    @property
    def iterations(self):
        """Number of iterations available."""
        return self._iterations
        
    @property
    def length(self):
        """Maximum object length."""
        if self._length is None:
            raise ValueError("Length has not been set.")
        return self._length
        
    @length.setter
    def length(self, value):
        """Maximum object length."""
        if self._length is None:
            self._length = value
        else:
            raise ValueError("Can't adjust length.")
            
    def initialize_arrays(self):
        """Set the array length before initializing."""
        if self._length is None:
            self._length = 100
        return super(NumpyArrayEngine, self).initialize_arrays()
    
    @property
    def dtype(self):
        """dtype"""
        return np.dtype(self._dtype).str
        
    @classmethod
    def get_parameter_list(cls):
        """Get the parameter list pairs."""
        return ['_dtype'] + super(NumpyArrayEngine, cls).get_parameter_list()
        
    def allocate(self, name, shape):
        """Allocate arrays with empty numpy arrays"""
        self[name] = np.zeros(shape + tuple([self.length]), dtype=self._dtype)
        
        
class NumpyFrameEngine(dict, ArrayEngine):
    """A numpy-based array engine which only holds a single frame."""
    
    def __init__(self, system, dtype=np.float):
        """Initialize this object."""
        ArrayEngine.__init__(self, system)
        super(NumpyFrameEngine, self).__init__(system)
        self._dtype = dtype
        
    def allocate(self, name, shape):
        """Allocate arrays with empty numpy arrays"""
        self[name] = np.zeros(shape, dtype=self._dtype)
        
    @classmethod
    def get_parameter_list(cls):
        """Get the parameter list pairs."""
        return ['dtype']