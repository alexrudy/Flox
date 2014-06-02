# -*- coding: utf-8 -*-
# 
#  core.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-25.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import numpy as np
import collections
import abc
import six

from pyshell.util import setup_kwargs, configure_class, resolve

from ..process.packet import PacketInterface
from .views import DimensionalView, NonDimensionalView, TransformedView, BaseView, PerturbedTransformedView

class EngineIterator(collections.Iterator):
    """An iterator wrapper for the engine class."""
    def __init__(self, engine, system):
        super(EngineIterator, self).__init__()
        self.engine = engine
        self._system = system.__getstate__()
        self.iteration = 0
        self.expired = False
        
    def __iter__(self):
        """Return self as the iterator"""
        return self
        
    def __next__(self):
        """Advance this iterator."""
        if self.expired:
            raise StopIteration
        self.iteration += 1
        if self.iteration >= self.engine.iterations:
            self.expired = True
            raise StopIteration
        else:
            return self.system(self.iteration)
            
    def system(self, iteration):
        """Return a new system"""
        system = self.engine.type.__new__(self.engine.type)
        self._system['_iteration'] = iteration
        system.__setstate__(self._system)
        system.engine = self.engine
        return system
    
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
        self._system_type = type(system)
        self._setup_views()
    
    def _setup_views(self):
        """Set up the views."""
        self._views = {}
        self._views["dimensional"] = DimensionalView(self)
        self._views["nondimensional"] = NonDimensionalView(self)
        self._views["transformed"] = TransformedView(self)
        self._views["perturbation"] = PerturbedTransformedView(self)
        self._views["raw"] = BaseView(self)
        
    @property
    def type(self):
        """Return the system type."""
        return self._system_type
    
    @abc.abstractproperty
    def iterations(self):
        """Number of available iterations."""
        pass
        
    @abc.abstractproperty
    def length(self):
        """Total potential length of this object."""
        pass
    
    @property
    def free(self):
        """Number of free-space iterations available."""
        return self.length - self.iterations - 2
    
    def iterator(self, system):
        """Return an iterator."""
        return EngineIterator(self, system)
    
    def __getstate__(self):
        """Return the pickling state for this object. 
        Should return an empty dictionary, as we don't want to actually pickle the contents of this array engine."""
        return { name:getattr(self, name) for name in self.get_parameter_list() }
    
    def __setstate__(self, state):
        """Return the state, resetting the caching views."""
        [self.__setattr__(name, value) for name, value in state.items()]
        self._setup_views()
    
    @classmethod
    def get_parameter_list(cls):
        """Get the parameter list for YAML."""
        return [ '_system_type' ]
    
    def get_data_list(self):
        """Return the packet list."""
        return self.keys()
    
    def initialize_arrays(self, system):
        """Initialize data arrays for this engine."""
        for attr_name in system.list_arrays():
            self.allocate(attr_name, getattr(self.type, attr_name).shape(system))
    
    @abc.abstractmethod
    def allocate(self, name, shape):
        """Allocate a given array."""
        raise NotImplementedError()
        
    def shape(self, obj, shape):
        """Hook into modification of the array shape."""
        return shape
    
    def __getdata__(self, system, name):
        """Engine caller to the underlying get method."""
        return self[name]
    
    def __setdata__(self, obj, name, value):
        """Engine caller to the underlying set method."""
        self[name] = value
        
    def read_packet(self, system, packet):
        """Create the packet."""
        for key in self.get_data_list():
            self.check_array(packet[key], key)
            self.__setdata__(system, key, packet[key])
        
    def create_packet(self, system):
        """Create the packet."""
        packet = dict()
        for key in self.get_data_list():
            packet[key] = self.__getdata__(system, key)
        return packet
        
    def check_array(self, value, name):
        """Check array values."""
        if ~np.isfinite(value).all():
            raise ValueError("{} should be finite. {} entries are not finite.".format(name, np.sum(~np.isfinite(value))))

class TimekeepingEngine(ArrayEngine):
    """An engine which handles timekeeping using the system iteration interface."""
    
    def __getdata__(self, obj, name):
        """Engine caller to the underlying get method."""
        return self[name][...,obj.iteration]
        
    def __setdata__(self, obj, name, value):
        """Engine caller to the underlying set method."""
        if self._iterations < obj.iteration:
            self._iterations = obj.iteration
        self[name][...,obj.iteration] = value

class EngineInterface(object):
    """An engine interface."""
    _engine = None
    _iteration = None
    
    def __init__(self, nt=None, engine=ArrayEngine, **kwargs):
        self.engine = engine
        if nt is not None:
            self.engine.length = nt
        super(EngineInterface, self).__init__(**kwargs)
        self.engine.initialize_arrays(self)
        self.iteration = 0
    
    def __iter__(self):
        """Iterator"""
        return self.engine.iterator(self)
    
    @property
    def iteration(self):
        """The iteration number."""
        if self._iteration is None:
            return 0
        return self._iteration
    
    @iteration.setter
    def iteration(self, value):
        """Setter for the iteration."""
        if self._iteration is None:
            self._iteration = value
        else:
            raise AttributeError("Can't set read-only attribute.")
    
    @property
    def engine(self):
        """Get the engine property."""
        return self._engine
        
    @engine.setter
    def engine(self, engine):
        """Set the engine property."""
        if isinstance(engine, ArrayEngine):
            engine.system = self
        elif isinstance(engine, collections.Mapping):
            config = dict(**engine)
            engine = resolve(config.pop('()'))(self, **config)
        elif isinstance(engine, six.text_type):
            engine = resolve(engine)(self)
        elif issubclass(engine, ArrayEngine):
            engine = engine(self)
        if not isinstance(engine, ArrayEngine):
            raise ValueError("Can't set an array engine to a non-subclass of {0}: {1!r}".format(ArrayEngine, engine))
        self._engine = engine
    
    def read_packet(self, packet):
        """Read the packet."""
        try:
            self._iteration += 1
            r = self.engine.read_packet(self, packet)
        except:
            self._iteration -= 1
            raise
        return r
    
    def create_packet(self):
        """Create the packet."""
        return self.engine.create_packet(self)
        
    @classmethod
    def get_parameter_list(cls):
        """Return the iteration."""
        return ['_iteration'] + super(EngineInterface, cls).get_parameter_list()
    