# -*- coding: utf-8 -*-
# 
#  box.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-07.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import abc
import collections
import six
import numpy as np
import astropy.units as u
from pyshell.astron.units import UnitsProperty, HasUnitsProperties, recompose, ComputedUnitsProperty, recompose_unit
from pyshell.util import setup_kwargs, configure_class, resolve

from .input import FloxConfiguration
from .engine.descriptors import SpectralArrayProperty, ArrayProperty
from .engine.core import EngineInterface
from .io import WriterInterface
from .util import fullname
from .process.packet import PacketInterface
from .engine.units import WithUnitBases

@six.add_metaclass(abc.ABCMeta)
class SystemBase(EngineInterface, WriterInterface, HasUnitsProperties, WithUnitBases):
    """Base functions and properties for a system."""
    
    def __init__(self, nx, nz, nn, **kwargs):
        
        self.nx = nx
        self.nz = nz
        self.nn = nn
        
        super(SystemBase, self).__init__(**kwargs)
        
    def __repr__(self):
        """Represent this object!"""
        return "<{0} ({1}x{2})@({3})>".format(self.__class__.__name__, self.nz, self.nn, self.iteration)
        
    @classmethod
    def get_parameter_list(cls):
        """Get a list of the parameters which can be changed/modified directly"""
        return ['nz', 'nx', 'nn'] + super(SystemBase, cls).get_parameter_list()
        
    @classmethod
    def get_attribute_list(cls):
        """Return a full list of attributes, abstract and not."""
        return set(cls._list_attributes(UnitsProperty))
        
    @classmethod
    def from_params(cls, parameters):
        """Load a box from a parameter file."""
        return cls(**parameters)
        
    def to_params(self):
        """Create a parameter file."""
        parameters = FloxConfiguration()
        argnames = self.get_parameter_list()
        for argname in argnames:
            parameters[argname] = getattr(self, argname)
        return parameters
        
    def list_arrays(self):
        """Return an iterator over the array property names."""
        return self._list_attributes(ArrayProperty)
    

class System2D(SystemBase):
    """An abstract 2D Fluid box, with some basic properties."""
    
    @abc.abstractproperty
    def depth(self):
        """Box depth"""
        raise NotImplementedError()
        
    @ComputedUnitsProperty
    def dz(self):
        """The z-grid spacing"""
        return self.depth / (self.nz + 2)
        
    @property
    def z(self):
        return (np.arange(self.nz + 2))[1:-1] * self.dz
        
    @abc.abstractproperty
    def aspect(self):
        """The aspect ratio of the box."""
        raise NotImplementedError()
    
    @ComputedUnitsProperty
    def width(self):
        """The box width."""
        return self.aspect * self.depth
        
    @ComputedUnitsProperty
    def dx(self):
        """x grid spacing."""
        return self.width / self.nn
    
    @ComputedUnitsProperty
    def npa(self):
        """(n * pi / a)"""
        return np.arange(self.nn).astype(np.float) * np.pi / self.aspect
    
    def diagnostic_string(self, z=None, n=None):
        """A longer diagnostic string."""
        if n is None:
            n = [0, 1]
            ns = np.arange(self.nm)[n]
        if z is None:
            z = self.nz // 3
            zs = np.arange(self.nz)[z]
        
        output = []
        output.append("At mode n={} and z={} (it={})".format(ns, zs, self.it))
        output.append("    Time: {}".format(self.time))
        for array_name in self.list_arrays():
            if getattr(self, array_name).ndim == 2:
                output.append("    {name:15.15s}: [{value}]".format(
                    name = array_name,
                    value = ",".join([ "{:12.8g}".format(x) for x in getattr(self, array_name)[z,n]])
                ))
        
        return "\n".join(output)
        
    @ComputedUnitsProperty
    def time(self):
        """The current time of this simulation"""
        return self.Time.dimensional
    
    Time = ArrayProperty("Time", u.s, shape=tuple(), latex=r"$t$")

        