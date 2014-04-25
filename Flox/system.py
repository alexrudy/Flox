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
import six
import numpy as np
import astropy.units as u
from .units import HasUnitsProperties, ComputedUnitsProperty, UnitsProperty
from pyshell.util import setup_kwargs

from .input import FloxConfiguration

class ArrayProperty(UnitsProperty):
    """Custom subclass used to assist with Array allocation."""
    pass

@six.add_metaclass(abc.ABCMeta)
class System2D(HasUnitsProperties):
    """An abstract 2D Fluid box, with some basic properties."""
    
    nx = 0
    nz = 0
    nt = 0
    it = 0
    
    def __init__(self, nx, nz, nt, it=0, dtype=np.float):
        super(System2D, self).__init__()
        self.nx = nx
        self.nz = nz
        self.nt = nt
        self.it = it
        self.dtype = dtype
        self.initialize_arrays()
        
    
    def __repr__(self):
        """Represent this object!"""
        try:
            Pr = self.Prandtl
            Re = self.Reynolds
            time = self.Time[self.it]
            return "<{0} with Re={1.value} and Pr={2.value} at {3}>".format(self.__class__.__name__, Re, Pr, time)
        except NotImplementedError:
            return super(System2D, self).__repr__()
    
    def infer_iteration(self):
        """Infer the iteration number from loaded data."""
        # TODO Ensure Time is sorted!
        self.it = np.argmax(self.Time)
    
    @abc.abstractproperty
    def Prandtl(self):
        """Prandtl number."""
        raise NotImplementedError()
        
    
    @abc.abstractproperty
    def Reynolds(self):
        """Reynolds number."""
        raise NotImplementedError()
        
    @abc.abstractproperty
    def deltaT(self):
        """Temperature differential."""
        raise NotImplementedError()
        
    @abc.abstractproperty
    def depth(self):
        """Box depth"""
        raise NotImplementedError()
        
    @ComputedUnitsProperty
    def dz(self):
        """The z-grid spacing"""
        return self.depth / self.nz
        
    @abc.abstractproperty
    def aspect(self):
        """docstring for aspect"""
        raise NotImplementedError()
    
    @ComputedUnitsProperty
    def width(self):
        """The box width."""
        return self.aspect * self.depth
    
    @abc.abstractproperty
    def kinematic_viscosity(self):
        """Kinematic Viscosity"""
        raise NotImplementedError()
    
    @abc.abstractproperty
    def thermal_diffusivity(self):
        """Thermal Diffusivity"""
        raise NotImplementedError()
    
    @ComputedUnitsProperty
    def primary_viscosity(self):
        """Primary Viscosity component"""
        if self.kinematic_viscosity > self.thermal_diffusivity:
            return self.kinematic_viscosity
        else:
            return self.thermal_diffusivity
        
    @ComputedUnitsProperty
    def npa(self):
        """(n * pi / a)"""
        return np.arange(self.nx).astype(np.float) * np.pi / self.aspect
        
    @classmethod
    def get_parameter_list(cls):
        """Get a list of the parameters which can be changed/modified directly"""
        return ['nx', 'ny']
        
    def _setup_standard_bases(self):
        """Set the standard, non-dimensional bases"""
        self.add_bases("standard", set([u.K, u.m, u.s]))
        with self.bases("standard"):
            temperature_unit = u.def_unit("Box-delta-T", self.deltaT)
            length_unit = u.def_unit("Box-D", self.depth)
            time_unit = u.def_unit("Box-D^2/kappa", self.depth**2 / self.primary_viscosity)
            self.add_bases("nondimensional", set([temperature_unit, length_unit, time_unit]))
        
    @classmethod
    def from_params(cls, parameters):
        """Load a box from a parameter file."""
        return cls(**setup_kwargs(cls.__init__,parameters))
        
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
        
    def initialize_arrays(self):
        """Initialize data arrays"""
        shape = (self.nz, self.nx, self.nt)
        for attr_name in self.list_arrays():
            setattr(self, attr_name, np.zeros(shape, dtype=self.dtype))
        self.Time = np.zeros((self.nt,), dtype=self.dtype)
    
    Time = ArrayProperty("Time", u.s, latex=r"$t$")
    Temperature = ArrayProperty("Temperature", u.K, latex=r"$T$")
    Vorticity = ArrayProperty("Vorticity", 1.0 / u.s, latex=r"$\omega$")
    StreamFunction = ArrayProperty("StreamFunction", u.m**2 / u.s, latex=r"$\psi$")
    
    

class NDSystem2D(System2D):
    """A primarily non-dimensional 2D system."""
    def __init__(self,
        nx=0, nz=0, nt=0,
        deltaT=0, depth=0, aspect=0, Prandtl=0, Reynolds=0, kinematic_viscosity=0):
        super(NDSystem2D, self).__init__(nx=nx, nz=nz, nt=nt)
        self.deltaT = deltaT
        self.depth = depth
        self.aspect = aspect
        self.Prandtl = Prandtl
        self.Reynolds = Reynolds
        self.kinematic_viscosity = kinematic_viscosity
        self._setup_standard_bases()
        
        
    deltaT = UnitsProperty("deltaT", u.K, latex=r"$\Delta T$")
    depth = UnitsProperty("depth", u.m, latex=r"$D$")
    aspect = UnitsProperty("aspect", u.dimensionless_unscaled, latex=r"$a$")
    
    kinematic_viscosity = UnitsProperty("kinematic viscosity", u.m**2.0 / u.s, latex=r"$\kappa$")
    Prandtl = UnitsProperty("Prandtl", u.dimensionless_unscaled, latex=r"$Pr$")
    Reynolds = UnitsProperty("Reynolds", u.dimensionless_unscaled, latex=r"$Re$")
    
    @ComputedUnitsProperty
    def thermal_diffusivity(self):
        """Thermal Diffusivity"""
        return self.kinematic_viscosity * self.Prandtl
        
    @classmethod
    def get_parameter_list(cls):
        """Get a list of the parameters which can be changed/modified directly"""
        import inspect
        return inspect.getargspec(cls.__init__)[0][1:]
        
    

class PhysicalSystem2D(System2D):
    """A 2D Fluid box, with some basic properties.
    
    :param deltaT: The temperature change between the bottom and top of the box.
    :param depth: The depth of the box.
    :param aspect: The aspect ratio of the box.
    :param kinematic_viscosity: The kinematic viscosity of the box.
    :param thermal_diffusivity: The thermal diffusivity of the box.
    :param thermal_expansion: The coefficient of thermal expansion.
    :param gravitaional_acceleration: The rate of gravitational acceleration.
    
    """
    def __init__(self, deltaT=0, depth=0, aspect=0,
        nx=0, nz=0, nt=0,
        kinematic_viscosity=0, thermal_diffusivity=0, thermal_expansion=0, gravitaional_acceleration=0):
        super(PhysicalSystem2D, self).__init__(nx=nx, nz=nz, nt=nt)
        
        # Box Physical Variables
        self.deltaT = deltaT
        self.depth = depth
        self.aspect = aspect
        
        # Fluid Variables
        self.kinematic_viscosity = kinematic_viscosity
        self.thermal_diffusivity = thermal_diffusivity
        self.thermal_expansion = thermal_expansion
        self.gravitaional_acceleration = gravitaional_acceleration
        self._setup_standard_bases()
        
        
    deltaT = UnitsProperty("deltaT", u.K, latex=r"$\Delta T$")
    depth = UnitsProperty("depth", u.m, latex=r"$D$")
    aspect = UnitsProperty("aspect", u.dimensionless_unscaled, latex=r"$a$")
    
    kinematic_viscosity = UnitsProperty("kinematic viscosity", u.m**2.0 / u.s, latex=r"$\kappa$")
    thermal_diffusivity = UnitsProperty("thermal diffusivity", u.m**2.0 / u.s, latex=r"$\nu$")
    thermal_expansion = UnitsProperty("thermal expansion", 1.0 / u.K, latex=r"$\alpha$")
    gravitaional_acceleration = UnitsProperty("gravitational acceleration", u.m / u.s**2.0, latex=r"$g$")
    
    @ComputedUnitsProperty
    def width(self):
        """The box width."""
        return self.aspect * self.depth
    
    @ComputedUnitsProperty
    def Prandtl(self):
        """The Prandtl number."""
        return (self.thermal_diffusivity / self.kinematic_viscosity)
    
    @ComputedUnitsProperty
    def Reynolds(self):
        """The Reynolds number."""
        return (self.gravitaional_acceleration * self.thermal_expansion * self.deltaT * self.depth**3.0) / (self.thermal_diffusivity * self.kinematic_viscosity)
    
    @classmethod
    def get_parameter_list(cls):
        """Get a list of the parameters which can be changed/modified directly"""
        import inspect
        return inspect.getargspec(cls.__init__)[0][1:]
        
    
        