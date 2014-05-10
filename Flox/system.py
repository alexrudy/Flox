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
from .array import SpectralArrayProperty, ArrayEngine, ArrayProperty

from .process.packet import PacketInterface, Packet

@six.add_metaclass(abc.ABCMeta)
class System2D(PacketInterface, HasUnitsProperties):
    """An abstract 2D Fluid box, with some basic properties."""
    
    # nx = 0
    # nz = 0
    # nt = 0
    # it = 0
    _bases = {}
    _engine = None
    
    def __init__(self, nx, nz, nt, it=0, engine=ArrayEngine, dtype=np.float):
        super(System2D, self).__init__()
        self.nx = nx
        self.nz = nz
        self.nt = nt
        self.it = it
        self.dtype = dtype
        self._bases = {}
        self.engine = engine
        self.initialize_arrays()
        
    
    def __repr__(self):
        """Represent this object!"""
        try:
            Pr = self.Prandtl
            Ra = self.Rayleigh
            time = self.time
            return "<{0} with Ra={1.value} and Pr={2.value} at {3}>".format(self.__class__.__name__, Ra, Pr, time)
        except (NotImplementedError, IndexError):
            return super(System2D, self).__repr__()
    
    def infer_iteration(self):
        """Infer the iteration number from loaded data."""
        # TODO Ensure Time is sorted!
        self.it = self.nit
        
    @property
    def nit(self):
        """Get the total number of iterations."""
        return np.argmax(self.engine["Time"])
    
    @property
    def engine(self):
        """Get the engine property."""
        return self._engine
        
    @engine.setter
    def engine(self, engine):
        """Set the engine property."""
        if isinstance(engine, collections.Mapping):
            engine = configure_class(engine)
        elif isinstance(engine, six.text_type):
            engine = resolve(engine)()
        elif issubclass(engine, ArrayEngine):
            engine = engine()
        if not isinstance(engine, ArrayEngine):
            raise ValueError("Can't set an array engine to a non-subclass of {0}: {1!r}".format(ArrayEngine, engine))
        self._engine = engine
    
    @abc.abstractproperty
    def Prandtl(self):
        """Prandtl number."""
        raise NotImplementedError()
        
    
    @abc.abstractproperty
    def Rayleigh(self):
        """Rayleigh number."""
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
        """The aspect ratio of the box."""
        raise NotImplementedError()
    
    @ComputedUnitsProperty
    def width(self):
        """The box width."""
        return self.aspect * self.depth
        
    @ComputedUnitsProperty
    def dx(self):
        """x grid spacing."""
        return self.width / self.nx
    
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
        return ['nz', 'nx', 'nt', 'engine']
        
    def get_packet_list(self):
        """Return the packet list."""
        return self._list_attributes(ArrayProperty)
        
    def _setup_standard_bases(self):
        """Set the standard, non-dimensional bases"""
        temperature_unit = u.def_unit("Box-delta-T", self.deltaT)
        length_unit = u.def_unit("Box-D", self.depth)
        viscosity_unit = u.def_unit("kappa", self.primary_viscosity)
        time_unit = length_unit**2 / viscosity_unit
        self._bases['nondimensional'] = { unit.physical_type:unit for unit in [temperature_unit, length_unit, viscosity_unit, time_unit] }
        
        temperature_unit = self.deltaT.unit
        length_unit = self.depth.unit
        time_unit = type(self).Time.unit(self)
        viscosity_unit = length_unit**2 / time_unit
        self._bases['standard'] = { unit.physical_type:unit for unit in [temperature_unit, length_unit, time_unit, viscosity_unit] }
        
    def nondimensionalize(self, quantity):
        """Nondimensionalize a given quantity for use somewhere."""
        return recompose(quantity, list(self._bases['nondimensional'].values()))
        
    def dimensionalize(self, quantity):
        """Dimensionalize a given quantity or value."""
        return recompose(quantity, list(self._bases['standard'].values()))
    
    def dimesnional_array(self, name):
        """Return a dimensionalized array"""
        array_desc = getattr(type(self), name)
        array_dunit = array_desc.unit(self)
        array_ndunit = self.nondimensional_unit(array_dunit)
        return (array_desc.get(self) * array_ndunit).to(array_dunit)
        
    def dimensional_full_array(self, name):
        """Return a dimensionalized array"""
        array_desc = getattr(type(self), name)
        array_dunit = array_desc.unit(self)
        array_ndunit = self.nondimensional_unit(array_dunit)
        return (self.engine[name] * array_ndunit).to(array_dunit)
        
    def nondimensional_unit(self, unit):
        """Create a nondimensional unit for this system."""
        return recompose_unit(unit, set(self._bases['nondimensional'].values()))
        
    def transformed_array(self, name, perturbed=False):
        """Return a transformed array for a given name"""
        array_desc = getattr(type(self), name)
        array_dunit = array_desc.unit(self)
        array_ndunit = self.nondimensional_unit(array_dunit)
        if perturbed:
            return (array_desc.p_itransform(self) * array_ndunit).to(array_dunit)
        return (array_desc.itransform(self) * array_ndunit).to(array_dunit)
        
    def diagnostic_string(self, z=None, n=None):
        """A longer diagnostic string."""
        if n is None:
            n = [0, 1, 2, -3, -2, -1]
            ns = np.arange(self.nx)[n]
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
        
    def initialize_arrays(self):
        """Initialize data arrays"""
        for attr_name in self.list_arrays():
            getattr(type(self), attr_name).allocate(self)
            
    def read_packet(self, packet):
        """Read the packet."""
        self.it += 1
        super(System2D, self).read_packet(packet)
    
    @ComputedUnitsProperty
    def time(self):
        """The current time of this simulation"""
        return self.dimensionalize(self.Time * self.nondimensional_unit(type(self).Time.unit(self)))
    
    Time = ArrayProperty("Time", u.s, shape=tuple(), latex=r"$t$")
    Temperature = SpectralArrayProperty("Temperature", u.K, func=np.cos, shape=('nz','nx'), latex=r"$T$")
    Vorticity = SpectralArrayProperty("Vorticity", 1.0 / u.s, func=np.sin, shape=('nz','nx'), latex=r"$\omega$")
    StreamFunction = SpectralArrayProperty("StreamFunction", u.m**2 / u.s, func=np.sin, shape=('nz','nx'), latex=r"$\psi$")
    dTemperature = SpectralArrayProperty("dTemperature", u.K / u.s, func=np.cos, shape=('nz','nx'), latex=r"$\frac{d T}{dt}$")
    dVorticity = SpectralArrayProperty("dVorticity", 1.0 / u.s / u.s, func=np.sin, shape=('nz','nx'), latex=r"$\frac{d \omega}{dt}$")

class NDSystem2D(System2D):
    """A primarily non-dimensional 2D system."""
    def __init__(self,
        deltaT=0, depth=0, aspect=0, Prandtl=0, Rayleigh=0, kinematic_viscosity=0, **kwargs):
        super(NDSystem2D, self).__init__(**kwargs)
        self.deltaT = deltaT
        self.depth = depth
        self.aspect = aspect
        self.Prandtl = Prandtl
        self.Rayleigh = Rayleigh
        self.kinematic_viscosity = kinematic_viscosity
        self._setup_standard_bases()
        
        
    deltaT = UnitsProperty("deltaT", u.K, latex=r"$\Delta T$")
    depth = UnitsProperty("depth", u.m, latex=r"$D$")
    aspect = UnitsProperty("aspect", u.dimensionless_unscaled, latex=r"$a$")
    
    kinematic_viscosity = UnitsProperty("kinematic viscosity", u.m**2.0 / u.s, latex=r"$\kappa$")
    Prandtl = UnitsProperty("Prandtl", u.dimensionless_unscaled, latex=r"$Pr$")
    Rayleigh = UnitsProperty("Rayleigh", u.dimensionless_unscaled, latex=r"$Re$")
    
    @ComputedUnitsProperty
    def thermal_diffusivity(self):
        """Thermal Diffusivity"""
        return self.kinematic_viscosity * self.Prandtl
        
    @classmethod
    def get_parameter_list(cls):
        """Get a list of the parameters which can be changed/modified directly"""
        import inspect
        return inspect.getargspec(cls.__init__)[0][1:] + super(NDSystem2D, cls).get_parameter_list()
    

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
        kinematic_viscosity=0, thermal_diffusivity=0, thermal_expansion=0, gravitaional_acceleration=0,
        **kwargs):
        super(PhysicalSystem2D, self).__init__(**kwargs)
        
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
    def Rayleigh(self):
        """The Rayleigh number."""
        return (self.gravitaional_acceleration * self.thermal_expansion * self.deltaT * self.depth**3.0) / (self.thermal_diffusivity * self.kinematic_viscosity)
    
    @classmethod
    def get_parameter_list(cls):
        """Get a list of the parameters which can be changed/modified directly"""
        import inspect
        return inspect.getargspec(cls.__init__)[0][1:] + super(PhysicalSystem2D, cls).get_parameter_list()        
    
        