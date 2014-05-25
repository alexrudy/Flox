# -*- coding: utf-8 -*-
# 
#  system.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-24.
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

from ..engine.descriptors import SpectralArrayProperty, ArrayProperty
from ..system import System2D

class HydroSystem(System2D):
    """A hydrodynamical system."""
    
    def __repr__(self):
        """Represent this system."""
        try:
            return "<{0} Pr={Prandtl.value:.1g} Ra={Rayleigh.value:.1g}>".format(self.__class__.__name__, Prandtl=self.Prandtl, Rayleigh=self.Rayleigh)
        except NotImplementedError:
            return super(System2D, self).__repr__()
    
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
    
    def setup_bases(self):
        """Set the standard, non-dimensional bases"""
        temperature_unit = u.def_unit("Box-delta-T", self.deltaT)
        length_unit = u.def_unit("Box-D", self.depth)
        viscosity_unit = u.def_unit("kappa", self.primary_viscosity)
        time_unit = length_unit**2 / viscosity_unit
        self.add_bases("nondimensional", set([temperature_unit, length_unit, viscosity_unit, time_unit]))
        
        temperature_unit = self.deltaT.unit
        length_unit = self.depth.unit
        time_unit = type(self).Time.unit
        viscosity_unit = length_unit**2 / time_unit
        self.add_bases("standard", set([temperature_unit, length_unit, time_unit, viscosity_unit]))
        
    
    Temperature = SpectralArrayProperty("Temperature", u.K, func=np.cos, shape=('nz','nx'), latex=r"$T$")
    Vorticity = SpectralArrayProperty("Vorticity", 1.0 / u.s, func=np.sin, shape=('nz','nx'), latex=r"$\omega$")
    Stream = SpectralArrayProperty("Stream", u.m**2 / u.s, func=np.sin, shape=('nz','nx'), latex=r"$\psi$")
    dTemperature = SpectralArrayProperty("dTemperature", u.K / u.s, func=np.cos, shape=('nz','nx'), latex=r"$\frac{d T}{dt}$")
    dVorticity = SpectralArrayProperty("dVorticity", 1.0 / u.s / u.s, func=np.sin, shape=('nz','nx'), latex=r"$\frac{d \omega}{dt}$")
    


class NDSystem2D(HydroSystem):
    """A primarily non-dimensional 2D system."""
    def __init__(self,
        deltaT=0, depth=0, aspect=0, Prandtl=0, Rayleigh=0, kinematic_viscosity=0, **kwargs):
        self.deltaT = deltaT
        self.depth = depth
        self.aspect = aspect
        self.Prandtl = Prandtl
        self.Rayleigh = Rayleigh
        self.kinematic_viscosity = kinematic_viscosity
        super(NDSystem2D, self).__init__(**kwargs)
        
        
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
        properties = list(cls._list_attributes(UnitsProperty, strict=True))
        return properties + super(NDSystem2D, cls).get_parameter_list()
    

class PhysicalSystem2D(HydroSystem):
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
        
        # Box Physical Variables
        self.deltaT = deltaT
        self.depth = depth
        self.aspect = aspect
        
        # Fluid Variables
        self.kinematic_viscosity = kinematic_viscosity
        self.thermal_diffusivity = thermal_diffusivity
        self.thermal_expansion = thermal_expansion
        self.gravitaional_acceleration = gravitaional_acceleration
        super(PhysicalSystem2D, self).__init__(**kwargs)
        
        
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
    
