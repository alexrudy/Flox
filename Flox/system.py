# -*- coding: utf-8 -*-
# 
#  box.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-07.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import numpy
import inspect
import astropy.units as u
from pyshell.astron.units import HasNonDimensonals, HasInitialValues, UnitsProperty, ComputedUnitsProperty
from pyshell.util import setup_kwargs

from .input import FloxConfiguration

class System2D(HasNonDimensonals, HasInitialValues):
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
        time=0, nx=0, nz=0,
        kinematic_viscosity=0, thermal_diffusivity=0, thermal_expansion=0, gravitaional_acceleration=0):
        super(System2D, self).__init__()
        
        # Simulation State Variables
        self.time = time
        self.nx = nx
        self.nz = nz
        
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
        
    nx = 0
    nz = 0
    
    time = UnitsProperty("time", u.s, latex=r"$t$")
    
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
        
    def _setup_standard_bases(self):
        """Set the standard, non-dimensional bases"""
        temperature_unit = u.def_unit("Box ∆T", self.deltaT)
        length_unit = u.def_unit("Box D", self.depth)
        time_unit = u.def_unit("Box D²/κ", self.depth**2 / self.kinematic_viscosity)
        self._nondimensional_bases = set([temperature_unit, length_unit, time_unit])
        
    @classmethod
    def get_parameter_list(cls):
        """Get a list of the parameters which can be changed/modified directly"""
        return inspect.getargspec(cls.__init__)[0][1:]
        
    @classmethod
    def from_param_file(cls, filename):
        """Load a box from a parameter file."""
        parameters = FloxConfiguration.fromfile(filename)
        return cls(**setup_kwargs(cls.__init__,parameters))
        
    def to_param_file(self, filename):
        """Create a parameter file."""
        parameters = FloxConfiguration()
        argnames = self.get_parameter_list()
        for argname in argnames:
            parameters[argname] = getattr(self, argname)
        parameters.save(filename)
        
    
        