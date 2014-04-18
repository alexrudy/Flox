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

import astropy.units as u
from pyshell.astron.units import HasNonDimensonals, HasInitialValues, UnitsProperty, ComputedUnitsProperty

class Box2D(HasNonDimensonals, HasInitialValues):
    """A 2D Fluid box, with some basic properties.
    
    :param deltaT: The temperature change between the bottom and top of the box.
    :param depth: The depth of the box.
    :param aspect: The aspect ratio of the box.
    :param kinematic_viscosity: The kinematic viscosity of the box.
    :param thermal_diffusivity: The thermal diffusivity of the box.
    :param thermal_expansion: The coefficient of thermal expansion.
    :param gravitaional_acceleration: The rate of gravitational acceleration.
    
    """
    def __init__(self, deltaT, depth, aspect,
        time, nx, nz
        kinematic_viscosity, thermal_diffusivity, thermal_expansion, gravitaional_acceleration):
        super(Box, self).__init__()
        
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
        
    nx = 0
    nz = 0
    
    time = UnitsProperty("time", u.s, latex=r"$t$")
    
    deltaT = UnitsProperty("deltaT", u.K, latex=r"$\Delta T$")
    depth = UnitsProperty("depth", u.m, latex=r"$D$")
    aspect = UnitsProperty("aspect", u.dimensionless_unscaled, latex=r"$a$")
    
    kinematic_viscosity = UnitsProperty("kinematic viscosity", u.m**2.0 / u.s, latex=r"$\kappa$")
    thermal_diffusivity = UnitsProperty("thermal diffusivity", u.m**2.0 / u.s, latex=r"$\nu$")
    thermal_expansion = UnitsProperty("thermal expansion", 1.0 / u.K, latex=r"$\alpha$")
    gravitaional_acceleration = UnitsProperty("gravitaional acceleration", u.m / u.s**2.0, latex=r"$g$")
    
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
    def from_param_file(cls, filename):
        """Load a box from a parameter file."""
        