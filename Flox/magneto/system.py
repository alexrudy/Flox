# -*- coding: utf-8 -*-
# 
#  system.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-19.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import six
import numpy as np
import astropy.units as u
from pyshell.astron.units import UnitsProperty, HasUnitsProperties, recompose, ComputedUnitsProperty, recompose_unit
from pyshell.util import setup_kwargs, configure_class, resolve

from ...array import SpectralArrayProperty, ArrayEngine, ArrayProperty
from ...system import NDSystem2D

class MagnetoSystem(NDSystem2D):
    """docstring for MagnetoSystem"""
    def __init__(self, Roberts=0, Chandrasekhar=0, **kwargs):
        super(MagnetoSystem, self).__init__()
        self.Roberts = Roberts
        self.Chandrasekhar = Chandrasekhar
    
    VectorPotential = SpectralArrayProperty("VectorPotential", u.V * u.s / u.m, func=np.sin, shape=('nz','nx'), latex=r"$A$")
    CurrentDensity = SpectralArrayProperty("CurrentDensity", u.A / u.m**2, func=np.sin, shape=('nz','nx'), latex=r"$J$")
    
    Roberts = UnitsProperty("Roberts", u.dimensionless_unscaled, latex=r"$q$")
    Chandrasekhar = UnitsProperty("Chandrasekhar", u.dimensionless_unscaled, latex=r"$Q$")
    
    @classmethod
    def get_parameter_list(cls):
        """Get a list of the parameters which can be changed/modified directly"""
        import inspect
        return inspect.getargspec(cls.__init__)[0][1:] + super(NDSystem2D, cls).get_parameter_list()