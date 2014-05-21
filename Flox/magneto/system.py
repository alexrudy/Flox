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

from ..array import SpectralArrayProperty, ArrayEngine, ArrayProperty
from ..system import NDSystem2D

class MagnetoSystem(NDSystem2D):
    """docstring for MagnetoSystem"""
    def __init__(self, Roberts=0, Chandrasekhar=0, B0=0, **kwargs):
        self.Roberts = Roberts
        self.Chandrasekhar = Chandrasekhar
        self.B0 = B0
        super(MagnetoSystem, self).__init__(**kwargs)
    
    VectorPotential = SpectralArrayProperty("VectorPotential", u.T * u.m, func=np.sin, shape=('nz','nx'), latex=r"$A$")
    CurrentDensity = SpectralArrayProperty("CurrentDensity", u.A / u.m**2, func=np.sin, shape=('nz','nx'), latex=r"$J$")
    
    Roberts = UnitsProperty("Roberts", u.dimensionless_unscaled, latex=r"$q$")
    Chandrasekhar = UnitsProperty("Chandrasekhar", u.dimensionless_unscaled, latex=r"$Q$")
    B0 = UnitsProperty("Magnetic field", u.T, latex=r"$B_{0}$")
    
    def _setup_standard_bases(self):
        """Setup the standard bases."""
        super(MagnetoSystem, self)._setup_standard_bases()
        magnetic_unit = u.def_unit("B0", self.B0)
        mass_unit = magnetic_unit * u.A * u.s**2
        self._bases["standard"][u.T.physical_type] = u.T
        self._bases["standard"][u.kg.physical_type] = u.kg
        self._bases["standard"][u.A.physical_type] = u.A
        self._bases["nondimensional"][magnetic_unit.physical_type] = magnetic_unit
        self._bases["nondimensional"][mass_unit.physical_type] = mass_unit
        self._bases["nondimensional"][u.A.physical_type] = u.A
    
    def __repr__(self):
        """Represent this object!"""
        try:
            Pr = self.Prandtl
            Ra = self.Rayleigh
            Q = self.Chandrasekhar
            q = self.Roberts
            time = self.time
            return "<{0} with Ra={Ra.value} Pr={Pr.value} Q={Q.value} q={q.value} at {time}>".format(self.__class__.__name__, Ra=Ra, Pr=Pr, Q=Q, q=q, time=time)
        except (NotImplementedError, IndexError, KeyError):
            return super(MagnetoSystem, self).__repr__()