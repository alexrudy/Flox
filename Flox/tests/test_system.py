# -*- coding: utf-8 -*-
# 
#  test_system.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-05.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

from Flox.system import NDSystem2D, PhysicalSystem2D
from Flox.magneto.system import MagnetoSystem
import astropy.units as u

system_args = dict(nx = 100, nz = 100, nt = 100, deltaT = 10, depth = 1, aspect = 1, engine = 'Flox.array.NumpyArrayEngine',)

system_classes = [
    (NDSystem2D, dict(deltaT=1, depth=1, aspect=1, Prandtl=1, Rayleigh=1, kinematic_viscosity=1)), 
    (PhysicalSystem2D, dict(deltaT=1, depth=1, aspect=1, kinematic_viscosity=1, thermal_diffusivity=1, thermal_expansion=1, gravitaional_acceleration=1)),
    (MagnetoSystem, dict(Roberts=1, Chandrasekhar=1, B0=1, deltaT=1, depth=1, aspect=1, Prandtl=1, Rayleigh=1, kinematic_viscosity=1))
    ]

def system_from_args(klass, args, extras):
    """Create a system"""
    args.update(extras)
    return klass(**args)
    

def test_system_repr(system):
    """Represent system"""
    repr(system)
    
def check_quantity(obj, name, unit):
    """Check units."""
    if unit is not None:
        assert isinstance(getattr(obj, name), u.Quantity)
        assert getattr(obj, name).unit.is_equivalent(u.Unit(unit))
    else:
        assert not isinstance(getattr(obj, name), u.Quantity)
    
system_units = {
    'npa' : 1,
    'primary_viscosity': 'St',
    'thermal_diffusivity' : 'St',
    'kinematic_viscosity' : 'St',
    'dx' : 'm',
    'width' : 'm',
    'aspect': 1,
    'dz' : 'm',
    'time' : 's',
    'depth' : 'm',
    'deltaT' : 'K',
    'Rayleigh' : 1,
    'Prandtl' : 1,
    'gravitaional_acceleration' : 'm / s**2',
    'thermal_expansion' : '1 / K',
    'B0' : 'T',
    'Roberts' : 1,
    'Chandrasekhar' : 1
}

def test_system_quantities(system):
    """Check for quantities."""
    for attribute in system.get_attribute_list():
        check_quantity(system, attribute, system_units[attribute])
    
def test_system_pickle(system):
    """Pickling."""
    import pickle
    newsys = pickle.loads(pickle.dumps(system))
