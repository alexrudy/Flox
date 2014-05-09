# -*- coding: utf-8 -*-
# 
#  test_system.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-18.
#  Copyright 2014 University of California. All rights reserved.
# 
"""
Integration testing for the System objects.
"""

from __future__ import (absolute_import, unicode_literals, division, print_function)

import nose.tools as nt
import os, os.path

filename_stem = os.path.splitext(os.path.basename(__file__))[0]
directory_stem = os.path.dirname(__file__)
make_parameter_filename = os.path.join(directory_stem, filename_stem+"-make.yml")
make_data_filename = os.path.join(directory_stem, filename_stem+"-make.hdf5")
read_parameter_filename = os.path.join(directory_stem, filename_stem+"-read.yml")
read_data_filename = os.path.join(directory_stem, filename_stem+"-read.hdf5")
pickle_data_filename = os.path.join(directory_stem, filename_stem+"-make.pkl")

def setup():
    """Test setup!"""
    remove(make_parameter_filename)
    remove(make_data_filename)
    remove(pickle_data_filename)
    
def remove(filename):
    """Remove a filename, but don't fail if it isn't there."""
    try:
        os.remove(filename)
    except OSError as e:
        pass
    

def create_system():
    """Create a system object."""
    from Flox.system import PhysicalSystem2D
    from astropy.constants import g0
    my_system = PhysicalSystem2D(
        nx = 100,
        nz = 100,
        nt = 100,
        deltaT = 10,
        depth = 1,
        aspect = 1,
        kinematic_viscosity = 1,
        thermal_diffusivity = 1,
        thermal_expansion = 1,
        gravitaional_acceleration = g0,
        engine = 'Flox.array.NumpyArrayEngine',
    )
    return my_system
    
def test_make_parameter_file():
    """Test creating a parameter file."""
    system = create_system()
    system.to_params().save(make_parameter_filename)
    
def test_make_data_file():
    """Write data to a file."""
    from Flox.io import HDF5Writer
    system = create_system()
    writer = HDF5Writer(make_data_filename)
    writer.write(system,'main')
    
def test_read_data_file():
    """Read a data file."""
    from Flox.io import HDF5Writer
    from astropy.constants import g0
    system = create_system()
    writer = HDF5Writer(read_data_filename)
    writer.read(system,'main')
    assert system.Rayleigh == (10 * g0).value
    
def test_make_ND_parameter_file():
    """Non-dimensional Parameter File"""
    from Flox.system import NDSystem2D
    my_system = NDSystem2D(
        nx = 100,
        nz = 100,
        nt = 100,
        deltaT = 10,
        depth = 1,
        aspect = 1,
        kinematic_viscosity = 1,
        Prandtl = 1.0,
        Rayleigh = 1.0,
        engine = 'Flox.array.NumpyArrayEngine',
        )
    my_system.to_params().save(make_parameter_filename)
    
def test_read_parameter_file():
    """Test reading a parameter file."""
    from Flox.system import PhysicalSystem2D
    from Flox.input import FloxConfiguration
    PhysicalSystem2D.from_params(FloxConfiguration.fromfile(read_parameter_filename))
    
def test_pickle_NDSystem():
    """Pickle an NDSystem"""
    from six.moves import cPickle as pickle
    from Flox.system import NDSystem2D
    my_system = NDSystem2D(
        nx = 100,
        nz = 100,
        nt = 100,
        deltaT = 10,
        depth = 1,
        aspect = 1,
        kinematic_viscosity = 1,
        Prandtl = 1.0,
        Rayleigh = 1.0,
        engine = 'Flox.array.NumpyArrayEngine',
        )
    
    with open(pickle_data_filename, 'wb') as f:
        pickle.dump(my_system, f)
    
    