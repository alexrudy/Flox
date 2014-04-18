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
read_parameter_filename = os.path.join(directory_stem, filename_stem+"-read.yml")

def setup():
    """Test setup!"""
    remove(make_parameter_filename)
    
def remove(filename):
    """Remove a filename, but don't fail if it isn't there."""
    try:
        os.remove(filename)
    except OSError, e:
        pass
    

def create_system():
    """Create a system object."""
    from Flox.system import System2D
    from astropy.constants import g0
    my_system = System2D(
        time = 0,
        nx = 100,
        nz = 100,
        deltaT = 10,
        depth = 1,
        aspect = 1,
        kinematic_viscosity = 1,
        thermal_diffusivity = 1,
        thermal_expansion = 1,
        gravitaional_acceleration = g0,
    )
    return my_system
    
def test_make_parameter_file():
    """Test creating a parameter file."""
    system = create_system()
    system.to_params().save(make_parameter_filename)
    
def test_read_parameter_file():
    """Test reading a parameter file."""
    from Flox.system import System2D
    from Flox.input import FloxConfiguration
    System2D.from_params(FloxConfiguration.fromfile(read_parameter_filename))
    