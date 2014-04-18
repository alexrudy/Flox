# -*- coding: utf-8 -*-
# 
#  input.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-07.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from pyshell.yaml import PyshellLoader, PyshellDumper
from pyshell.config import DottedConfiguration
from pyshell.astron.yaml_tools import astropy_quantity_yaml_factory
import astropy.constants
import astropy.units

class FloxDumper(PyshellDumper):
    """A YAML Dumper which permits quantities!"""
    
    def __init__(self, *args, **kwargs):
        """Initialize the dumper"""
        super(FloxDumper, self).__init__(*args, **kwargs)
        self.add_representer(astropy.constants.Constant, type(self).constant_representer)
    
    def constant_representer(self, data):
        """Represent an Astropy Constant"""
        return self.represent_scalar("!constant", "{0:s}".format(data.abbrev))
        

class FloxLoader(PyshellLoader):
    """A YAML loader which permits quantities!"""
    
    def __init__(self, *args, **kwargs):
        """Initialize the dumper"""
        super(FloxLoader, self).__init__(*args, **kwargs)
        self.add_constructor("!constant", type(self).constant_constructor)
        
    def constant_constructor(self, node):
        """Construct an Astropy Constant"""
        scalar_value = self.construct_scalar(node).strip()
        return getattr(astropy.constants, scalar_value)

astropy_quantity_yaml_factory(astropy.units.Quantity, FloxLoader, FloxDumper)

class FloxConfiguration(DottedConfiguration):
    """A configuration object for Flox data."""
    
    _loader = FloxLoader
    _dumper = FloxDumper
    
