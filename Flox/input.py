# -*- coding: utf-8 -*-
# 
#  input.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-07.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

from pyshell.yaml import PyshellLoader, PyshellDumper
from pyshell.config import DottedConfiguration
from pyshell.astron.yaml_tools import astropy_quantity_yaml_factory
import astropy.constants
import astropy.units

from .array import ArrayEngine, ArrayYAMLSupport


class FloxDumper(PyshellDumper):
    """A YAML Dumper which permits quantities!"""
    
    def __init__(self, *args, **kwargs):
        """Initialize the dumper"""
        super(FloxDumper, self).__init__(*args, **kwargs)
        self.add_representer(astropy.constants.Constant, type(self).constant_representer)
        self.add_representer(astropy.units.Quantity, type(self).astropy_representer)
        self.add_multi_representer(ArrayEngine, type(self).arrayengine_representer)
        self.add_multi_representer(ArrayYAMLSupport, type(self).arrayengine_representer)
    
    def constant_representer(self, data):
        """Represent an Astropy Constant"""
        return self.represent_scalar("!constant", "{0:s}".format(data.abbrev))
        
    def arrayengine_representer(self, data):
        """Represent an ArrayEngine object."""
        params = {}
        params['()'] = ".".join([data.__class__.__module__,data.__class__.__name__])
        for param in data.get_parameter_list():
            params[param] = getattr(data, param)
        return self.represent_mapping('tag:yaml.org,2002:map',params)
        
    def astropy_representer(self, data):
        if data.unit.decompose() == astropy.units.Unit(1):
            return self.represent_scalar('tag:yaml.org,2002:float', "{!s}".format(float(data.value)))
        return self.represent_scalar("!quantity", "{0.value} {0.unit:generic}".format(data))
        

class FloxLoader(PyshellLoader):
    """A YAML loader which permits quantities!"""
    
    def __init__(self, *args, **kwargs):
        """Initialize the dumper"""
        super(FloxLoader, self).__init__(*args, **kwargs)
        self.add_constructor("!constant", type(self).constant_constructor)
        self.add_constructor("!quantity", type(self).astropy_constructor)
        
    def constant_constructor(self, node):
        """Construct an Astropy Constant"""
        scalar_value = self.construct_scalar(node).strip()
        return getattr(astropy.constants, scalar_value)
        
    def astropy_constructor(self, node):
        scalar_value = self.construct_scalar(node)
        parts = scalar_value.split(" ")
        value = parts[0]
        unit = " ".join(parts[1:])
        return astropy.units.Quantity(float(value), unit=unit)

class FloxConfiguration(DottedConfiguration):
    """A configuration object for Flox data."""
    
    _loader = FloxLoader
    _dumper = FloxDumper
    
