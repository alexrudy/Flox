# -*- coding: utf-8 -*-
# 
#  input.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-07.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from pyshell.yaml import PyshellLoader, PyshellDumper
from pyshell.astron.yaml_tools import astropy_quantity_yaml_factory

class FloxLoader(PyshellLoader):
    """A YAML loader which permits quantities!"""
    pass
        

astropy_quantity_yaml_factory(u.Quantity, FloxLoader, None)