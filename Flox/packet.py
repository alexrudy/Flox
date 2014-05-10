# -*- coding: utf-8 -*-
# 
#  packet.py
#  flox
#  
#  Created by Alexander Rudy on 2014-05-09.
#  Copyright 2014 University of California. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import numpy as np
import abc
import six

@six.add_metaclass(abc.ABCMeta)
class Packet2D(dict):
    """A 2D packet, containing the data required for the setup of an evolver."""
    
    def __init__(self):
        super(Packet2D, self).__init__()
        
    def __setitem__(self, key, value):
        """Set the item """
        super(Packet2D, self).__setitem__(key, np.asanyarray(value))
        
