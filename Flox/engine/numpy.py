# -*- coding: utf-8 -*-
# 
#  numpy.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-25.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import numpy as np
from .core import TimekeepingEngine, ArrayEngine

class NumpyArrayEngine(dict, TimekeepingEngine):
    """A numpy-based array engine"""
    
    def __init__(self, system, length=None, dtype=np.float):
        """Initialize this object."""
        TimekeepingEngine.__init__(self, system)
        super(NumpyArrayEngine, self).__init__()
        self._dtype = dtype
        self._length = length
        self._iterations = 0
        
    @property
    def iterations(self):
        """Number of iterations available."""
        return self._iterations
        
    @iterations.setter
    def iterations(self, value):
        """Set the iterations"""
        if value < self.length:
            self._iterations = value
        
    @property
    def length(self):
        """Maximum object length."""
        if self._length is None:
            raise ValueError("Length has not been set.")
        return self._length
        
    @length.setter
    def length(self, value):
        """Maximum object length."""
        if self._length is None:
            self._length = value
        else:
            raise ValueError("Can't adjust length.")
            
    def initialize_arrays(self, system):
        """Set the array length before initializing."""
        self._iterations = 0
        return super(NumpyArrayEngine, self).initialize_arrays(system)
    
    @property
    def dtype(self):
        """dtype"""
        return np.dtype(self._dtype).str
        
    @classmethod
    def get_parameter_list(cls):
        """Get the parameter list pairs."""
        return ['_dtype'] + super(NumpyArrayEngine, cls).get_parameter_list()
        
    def allocate(self, name, shape):
        """Allocate arrays with empty numpy arrays"""
        self[name] = np.zeros(shape + tuple([self.length]), dtype=self._dtype)
        
        
class NumpyFrameEngine(dict, ArrayEngine):
    """A numpy-based array engine which only holds a single frame."""
    
    def __init__(self, system, dtype=np.float):
        """Initialize this object."""
        ArrayEngine.__init__(self, system)
        super(NumpyFrameEngine, self).__init__()
        self._dtype = dtype
        
    def allocate(self, name, shape):
        """Allocate arrays with empty numpy arrays"""
        self[name] = np.zeros(shape, dtype=self._dtype)
        
    @classmethod
    def get_parameter_list(cls):
        """Get the parameter list pairs."""
        return ['dtype']