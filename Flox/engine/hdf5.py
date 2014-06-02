# -*- coding: utf-8 -*-
# 
#  hdf5.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-06-02.
#  Copyright 2014 University of California. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import numpy as np
import h5py
from threading import Thread
import os, os.path
from .core import TimekeepingEngine, ArrayEngine

class HDF5ArrayEngine(h5py.File, TimekeepingEngine):
    """A numpy-based array engine"""
    
    def __init__(self, system, filename, dtype=np.float):
        """Initialize this object."""
        TimekeepingEngine.__init__(self, system)
        super(HDF5ArrayEngine, self).__init__(os.path.normpath(os.path.expanduser(filename)))
        self._dtype = dtype
        self._iterations = 0
        self._length = None
        self._threads = []
        
    def __del__(self):
        """Delete this object, closing threads."""
        for thread in self._threads:
            thread.join()
        
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
            
    @property
    def free(self):
        """Number of free-space iterations available."""
        return self.length
            
    def initialize_arrays(self, system):
        """Set the array length before initializing."""
        self._iterations = 0
        return super(HDF5ArrayEngine, self).initialize_arrays(system)
    
    @property
    def dtype(self):
        """dtype"""
        return np.dtype(self._dtype).str
        
    @classmethod
    def get_parameter_list(cls):
        """Get the parameter list pairs."""
        return ['_dtype'] + ['filename'] + super(HDF5ArrayEngine, cls).get_parameter_list()
        
    def allocate(self, name, shape):
        """Allocate arrays with empty numpy arrays"""
        try:
            self.require_dataset(name, shape + (self.length,), maxshape = shape + (None,), chunks=True, dtype=self.dtype)
        except TypeError:
            del self[name]
        self.require_dataset(name, shape + (self.length,), maxshape = shape + (None,),  chunks=True, dtype=self.dtype)
        
    def __setdata__(self, obj, name, value):
        """Engine caller to the underlying set method."""
        if self._iterations < obj.iteration:
            self._iterations = obj.iteration
        thread = Thread(target=self.write, args=(name, value, obj.iteration), daemon=True)
        thread.start()
        self._threads.append(thread)
    
    def write(self, name, value, i):
        """docstring for write"""
        if self[name].shape[-1] < (i + 1):
            self[name].resize(i+1, axis=len(self[name].shape)-1)
        self[name][...,i] = value
