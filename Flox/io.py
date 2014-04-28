# -*- coding: utf-8 -*-
# 
#  io.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-20.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import astropy.units as u
import numpy as np
import abc
import six
import h5py

@six.add_metaclass(abc.ABCMeta)
class GridWriter(object):
    """A grid writing object."""
    def __init__(self, filename):
        super(GridWriter, self).__init__()
        self.filename = filename
        
    @abc.abstractmethod
    def write(self, data, name=""):
        """Write this data object to a file."""
        pass
        
    @abc.abstractmethod
    def read(self, data_cls, name=""):
        """Read to a data class."""
        pass

class HDF5Writer(GridWriter):
    """Write an HDF5 file."""
    
    def write(self, data, name=""):
        """Write the data to a file."""
        with h5py.File(self.filename) as file_context:
            group = file_context.require_group(name)
            for array_name in data.list_arrays():
                self.create_array(group, data, array_name)
    
    def create_array(self, group, data, array_name):
        """Write the array object"""
        array_obj = getattr(type(data), array_name)
        array_data = getattr(data, array_name)
        dataset = group.require_dataset(array_name, array_data.shape, dtype=array_data.dtype)
        dataset[...] = array_data
        dataset.attrs['name'] = six.text_type(array_obj.name)
        dataset.attrs['unit'] = six.text_type(array_obj.unit(data))
        dataset.attrs['latex'] = array_obj.latex
        
    def read(self, data, name=""):
        """Read from a data file."""
        with h5py.File(self.filename) as file_context:
            group = file_context.require_group(name)
            for array_name in group.keys():
                self.read_array(group, data, array_name)
        data.infer_iteration()
    
    def read_array(self, group, data, array_name):
        """Read the dataset."""
        array_obj = getattr(type(data), array_name)
        dataset = group[array_name]
        array_data = dataset[...] # * u.Unit(dataset.attrs['unit'])
        setattr(data, array_name, array_data)
        
        
        