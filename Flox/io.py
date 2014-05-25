# -*- coding: utf-8 -*-
# 
#  io.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-04-20.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import os, os.path
import astropy.units as u
import numpy as np
import abc
import six
import h5py
from pyshell.util import resolve

@six.add_metaclass(abc.ABCMeta)
class GridWriter(object):
    """A grid writing object."""
    def __init__(self, filename):
        super(GridWriter, self).__init__()
        self.filename = os.path.normpath(os.path.expanduser(filename))
        
    @abc.abstractmethod
    def write(self, data, name=""):
        """Write this data object to a file."""
        pass
        
    def write_frame(self, data, name=""):
        """Write a single frame."""
        raise NotImplementedError("Not Supported!")
        
    @abc.abstractmethod
    def read(self, data, name=""):
        """Read to a data class."""
        pass
        
class WriterInterface(object):
    """A mixin for classes which can use a writer interface."""
    
    @staticmethod
    def _get_writer(writer, filename):
        """Get a writer class for the appropriate type."""
        return resolve(writer)(filename)
    
    def write(self, writer, filename, dataname):
        """Get the writer and write!"""
        self._get_writer(writer, filename).write(self, dataname)
    
    def read(self, writer, filename, dataname):
        """Read based on a reader class."""
        self._get_writer(writer, filename).read(self, dataname)

class HDF5Writer(GridWriter):
    """Write an HDF5 file."""
    
    def write(self, data, name=""):
        """Write the data to a file."""
        with h5py.File(self.filename) as file_context:
            group = file_context.require_group(name)
            for array_name in data.list_arrays():
                self.create_array(group, data, array_name)
            for param, value in data.to_params().items():
                if isinstance(value, u.Quantity):
                    group.attrs[param] = value.value
                elif isinstance(value, six.string_types):
                    group.attrs[param] = value
            group.attrs['iterations'] = data.engine.iterations
    
    def create_array(self, group, data, array_name):
        """Write the array object"""
        array_obj = getattr(type(data), array_name)
        array_data = data.engine[array_name]
        try:
            dataset = group.require_dataset(array_name, array_data.shape, dtype=array_data.dtype)
        except TypeError as e:
            del group[array_name]
            dataset = group.create_dataset(array_name, array_data.shape, dtype=array_data.dtype)
        dataset[...] = array_data
        dataset.attrs['name'] = six.text_type(array_obj.name)
        dataset.attrs['unit'] = six.text_type(array_obj.unit)
        dataset.attrs['latex'] = six.text_type(array_obj.latex)
        
    def read(self, data, name=""):
        """Read from a data file."""
        with h5py.File(self.filename) as file_context:
            group = file_context.require_group(name)
            for array_name in data.engine.get_data_list():
                self.read_array(group, data, array_name)
            for param in data.get_parameter_list():
                if param in group.attrs:
                    setattr(data, param, group.attrs[param])
            data.engine.iterations = group.attrs.get('iterations',np.argmax(group['Time']))
    
    def read_array(self, group, data, array_name):
        """Read the dataset."""
        dataset = group[array_name]
        data.engine[array_name] = dataset[...]
        
        
        