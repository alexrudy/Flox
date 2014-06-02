# -*- coding: utf-8 -*-
# 
#  descriptors.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-25.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import six

from pyshell.util import descriptor__get__

from ..transform import spectral_transform

def EvolverProperty(propname):
    """A simulation property decorator."""
    if not isinstance(propname, six.text_type):
        raise ValueError("Must provide a property name.")
    
    def decorator(f):
        """Decorate the function."""
        f.propname = propname
        return f
    
    return decorator
    
    

class ArrayValue(object):
    """An array value instance"""
    def __init__(self, engine, system, attribute):
        super(ArrayValue, self).__init__()
        self._engine = engine
        self._attr = attribute
        self._system = system
        
    def __getattr__(self, attribute):
        """Return an inverted view of this array."""
        if attribute in self._engine._views:
            return self._engine._views[attribute].__getdata__(self._system, self._attr)
        raise AttributeError("{0}: No attribute '{1}' on Array '{2}'".format(self._system.__class__.__name__, attribute, self._attr))
            
    
    def __setattr__(self, attribute, value):
        """Set the attribute."""
        if attribute[:1] == "_":
            return super(ArrayValue, self).__setattr__(attribute, value)
        if attribute in self._engine._views:
            return self._engine._views[attribute].__setdata__(self._system, self._attr, value)
        else:
            raise AttributeError("{0}: Can't set attribute '{1}' on Array '{2}'".format(self._system.__class__.__name__, attribute, self._attr))
    
    def __repr__(self):
        """A string representation of this array value."""
        return "<{0}.{1} view>".format(self._engine.type.__name__, self._attr)
    

class ArrayProperty(object):
    """Custom subclass used to assist with Array allocation."""
    def __init__(self, name, unit, shape=tuple(), engine='engine', latex=''):
        super(ArrayProperty, self).__init__()
        self._shape = shape
        self._engine = engine
        self._attr = name
        self._unit = unit
        self._latex = latex
    
    @property
    def name(self):
        return self._attr
    
    @property
    def unit(self):
        return self._unit
        
    @property
    def latex(self):
        return self._latex
    
    def shape(self, obj):
        """Retrieve inexplicit shapes"""
        shape = tuple([self._get_shape_part(obj, part) for part in self._shape])
        if shape == tuple((0,)):
            raise ValueError("Got unindexable array: {!r}".format(shape))
        return getattr(obj, self._engine).shape(obj, shape)
        
    def _get_shape_part(self, obj, part):
        """Get a part of the shape tuple"""
        if isinstance(part, int):
            return part
        elif isinstance(part, six.string_types):
            return getattr(obj, part)
        else:
            raise ValueError("Implicit shape '{!r}' is not a valid type: {!r}".format(part, [int, six.string_types]))
    
    def __set__(self, obj, value):
        """Short out the setter so that it doesn't use units, but uses the allocated array space."""
        raise AttributeError("Can't directly set an ArrayProperty.")
        
    @descriptor__get__
    def __get__(self, obj, objtype):
        """Get this object."""
        return ArrayValue(getattr(obj, self._engine), obj, self._attr)
        
class SpectralArrayProperty(ArrayProperty):
    """An array with spectral property support"""
    def __init__(self, name, unit, func, **kwargs):
        super(SpectralArrayProperty, self).__init__(name, unit, **kwargs)
        self._func = func
        
    def itransform(self, obj, perturbed=False):
        """Perform the inverse transform."""
        return spectral_transform(self._func, self.__get__(obj, type(obj)).raw, obj.nx, obj.aspect.value, perturbed)
        
