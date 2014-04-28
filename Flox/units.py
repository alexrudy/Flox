# -*- coding: utf-8 -*-
# 
#  units.py
#  pyshell
#  
#  Created by Alexander Rudy on 2013-12-02.
#  Copyright 2013 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

import numpy as np
import six
import abc
import warnings
import contextlib
from collections import OrderedDict

try:
    import astropy.units as u
except ImportError:
    raise ImportError("This module requires the 'astropy' module to function properly.")

from pyshell.util import descriptor__get__

u.du = u.dimensionless_unscaled
UNIT_MAX_DEPTH = 3
NON_DIMENSIONAL_FLAG = '_is_nondimensional'
INITIAL_VALUE_FLAG = '_is_initial'

def recompose(quantity, bases, scaled=False, warn_compositons=False, max_depth=UNIT_MAX_DEPTH):
    """Recompose a quantity in terms of the given bases.
    
    :param unit: The unit to recompose.
    :param bases: The set of units allowed in the final result.
    :param bool scaled: Whether to allow units with a scale or not.
    :param bool warn_compositons: Whether to warn when there were multiple compositions found.
    """
    result_unit = recompose_unit(quantity.unit, bases, scaled, warn_compositons, max_depth)
    result = quantity.to(result_unit)
    return result

def recompose_unit(unit, bases, scaled=False, warn_compositons=False, max_depth=UNIT_MAX_DEPTH):
    """Recompose a unit in terms of the provided bases.
    
    :param unit: The unit to recompose.
    :param bases: The set of units allowed in the final result.
    :param bool scaled: Whether to allow units with a scale or not.
    :param bool warn_compositons: Whether to warn when there were multiple compositions found.
    
    """
    composed = unit.compose(units=bases, max_depth=UNIT_MAX_DEPTH)
    if len(composed) != 1 and warn_compositons:
        warnings.warn("Multiple compositions are possible for {!r}: {!r}".format(unit,composed))
    result = composed[0]
    if result.scale == 1.0:
        return result
    elif scaled:
        return result
    else:
        unscaled = result / u.Unit(result.scale)
        return unscaled

class UnitsProperty(object):
    """A descriptor which enforces units."""
    def __init__(self, name, unit, latex=None, nonnegative=False, finite=False, readonly=False, scale=False, warn_for_unit_composition=False):
        super(UnitsProperty, self).__init__()
        self.name = name
        self.latex = name if latex is None else latex
        self._unit = u.Unit(unit) if unit is not None else None
        self._attr = '_{}_{}'.format(self.__class__.__name__, name.replace(" ", "_"))
        self._nn = nonnegative
        self._ff = finite
        self._readonly = readonly
        self._scale = scale
        self._warncompositon = warn_for_unit_composition
        
    def bases(self, obj):
        """Return the bases."""
        return getattr(obj, '_bases', None)
        
    def recompose(self, quantity, bases):
        """Recompose a unit into a new base set."""
        return recompose(quantity, bases, scaled=self._scale, warn_compositons=self._warncompositon)
        
    def unit(self, obj):
        """Get the recomposed unit."""
        return recompose_unit(self._unit, self.bases(obj), scaled=self._scale, warn_compositons=self._warncompositon)
        
    def __set__(self, obj, value):
        """Set this property's value"""
        if self._readonly:
            raise AttributeError("{} cannot set read-only attribute {}".format(obj, self.name))
        if self._ff and not np.isfinite(value).all():
            raise ValueError("{} must be finite!".format(self.name))
        if self._nn and not np.all(value >= 0.0):
            raise ValueError("{} must be non-negative!".format(self.name))
        return self.set(obj, value)
        
    def set(self, obj, value):
        """Shortcut for the setter."""
        if isinstance(value, u.Quantity):
            quantity = value
        else:
            quantity = u.Quantity(value, unit=self.unit(obj))
        if not quantity.unit.is_equivalent(self.unit(obj)):
            raise ValueError("{} must have units of {}".format(quantity, self.unit(obj)))
        return setattr(obj, self._attr, quantity.to(self._unit).value)
        
    @descriptor__get__
    def __get__(self, obj, objtype):
        """Get the property's value"""
        return self.get(obj)
        
    def get(self, obj):
        """Shortcut get method."""
        value = getattr(obj, self._attr)
        bases = self.bases(obj)
        if bases is not None:
            recomposed = self.recompose(value * self._unit, bases)
            return recomposed
        return value
        

class ComputedUnitsProperty(UnitsProperty):
    """A units property computed from source."""
    def __init__(self, fget=None, unit=None, **kwargs):
        super(ComputedUnitsProperty, self).__init__(fget.__name__, None, **kwargs)
        self.fget = fget
        
    def __call__(self, fget):
        """Use descriptor twice."""
        if self.fget is not None:
            raise ValueError("Can't set getter twice: {}".format(self.fget))
        self.fget = fget
        return self
        
    @descriptor__get__
    def __get__(self, obj, objtype):
        """Getter which calls the property function."""
        value = self.fget(obj)
        if isinstance(value, u.Quantity):
            return self.recompose(value, self.bases(obj))
        else:
            return value
        
class HasUnitsProperties(object):
    """Mixin for objects with UnitsProperty"""
    
    _bases_index = {}
    _active_base = None
    
    def _list_attributes(self, klass):
        """Generate attributes matching a certain class."""
        for element in dir(type(self)):
            attr = getattr(type(self), element)
            if isinstance(attr, klass):
                yield element
    
    def _get_attr_by_name(self, name):
        """Get an attribute by its full name."""
        for element in dir(type(self)):
            attr = getattr(type(self), element)
            if isinstance(attr, UnitsProperty):
                if attr.name == name:
                    return attr
        raise AttributeError("{} has no property named {}".format(self, name))
        
    def get_unit(self, attr=None, name=None):
        """Get a property unit by attribute name or full name."""
        if attr is None:
            attr = self._get_attr_by_name(name)
        else:
            attr = getattr(type(self), attr)
        return attr.recomposed_unit(self)
        
    def list_units_properties(self):
        """List all of the unit properties"""
        return self._list_attributes(UnitsProperty)
        
    def add_bases(self, name, bases):
        """Add a base set."""
        self._bases_index[name] = set(bases)
        
    @contextlib.contextmanager
    def bases(self, name):
        """Use this object in a specified base."""
        self._active_base, previous_base = name, self._active_base
        yield self
        self._active_base = previous_base
        
    @property
    def _bases(self):
        """Get the current unit base set."""
        if self._active_base is None:
            return None
        return self._bases_index[self._active_base]
