# -*- coding: utf-8 -*-
# 
#  test_setup.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-02.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

"""
Functions for setting up tests on analytical functions in evolution mode.

"""

from __future__ import (absolute_import, unicode_literals, division, print_function)

import numpy as np
import six
import abc
from six.moves import reduce

from Flox.tests.functional_forms import d_polynomial

def _second_derivative(f, f_p, f_m, dz):
    """Second derivative handler"""
    from Flox.finitedifference import second_derivative2D
    ddf = np.zeros_like(f)
    assert not second_derivative2D(f.shape[0], f.shape[1], ddf, f, dz, f_p, f_m, 1.0)
    return ddf

def second_derivative(system, fluid_componet):
    """Finite differenced second derivative of an array."""
    from Flox.finitedifference import second_derivative2D
    f = getattr(system, fluid_componet)
    ddf = np.zeros_like(f)
    f_m, f_p = getattr(system, 'b_'+fluid_componet)
    assert not second_derivative2D(f.shape[0], f.shape[1], ddf, f, system.dz, f_p, f_m, 1.0)
    return ddf

six.add_metaclass(abc.ABCMeta)
class AnalyticalSystem(object):
    """Setup an analytical system."""
    def __init__(self, nz, nx, nn=None, dz=1, dt=1, a=1, Ra=1, Pr=1):
        super(AnalyticalSystem, self).__init__()
        self.nz = nz
        self.nx = nx
        nn = nn if nn is not None else nx
        self.nn = nn
        self.dz = dz
        self.dt = dt
        self.a = a
        self.Ra = Ra
        self.Pr = Pr
        
    
    @property
    def npa(self):
        """The NPA array."""
        return self.n * np.pi / self.a
        
    def grids(self):
        """Make the grids"""
        n, z = np.meshgrid(np.arange(self.nn), np.linspace(0, (self.nz-1) * self.dz, self.nz))
        return n, z
        
    @property
    def z(self):
        """The z array"""
        n, z = self.grids()
        return z
        
    @property
    def z_i(self):
        """The z indicies"""
        return self.igrids()[1]
        
    def igrids(self):
        """docstring for igrids"""
        x_i, z_i = np.meshgrid(np.linspace(0, 1, self.nn),np.linspace(0, 1, self.nz+2)[1:-1])
        return x_i, z_i
        
    @property
    def n(self):
        """Mode number."""
        n, z = self.grids()
        return n
        
    @property
    def zp(self):
        """Positive boundary for z"""
        return (np.max(self.z) + self.dz) * np.ones((self.nn))
    
    @property
    def zm(self):
        """Negative boundray for z"""
        return (np.min(self.z) - self.dz) * np.ones((self.nn))
    
    @abc.abstractproperty
    def Temperature(self):
        """Return the analytic temperature."""
        pass
        
    @property
    def dd_Temperature(self):
        """Return an analytic second derivative of temperature."""
        return second_derivative(self, 'Temperature')
        
    @abc.abstractproperty
    def b_Temperature(self):
        """Temperature ghost point values."""
        pass
        
    @abc.abstractproperty
    def Vorticity(self):
        """Return the analytic Vorticity."""
        pass
        
    @property
    def dd_Vorticity(self):
        """Return an analytic second derivative of Vorticity."""
        return second_derivative(self, 'Vorticity')
        
    @abc.abstractproperty
    def b_Vorticity(self):
        """Vorticity ghost point values."""
        pass
        
    @abc.abstractproperty
    def Stream(self):
        """Return the analytic Stream."""
        pass
        
    @property
    def dd_Stream(self):
        """Return an analytic second derivative of Stream."""
        return second_derivative(self, 'Stream')
        
    @abc.abstractproperty
    def b_Stream(self):
        """Stream ghost point values."""
        pass
        
    @property
    def d_Temperature(self):
        """Derivative of temperature."""
        return self.npa * self.Stream + self.dd_Temperature - self.Temperature * self.npa * self.npa
        
    @property
    def d_Temperature_simple(self):
        """Simple temperature derivative."""
        return self.dd_Temperature - self.Temperature * self.npa * self.npa
    
    
    @property
    def d_Vorticity(self):
        """The derivative of vorticity, given the analytic components."""
        return (self.Ra * self.Pr * self.Temperature * self.npa) - (self.Pr * self.npa * self.npa * self.Vorticity) + self.Pr * self.dd_Vorticity
        
    
    def vorticity_from_stream(self, stream, dd_stream):
        """Compute the round-trip vorticity from the stream function."""
        return dd_stream - self.npa**2 * stream
        
    def evolved(self, variable):
        """Evolved variable"""
        return self.dt/2.0 * 3.0 * getattr(self,"d_{}".format(variable)) + getattr(self,variable)


class PolynomialSystem(AnalyticalSystem):
    """A polynomial based, analytically solved system."""
    
    _T = (0, 0, 1, 2)
    
    @property
    def Temperature(self):
        """Return the analytic temperature."""
        return d_polynomial(0, self.z, *self._T)
        
    @property
    def b_Temperature(self):
        z = np.array([self.zm, self.zp])
        return d_polynomial(0, z, *self._T)
        
    _V = (0, 3, 1, 8)
        
    @property
    def Vorticity(self):
        """Return the analytic Vorticity."""
        return d_polynomial(0, self.z, *self._V)
                
    @property
    def b_Vorticity(self):
        z = np.array([self.zm, self.zp])
        return d_polynomial(0, z, *self._V)
        
    _S = (0, -2, 0, 1)
        
    @property
    def Stream(self):
        """Return the analytic Stream."""
        return d_polynomial(0, self.z, *self._S)
        
    @property
    def b_Stream(self):
        z = np.array([self.zm, self.zp])
        return d_polynomial(0, z, *self._S)
    
class ConstantSystem(AnalyticalSystem):
    """A polynomial based, analytically solved system."""
    
    @property
    def Temperature(self):
        """Return the analytic temperature."""
        return self.z
        
    @property
    def dd_Temperature(self):
        """Return an analytic second derivative of temperature."""
        return 0
        
    @property
    def b_Temperature(self):
        return np.array([self.zm, self.zp])
        
    @property
    def Vorticity(self):
        """Return the analytic Vorticity."""
        return self.z
        
        
    @property
    def dd_Vorticity(self):
        """Return an analytic second derivative of Vorticity."""
        return 0
        
    @property
    def b_Vorticity(self):
        return np.array([self.zm, self.zp])
        
    @property
    def Stream(self):
        """Return the analytic Stream."""
        return self.z
        
    
    @property
    def b_Stream(self):
        return np.array([self.zm, self.zp])
    
    @property
    def dd_Stream(self):
        """Second derivative of the stream function"""
        return 0

class FourierSystem(AnalyticalSystem):
    """A system expanded in vertical and horizontal fourier modes."""
    
    m = 2
    
    @property
    def Temperature(self):
        """Return the analytic temperature."""
        return 3 * np.sin((self.m * np.pi)/(self.nz * self.dz) * self.z)
        
    @property
    def b_Temperature(self):
        return [np.zeros(self.nn),np.zeros(self.nn)]
        
    @property
    def Vorticity(self):
        """Return the analytic Vorticity."""
        return 2 * np.sin((self.m * np.pi)/(self.nz * self.dz) * self.z)
        
    @property
    def b_Vorticity(self):
        return [np.zeros(self.nn),np.zeros(self.nn)]
        
    @property
    def Stream(self):
        """Return the analytic Stream."""
        return 4 * np.sin((self.m * np.pi)/(self.nz * self.dz) * self.z)
        
    @property
    def b_Stream(self):
        return [np.zeros(self.nn),np.zeros(self.nn)]