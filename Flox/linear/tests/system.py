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


six.add_metaclass(abc.ABCMeta)
class AnalyticalSystem(object):
    """Setup an analytical system."""
    def __init__(self, nz, nx, dz=1, dt=1, a=1, Ra=1, Pr=1):
        super(AnalyticalSystem, self).__init__()
        self.nz = nz
        self.nx = nx
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
        n, z = np.meshgrid(np.arange(self.nx), np.linspace(0, (self.nz-1) * self.dz, self.nz))
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
        x_i, z_i = np.meshgrid(np.linspace(0, 1, self.nx),np.linspace(0, 1, self.nz+2)[1:-1])
        return x_i, z_i
        
    @property
    def n(self):
        """Mode number."""
        n, z = self.grids()
        return n
        
    @property
    def zp(self):
        """Positive boundary for z"""
        return np.max(self.z) + self.dz
    
    @property
    def zm(self):
        """Negative boundray for z"""
        return np.max(self.z) + self.dz
    
    @abc.abstractproperty
    def Temperature(self):
        """Return the analytic temperature."""
        pass
        
    @abc.abstractproperty
    def dd_Temperature(self):
        """Return an analytic second derivative of temperature."""
        pass
        
    @abc.abstractproperty
    def Vorticity(self):
        """Return the analytic Vorticity."""
        pass
        
    @abc.abstractproperty
    def dd_Vorticity(self):
        """Return an analytic second derivative of Vorticity."""
        pass
        
    @abc.abstractproperty
    def Stream(self):
        """Return the analytic Stream."""
        pass
        
    @abc.abstractproperty
    def dd_Stream(self):
        """Return an analytic second derivative of Stream."""
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
        return self.dt/2.0 * 3.0 * getattr(self,"d_{}".format(variable)) + getattr(self,"{}".format(variable))


class PolynomialSystem(AnalyticalSystem):
    """A polynomial based, analytically solved system."""
    
    @property
    def Temperature(self):
        """Return the analytic temperature."""
        return self.z**3 + 2 * self.z**2
        
    @property
    def dd_Temperature(self):
        """Return an analytic second derivative of temperature."""
        return 6 * self.z + 4
        
    @property
    def Vorticity(self):
        """Return the analytic Vorticity."""
        return 8 * self.z**3 - 1 * self.z**2
        
        
    @property
    def dd_Vorticity(self):
        """Return an analytic second derivative of Vorticity."""
        return 6 * 8 * self.z - 2
        
    @property
    def Stream(self):
        """Return the analytic Stream."""
        return self.z**3 - 2 * self.z
        
    @property
    def dd_Stream(self):
        """Second derivative of the stream function"""
        return 6 * self.z
    
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
    def Vorticity(self):
        """Return the analytic Vorticity."""
        return self.z
        
        
    @property
    def dd_Vorticity(self):
        """Return an analytic second derivative of Vorticity."""
        return 0
        
    @property
    def Stream(self):
        """Return the analytic Stream."""
        return self.z
        
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
    def dd_Temperature(self):
        """Return an analytic second derivative of temperature."""
        return -1 * ((self.m * np.pi)/(self.nz * self.dz))**2.0 * self.Temperature
        
    @property
    def Vorticity(self):
        """Return the analytic Vorticity."""
        return 2 * np.sin((self.m * np.pi)/(self.nz * self.dz) * self.z)
        
        
    @property
    def dd_Vorticity(self):
        """Return an analytic second derivative of Vorticity."""
        return -1 * ((self.m * np.pi)/(self.nz * self.dz))**2.0 * self.Vorticity
        
    @property
    def Stream(self):
        """Return the analytic Stream."""
        return 4 * np.sin((self.m * np.pi)/(self.nz * self.dz) * self.z)
        
    @property
    def dd_Stream(self):
        """Second derivative of the stream function"""
        return -1 * ((self.m * np.pi)/(self.nz * self.dz))**2.0 * self.Stream