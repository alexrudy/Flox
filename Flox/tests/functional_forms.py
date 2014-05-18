# -*- coding: utf-8 -*-
# 
#  functional_forms.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-17.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import numpy as np
import six

from six.moves import reduce

class FunctionalForm(object):
    """docstring for FunctionalForm"""
    def __init__(self, name, n, dx, func, args, analytic=True, ndim=1, n2=10):
        super(FunctionalForm, self).__init__()
        self.name = name
        self.dx = dx
        self.n = n
        self.func = func
        self.args = args
        self.ndim = ndim
        self.n2 = n2
        self.analytic = analytic
        
    def __repr__(self):
        """Represent this functional form."""
        return "<{name}: {self.name} {func.__name__}{self.args!r} {analytic}>".format(self=self, name=self.__class__.__name__, func=self.func,
            analytic = "analytic" if self.analytic else "finite-difference")
        
    def __getattr__(self, varname):
        if varname[-2:] == "fx":
            if self.analytic:
                return self.func(len(varname[:-2]), self.x, *self.args)
            elif len(varname[:-2]) == 0:
                return self.func(0, self.x, *self.args)
            elif len(varname[:-2]) == 1:
                return self._finite_d()
            elif len(varname[:-2]) == 2:
                return self._finite_dd()
            else:
                raise AttributeError("Can't compute derivatives above {}".format(len(varname[:-2])))
        raise AttributeError("No attribute {}".format(varname))
        
    def _finite_d(self):
        """Solve a derivative analytically."""
        d_f = np.empty_like(self.fx)
        j = 0
        d_f[j] = (self.fx[j+1] - self.f_m)/(2.0 * self.dx)
        for j in range(1,self.fx.shape[0]-1):
            d_f[j] = (self.fx[j+1] - self.fx[j-1])/(2.0 * self.dx)
        j = self.fx.shape[0]-1
        d_f[j] = (self.f_p - self.fx[j-1])/(2.0 * self.dx)
        return d_f
        
    def _finite_dd(self):
        """Solve a derivative analytically."""
        dd_f = np.empty_like(self.fx)
        j = 0
        dd_f[j] = (self.fx[j+1] - 2.0 * self.fx[j] + self.f_m)/self.dx**2
        for j in range(1,self.fx.shape[0]-1):
            dd_f[j] = (self.fx[j+1] - 2.0 * self.fx[j] + self.fx[j-1])/self.dx**2
        j = self.fx.shape[0]-1
        dd_f[j] = (self.f_p - 2.0 * self.fx[j] + self.fx[j-1])/self.dx**2
        return dd_f
        
    @property
    def x(self):
        _x = np.arange(0.0, self.n, self.dx)
        x = np.empty(_x.shape + tuple([self.n2] * (self.ndim - 1)))
        x[...] = _x[(Ellipsis,) + tuple([None] * (self.ndim - 1))]
        return x
    
    @property
    def x_p(self):
        return np.max(self.x) + self.dx
    
    @property
    def x_m(self):
        return np.min(self.x) - self.dx
    
    @property
    def f_p(self):
        """Positive boundary"""
        return self.func(0, self.x_p, *self.args)
    
    @property
    def f_m(self):
        """Positive boundary"""
        return self.func(0, self.x_m, *self.args)

def polynomial(data, *args):
    """A polynomial evaluator."""
    return d_polynomial(0, data, *args)
    
def d_polynomial(n_d, data, *args):
    """Nth Derivative of a polynomial"""
    ans = np.zeros_like(data)
    for power, coeff in enumerate(args):
        if power >= n_d:
            p_coeff = reduce(lambda x,y : x*y, [ power - i for i in range(n_d) ], 1)
            ans += coeff * np.power(data, power - n_d)
    return ans
    
def d_fourier(n_d, data, *args):
    """Fourier modes A_m"""
    ans = np.zeros_like(data)
    for mode, amp in enumerate(args):
        coeff = amp * np.power(mode * np.pi, n_d)
        ans += np.real(coeff * np.exp(1j * np.pi * mode * data))
    return ans