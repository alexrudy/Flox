# -*- coding: utf-8 -*-
# 
#  ic.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-02.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

"Handle initial conditions."

from __future__ import (absolute_import, unicode_literals, division, print_function)

import numpy as np

class InitialConditioner(object):
    """A class which builds up initial conditions."""
    def __init__(self, params):
        super(InitialConditioner, self).__init__()
        self.params = params
        
    def run(self, System):
        """Apply the initial conditions to a system."""
        self.stable(System)
        self.sin(System)
        
    def stable(self, System):
        """Apply the stable perturbation."""
        if self.params.get('stable',False):
            stable_temperature_gradient(System)
    
    def sin(self, System):
        """Sin perturbations in each mode."""
        if self.params.get('sin.enable', False):
            if self.params.get('sin.limits', True):
                kmin, kmax = self.params.get('sin.k',[1,2])
                ks = range(kmin, kmax)
                lmin, lmax = self.params.get('sin.l',[1,2])
                ls = range(lmin, lmax)
            else:
                ks = self.params.get('sin.k',[1])
                ls = self.params.get('sin.l',[1])
            amplitude = self.params.get('sin.epsilon',0.5)
            amp_mode = self.params.get('sin.amplitude','fixed')
            for k in ks:
                for l in ls:
                    if amp_mode == 'fixed':
                        eps = amplitude
                    if amp_mode == 'random':
                        eps = np.random.randn(1) * amplitude
                    if amp_mode == 'powerlaw':
                        eps = k**self.params.get('sin.powerlaw',-1) * amplitude
                    if amp_mode == 'powerlaw-random':
                        eps = np.random.randn(1) * k**self.params.get('sin.powerlaw',-1) * amplitude
                    single_mode_perturbation(System, k, l, eps=eps)

def z_array(System):
    """Z position array, appropriate for initialzing."""
    return (np.arange(System.nz + 2) / (System.nz + 1))[1:-1]

def stable_temperature_gradient(System):
    """Apply the stable temperature gradient to the system."""
    if System.deltaT > 0.0:
        step = -1
    else:
        step = 1
    System.Temperature[:,0] = z_array(System)[::-1]

def single_mode_perturbation(System, k, l=1, eps=1):
    """Single mode perturbation."""
    System.Temperature[:,k] = eps * np.sin(l * np.pi * z_array(System))

