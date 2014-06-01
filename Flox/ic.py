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
        self.forcing(System)
        
    def forcing(self, System):
        """Thermal forcing intial condition."""
        if self.params.get('thermal.enable', False):
            print("Adding thermal.")
            fks = self.params.get('thermal.fks', 0.5)
            ks = int(np.fix(System.nz * fks))
            top = self.params.get('thermal.top', False)
            z = z_array(System)
            T0 = 0.5
            System.Temperature.raw[:,0] = z/z[ks]
            zp = 1 - (1 - z[ks:])/(1 - z[ks])
            System.Temperature.raw[ks:,0] = (1 - zp) + zp * T0
            
        
    def stable(self, System):
        """Apply the stable perturbation."""
        if self.params.get('stable',False):
            if System.deltaT > 0.0:
                System.Temperature.raw[:,0] = System.z / System.dz
            elif System.deltaT < 0.0:
                System.Temperature.raw[:,0] = 1.0 - (System.z / System.dz)
    
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
                    System.Temperature.raw[:,k] = eps * np.sin(l * np.pi * System.z)



