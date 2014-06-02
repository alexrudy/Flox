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
import astropy.units as u
import logging

log = logging.getLogger(__name__)

class InitialConditioner(object):
    """A class which builds up initial conditions."""
    def __init__(self, params):
        super(InitialConditioner, self).__init__()
        self.params = params
        
    def run(self, System):
        """Apply the initial conditions to a system."""
        self.stable(System)
        self.forcing(System)
        self.sin(System)
        
    def forcing(self, System):
        """Thermal forcing intial condition."""
        if self.params.get('thermal.enable', False):
            log.info("Adding Thermal Forcing IC: {!r}".format(self.params.get('thermal',{})))
            log.debug("Adding Stable Region: [{} to {}]".format(System.fzm, System.fzp))
            
            System.Temperature.raw[:,0] += System._T_Stability()
            if System.deltaT > 0.0:
                log.info("Adding unstable regions: [{} to {}] and [{} to {}]".format(0.0 * u.m, System.fzm, System.fzp, System.depth))
                dTdz = (System.fTm) / System.fzm
                System.Temperature.raw[:System.fzmi,0] += System.nondimensionalize(dTdz * System.z[:System.fzmi])
                dTdz = (System.deltaT - System.fTp) / (System.depth - System.fzp)
                System.Temperature.raw[System.fzpi:,0] += System.nondimensionalize(dTdz * (System.z[System.fzpi:] - System.fzp) + System.fTp)
            elif System.deltaT < 0.0:
                log.info("Adding unstable regions: [{} to {}] and [{} to {}]".format(0.0 * u.m, System.fzm, System.fzp, System.depth))
                dTdz = (System.fTm + System.deltaT) / System.fzm
                System.Temperature.raw[:System.fzmi,0] += System.nondimensionalize(dTdz * System.z[:System.fzmi] + System.deltaT)
                dTdz = (-System.fTp) / (System.depth - System.fzp)
                System.Temperature.raw[System.fzpi:,0] += System.nondimensionalize(dTdz * (System.z[System.fzpi:] - System.fzp) + System.fTp)
            else:
                log.warning("No unstable region added, DeltaT={}".format(System.detlaT))
        
    def stable(self, System):
        """Apply the stable perturbation."""
        if self.params.get('stable',False):
            log.info("Adding stable background")
            if System.deltaT > 0.0:
                System.Temperature.raw[:,0] += System.nondimensionalize(System.z * System.deltaT/System.depth)
            elif System.deltaT < 0.0:
                System.Temperature.raw[:,0] += System.nondimensionalize(System.z * System.deltaT/System.depth - System.deltaT)
    
    def sin(self, System):
        """Sin perturbations in each mode."""
        if self.params.get('sin.enable', False):
            log.info("Adding sin(πz) perturbations")            
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
            log.debug("A*sin(πz)={}".format(amp_mode))
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
                    log.debug("Adding to mode k={} with A={} and l={}".format(k, eps, l))
                    System.Temperature.raw[:,k] += eps * np.sin(l * np.pi * (System.z/System.depth).to(1).value)



