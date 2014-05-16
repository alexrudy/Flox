# -*- coding: utf-8 -*-
# 
#  manager.py
#  EART275
#  
#  Created by Alexander Rudy on 2014-05-15.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import argparse
import six

import numpy as np
import numpy.random

import multiprocessing.managers as mm
import multiprocessing as mp

from pyshell.util import resolve, ipydb

from Flox.input import FloxConfiguration
from Flox.process.manager import AsynchronousManager
from Flox.process.evolver import EvolverManager
from Flox.process.queue import MultiplexedQueue


class FloxManager(object):
    """Manage Flox Instances"""
    def __init__(self):
        super(FloxManager, self).__init__()
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('configfile', type=six.text_type, help="Configuration file name.")
        self._finish = []
        
    def load_configuration(self):
        """docstring for fname"""
        self.config = FloxConfiguration.fromfile(self.opt.configfile)
        
    def finish(self):
        """Finish processes."""
        [ p() for p in self._finish ]
        
    def run(self):
        """Run this management object."""
        # ipydb()
        self.opt = self.parser.parse_args()
        self.load_configuration()
        System_config = self.config["system"]
        System_Class = resolve(System_config.pop("()"))
        System = System_Class.from_params(System_config)
        self.construct_initial_conditions(System, self.config["ic"])
        self.processes(System)
        self.finish()
        
    def construct_initial_conditions(self, System, config):
        """Construct configurable initial conditions for a simulation."""
        from .ic import stable_temperature_gradient, single_mode_linear_perturbation
        if config.get('stable',False):
            stable_temperature_gradient(System)
        if 'sin' in config:
            for m in range(config['sin'].get('mink',1),config['sin'].get('maxk',2)):
                eps = config['sin'].get('eps',5e-1)
                if config['sin'].get('random',False):
                    eps *= np.random.rand(1)
                single_mode_linear_perturbation(System, mode=m, eps=eps)
                
    def processes(self, System):
        """Construct the required processes."""
        # mp.log_to_stderr()
        SM = mm.SyncManager()
        SM.start()
        
        AM = AsynchronousManager()
        AM.name = AM.name.replace("AsynchronousManager","WritingManager")
        AM.start()
        
        EQ = SM.Queue()
        Qs = [EQ]
        if 'animate' in self.config and self.config.get('animate.enable', True):
            AQ = SM.Queue()
            Qs += [AQ]
        PSystem = AM.send(System)
        EM = self.evolve(Qs, System)
        PSystem.read_queue(EQ, timeout=60)
        if 'write' in self.config:
            PSystem.write(**self.config.get('write',{}))
        if 'animate' in self.config and self.config.get('animate.enable', True):
            AN = self.animate(AQ, System)
        EM.stop()
        AM.stop()
        AM.join()
        EM.join()
        if 'animate' in self.config and self.config.get('animate.enable', True):
            AN.stop()
            AN.join()
        SM.shutdown()
        
        
        
    def animate(self, queue, system):
        """Animate evolution"""
        from Flox.plot import MultiViewController
        AM = AsynchronousManager()
        AM.name = AM.name.replace("AsynchronousManager", "AnimationManager")
        AM.register(MultiViewController.__name__, MultiViewController.from_config)
        AM.start()
        MVC = getattr(AM, MultiViewController.__name__)(self.config['animate'].store)
        MVC.animate(queue, system, buffer_length=self.config.get('animate.buffer'), timeout=2)
        return AM
        
    def evolve(self, queue, system):
        """Evolve the system forward in time."""
        EM = EvolverManager()
        evolver = resolve(self.config['evolve.class'])
        EM.register_evolver(evolver)
        EM.start()
        EV = getattr(EM, evolver.__name__)(system, self.config.get('evolve.saftey', 0.1))
        EV.read_packet(system.create_packet())
        nd_time = system.nondimensionalize(self.config['evolve.time']).value
        EV.evolve_async(nd_time, chunks=int(self.config.get('evolve.nt',system.nt-1)), chunksize=int(self.config.get('evolve.iterations',1)), queue=queue)
        return EM
        
        