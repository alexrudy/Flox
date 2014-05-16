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
import copy
import numpy as np
import numpy.random

import multiprocessing.managers as mm
import multiprocessing as mp

from pyshell.util import resolve, ipydb

from .ic import InitialConditioner
from .input import FloxConfiguration
from .process.manager import AsynchronousManager
from .process.evolver import EvolverManager
from .plot import MultiViewController


class FloxManager(object):
    """Manage Flox Instances"""
    def __init__(self):
        super(FloxManager, self).__init__()
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('configfile', type=six.text_type, help="Configuration file name.")
        self.parser.add_argument('-mp','--multiprocess', action='store_true', help="Use only a single process, do no extra work.")
        self.parser.add_argument('-n','--no-evolve',action='store_false', dest='evolve', help="Dry run, don't actually evolve.")
        self.parser.add_argument('-d', '--debug', action='store_true', help='Enable logging and debug mode.')
        self.parser.add_argument('--movie', action='store_true', help="Create a movie at the end.")
        self.parser.add_argument('--view', action='store_true', help="View a saved simulation")
        self.parser.add_argument('--restart', action='store_true', help="Restart the simulation from a partial simulation elsewhere.")
        
    def load_configuration(self):
        """Load the configuration for this module."""
        self.config = FloxConfiguration.fromfile(self.opt.configfile)
        
    def run(self):
        """Run this management object."""
        self.opt = self.parser.parse_args()
        # if self.opt.debug:
        #     ipydb()
        self.load_configuration()
        
        # Build the system.
        System_config = self.config["system"]
        System_Class = resolve(System_config.pop("()"))
        System = System_Class.from_params(System_config)
        
        if self.opt.restart:
            System.read(**self.config.get('write',{}))
            print(System)
            print(System.it)
        else:
            # Build the initial conditions
            ICs = InitialConditioner(self.config['ic'])
            ICs.run(System)
        
        if self.opt.evolve:
            if not self.opt.multiprocess:
                self.mo_evolve(System, debug=self.opt.debug)
            else:
                self.mp_evolve(System, debug=self.opt.debug)
        else:
            System.read(**self.config.get('write',{}))
        
        System.it = 0
        if self.opt.movie:
            self.mo_movie(System, debug=self.opt.debug)
        if self.opt.view:
            self.mo_view(System, debug=self.opt.debug)
    
    def mo_evolve(self, System, debug=False):
        """Single process evolver!"""
        
        # Set up the evolver object.
        evolver = resolve(self.config['evolve.class'])
        EV = evolver.from_system(System)
        EV.evolve_system(System, self.config['evolve.time'], chunks=int(self.config.get('evolve.nt',System.nt-System.nit-2)), chunksize=int(self.config.get('evolve.iterations',1)))
        System.write(**self.config.get('write',{}))
    
    def mo_movie(self, System, debug=False):
        """Create a movie."""
        MVC = MultiViewController.from_config(self.config['animate'].store)
        MVC.movie(self.config.get('animate.filename','movie.mp4'), System, **self.config.get('animate.movie',{}))
    
    def mo_view(self, System, debug=False):
        """Create a movie."""
        MVC = MultiViewController.from_config(self.config['animate'].store)
        MVC.animate(System, **self.config.get('animate.view',{}))
    
    def mp_evolve(self, System, debug=False):
        """Construct the required processes."""
        if debug:
            mp.log_to_stderr()
        
        # Launch the necessary managers.
        
        # Queue Synchronization Manager
        SM = mm.SyncManager()
        
        # Evolution Manager
        EM = EvolverManager()
        evolver = resolve(self.config['evolve.class'])
        EM.register_evolver(evolver)
        
        # Output/Writing Manager
        WM = AsynchronousManager()
        WM.name = WM.name.replace("AsynchronousManager","WritingManager")
        
        # Animation/Display Manager
        animate = self.config.get('animate.enable', False)
        if animate:
            AM = AsynchronousManager()
            AM.name = AM.name.replace("AsynchronousManager", "AnimationManager")
            AM.register(MultiViewController.__name__, MultiViewController.from_config)
        
        try:
            # Start all of the processes.
            SM.start()
            EM.start()
            WM.start()
            Qs = []
            
            # Set up animation
            if animate:
                AM.start()
                AQ = SM.Queue()
                Qs.append(AQ)
            
            # Set up writing.
            WQ = SM.Queue()
            Qs.append(WQ)
            WS = WM.send(System)
            WQ.put(System.create_packet())
            # Launch the evolver.
            EV = getattr(EM, evolver.__name__)(System, self.config.get('evolve.saftey', 0.1))
            EV.read_packet(System.create_packet())
            nd_time = System.nondimensionalize(self.config['evolve.time'] + System.time).value
            EV.evolve_queues(nd_time, chunks=int(self.config.get('evolve.nt',System.nt-System.nit-2)), chunksize=int(self.config.get('evolve.iterations',1)), queues=Qs)
            
            # Launch the reader
            WS.read_queue(WQ, timeout=60)
            
            # Launch the animator
            if animate:
                ASystem = copy.copy(System)
                ASystem.engine = "Flox.array.NumpyFrameEngine"
                MVC = getattr(AM, MultiViewController.__name__)(self.config['animate'].store)
                MVC.animate(System, AQ, buffer=self.config.get('animate.buffer',10), timeout=self.config.get('animate.timeout',2), **self.config.get('animate.view',{}))
            
            WS.write(**self.config.get('write',{}))
            
        except Exception as e:
            raise e
        finally:
            if animate:
                AM.shutdown()
            WM.shutdown()
            EM.shutdown()
            SM.shutdown()
        
        