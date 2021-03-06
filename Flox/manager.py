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
import os

import multiprocessing.managers as mm
import multiprocessing as mp

from logging import getLogger
from pyshell.util import resolve, ipydb, askip
from pyshell.loggers import configure_logging
import os, os.path
import glob
from pkg_resources import resource_listdir

from .ic import InitialConditioner
from .input import FloxConfiguration
from .process.manager import AsynchronousManager
from .process.evolver import EvolverManager
from .plot import MultiViewController
from .process._threads import omp_set_num_threads, omp_get_num_threads

log = getLogger(__name__)

class FloxManager(object):
    """Manage Flox Instances"""
    def __init__(self):
        super(FloxManager, self).__init__()
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('configfile', type=six.text_type, help="Configuration file name.")
        self.parser.add_argument('-nt','--snapshots', type=int, default=0, help="Number of snapshots to record at maximum.")
        self.parser.add_argument('-mp','--multiprocess', action='store_true', help="Use only a single process, do no extra work.")
        self.parser.add_argument('-n','--no-evolve',action='store_false', dest='evolve', help="Dry run, don't actually evolve.")
        self.parser.add_argument('-d', '--debug', action='store_true', help='Enable logging and debug mode.')
        self.parser.add_argument('--movie', action='store_true', help="Create a movie at the end.")
        self.parser.add_argument('--view', action='store_true', help="View a saved simulation")
        self.parser.add_argument('--restart', action='store_true', help="Restart the simulation from a partial simulation elsewhere.")
        self.parser.add_argument('--nthreads', dest='num_threads', type=int, help="Number of threads for OpenMP.", default=1)
        
    def load_configuration(self):
        """Load the configuration for this module."""
        self.config = FloxConfiguration()
        
        for defaults in resource_listdir(__name__, "defaults"):
            self.config.load_resource(__name__, os.path.join("defaults",defaults))
        self.config.load(self.opt.configfile)
        configure_logging(self.config)
        
        if getattr(self.opt, 'snapshots', 0) > 0:
            log.info("Setting 'evolve.nt'={}".format(self.opt.snapshots))
            self.config['evolve.nt'] = self.opt.snapshots
        
        if self.opt.num_threads != 1:
            log.info("Setting 'OMP_NUM_THREADS'={}".format(self.opt.num_threads))
        omp_set_num_threads(self.opt.num_threads)
        log.debug("Got 'OMP_NUM_THREADS'={}".format(omp_get_num_threads()))
        
    def run(self):
        """Run this management object."""
        self.opt = self.parser.parse_args()
        self.load_configuration()
        if self.opt.debug:
            log.info("Set DEBUG mode.")
            ipydb()
        
        # Build the system.
        System_config = self.config["system"]
        System_Class = resolve(System_config.pop("()"))
        System = System_Class.from_params(System_config)
        log.debug("Created a system {}".format(System))
        
        if self.opt.restart:
            log.info("Restarting")
            System.read(**self.config.get('write',{}))
            log.info(System)
        else:
            log.info("Initializing")
            # Build the initial conditions
            ICs = InitialConditioner(self.config['ic'])
            ICs.run(System)
            log.info(System)
            # askip()()
        
        if self.opt.evolve:
            log.info("Evolving")
            if not self.opt.multiprocess:
                self.mo_evolve(System, debug=self.opt.debug)
            else:
                self.mp_evolve(System, debug=self.opt.debug)
        else:
            log.info("Reading")
            System.read(**self.config.get('write',{}))
        
        System.it = 0
        if self.opt.movie:
            log.info("Making Movie")
            self.mo_movie(System, debug=self.opt.debug)
        if self.opt.view:
            log.info("Viewing")
            self.mo_view(System, debug=self.opt.debug)
    
    def mo_evolve(self, System, debug=False):
        """Single process evolver!"""
        
        # Set up the evolver object.
        evolver = resolve(self.config['evolve.class'])
        EV = evolver.from_system(System, **self.config.get('evolve.settings',{}))
        EV.evolve_system(System, self.config['evolve.time'], chunks=int(self.config.get('evolve.nt',System.engine.free)), chunksize=int(self.config.get('evolve.iterations',1)), quiet=debug)
        if self.config.get('write', False) is not False:
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
        writing = self.config.get('write', False) is not False
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
            Qs = []
            # Set up writing
            WM.start()
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
            EV = getattr(EM, evolver.__name__)(System, **self.config.get('evolve.settings',{}))
            EV.read_packet(System.create_packet())
            nd_time = System.nondimensionalize(self.config['evolve.time'] + System.time).value
            EV.evolve_queues(nd_time, chunks=int(self.config.get('evolve.nt',System.engine.free)), chunksize=int(self.config.get('evolve.iterations',1)), queues=Qs)
            
            # Launch the reader
            WS.read_queue(WQ, timeout=60)
            
            # Launch the animator
            if animate:
                ASystem = copy.copy(System)
                ASystem.engine = "Flox.array.NumpyFrameEngine"
                MVC = getattr(AM, MultiViewController.__name__)(self.config['animate'].store)
                MVC.animate(System, AQ, buffer=self.config.get('animate.buffer',10), timeout=self.config.get('animate.timeout',2), **self.config.get('animate.view',{}))
            
            if writing:
                WS.write(**self.config.get('write',{}))
            
        except Exception as e:
            raise e
        finally:
            if animate:
                AM.shutdown()
            WM.shutdown()
            EM.shutdown()
            SM.shutdown()
        
        