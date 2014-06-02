# -*- coding: utf-8 -*-
# 
#  evolver.py
#  Flox
#  
#  Created by Alexander Rudy on 2014-05-09.
#  Copyright 2014 Alexander Rudy. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import numpy as np
import six
import abc
import io
import sys
import collections.abc
import logging
import functools

from astropy.utils.console import ProgressBar

from ..process.packet import PacketInterface
from ..util import callback_progressbar_wrapper

log = logging.getLogger(__name__)

@six.add_metaclass(abc.ABCMeta)
class Evolver(PacketInterface):
    """The python side of the evolver."""
    
    def __repr__(self):
        """Represent this Linear Evolver."""
        try:
            return "<{} at Time={}>".format(self.__class__.__name__, self.time)
        except:
            return super(Evolver, self).__repr__()
        
    def evolve(self, time, chunksize):
        """Information on evolution."""
        log.debug("Called evolve for t={}/n={}".format(time, chunksize))
        time_start = self.Time
        r = super(Evolver, self).evolve(time, chunksize)
        log.debug("Evolved for {}".format(self.Time-time_start))
        log.debug("Timestep currently {}".format(self.delta_time()))
        return r
        
    def evolve_system(self, system, total_time, chunksize=int(1e3), chunks=1000, quiet=False, callback=None):
        """Evolve over many iterations with a given total time."""
        log.info("Evolving {}".format(system))
        
        self.read_packet(system.create_packet())
        end_time = self.Time + system.nondimensionalize(total_time).value
        log.debug("Total evolution time: {}".format(end_time))
        
        file = None if not quiet else io.StringIO()
        callback = callback if callback is not None else lambda i,p : system.read_packet(p)
        with ProgressBar(chunks, file=file) as pbar:
            iters = self.evolve_cb(end_time, chunksize=chunksize, chunks=chunks, callback=callback_progressbar_wrapper(callback, pbar))
        return iters
    
    def evolve_cb(self, total_time, chunksize=int(1e3), chunks=int(1e3), callback=lambda i,p : None):
        """Run the evolution with a callback."""
        for i in range(chunks):
            if self.Time >= total_time:
                break
            else:
                self.evolve(total_time - self.Time, chunksize)
                callback(i, self.create_packet())
        return i
    
    def evolve_queue(self, total_time, chunksize=int(1e3), chunks=int(1e3), queue=None):
        """An event evolution tool, using a queue. The queue should be ready to receive all
        of the read packets."""
        iters = self.evolve_cb(total_time, chunksize=chunksize, chunks=chunks, callback=lambda i,p : queue.put(p))
        return iters
    
    
    def evolve_queues(self, total_time, chunksize=int(1e3), chunks=int(1e3), queues=None):
        """An event evolution tool, using queues. The queues should be ready to receive all
        of the read packets."""
        def callback(i, p):
            for q in queues:
                q.put(p)
        iters = self.evolve_cb(total_time, chunksize=chunksize, chunks=chunks, callback=callback)
        return iters


