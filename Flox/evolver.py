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
from astropy.utils.console import ProgressBar

from .process.packet import PacketInterface



@six.add_metaclass(abc.ABCMeta)
class Evolver(PacketInterface):
    """The python side of the evolver."""
    
    def __repr__(self):
        """Represent this Linear Evolver."""
        try:
            return "<{} at Time={}>".format(self.__class__.__name__, self.time)
        except:
            return super(Evolver, self).__repr__()
        
    def evolve_system(self, system, total_time, chunksize=int(1e3), chunks=1000, quiet=False):
        """Evolve over many iterations with a given total time."""
        self.read_packet(system.create_packet())
        end_time = self.Time + system.nondimensionalize(total_time).value
        file = sys.stdout if not quiet else io.StringIO()
        with ProgressBar(chunks, file=file) as pbar:
            for i in range(chunks):
                if self.Time >= end_time:
                    break
                else:
                    self.evolve(end_time, chunksize)
                    system.read_packet(self.create_packet())
                    pbar.update(i)
                    
    def evolve_async(self, total_time, chunksize=int(1e3), chunks=int(1e3), queue=None):
        """An event evolution tool, using a queue. The queue should be ready to recieve all
        of the read packets."""
        if not isinstance(queue, collections.abc.Iterable):
            queue = [queue]
        for i in range(chunks):
            if self.Time >= total_time:
                break
            else:
                self.evolve(total_time - self.Time, chunksize)
                packet = self.create_packet()
                for q in queue:
                    q.put(packet, block=False)
        return i
    

