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

from queue import Queue, Empty

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
        
    def evolve_system(self, system, total_time, chunksize=int(1e3), chunks=1000):
        """Evolve over many iterations with a given total time."""
        self.read_packet(system.create_packet())
        end_time = self.Time + system.nondimensionalize(total_time).value
        with ProgressBar(chunks) as pbar:
            for i in range(chunks):
                if self.Time >= total_time:
                    break
                else:
                    self.evolve(end_time, chunksize)
                    system.read_packet(self.create_packet())
                    pbar.update(i)
                    
    def evolve_async(self, total_time, chunksize=int(1e3), chunks=int(1e3), queue=None):
        """An event evolution tool, using a queue. The queue should be ready to recieve all
        of the read packets."""
        for i in range(chunks):
            if self.Time >= total_time:
                break
            else:
                self.evolve(total_time - self.Time, chunksize)
                queue.put(self.create_packet(), block=False)
    

