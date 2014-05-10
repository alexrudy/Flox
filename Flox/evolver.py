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

@six.add_metaclass(abc.ABCMeta)
class Evolver(object):
    """The python side of the evolver."""
    
    def __repr__(self):
        """Represent this Linear Evolver."""
        try:
            return "<{} at Time={}>".format(self.__class__.__name__, self.time)
        except:
            return super(Evolver, self).__repr__()
        
    def evolve_many(self, grids, total_time, chunksize=int(1e3), chunks=1000):
        """Evolve over many iterations with a given total time."""
        self.update_from_grids(grids)
        start_time = grids.dimensionalize(self.time * grids.nondimensional_unit(total_time.unit))
        end_time = self.time + grids.nondimensionalize(total_time).value
        with ProgressBar(chunks) as pbar:
            for i in range(chunks):
                if grids.time >= total_time:
                    break
                else:
                    self.evolve(end_time, chunksize)
                    self.to_grids(grids, grids.it+1)
                    pbar.update(i)
    
    @abc.abstractmethod
    def create_packet(self):
        """Create a packet from this evolver."""
        pass
        
    @abc.abstractmethod
    def read_packet(self, packet):
        """Read an incoming packet into this system."""
        pass
