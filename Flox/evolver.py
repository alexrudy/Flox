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

from .packet import PacketInterface

from astropy.utils.console import ProgressBar


@six.add_metaclass(abc.ABCMeta)
class Evolver(PacketInterface):
    """The python side of the evolver."""
    
    def __repr__(self):
        """Represent this Linear Evolver."""
        try:
            return "<{} at Time={}>".format(self.__class__.__name__, self.time)
        except:
            return super(Evolver, self).__repr__()
        
    def evolve_many(self, system, total_time, chunksize=int(1e3), chunks=1000):
        """Evolve over many iterations with a given total time."""
        start_time = system.dimensionalize(self.time * system.nondimensional_unit(total_time.unit))
        self.read_packet(system.create_packet())
        end_time = self.time + system.nondimensionalize(total_time).value
        with ProgressBar(chunks) as pbar:
            for i in range(chunks):
                if system.time >= total_time:
                    break
                else:
                    self.evolve(end_time, chunksize)
                    system.read_packet(self.create_packet())
                    pbar.update(i)
    

