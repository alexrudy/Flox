# -*- coding: utf-8 -*-
# 
#  packet.py
#  flox
#  
#  Created by Alexander Rudy on 2014-05-09.
#  Copyright 2014 University of California. All rights reserved.
# 

from __future__ import (absolute_import, unicode_literals, division, print_function)

import numpy as np
import abc
import six


class Packet(dict):
    """A 2D packet, containing the data required for the setup of an evolver."""
    
    def __init__(self):
        super(Packet, self).__init__()
        
    def __setitem__(self, key, value):
        """Set the item, converting it to a numpy array in the process."""
        super(Packet, self).__setitem__(key, np.asanyarray(value))
        
@six.add_metaclass(abc.ABCMeta)
class PacketInterface(object):
    """The interface for packet consumers and producers."""
    
    @abc.abstractmethod
    def get_packet_list(self):
        """Return the parameter list."""
        return []
    
    def create_packet(self):
        """Create a packet from the LinearEvolver state."""
        packet = Packet()
        for variable in self.get_packet_list():
            packet[variable] = getattr(self, variable)
        return packet
        
    def read_packet(self, packet):
        """Read an imcoming packet"""
        for variable in packet.keys():
            setattr(self, variable, packet[variable])
